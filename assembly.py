import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import utils
from model.connect import Model as ConnectModel
from model.retrieval import Model as RetrievalModel, gaussian_mixture_loss
from model.contact import Model as ContactModel
from data.dataset import *
import copy
from pyod.models.ocsvm import OCSVM
from train_od import get_features
import argparse
import time
import yaml

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

def get_obj(node, partnet_root_dir):
    objs = node.objs

    all_verts = []
    all_faces = []
    offset = 0
    for obj in objs:
        verts, faces = utils.load_obj(f"{partnet_root_dir}/{node.partnet_shape_dir.split('/')[-1]}/objs/{obj}.obj")

        faces -= 1
        used_vs = set()
        for fidx in range(faces.shape[0]):
            face = faces[fidx]
            for vidx in range(3):
                used_vs.add(face[vidx])
        used_vs = sorted(list(used_vs))

        verts = verts[used_vs]

        vert_map = {used_vs[a]:a for a in range(len(used_vs))}

        for fidx in range(faces.shape[0]):
            for vidx in range(3):
                faces[fidx][vidx] = vert_map[faces[fidx][vidx]]

        faces += offset
        offset += verts.shape[0]

        all_verts.append(verts)
        all_faces.append(faces)

    verts = np.concatenate(all_verts, axis=0)
    faces = np.concatenate(all_faces, axis=0)

    return verts, faces

class Part():
    def __init__(self):
        pass

class Assembly():
    def __init__(self, config):
        self.__dict__.update(config) #I hate typing quotation marks

        with open(f"{self.model_contact_dir}/config.yaml", 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        model_params = config['model_params']
        self.model_contact = ContactModel(
            feature_size=13,
            edge_size=1,
            **model_params
        )
        with open(f"{self.model_retrieval_dir}/config.yaml", 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        model_params = config['model_params']
        self.model_retrieval = RetrievalModel(
            feature_size=13,
            edge_size=1,
            **model_params
        )
        with open(f"{self.model_connect_dir}/config.yaml", 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        model_params = config['model_params']
        self.model_connect = ConnectModel(
            feature_size=13,
            edge_size=1,
            **model_params
        )
        
        self.model_contact.load_state_dict(torch.load(self.model_contact_checkpoint))
        self.model_retrieval.load_state_dict(torch.load(self.model_retrieval_checkpoint))
        self.model_connect.load_state_dict(torch.load(self.model_connect_checkpoint))
        self.model_ods = {}
        for i in range(2, self.node_threshold):
            try:
                with open(f"{self.model_od_dir}/svm_{i}.pkl", 'rb') as f:
                    self.model_ods[i] = pickle.load(f)
            except:
                self.model_ods[i] = None

        self.model_retrieval.to(self.device).eval()
        self.model_contact.to(self.device).eval()
        self.model_connect.to(self.device).eval()

        assembly_dataset = SPSMDataset(data_dir = data_dir, data_root_dir = data_root_dir, task='assembly')

        all_imgs = []
        all_embeddings = []
        all_attachments = []
        all_nodes = []
        all_shapes = []

        with torch.no_grad():
            for i in range(self.test_size):
                print(f"Loading shapes {i}/{self.test_size}", end="\r")

                offset = len(all_nodes)

                boxes, edges, N, splits, vis_img, bboxes, nodes = assembly_dataset.get_data(i)
                if len(nodes) > self.node_threshold:
                    continue                                               

                for node in nodes:
                    node.sym_pair_idxs = []
                    node.sym_idxs = {}

                    node.sym_idxs['trans'] = []
                    node.sym_idxs['ref'] = []
                    node.sym_idxs['rot'] = []

                    for sym_part, sym_info in node.sym_parts:
                        if sym_info == 'trans':
                            sym_type = sym_info
                        else:
                            sym_type, sym_params = sym_info
                        if sym_type == 'refr':
                            sym_type = 'ref'
                        if sym_type == 'rotr':
                            sym_type = 'rot'
                        sym_part_idx = nodes.index(sym_part)
                        node.sym_idxs[sym_type].append(sym_part_idx + offset)
                    for syma, symb, params in node.sym_pairs:
                        idxa = node.adj.index(syma)
                        idxb = node.adj.index(symb)
                        node.sym_pair_idxs.append((idxa, idxb, params))

                shape_pc = torch.cat([vis_img[k] + bboxes[k] for k in range(len(vis_img))], dim=0)
                all_shapes.append(shape_pc.unsqueeze(0).cuda())

                boxes = boxes.to(self.device)
                if edges.shape[0] < 1:
                    continue

                edge_label = torch.zeros(len(edges)).to(self.device)
                embeddings = self.model_retrieval.predict_embedding(boxes, edges, edge_label, splits)

                all_imgs += vis_img

                all_embeddings.append(embeddings)
                all_nodes += nodes

        all_embeddings = torch.cat(all_embeddings, dim=0)
        total_rank = 0


        all_imgs = torch.stack(all_imgs, dim=0).to(self.device)
        print(f"Loaded {all_imgs.shape} parts")

        self.all_nodes = all_nodes
        self.all_embeddings = all_embeddings
        self.all_imgs = all_imgs
        self.all_shapes = all_shapes
        self.N_data = len(all_nodes)        
        self.trans_syms = []
        self.rot_syms = []
        self.ref_syms = []

    def global_idx_to_node_adj_idxs(self, idx):
        i = 0
        while i < self.N - 1 and idx >= self.cum_adj_counts[i+1]:
            i += 1

        return i, idx-self.cum_adj_counts[i]

    def update_info(self):
        self.adj_counts = [len(node.adj) for node in self.nodes]
        self.total_contacts = sum(self.adj_counts)
        self.cum_adj_counts = [sum(self.adj_counts[:i]) for i in range(len(self.nodes))]
        self.N = len(self.nodes)

        self.unattached = []
        for (i, node) in enumerate(self.nodes):
            for (j, adj) in enumerate(node.adj):
                if adj is None:
                    self.unattached.append(self.cum_adj_counts[i] + j)

        self.remaining_adjs = len(self.unattached)

        self.boxes = [node.adj_descriptor for node in self.nodes]
        self.boxes = torch.cat(self.boxes, dim=0).float().to(self.device)
        self.boxes[self.boxes.isnan()] = 0
        self.part_edges = get_node_edges(self.nodes)
        self.adj_edges = get_adj_edges(self.nodes, return_adj_count=False)
        self.all_edges = self.part_edges + self.adj_edges
        self.edges = torch.LongTensor(self.all_edges)
        self.edge_label = torch.zeros(len(self.all_edges)).to(self.device)
        self.edge_label[len(self.part_edges):] = 1
        self.dists = None

    def get_contact_probability(self, input_label, predict_idxs):
        input_label = input_label.unsqueeze(1)
        input_features = torch.cat([input_label, self.boxes], dim=1)
        p, logits = self.model_contact.predict_likelihood(input_features, self.edges, self.edge_label, torch.LongTensor([self.total_contacts]), input_label.detach(), predict_idxs)

        return p, logits

    def predict_contact(self, first_contact=None, continue_temp=2, select_temp=2):
        if first_contact is not None:
            assert first_contact in self.unattached
            self.unattached.remove(first_contact)
            target_contacts = [first_contact]
        else:
            target_contacts = []

        unattached = copy.copy(self.unattached)
        random.shuffle(unattached)

        should_continue = True
        while should_continue:
            input_label = torch.zeros(self.total_contacts).to(self.device)
            input_label[target_contacts] = 1
            
            predict_idxs = torch.LongTensor(self.unattached).unsqueeze(0)
            p, logits = self.get_contact_probability(input_label, predict_idxs)
            continue_p = float(F.softmax(p*continue_temp, dim=1)[:,1])
            if len(target_contacts) == 0 and continue_p < 0.5:
                return None
            if (0.5 < continue_p) and len(self.unattached) > 0:
                next_slot_idx = int(torch.argmax(logits))
                next_slot = self.unattached[next_slot_idx]
                self.unattached.remove(next_slot)
                target_contacts.append(next_slot)
            else:
                should_continue = False

        return target_contacts

    def get_od_score(self):
        input_label = torch.zeros(self.total_contacts).to(self.device)
        predict_idxs = torch.LongTensor(self.unattached).unsqueeze(0)
        input_label = input_label.unsqueeze(1)
        input_features = torch.cat([input_label, self.boxes], dim=1)
        h = self.model_contact.get_latent(input_features, self.edges, self.edge_label, torch.LongTensor([self.total_contacts]), input_label.detach(), predict_idxs)
        od_score = self.model_od.decision_function(h.detach().cpu().numpy())
        return od_score

    def get_retrieval_gmm(self, target_contacts):
        input_label = torch.zeros(self.total_contacts).to(self.device)
        input_label[target_contacts] = 1

        input_label = input_label.unsqueeze(1)
        input_features = torch.cat([input_label, self.boxes], dim=1)

        log_logits, mus, sigmas = self.model_retrieval.predict_retrieval(input_features, self.edges, self.edge_label, torch.LongTensor([self.total_contacts]), input_label.detach())

        return log_logits, mus, sigmas

    def compute_retrieval_distances(self, target_contacts):
        log_logits, mus, sigmas = self.get_retrieval_gmm(target_contacts)

        dists = gaussian_mixture_loss(self.all_embeddings, log_logits, mus, sigmas)
        dists = list(dists.cpu().numpy())

        dists = list(enumerate(dists))
        dists = sorted(dists, key=lambda x:x[1])
        self.dists = dists

    def compute_retrieval_order(self, sym_candidates):
        min_dist = self.dists[0][1]
        cur_dist = self.dists[0][1]
        cur_idx = 0

        candidate_idxs = []
        non_candidate_idxs = []
        while cur_dist - min_dist < 60 and cur_idx < 200: #Hardcoded for now
            next_idx = self.dists[cur_idx][0]
            if next_idx in sym_candidates:
                candidate_idxs.append((next_idx, cur_idx))
            else:
                non_candidate_idxs.append((next_idx, cur_idx))
            cur_idx += 1
            cur_dist = self.dists[cur_idx][1]
        while cur_dist - min_dist < 100 and cur_idx < 200:
            next_idx = self.dists[cur_idx][0]
            if next_idx in sym_candidates:
                candidate_idxs.append((next_idx, cur_idx))
            cur_idx += 1
            cur_dist = self.dists[cur_idx][1]

        return candidate_idxs, non_candidate_idxs

    def connect_edges(self, target_contacts, new_node):
        non_target_contacts = [i for i in self.unattached if i not in target_contacts]
        if len(non_target_contacts) > 0:
            noise_contact = [random.choice(non_target_contacts)]
        else:
            noise_contact = []

        a = len(target_contacts)
        b = len(new_node.adj)
        u_indices = [[i for _ in range(b)] for i in target_contacts]
        u_indices = torch.LongTensor([item for sublist in u_indices for item in sublist])
        v_indices = [list(range(b)) for _ in range(a)]
        v_indices = torch.LongTensor([item for sublist in v_indices for item in sublist])

        boxes_p = new_node.adj_descriptor.float().to(self.device)
        boxes_p[boxes_p.isnan()] = 0
        edges_p = get_node_edges([new_node])
        edge_label_p = torch.zeros(len(edges_p)).to(self.device)
        edges_p = torch.LongTensor(edges_p)

        boxes = self.boxes.clone()
        input_label = torch.zeros(self.total_contacts).to(self.device).unsqueeze(1)
        input_label[target_contacts] = 1
        input_label[noise_contact] = 1
        input_features = torch.cat([input_label, self.boxes], dim=1)

        output = self.model_connect.predict_likelihood(input_features, self.edges, self.edge_label, torch.LongTensor([self.total_contacts]), u_indices, v_indices, boxes_p, edges_p, edge_label_p, torch.LongTensor([len(new_node.adj)]))

        output = F.softmax(output, dim=1)[:,1]                

        attached_u = []
        attached_v = []
        output = list(output.cpu().numpy())

        output = [(i, output[i]) for i in range(len(output))]
        output = sorted(output, key=lambda x:-x[1])
        c = len(self.unattached)
        p_threshold = 1/(max(a+1,b)+0.5)
        for max_idx, p in output:
            u_idx = int(u_indices[max_idx])
            v_idx = int(v_indices[max_idx])
            if p > p_threshold and (u_idx not in attached_u) and (v_idx not in attached_v):                       
                attached_u.append(u_idx)
                attached_v.append(v_idx)
        assert len(attached_u) == len(attached_v)
        
        if len(attached_u) == len(target_contacts):
            return attached_u, attached_v
        else:
            return None, None

    def add_node(self, node, update_original_adj=True):
        new_node = copy.deepcopy(node)
        if update_original_adj:
            new_node.original_adj = new_node.adj
        new_node.adj = [None for _ in range(len(new_node.adj))]
        new_node.original_node = node
        self.nodes.append(new_node)

    def optimize(self, num_epochs = 1500):
        num_epochs = num_epochs

        n_parts = len(self.nodes)
        initial_offset = torch.randn(n_parts, 3) * 0.2
        initial_scale = torch.ones(n_parts, 3) * 1
               
        scale = initial_scale.cuda()
        offset = initial_offset.cuda()
        offset.requires_grad = True
        scale.requires_grad = True

        points_opt = torch.stack([node.points - node.box[0:3] for node in self.nodes]).cuda()

        optimizer = optim.Adam([offset, scale], lr=0.005, eps=1e-4)

        indices = []

        adj_edges = self.adj_edges

        prev_loss = 100000000000
        skip_first = False
        skip_second = False


        ods = []
        for u,v in adj_edges:

            u1, u2 = self.global_idx_to_node_adj_idxs(u)
            v1, v2 = self.global_idx_to_node_adj_idxs(v)

            od1 = self.nodes[u1].adj_dists[u2].cuda()
            od2 = self.nodes[v1].adj_dists[v2].cuda()

            od1.requires_grad = False
            od2.requires_grad = False

            ods.append((u,v,od1,od2))

        for iters in range(num_epochs):
            optimizer.zero_grad()
            if iters < 1000:
                if skip_first:
                    continue
                points_translated = points_opt + offset.unsqueeze(1)
            else:
                if skip_second:
                    continue
                if iters % 100 == 0:
                    offset.requires_grad = False
                    scale.requires_grad = True
                elif iters % 50 == 0:
                    offset.requires_grad = True
                    scale.requires_grad = False
                
                points_translated = points_opt * scale.unsqueeze(1) + offset.unsqueeze(1)

            loss = 0
            for u,v,od1,od2 in ods:

                u1, u2 = self.global_idx_to_node_adj_idxs(u)
                v1, v2 = self.global_idx_to_node_adj_idxs(v)

                p1 = points_translated[u1][self.nodes[u1].adj_idxs[u2]]
                p2 = points_translated[v1][self.nodes[v1].adj_idxs[v2]]
                dists_cur = utils.pairwise_dist(p1, p2) ** 0.5
                d1 = dists_cur.min(axis=1)[0]
                d2 = dists_cur.min(axis=0)[0]
                
                loss += ((d1-od1) ** 2).sum()
                loss += ((d2-od2) ** 2).sum()

                loss_reg =  ((1-scale) ** 4 * 100).sum()

            if iters % (50) == 0:
                print("Optimizing...")
                print(f"Iteration {iters}, loss {float(loss.item())}")

                if iters < 1000 or (iters > 1099 and iters % 100 == 0):
                    if abs(loss-prev_loss) < 1:
                        if iters < 1000:
                            skip_first = True
                        else:
                            skip_second = True
                    prev_loss = loss

            loss.backward()
            optimizer.step()

            if iters == 1000:
                offset.requires_grad=False
                for g in optimizer.param_groups:
                    g['lr'] = 0.001

        voffset = 0
        all_verts = []
        all_faces = []
        all_source = []
        for (i, node) in enumerate(self.nodes):
            verts, faces = get_obj(node, partnet_root_dir)

            verts = verts - node.box[0:3].numpy()
            verts = verts * scale[i].detach().cpu().numpy()
            verts = verts + offset[i].detach().cpu().numpy()

            faces += voffset
            voffset += verts.shape[0]

            all_verts.append(verts)
            all_faces.append(faces)
            all_source.append(node.partnet_shape_dir)

        verts = np.concatenate(all_verts, axis=0)
        faces = np.concatenate(all_faces, axis=0)
        
        return verts, faces

    def check_for_dataset_pairs(self, new_node, target_contacts):
        n_psd = new_node.partnet_shape_dir
        n_id = new_node.id
        for contact in target_contacts:
            n_idx, _ = self.global_idx_to_node_adj_idxs(contact)
            for adj in self.nodes[n_idx].original_adj:
                if n_psd == adj.partnet_shape_dir and n_id == adj.id:
                    return True
        return False

    def check_symmetry(self, target_contacts):
        all_candidates = []
        geo_checks = []
        for contact in target_contacts:
            n_idx, adj_idx = self.global_idx_to_node_adj_idxs(contact)

            node = self.nodes[n_idx]
            for a, b, edge in node.sym_pair_idxs:
                if edge == 'trans':
                    sym_type, sym_params = edge, None
                else:
                    sym_type, sym_params = edge

                if a == adj_idx:
                    if node.adj[b] is not None:
                        all_candidates += (node.adj[b].sym_idxs[sym_type])
                        geo_checks.append((node.adj[b], sym_type, sym_params))
                elif b == adj_idx:
                    if node.adj[a] is not None:
                        all_candidates += (node.adj[a].sym_idxs[sym_type])
                        geo_checks.append((node.adj[a], sym_type, sym_params))

        return list(set(all_candidates)), geo_checks

    def check_longest_path(self, nodes):
        dists = []
        degrees = []
        degrees2 = []
        for node in nodes:
            visited = []
            cur_level = [node]
            next_level = []
            dist = -1
            count = 0
            while len(visited) != len(nodes):
                count += 1
                if count > 50:
                    raise NotImplementedError
                for cur in cur_level:
                    for adj in cur.adj:
                        if (adj in nodes) and (adj not in visited) and (adj not in next_level) and (adj not in cur_level):
                            next_level.append(adj)
                visited += cur_level
                cur_level = next_level
                next_level = []
                dist += 1
                if dist == 1:
                    assert len(visited) - 1 == len(node.adj)
                if dist == 2:
                    two_hop = len(visited) - 1
            dists.append(dist)
            assert dist > 0
            if dist < 2:
                two_hop = len(nodes)
            degrees.append(len(node.adj))
            degrees2.append(two_hop)
        return dists, degrees, degrees2

        
    def synth(self, idx, start_node=None):
        all_nodes = self.all_nodes
        self.idx = idx
        self.nodes = []

        if start_node is None:
            start_node = random.choice(all_nodes)

        self.add_node(start_node)
        self.update_info()


        try:
            self.failure = False
            self.od_score = 9999999
            with torch.no_grad():
                while self.remaining_adjs > 0:
                    print(f"Total number of parts: {len(self.nodes)}, number of unattached slots: {self.remaining_adjs}")
                    print('---------------------')
                    if len(self.nodes) >= self.node_threshold:
                        print(f"Failure: more than {self.node_threshold} nodes")
                        self.failure = True
                        break

                    self.od_score = 0

                    target_contacts = self.predict_contact()
                    if target_contacts is None:
                        print("No viable contacts identified")
                        return None
                    
                    if self.use_symmetry:
                        sym_candidates, geo_checks = self.check_symmetry(target_contacts)
                    else:
                        sym_candidates, geo_checks = [], []
                    
                    saved_is = []
                    self.compute_retrieval_distances(target_contacts)
                    candidate_idxs, non_candidate_idxs = self.compute_retrieval_order(sym_candidates)
                    attached_u = None
                    for next_idx, rank in candidate_idxs:
                        new_node = self.all_nodes[next_idx]

                        if (self.no_duplicate or self.duplicate_last) and self.check_for_dataset_pairs(new_node, target_contacts) is True:
                            print("Retrieved an original attached part from the dataset")
                            if self.duplicate_last:
                                saved_is.append(i)
                            continue

                        attached_u, attached_v = self.connect_edges(target_contacts, new_node)
                        if attached_u is not None:
                            break


                    if attached_u is None:
                        no_sym_saves = []
                        check_pcs = []
                        if len(geo_checks) > 0:
                            for node, sym_type, sym_params in geo_checks:
                                if sym_type == 'rot':
                                    continue
                                pc = node.points.float()
                                pc -= pc.mean(dim=0)
                                if sym_type == 'ref':
                                    _, direction = sym_params
                                    www = torch.ones_like(direction).float()
                                    if direction.abs()[0] > 0.9:
                                        www[0] = -1
                                    elif direction.abs()[1] > 0.9:
                                        www[1] = -1
                                    elif direction.abs()[2] > 0.9:
                                        www[2] = -1
                                    pc = pc * www.unsqueeze(0)
                                check_pcs.append(pc)
                        if len(check_pcs) > 0:
                            check_chamfer = True
                        else:
                            check_chamfer = False

                        for next_idx, rank in non_candidate_idxs:
                            new_node = self.all_nodes[next_idx]
                            if check_chamfer:
                                min_error = 10000
                                pc = new_node.points.float()
                                pc -= pc.mean(dim=0)
                                for check_pc in check_pcs:
                                    error = utils.get_chamfer_distance(pc, check_pc)
                                    min_error = min(error, min_error)
                                if min_error > 0.02:
                                    no_sym_saves.append((next_idx, rank))
                                    continue

                            if (self.no_duplicate or self.duplicate_last) and self.check_for_dataset_pairs(new_node, target_contacts) is True:
                                
                                continue
                            attached_u, attached_v = self.connect_edges(target_contacts, new_node)
                            if attached_u is not None:
                                break
                        
                        if attached_u is None:
                            for next_idx, rank in no_sym_saves:
                                new_node = self.all_nodes[next_idx]
                                attached_u, attached_v = self.connect_edges(target_contacts, new_node)
                                if attached_u is not None:
                                    break
                    
                    if (attached_u is None) and self.duplicate_last:
                        for i in saved_is:
                            new_node = self.retrieve_node(target_contacts, nth=i) 
                            attached_u, attached_v = self.connect_edges(target_contacts, new_node)
                            if attached_u is not None:
                                break

                    if attached_u is None:
                        print("Failure, can't connect...")
                        self.failure = True
                        break

                    self.add_node(new_node)

                    for i in range(len(attached_u)):
                        u_idx = attached_u[i]
                        v_idx = attached_v[i]
                        n_idx, a_idx = self.global_idx_to_node_adj_idxs(u_idx)
                        self.nodes[n_idx].adj[a_idx] = self.nodes[-1]
                        self.nodes[-1].adj[v_idx] = self.nodes[n_idx]

                    self.update_info()

            if len(self.nodes) < 20 and not self.failure:
                dists, degrees, degrees2 = self.check_longest_path(self.nodes)
                features = get_features(self.nodes, degrees, degrees2, dists)
                if self.model_ods[len(self.nodes)] is None: #No dataset shape contains this many parts
                    self.od_score = 10000000
                    self.od_threshold = 0
                    self.failure = True
                else:
                    self.od_score = self.model_ods[len(self.nodes)].decision_function(features.unsqueeze(0).numpy())
                    self.od_threshold = self.model_ods[len(self.nodes)].threshold_
                    self.od_score = round(float(self.od_score), 2)
                    self.od_threshold = round(float(self.od_threshold), 2)
            else:
                self.od_score = 10000000
                self.od_threshold = 0
                self.failure = True

            if not self.failure:           
                shape_dirs = [node.partnet_shape_dir for node in self.nodes]
                shape_dirs_set = list(set(shape_dirs))

            if self.failure and self.skip_failure:
                return None
            for i in range(self.opt_attempts):
                try:
                    verts, faces = self.optimize()
                    break
                except Exception as e:
                    print(e)
                    print("Optimize Error")
                    if i+1 == self.opt_attempts:
                        return None
            utils.write_obj(verts, faces+1, f"{vis_dir}/{attempts}_{self.failure}_{len(self.nodes)}_{self.od_score}_{self.od_threshold}.obj")
        except:
            raise

if __name__ == "__main__":
    from datetime import datetime

    parser = argparse.ArgumentParser(description='retrieval')
    parser.add_argument('--vis-root-dir', type=str, default='assembly_outputs', metavar='N')
    parser.add_argument('--synth-category', type=str, default='chair', metavar='N')
    parser.add_argument('--data-root-dir', type=str, default="/data_hdd/part-data", metavar='N')
    parser.add_argument('--partnet-root-dir', type=str, default='/data_hdd/data_v0', metavar='N')
    parser.add_argument('--config-dir', type=str, default="config/assembly.yaml", metavar='N')
    args = parser.parse_args()
    synth_category = args.synth_category

    partnet_root_dir = args.partnet_root_dir
    data_root_dir = args.data_root_dir
    data_dir = f"graph_{synth_category}_test_final"

    vis_root_dir = args.vis_root_dir
    vis_dir = f"{vis_root_dir}/gen_{synth_category}/{str(datetime.now())}"
    utils.ensuredir(vis_dir)

    config_dir = args.config_dir
    with open(config_dir, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    assert 'model_root_dir' in config
    model_root_dir = config['model_root_dir']
    config['synth_category'] = synth_category

    if not 'model_contact_dir' in config:
        config['model_contact_dir'] = f"{model_root_dir}/{synth_category}_contact"
    if not 'model_retrieval_dir' in config:
        config['model_retrieval_dir'] = f"{model_root_dir}/{synth_category}_retrieval"
    if not 'model_connect_dir' in config:
        config['model_connect_dir'] = f"{model_root_dir}/{synth_category}_connect"
    if not 'model_od_dir' in config:
        config['model_od_dir'] = f"{model_root_dir}/{synth_category}_od"

    if not 'model_contact_checkpoint' in config:
        contact_epoch = 'best'
    config['model_contact_checkpoint'] = f"{config['model_contact_dir']}/{contact_epoch}.pt"
    if not 'model_retrieval_checkpoint' in config:
        retrieval_epoch = 'best'
    config['model_retrieval_checkpoint'] = f"{config['model_retrieval_dir']}/{retrieval_epoch}.pt"
    if not 'model_connect_checkpoint' in config:
        connect_epoch = 'best'
    config['model_connect_checkpoint'] = f"{config['model_connect_dir']}/{connect_epoch}.pt"

    training_size = 0

    idxs = []
    for (i,filename) in enumerate(Path(f"{data_root_dir}/{data_dir}").glob('*.pkl')):
        if "full" not in str(filename):
            idxs.append(int(str(filename).split("/")[-1][:-4]))
    test_set_size = len(idxs)

    with open(f"{vis_dir}/config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    if 'test_size' not in config or config['test_size'] == -1:
        config['test_size'] = test_set_size
    else:
        assert config['test_size'] <= test_set_size

    assembly = Assembly(config)

    for attempts in range(0,10000):
        print('=========================================')
        print(f"Assembly attempt {attempts}")
        assembly.synth(attempts)
        print('=========================================')
