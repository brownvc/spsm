import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import random
from pathlib import Path

def get_node_edges(nodes, offset=0):
    splits = [len(node.adj) for node in nodes]
    N = sum(splits)

    cum = offset

    edges = []
    for node in nodes:
        for i in range(len(node.adj)):
            for j in range(i+1, len(node.adj)):
                edges.append([cum+i,cum+j])
        cum += len(node.adj)
    
    assert cum == N+offset

    return edges

def get_adj_edges(nodes, offset=0, return_adj_count=True):
    cum = offset
    cum_adj_count = []
    for node in nodes:
        cum_adj_count.append(cum)
        cum += len(node.adj)

    node_to_idx = {node:idx for idx, node in enumerate(nodes)}
    adj_edges = []
    for (i, node) in enumerate(nodes):
        for (u_index, adj_node) in enumerate(node.adj):
            if adj_node in nodes:
                j = node_to_idx[adj_node]
                if i < j:
                    for (v_index, adj_node2) in enumerate(adj_node.adj):
                        if adj_node2 == node:
                            adj_edges.append([cum_adj_count[i] + u_index, cum_adj_count[j] + v_index])
    if return_adj_count:
        return adj_edges, cum_adj_count
    else:
        return adj_edges

def get_node_adj_boxes(node):
    boxes = torch.zeros(len(node.adj), 12)

    centered_pc = node.points - torch.from_numpy(node.box[0:3])
    scale = centered_pc.max(dim=0)[0] - centered_pc.min(dim=0)[0] + 1e-5

    for i in range(len(node.adj)):
        pts = centered_pc[node.adj_idxs[i]]
        if len(node.adj_idxs[i]) == 0:
            pts = centered_pc[0:1]*0
        mins = pts.min(dim=0)[0]
        maxs = pts.max(dim=0)[0]
        box = torch.cat([mins,maxs], dim=0)
        box_scaled = torch.cat([mins/scale,maxs/scale], dim=0)
        boxes[i, :6] = box
        boxes[i, 6:] = box_scaled
    
    return boxes

def get_nodes_adj_boxes(nodes):
    boxes = []
    for node in nodes:
        boxes.append(get_node_adj_boxes(node))
    boxes = torch.cat(boxes, dim=0)
    return boxes

def get_pc_scale(pc):
    return torch.sqrt(torch.max(torch.sum((pc - torch.mean(pc, axis=0))**2, axis=1)))

class SPSMDataset():
    def __init__(self, data_dir=None, indices=None, data_root_dir='data', task='retrieval', od=False, preload=False):
        self.indices = indices
        self.data_dir = data_dir
        self.data_root_dir = data_root_dir
        self.task = task

        idxs = []
        for (i,filename) in enumerate(Path(f"{data_root_dir}/{data_dir}").glob('*.pkl')):
            if "full" not in str(filename):
                idxs.append(int(str(filename).split("/")[-1][:-4]))
        self.idxs = sorted(idxs)
        if indices is None:
            self.indices = (0, len(idxs))
        self.od = od
        self.datas = []
        if preload:
            assert task != 'assembly' #probably can't preload that stuff
            for i in range(self.indices[0], self.indices[1]):
                print(i, end="\r")
                with open(f"{self.data_root_dir}/{self.data_dir}/{self.idxs[i]}.pkl", 'rb') as f:
                    nodes, edges = pickle.load(f)
                    self.datas.append((nodes, edges))
            print()
        self.preload = preload

    def __len__(self):
        return self.indices[1] - self.indices[0]

    def __getitem__(self, index):
        return self.get_data(index)

    def get_data(self, index):
        i = self.indices[0] + index
        if self.preload:
            nodes, edges = self.datas[i]
        else:
            if self.task == 'assembly':
                with open(f"{self.data_root_dir}/{self.data_dir}/{self.idxs[i]}_full.pkl", 'rb') as f:
                    nodes, edges = pickle.load(f)
            else:
                with open(f"{self.data_root_dir}/{self.data_dir}/{self.idxs[i]}.pkl", 'rb') as f:
                    nodes, edges = pickle.load(f)

        if self.task in ['neg', 'assembly']:
            splits = [len(node.adj) for node in nodes]
            N = sum(splits)
            splits = torch.LongTensor(splits)

            edges = torch.LongTensor(get_node_edges(nodes))
            boxes = [node.adj_descriptor for node in nodes]
            boxes = torch.cat(boxes, dim=0).float()
            boxes[boxes.isnan()] = 0

            assert boxes.shape[0] == N

            if self.task == 'neg':
                return boxes, edges, N, splits
            else:
                points_vis = [node.points - node.box[0:3] for node in nodes]
                bboxes = [node.box[0:3] for node in nodes]
                return boxes, edges, N, splits, points_vis, bboxes, nodes
        else:
            N = len(nodes)

            next_idx = random.randint(0, N-1)
            existing_count = random.randint(1, N-1)
            existing_indices = []

            node_to_idx = {node:idx for idx, node in enumerate(nodes)}
            
            candidates = [next_idx]
            while len(existing_indices) < existing_count:
                existing_indices.append(next_idx)
                candidates.remove(next_idx)

                for adj_node in nodes[next_idx].adj:
                    adj_idx = node_to_idx[adj_node]
                    if not adj_idx in (existing_indices+candidates):
                        candidates.append(adj_idx)

                next_idx = random.choice(candidates)

            to_predict = next_idx
            existing_indices = sorted(existing_indices)

            N = len(existing_indices)

            existing_nodes = [nodes[idx] for idx in existing_indices]

            ex_node_to_idx = {node:idx for idx, node in enumerate(existing_nodes)}

            existing_adj_count = [len(node.adj) for node in existing_nodes]
            total_contacts = sum(existing_adj_count)

            part_edges = get_node_edges(existing_nodes)
            
            boxes = [node.adj_descriptor for node in existing_nodes]
            boxes = torch.cat(boxes, dim=0).float()
            boxes[boxes.isnan()] = 0

            assert boxes.shape[0] == total_contacts

            adj_edges, cum_adj_count = get_adj_edges(existing_nodes)
            all_edges = part_edges + adj_edges
            edge_label = torch.zeros(len(all_edges))
            edge_label[len(part_edges):] = 1


            target_node = nodes[to_predict]
            target_contacts = []
            for (i, node) in enumerate(existing_nodes):
                if target_node in node.adj:
                    target_contacts.append(node.adj.index(target_node) + cum_adj_count[i])

            edges = torch.LongTensor(all_edges)
            N = torch.LongTensor([total_contacts])

            if self.task != 'contact':
                N_p = len(target_node.adj)
                edges_p = get_node_edges([target_node])
                boxes_p = target_node.adj_descriptor.float()
                boxes_p[boxes_p.isnan()] = 0

                edges_p = torch.LongTensor(edges_p)

            if self.task == 'retrieval':
                input_label = torch.zeros(total_contacts)
                input_label[target_contacts] = 1

                return boxes, edges, input_label, edge_label, N, boxes_p, edges_p, N_p

            elif self.task == 'contact':
                attached_nodes = sorted([item for sublist in adj_edges for item in sublist])
                assert len(list(set(attached_nodes).intersection(set(target_contacts)))) == 0

                random.shuffle(target_contacts)
                split_at = random.randint(0, len(target_contacts))
                if self.od:
                    split_at = 0

                input_target_contacts = target_contacts[:split_at]

                predict_idxs = [i for i in range(total_contacts) if i not in (attached_nodes + input_target_contacts)]
                N2 = torch.LongTensor([len(predict_idxs)])

                should_continue = torch.zeros(1).long()
                if split_at != len(target_contacts):
                    should_continue += 1
                    next_contact = predict_idxs.index(target_contacts[split_at])
                else:
                    next_contact = -1

                target_label = torch.zeros(total_contacts).long()               
                input_label = torch.zeros(total_contacts)
                input_label[input_target_contacts] = 1

                next_contact = torch.LongTensor([next_contact])
                predict_idxs = torch.LongTensor(predict_idxs)

                return boxes, edges, input_label, should_continue, next_contact, edge_label, N, predict_idxs, N2

            elif self.task == 'connect':
                attached_nodes = sorted([item for sublist in adj_edges for item in sublist])
                non_target_contacts = [i for i in range(total_contacts) if i not in (target_contacts+attached_nodes)]
                if len(non_target_contacts) > 0:
                    noise_contact = [random.choice(non_target_contacts)]
                else:
                    noise_contact = []

                input_label = torch.zeros(total_contacts)
                input_label[target_contacts] = 1
                if len(noise_contact) > 0:
                    input_label[noise_contact] = 1

                total_prev_contacts = cum_adj_count[-1] + len(existing_nodes[-1].adj)
                total_target_contacts = len(target_contacts)

                assert total_prev_contacts == total_contacts
                a = total_target_contacts
                if len(noise_contact) > 0:
                    a += len(noise_contact)
                    assert a == len(target_contacts + noise_contact)
                    assert noise_contact[0] not in target_contacts
                b = len(target_node.adj)
                n_edge_pairs = a * b
                
                u_indices = [[i for _ in range(b)] for i in (target_contacts+noise_contact)]
                v_indices = [list(range(b)) for _ in range(a)]

                u_indices = torch.LongTensor([item for sublist in u_indices for item in sublist])
                v_indices = torch.LongTensor([item for sublist in v_indices for item in sublist])
                assert u_indices.shape[0] == v_indices.shape[0] == n_edge_pairs

                target_label = torch.zeros(n_edge_pairs).long()

                c1 = -1
                for (i, node) in enumerate(existing_nodes):
                    if target_node in node.adj:
                        c1 += 1
                        c2 = target_node.adj.index(node)
                        pos_idx = b * c1 + c2

                        assert u_indices[pos_idx] == target_contacts[c1]
                        assert v_indices[pos_idx] == c2

                        target_label[pos_idx] = 1

                if len(noise_contact) > 0:
                    assert c1+1+len(noise_contact) == a
                else:
                    assert c1+1 == a

                return boxes, edges, target_label, edge_label, N, u_indices, v_indices, boxes_p, edges_p, N_p, input_label


def collate_data_retrieval(batch):
    boxes = torch.cat([item[0] for item in batch])
    N = torch.LongTensor([item[4] for item in batch])
    cum = [N[:i].sum() for i in range(len(batch)+1)]
    edges = torch.cat([batch[i][1] + cum[i] for i in range(len(batch))], dim=0)

    input_label = torch.cat([item[2] for item in batch], dim=0)
    edge_label = torch.cat([item[3] for item in batch], dim=0)

    N_p = torch.LongTensor([item[7] for item in batch])
    boxes_p = torch.cat([item[5] for item in batch])
    cum_p = [N_p[:i].sum() for i in range(len(batch)+1)]
    edges_p = torch.cat([batch[i][6] + cum_p[i] for i in range(len(batch))], dim=0)

    return boxes, edges, input_label, edge_label, N, boxes_p, edges_p, N_p
        
def collate_data_neg(batch):
    boxes = torch.cat([item[0] for item in batch])
    N = torch.LongTensor([item[2] for item in batch])
    cum = [N[:i].sum() for i in range(len(batch)+1)]
    edges = torch.cat([batch[i][1] + cum[i] for i in range(len(batch))], dim=0)

    splits = torch.cat([item[3] for item in batch])

    return boxes, edges, N, splits


def collate_data_contact(batch):
    boxes = torch.cat([item[0] for item in batch])
    N = torch.LongTensor([item[6] for item in batch])
    cum = [N[:i].sum() for i in range(len(batch)+1)]
    edges = torch.cat([batch[i][1] + cum[i] for i in range(len(batch))], dim=0)

    input_label = torch.cat([item[2] for item in batch], dim=0)
    should_continue = torch.cat([item[3] for item in batch], dim=0)
    next_contact = torch.cat([item[4] for item in batch], dim=0)
    edge_label = torch.cat([item[5] for item in batch], dim=0)
    predict_idxs = torch.cat([batch[i][7] + cum[i] for i in range(len(batch))], dim=0)
    N2 = torch.LongTensor([item[8] for item in batch])

    return boxes, edges, input_label, should_continue, next_contact, edge_label, N, predict_idxs, N2

def collate_data_connect(batch):
    boxes = torch.cat([item[0] for item in batch])
    N = torch.LongTensor([item[4] for item in batch])
    cum = [N[:i].sum() for i in range(len(batch)+1)]
    edges = torch.cat([batch[i][1] + cum[i] for i in range(len(batch))], dim=0)

    target_label = torch.cat([item[2] for item in batch], dim=0)
    edge_label = torch.cat([item[3] for item in batch], dim=0)

    u_indices = torch.cat([batch[i][5] + cum[i] for i in range(len(batch))], dim=0)

    N_p = torch.LongTensor([item[9] for item in batch])
    boxes_p = torch.cat([item[7] for item in batch])
    cum_p = [N_p[:i].sum() for i in range(len(batch)+1)]
    edges_p = torch.cat([batch[i][8] + cum_p[i] for i in range(len(batch))], dim=0)
    v_indices = torch.cat([batch[i][6] + cum_p[i] for i in range(len(batch))], dim=0)
    input_label = torch.cat([item[10] for item in batch], dim=0)

    return boxes, edges, target_label, edge_label, N, u_indices, v_indices, boxes_p, edges_p, N_p, input_label