import random
import os
import copy
import pickle
import json
import utils
from pathlib import Path
import numpy as np
import torch
import trimesh
from pyquaternion import Quaternion
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from tqdm import tqdm
from multiprocessing import Pool


class Part():
    def __init__(self):
        pass

def get_mesh(objs, in_dir):
    v = []; f = []; vid = 0;
    for item in objs:
        vv, ff = utils.load_obj(os.path.join(in_dir, 'objs', item+'.obj'))
        ff += vid
        v.append(vv)
        f.append(ff)
        vid += vv.shape[0]
    v = np.vstack(v)
    f = np.vstack(f)
    return v, f

def sample_points(v, f, num_points=3000):
    mesh = trimesh.Trimesh(vertices=v, faces=f-1)
    np.random.seed(42)
    pc, __ = trimesh.sample.sample_surface(mesh=mesh, count=num_points)
    return pc

def get_node_adj_boxes(node):
    boxes = []

    scale = (node.bbox[0:3] - node.bbox[3:6] + 1e-5) / 2

    for i in range(len(node.adj)):
        pts = node.centered_pc[node.adj_idxs[i]]
        if len(node.adj_idxs[i]) == 0:
            raise Exception("NO ADJ IDX")
            pts = node.centered_pc[0:1]*0
        mins = pts.min(dim=0)[0]
        maxs = pts.max(dim=0)[0]
        box = torch.cat([mins,maxs], dim=0)
        box_scaled = torch.cat([mins/scale,maxs/scale], dim=0)
        new_box = torch.zeros(12)
        new_box[:6] = box
        new_box[6:] = box_scaled
        boxes.append(new_box)
    
    return boxes

def get_nodes_adj_boxes(nodes):
    boxes = []
    for node in nodes:
        boxes.append(get_node_adj_boxes(node))
    boxes = torch.cat(boxes, dim=0)
    return boxes

def pairwise_dist(p1, p2):
    return ((p1**2).sum(dim=1).view(-1,1) + (p2**2).sum(dim=1).view(1,-1) - 2 * p1@p2.t())**0.5

def get_obj(objs, partnet_shape_dir):
    all_verts = []
    all_faces = []
    offset = 0
    for obj in objs:
        verts, faces = utils.load_obj(f"{partnet_shape_dir}/objs/{obj}.obj")

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

###Symmetry checking codes, taken from StructureNet
def get_pc_scale(pc):
    return torch.sqrt(torch.max(torch.sum((pc - torch.mean(pc, axis=0))**2, axis=1)))

def get_pc_center(pc):
    return torch.mean(pc, axis=0)

def compute_trans_sym(pc1, pc2):
    pc1_center = get_pc_center(pc1)
    pc2_center = get_pc_center(pc2)
    trans = pc2_center - pc1_center
    new_pc1 = atob_trans_sym(pc1, trans)
    error = utils.get_chamfer_distance(new_pc1, pc2)
    return error, trans

def atob_trans_sym(pc, trans):
    return pc + trans

def get_chamfer_distance_numpy(pc1, pc2):
    return utils.get_chamfer_distance(torch.from_numpy(pc1), torch.from_numpy(pc2)).numpy()

def compute_ref_sym(pc1, pc2):
    pc1_center = get_pc_center(pc1)
    pc2_center = get_pc_center(pc2)
    mid_pt = (pc1_center + pc2_center) / 2
    trans = pc2_center - pc1_center
    direction = trans / np.linalg.norm(trans.numpy())
    if direction.abs().max() < 0.9:
        return 10000, None, None
    new_pc1 = atob_ref_sym(pc1, mid_pt, direction)
    error = utils.get_chamfer_distance(new_pc1, pc2)
    return error, mid_pt, direction

def atob_ref_sym(pc, mid_pt, direction):
    #I got lazy
    pc = pc.numpy()
    pc2  = np.tile(np.expand_dims(np.matmul((mid_pt - pc), direction) * 2, axis=-1), [1, 3]) * \
            np.tile(np.expand_dims(direction, axis=0), [pc.shape[0], 1]) + pc
    return torch.from_numpy(pc2)

def compute_params(pc1_center, pc2_center, pc1_v1, pc1_v2, pc2_v1, pc2_v2):
    mid_v1 = (pc1_v1 + pc2_v1) / 2
    nor_v1 = pc1_v1 - pc2_v1
    nor_v1_len = np.linalg.norm(nor_v1)
    if nor_v1_len < 1e-6:
        return np.zeros((3), dtype=np.float32), np.zeros((3), dtype=np.float32), 0.0
    nor_v1 /= nor_v1_len
    
    mid_v2 = (pc1_v2 + pc2_v2) / 2
    nor_v2 = pc1_v2 - pc2_v2
    nor_v2_len = np.linalg.norm(nor_v2)
    if nor_v2_len < 1e-6:
        return np.zeros((3), dtype=np.float32), np.zeros((3), dtype=np.float32), 0.0
    nor_v2 /= nor_v2_len

    # compute the axis direction
    nor = np.cross(nor_v1, nor_v2)
    nor_len = np.linalg.norm(nor)
    if nor_len < 1e-6:
        return np.zeros((3), dtype=np.float32), np.zeros((3), dtype=np.float32), 0.0
    nor /= nor_len

    # compute one pivot point (any point along the axis is good)
    A = np.array([[nor_v1[0], nor_v1[1], nor_v1[2]], \
                  [nor_v2[0], nor_v2[1], nor_v2[2]], \
                  [nor[0], nor[1], nor[2]]], dtype=np.float32)
    b = np.array([np.dot(nor_v1, mid_v1), np.dot(nor_v2, mid_v2), np.dot(nor, mid_v1)])
    pt = np.matmul(np.linalg.inv(A), b)

    # compute rotation angle
    tv1 = pc1_center - pt - nor * np.dot(pc1_center - pt, nor)
    tv2 = pc2_center - pt - nor * np.dot(pc2_center - pt, nor)
    c = np.dot(tv1, tv2) / (np.linalg.norm(tv1) * np.linalg.norm(tv2))
    c = np.clip(c, -1.0, 1.0)
    angle = np.arccos(c)

    return pt, nor, angle

def get_pca_axes(pc):
    axes = PCA(n_components=3).fit(pc).components_
    return axes

def check_valid_rot(pt, nor, angle):
    if np.abs(nor)[1] < 0.9:
        return False
    if np.abs(pt)[0] > 0.2:
        return False
    if np.abs(pt)[2] > 0.2:
        return False
    if not 1.47 < abs(angle) < 1.67:
        return False
    return True

def compute_rot_sym(pc1, pc2):
    pc1_center = get_pc_center(pc1).numpy()
    pc2_center = get_pc_center(pc2).numpy()
    pc1_axes = get_pca_axes(pc1)
    pc2_axes = get_pca_axes(pc2)
    pc1 = pc1.numpy()
    pc2 = pc2.numpy()

    min_error = 1e8; min_pt = None; min_nor = None; min_angle = None;
    for axe_id in range(3):
        pc1_axis1 = pc1_axes[axe_id]
        pc1_axis2 = pc1_axes[(axe_id+1)%3]
        pc2_axis1 = pc2_axes[axe_id]
        pc2_axis2 = pc2_axes[(axe_id+1)%3]

        pt, nor, angle = compute_params(pc1_center, pc2_center, pc1_center + pc1_axis1, pc1_center + pc1_axis2, pc2_center + pc2_axis1, pc2_center + pc2_axis2)
        if check_valid_rot(pt, nor, angle):
            new_pc1 = atob_rot_sym(pc1, pt, nor, angle)
            error = get_chamfer_distance_numpy(new_pc1, pc2)
            if error < min_error:
                min_error = error; min_pt = pt; min_nor = nor; min_angle = angle;

        pt, nor, angle = compute_params(pc1_center, pc2_center, pc1_center + pc1_axis1, pc1_center + pc1_axis2, pc2_center - pc2_axis1, pc2_center + pc2_axis2)
        if check_valid_rot(pt, nor, angle):
            new_pc1 = atob_rot_sym(pc1, pt, nor, angle)
            error = get_chamfer_distance_numpy(new_pc1, pc2)
            if error < min_error:
                min_error = error; min_pt = pt; min_nor = nor; min_angle = angle;

        pt, nor, angle = compute_params(pc1_center, pc2_center, pc1_center + pc1_axis1, pc1_center + pc1_axis2, pc2_center + pc2_axis1, pc2_center - pc2_axis2)
        if check_valid_rot(pt, nor, angle):
            new_pc1 = atob_rot_sym(pc1, pt, nor, angle)
            error = get_chamfer_distance_numpy(new_pc1, pc2)
            if error < min_error:
                min_error = error; min_pt = pt; min_nor = nor; min_angle = angle;

        pt, nor, angle = compute_params(pc1_center, pc2_center, pc1_center + pc1_axis1, pc1_center + pc1_axis2, pc2_center - pc2_axis1, pc2_center - pc2_axis2)
        if check_valid_rot(pt, nor, angle):
            new_pc1 = atob_rot_sym(pc1, pt, nor, angle)
            error = get_chamfer_distance_numpy(new_pc1, pc2)
            if error < min_error:
                min_error = error; min_pt = pt; min_nor = nor; min_angle = angle;

    return min_error, min_pt, min_nor, min_angle

def atob_rot_sym(pc, pt, nor, angle):
    s = np.sin(angle); c = np.cos(angle); nx = nor[0]; ny = nor[1]; nz = nor[2];
    rotmat = np.array([[c + (1 - c) * nx * nx, (1 - c) * nx * ny - s * nz, (1 - c) * nx * nz + s * ny], \
                       [(1 - c) * nx * ny + s * nz, c + (1 - c) * ny * ny, (1 - c) * ny * nz - s * nx], \
                       [(1 - c) * nx * nz - s * ny, (1 - c) * ny * nz + s * nx, c + (1 - c) * nz * nz]], dtype=np.float32)
    return np.matmul(rotmat, (pc - pt).T).T + pt
######################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='How to Connect')
    parser.add_argument('--partnet-dir', type=str, default="/data_hdd/data_v0", metavar='N')
    parser.add_argument('--category', type=str, default='chair', metavar='N')
    parser.add_argument('--split', type=str, default='train', metavar='N')
    parser.add_argument('--save-dir', type=str, default=f'data', metavar='N')
    parser.add_argument('--split-info-dir', type=str, default=f'/data_hdd/data_v0/partnet_dataset/stats/train_val_test_split/', metavar='N')
    args = parser.parse_args()

    category = args.category
    split = args.split
    save_dir = f"{args.save_dir}/graph_{category}_{split}_final"
    part_data_dir = args.partnet_dir
    split_info_dir = args.split_info_dir
    utils.ensuredir(save_dir)

    #skip existing parsed data
    skip_existing = True
    detect_bad_part = True
    #save full version including pc and other stuff, used at test time
    save_data = True
    #save only the slot information for training, used at training time
    save_simple = True
    #edge pruning
    remove_edges = True

    split_name = category.title() if "stor" not in category else "StorageFurniture"
    with open(f"{split_info_dir}/{split_name}.{split}.json", 'rb') as f:
        files = json.load(f)


    count = 0
    with torch.no_grad():
        def parse(i):
            shape_id = files[i]['anno_id']

            partnet_shape_dir = f"{part_data_dir}/{shape_id}"

            if skip_existing:
                if os.path.exists(f"{save_dir}/{shape_id}_full.pkl"):
                    return

            with open(f"{partnet_shape_dir}/result_after_merging.json") as f:
                part_json = json.load(f)

            assert len(part_json) == 1

            root_node = part_json[0]
            parts = []

            missing_parts = [False] #lazy unprofessional hack
            def parse_partnet_level(shape):
                if 'children' not in shape:
                    parts.append(shape)
                else:
                    if 'level' not in shape:
                        shape['level'] = 0

                    objs_cur = shape['objs']
                    objs_child = []
                    for child in shape['children']:
                        objs_child += (child['objs'])
                        child['level'] = shape['level'] + 1
                    
                    if len(objs_cur) == len(objs_child) and all(obj in objs_child for obj in objs_cur):
                        for child in shape['children']:
                            child['parent_name'] = f"{shape['name']}_{shape['ori_id']}"

                            child['parent_level'] = shape['level']
                            if len(shape['children']) == 1 and shape['level'] > 1:
                                child['parent_name'] = shape['parent_name']
                                child['parent_level'] = shape['level'] - 1
                            parse_partnet_level(child)
                    else:
                        if category in ["lamp", "storage"] and shape['level'] > 0:                                    
                            #print("MISSING PARTS BUT IT's LAMP(rage) SURPIRSE!")
                            parts.append(shape)
                        else:
                            missing_parts[0] = True

            parse_partnet_level(root_node)
            if missing_parts[0]:
                #print("MISSING PARTS")
                return

            if len(parts) <=1:
                return

            num_children = len(parts)

            # get all part meshes and pc
            children_v = dict(); children_f = dict(); children_pc = dict();
            bad_part = False
            for i in range(num_children):
                id_i = parts[i]['id']
                v, f = get_mesh(parts[i]['objs'], partnet_shape_dir)
                children_v[id_i] = v; children_f[id_i] = f;
                children_pc[id_i] = torch.from_numpy(sample_points(v, f))

                pc = children_pc[id_i]
                if detect_bad_part:
                    clustering = DBSCAN(eps=0.3).fit(pc.numpy())
                    label = clustering.labels_
                    clusters = np.unique(clustering.labels_)
                    n_clusters = clusters.shape[0]
                    if n_clusters > 1:
                        for cluster in list(clusters):
                            n_pts = np.where(label==cluster)[0].shape[0]
                            if n_pts <= 3000/(n_clusters*2):
                                #print("BAD PART")
                                return

            if bad_part:
                #print("BAD PART")
                return

            parts_orig = parts
            parts = []
            cuboids = []
            
            graph_edges = []  

            part_dict = {}
            for part in parts_orig:
                new_part = Part()
                new_part.parent = None
                new_part.id = part['id']
                new_part.idx = None
                new_part.label = part['name']
                new_part.adj = []
                new_part.children = []
                new_part.sym_source = []
                new_part.sym_dest = []
                new_part.sym_pairs = []
                new_part.parent_name = part['parent_name']
                new_part.parent_level = part['parent_level']
                objs = part['objs']
                new_part.objs = objs
                new_part.partnet_shape_dir = partnet_shape_dir

                id_ = part['id']
                new_part.points = children_pc[id_].cpu()
                verts = children_v[id_]
                faces = children_f[id_]
                faces = faces - 1
                vs = torch.from_numpy(verts[faces])
                ab = vs[:,1] - vs[:,0]
                ac = vs[:,2] - vs[:,0]
                areas = torch.cross(ab, ac, dim=1).norm(dim=1)
                new_part.area = areas.sum()
                new_part.box_original = None
                new_part.box = new_part.points.mean(dim=0)
                new_part.centered_pc = new_part.points - new_part.box[0:3]
                new_part.bbox = torch.zeros(6)
                new_part.bbox[0:3] = new_part.centered_pc.max(dim=0)[0]
                new_part.bbox[3:6] = new_part.centered_pc.min(dim=0)[0]
                new_part.adj_idxs = []
                new_part.adj_dists = []
                new_part.adj_normals = []
                new_part.sym_type = None
                new_part.sym_parts = []
                new_part.sym_pairs = []
                new_part.secondary_sym_children = []

                
                parts.append(new_part)

                part_dict[new_part.id] = new_part

            if len(parts) > 30:
                return
            assert len(parts) == num_children
            found_edges = []

            adj_dist = [[None for _ in range(num_children)] for _ in range(num_children)]

            def create_adj(a,b,dists,threshold,scale):
                if dists is None:
                    dists = pairwise_dist(a.points, b.points)
                if scale is None:
                    pc1_scale = utils.get_pc_scale(a.points)
                    pc2_scale = utils.get_pc_scale(b.points)
                    scale = (pc1_scale + pc2_scale) / 2
                a.adj.append(b)
                b.adj.append(a)

                d2, didxs = dists.min(axis=1)
                d3 = d2.clone()
                d2 = d2 / scale
                idxs = torch.where(d2<threshold*1.5)[0]
                idxs = list(idxs.numpy())
                a.adj_idxs.append(idxs)
                a.adj_dists.append(d3[idxs])

                d2, didxs = dists.min(axis=0)
                d3 = d2.clone()
                d2 = d2 / scale
                idxs = torch.where(d2<threshold*1.5)[0]
                idxs = list(idxs.numpy())
                b.adj_idxs.append(idxs)
                b.adj_dists.append(d3[idxs])

            for i in range(1, num_children):
                for j in range(i):
                    a = parts[i]
                    b = parts[j]
                    pc1 = a.points
                    pc2 = b.points
                    pc1_scale = utils.get_pc_scale(pc1)
                    pc2_scale = utils.get_pc_scale(pc2)
                    scale = (pc1_scale + pc2_scale) / 2

                    dists = pairwise_dist(pc1, pc2)
                    min_d = torch.min(dists)
                    min_d /= scale

                    adj_dist[i][j] = min_d
                    adj_dist[j][i] = min_d
                    if min_d < 0.05:
                        create_adj(a,b,dists,0.05,scale)

            sym_groupings = []

            for i in range(1, num_children):
                for j in range(i):
                    a = parts[i]
                    b = parts[j]
                    if a.label != b.label:
                        continue
                    a_groups = []
                    b_groups = []
                    for group, groupsym in sym_groupings:
                        if a in group and groupsym == 'trans':
                            a_groups.append(group)
                        if b in group and groupsym == 'trans':
                            b_groups.append(group)
                    assert len(a_groups) < 2
                    assert len(b_groups) < 2
                    if len(a_groups) == 1 and len(b_groups) == 1: #either same group or not symmetrical
                        if a_groups[0] == b_groups[0]:
                            continue
                    error, trans = compute_trans_sym(a.points, b.points)
                    if error < 0.05:
                        if len(a_groups) == 1 and len(b_groups) == 1:
                            a_groups[0] += b_groups[0]
                            b_groups[0].clear()
                        if len(a_groups) > 0:
                            a_groups[0].append(b)
                        elif len(b_groups) > 0:
                            b_groups[0].append(a)
                        else:
                            sym_groupings.append([[a,b], 'trans'])

            ref_params = [[None for _ in range(num_children)] for _ in range(num_children)]
            for i in range(1, num_children):
                for j in range(i):
                    a = parts[i]
                    b = parts[j]
                    if a.label != b.label:
                        continue                            
                    a_groups = []
                    b_groups = []
                    in_trans_group = False
                    for group, groupsym in sym_groupings:
                        if a in group and b in group and groupsym == 'trans':
                            in_trans_group = True
                            break
                        if a in group and groupsym == 'ref':
                            a_groups.append(group)
                        if b in group and groupsym == 'ref':
                            b_groups.append(group)
                    if in_trans_group:
                        continue
                    assert len(a_groups) < 2
                    assert len(b_groups) < 2
                    if len(a_groups) == 1 and len(b_groups) == 1: #either same group or not symmetrical
                        if a_groups[0] == b_groups[0]:
                            continue
                    error, mid_pt, direction = compute_ref_sym(a.points, b.points)
                    if error < 0.05:
                        ref_params[i][j] = (mid_pt, direction)
                        if len(a_groups) + len(b_groups) == 2:
                            a_groups[0] += b_groups[0]
                            b_groups[0].clear()
                        if len(a_groups) > 0:
                            a_groups[0].append(b)
                        elif len(b_groups) > 0:
                            b_groups[0].append(a)
                        else:
                            sym_groupings.append([[a,b], 'ref'])

            rot_params = [[None for _ in range(num_children)] for _ in range(num_children)]
            for i in range(1, num_children):
                for j in range(i):
                    a = parts[i]
                    b = parts[j]
                    if a.label != b.label:
                        continue                            
                    a_groups = []
                    b_groups = []
                    in_trans_group = False
                    for group, groupsym in sym_groupings:
                        if a in group and b in group and groupsym == 'trans':
                            in_trans_group = True
                            break
                        if a in group and groupsym == 'rot':
                            a_groups.append(group)
                        if b in group and groupsym == 'rot':
                            b_groups.append(group)
                    if in_trans_group:
                        continue
                    assert len(a_groups) < 2
                    assert len(b_groups) < 2
                    if len(a_groups) == 1 and len(b_groups) == 1: #either same group or not symmetrical
                        if a_groups[0] == b_groups[0]:
                            continue
                    error, pt, nor, angle = compute_rot_sym(a.points, b.points)
                    if error < 0.05:
                        rot_params[i][j] = (pt, nor, angle)
                        if len(a_groups) + len(b_groups) == 2:                                   
                            a_groups[0] += b_groups[0]
                            b_groups[0].clear()
                        if len(a_groups) > 0:
                            a_groups[0].append(b)
                        elif len(b_groups) > 0:
                            b_groups[0].append(a)
                        else:
                            sym_groupings.append([[a,b], 'rot'])

            SYM_ORDER = {'rot': 1, 'ref': 2, 'trans':3}
            sym_groupings = [s for s in sym_groupings if len(s[0]) > 0 and (s[1]!='rot' or len(s[0]) ==4)]
            sym_groupings = sorted(sym_groupings, key=lambda x: -len(x[0])*1000 - SYM_ORDER[x[1]])

            def get_adj_dist(a,b):
                i = parts.index(a)
                j = parts.index(b)
                assert adj_dist[i][j] == adj_dist[j][i]
                return adj_dist[i][j]

            for group, _ in sym_groupings:
                common_parts = {}
                for part in group:
                    for adj in part.adj:
                        if adj not in group:
                            common_parts[adj] = common_parts.get(adj, 0) + 1

                for common_part, occurence in common_parts.items():
                    dist_threshold = 0.3 if len(group) > 3 and occurence >=3 else 0.1
                    for part in group:
                        if common_part not in part.adj:
                            dist = get_adj_dist(part, common_part)
                            if dist < dist_threshold:
                                create_adj(part, common_part, None, dist*1.01, None)

            def check_connected(starting_part, adj_to_remove=None, adjs_to_remove=None):
                connected_component = [starting_part]
                cur_size = 0
                while len(connected_component) != cur_size:
                    cur_size = len(connected_component)
                    for part in connected_component:
                        for adj in part.adj:
                            if adj_to_remove is not None:
                                if part == starting_part and adj == adj_to_remove:
                                    continue

                            if adjs_to_remove is not None:
                                valid_edge = True
                                for ra, rb in adjs_to_remove:
                                    if (ra == part and rb == adj) or (ra == adj and rb == part):
                                        valid_edge = False
                                if not valid_edge:
                                    continue
                            if not adj in connected_component:
                                connected_component.append(adj)
                if len(connected_component) != len(parts):
                    return False
                else:
                    return True

            if not check_connected:
                #print("Disconnected")
                return

            all_valid_contacts = True
            found = True
            for part in parts:
                part.adj_box = get_node_adj_boxes(part)

            to_delete = []

            def delete_adj(part, adj_part):
                adj_idx = part.adj.index(adj_part)
                del part.adj[adj_idx]
                del part.adj_dists[adj_idx]
                del part.adj_idxs[adj_idx]
                del part.adj_box[adj_idx]
            
            def delete_adj_bidirectional(parta, partb):
                delete_adj(parta, partb)
                delete_adj(partb, parta)

            def store_delete_triplet(parta, partb, anchor):
                if not (parta, partb, anchor) in to_delete:
                    to_delete.append((parta, partb, anchor))

            def adj_overlap(a, b, c, threshold=1):
                index_b = a.adj.index(b)
                index_c = a.adj.index(c)

                adj_idxs_b = a.adj_idxs[index_b]
                adj_idxs_c = a.adj_idxs[index_c]

                if threshold < 0:
                    threshold = min(len(adj_idxs_b), len(adj_idxs_c)) * -threshold

                intersection = list(set(adj_idxs_b) & set(adj_idxs_c))
                if (len(intersection) >= threshold):
                    return True
                else:
                    return False
            
            def adj_box_overlap(a,b,c):
                index_b = a.adj.index(b)
                index_c = a.adj.index(c)
                boxa = a.adj_box[index_b]
                boxb = a.adj_box[index_c]

                axmin, aymin, azmin = list(boxa[0:3]+a.box)
                axmax, aymax, azmax = list(boxa[3:6]+a.box)
                bxmin, bymin, bzmin = list(boxb[0:3]+b.box)
                bxmax, bymax, bzmax = list(boxb[3:6]+b.box)

                if axmax > bxmin and axmin < bxmax and aymax > bymin and aymin < bymax and azmax > bzmin and azmin < bzmax:
                    return True
                if bxmax > axmin and bxmin < axmax and bymax > aymin and bymin < aymax and bzmax > azmin and bzmin < azmax:
                    return True

                return False

            def adj_box_close(a,b,c):
                index_b = a.adj.index(b)
                index_c = a.adj.index(c)
                boxa = a.adj_box[index_b]
                boxb = a.adj_box[index_c]

                amin = boxa[0:3]
                amax = boxa[3:6]
                bmin = boxb[0:3]
                bmax = boxb[3:6]
                ac = (amin+amax) / 2
                bc = (bmin+bmax) / 2

                if (((ac-bc)**2).sum())**0.5 < 0.1:
                    return True
                else:
                    return False

            def three_way_or(func):
                def wrapper(a,b,c):
                    return func(a, b, c) or func(b, c, a) or func(c, a, b)
                return wrapper

            def three_way_and(func):
                def wrapper(a,b,c):
                    return func(a, b, c) and func(b, c, a) and func(c, a, b)
                return wrapper
            
            def three_way_execute(func):
                def wrapper(a,b,c):
                    func(a,b,c)
                    func(b,a,c)
                    func(c,a,b)
                return wrapper

            def three_way_execute_one_max(func):
                def wrapper(a,b,c):
                    if not func(a,b,c):
                        if not func(b,a,c):
                            func(c,a,b)
                return wrapper

            def remove_adj_subset(a,b,c):
                index_b = a.adj.index(b)
                index_c = a.adj.index(c)

                adj_idxs_b = a.adj_idxs[index_b]
                adj_idxs_c = a.adj_idxs[index_c]

                intersection = list(set(adj_idxs_b) & set(adj_idxs_c))
                
                ib = len(intersection) / len(adj_idxs_b)
                ic = len(intersection) / len(adj_idxs_c)

                if ib > 0.9 and ib > ic:
                    store_delete_triplet(a,b,c)
                    return True
                if ic > 0.9 and ic > ib:
                    store_delete_triplet(a,c,b)
                    return True
                return False

            def remove_least_connected(a,b,c):
                index_b = a.adj.index(b)
                index_c = a.adj.index(c)

                adj_idxs_b = a.adj_idxs[index_b]
                adj_idxs_c = a.adj_idxs[index_c]

                if len(adj_idxs_b) < len(adj_idxs_c):
                    store_delete_triplet(a,b,c)
                else:
                    store_delete_triplet(a,c,b)


            def remove_tiny_connections(a,b,c):
                index_b = a.adj.index(b)
                index_c = a.adj.index(c)
                adj_idxs_b = a.adj_idxs[index_b]
                adj_idxs_c = a.adj_idxs[index_c]

                if len(adj_idxs_b) < 10 and len(adj_idxs_c) > 20:
                    store_delete_triplet(a,b,c)
                    return True

                if len(adj_idxs_c) < 10 and len(adj_idxs_b) > 20:
                    store_delete_triplet(a,c,b)
                    return True
                return False

            def remove_adj_keep_same_parent(a,b,c):
                if b.parent_name == c.parent_name and b.parent_name != a.parent_name:
                    ac, bc, cc = a.box, b.box, c.box
                    dist_ab = ((ac-bc) ** 2).sum()**0.5
                    dist_ac = ((ac-cc) ** 2).sum()**0.5
                    if dist_ab / dist_ac > 1.2:
                        store_delete_triplet(a,b,c)
                    elif dist_ac / dist_ab > 1.2:
                        store_delete_triplet(a,c,b)
                    else:
                        if b.area < c.area:
                            store_delete_triplet(a,b,c)
                        elif c.area < b.area:
                            store_delete_triplet(a,c,b)
                        else:
                            remove_least_connected(a,b,c)
                        return True
                return False

            def remove_least_frequent_common_anchor(a, b, c):
                if a in all_in_group_parts:
                    agroupidx = part_to_group_idx[a]
                    acommons = group_common_parts[agroupidx]
                    assert b in acommons and c in acommons
                    if acommons[b] > acommons[c]:
                        store_delete_triplet(c,a,b)
                        return True
                    elif acommons[c] > acommons[b]:
                        store_delete_triplet(b,a,c)
                        return True
                    else:
                        return False
                else:
                    return False

            def always_true(a,b,c):
                return True

            def remove_adj_least_connections(a, b, c):
                parts = sorted([a,b,c], key=lambda x:-len(x.original_adj))
                if len(parts[1].adj) != len(parts[2].adj):
                    store_delete_triplet(parts[1], parts[2], parts[0])

            def remove_adj_smallest(a, b, c):
                parts = sorted([a,b,c], key=lambda x:-(x.area))
                if parts[1].area / parts[0].area < 0.7:
                    store_delete_triplet(parts[1], parts[2], parts[0])
                elif parts[1].area / parts[0].area < 0.8 and parts[2].area / parts[1].area > 0.95:
                    store_delete_triplet(parts[1], parts[2], parts[0])
                elif parts[1].area / parts[0].area > 0.95 and parts[2].area / parts[1].area < 0.8:
                    store_delete_triplet(parts[0], parts[1], parts[2])

            def remove_adj_gravity(a, b, c):
                parts = sorted([a,b,c], key=lambda x:float(x.box[1]))
                if (parts[1].box[1] - parts[0].box[1]) > 0.05 and (parts[2].box[1] - parts[1].box[1] > 0.05):
                    store_delete_triplet(parts[0], parts[2], parts[1])
                else:
                    parts = sorted([a,b,c], key=lambda x:float(x.bbox[4]))
                    if (parts[1].bbox[4] - parts[0].bbox[4]) > 0.05 and (parts[2].bbox[4] - parts[1].bbox[4] < 0.05):
                        store_delete_triplet(parts[1], parts[2], parts[0])

            def remove_adj(parts, filter_func, remove_target_func, mode=0):
                for a in parts:
                    temp_adj = copy.copy(a.adj)
                    N = len(temp_adj)
                    for i in range(N):
                        for j in range(i+1, N):
                            b = temp_adj[i]
                            c = temp_adj[j]
                            if b in a.adj and c in a.adj:
                                if b in c.adj:
                                    assert c in b.adj
                                    if mode == 0:
                                        if filter_func(a, b, c):
                                            remove_target_func(a, b, c)
                                    elif mode == 1:
                                        removed = False
                                        if filter_func(a, b, c):
                                            removed = removed or remove_target_func(a, b, c)
                                        if (not removed) and filter_func(b, c, a):
                                            removed = removed or remove_target_func(b, c, a)
                                        if (not removed) and filter_func(c, a, b):
                                            remove_target_func(c, a, b)
                                    else:
                                        assert False



            def reachable_otherwise(parta, partb, anchor, same_parent=False):
                visited = [parta]
                stack = [adj for adj in parta.adj if not adj == partb]
                candidates = []
                while len(stack) > 0:
                    cur_node = stack.pop()
                    visited.append(cur_node)
                    if same_parent and cur_node.parent_name != parta.parent_name:
                        continue
                    for adj in cur_node.adj:
                        if adj == partb:
                            if cur_node != anchor:
                                candidates.append(cur_node)
                        elif adj not in visited:
                            stack.append(adj)
                for adj in candidates:
                    if same_parent or adj_overlap(partb, parta, adj, threshold=-0.5):
                        return True
                return False

            connected_component = [parts[0]]
            cur_size = 0
            while len(connected_component) != cur_size:
                cur_size = len(connected_component)
                for part in connected_component:
                    for adj in part.adj:
                        if not adj in connected_component:
                            connected_component.append(adj)

            if len(connected_component) != len(parts):
                #print("????") #idk why
                return


            if remove_edges:
                parts2 = parts
                for part in parts:
                    part.original_adj = copy.copy(part.adj)
                parts = sorted(parts, key = lambda x: (x.area))

                for group, _ in sym_groupings:
                    common_parts = {}
                    for part in group:
                        for adj in part.adj:
                            if adj in group:
                                if check_connected(part, adj):
                                    delete_adj_bidirectional(part, adj)

                already_in_group_parts = []
                keep_idx = []
                for i in range(len(sym_groupings)):
                    group, _ = sym_groupings[i]
                    if any(part in already_in_group_parts for part in group):
                        continue
                    else:
                        already_in_group_parts += group
                        keep_idx.append(i)
                sym_groupings = [sym_groupings[i] for i in keep_idx]

                all_in_group_parts = already_in_group_parts

                group_common_parts = []
                max_common_part_deg = []
                already_in_group_parts = []
                secondary_target = []
                part_to_group_idx = {}
                for i, (group, _) in enumerate(sym_groupings):
                    for part in group:
                        part_to_group_idx[part] = i

                for i, (group, _) in enumerate(sym_groupings):
                    common_parts = {}
                    secondary = False
                    largest_parent_group = None

                    non_symmetry_anchor = False

                    for part in group:
                        part_to_group_idx[part] = i
                        for adj in part.adj:
                            if adj not in group:
                                common_parts[adj] = common_parts.get(adj, 0) + 1
                                if adj not in all_in_group_parts:
                                    non_symmetry_anchor = True

                    if not non_symmetry_anchor:
                        for part in group:
                            for adj in part.adj:
                                if adj in all_in_group_parts:
                                    parent_group_idx = part_to_group_idx[adj]
                                    parent_group = sym_groupings[parent_group_idx][0]
                                    if all(any(adj in group for adj in p.adj) for p in parent_group) and all(any(adj in parent_group for adj in p.adj) for p in group):
                                        secondary = True
                                        parent_group_idx = part_to_group_idx[adj]
                                        if largest_parent_group is None or largest_parent_group > parent_group_idx:
                                            largest_parent_group = parent_group_idx

                    for part in common_parts.keys():
                        if part in all_in_group_parts:
                            common_parts[part] -= 0.5
                    
                    if secondary:
                        for part in sym_groupings[largest_parent_group][0]:
                            try:
                                common_parts[part] += 1000
                            except:
                                pass

                    group_common_parts.append(common_parts)
                    if not secondary:
                        try:
                            max_common_part_deg.append(max([k for k in common_parts.values()]))
                        except:
                            #print("WTF IS THIS BUG")
                            return
                        secondary_target.append(None)
                    else:
                        max_common_part_deg.append(10000000)
                        assert largest_parent_group is not None
                        secondary_target.append(largest_parent_group)

                    already_in_group_parts += group

                #behavior of sym groups:
                #The largest common anchor is always protected
                #Removing edge -> remove all edge for other parts connected to this
                #Probably handle this with a priority queue

                #Priortize the following scenarios: a,b,c, c is in a sym group and b is attached to less nodes in the group than a

                remove_adj(parts, three_way_or(adj_overlap), three_way_execute(remove_adj_keep_same_parent))
                remove_adj(parts, three_way_or(adj_overlap), remove_adj_gravity)
                remove_adj(parts, three_way_or(adj_overlap), remove_adj_smallest)
                remove_adj(parts, three_way_or(adj_box_close), three_way_execute(remove_adj_keep_same_parent))
                remove_adj(parts, three_way_or(adj_box_close), remove_adj_gravity)
                remove_adj(parts, three_way_or(adj_box_close), remove_adj_smallest)

                def delete_adj_with_sym(parta, partb):
                    if (parta not in part_to_group_idx) and (partb not in part_to_group_idx):
                        delete_adj_bidirectional(parta, partb)
                        return True

                    edges_to_delete = [(parta, partb)]
                    #Both part should not be the most connected part in a symmetry
                    if parta in part_to_group_idx:
                        agroupidx = part_to_group_idx[parta]
                        acommons = group_common_parts[agroupidx]

                        if partb not in acommons:
                            return False

                        parent_idx = secondary_target[agroupidx]
                        if parent_idx is not None:
                            parent_group = sym_groupings[parent_idx][0]
                            if partb in parent_group:
                                return False

                        if acommons[partb] == max_common_part_deg[agroupidx]:
                            max_count = 0
                            for v in acommons.values():
                                if v == max_common_part_deg[agroupidx]:                                       
                                    max_count += 1
                            if max_count == 1:
                                return False
                        
                        del acommons[partb]
                        agroup = sym_groupings[agroupidx][0]
                        for part in agroup:
                            if part != parta and part in partb.adj:
                                edges_to_delete.append((part, partb))

                    if partb in part_to_group_idx:
                        bgroupidx = part_to_group_idx[partb]
                        bcommons = group_common_parts[bgroupidx]
                        if parta not in bcommons:
                            return False

                        parent_idx = secondary_target[bgroupidx]
                        if parent_idx is not None:
                            parent_group = sym_groupings[parent_idx][0]
                            if parta in parent_group:
                                return False

                        if bcommons[parta] == max_common_part_deg[bgroupidx]:
                            max_count = 0
                            for v in bcommons.values():
                                if v == max_common_part_deg[bgroupidx]:                                       
                                    max_count += 1
                            if max_count == 1:
                                return False

                        del bcommons[parta]
                        bgroup = sym_groupings[bgroupidx][0]
                        for part in bgroup:
                            if part != partb and part in parta.adj:
                                edges_to_delete.append((part, parta))   

                    if check_connected(parta, None, edges_to_delete):
                        for ra, rb in edges_to_delete:
                            if ra in rb.adj:
                                assert rb in ra.adj
                                delete_adj_bidirectional(ra, rb)
                        return True
                    else:
                        return False


                for parta, partb, anchor in to_delete:
                    if parta in partb.adj:
                        assert partb in parta.adj
                        if parta in anchor.adj and partb in anchor.adj:
                            delete_adj_with_sym(parta, partb)

                for parta, partb, anchor in to_delete:
                    if parta in partb.adj:
                        assert partb in parta.adj
                        if parta.parent_name == partb.parent_name:
                            if not (reachable_otherwise(parta, partb, anchor, same_parent=True) or reachable_otherwise(partb, parta, anchor, same_parent=True)):
                                continue
                        if reachable_otherwise(parta, partb, anchor) or reachable_otherwise(partb, parta, anchor):
                            delete_adj_with_sym(parta, partb)
                parts = parts2

            connected_component = [parts[0]]
            cur_size = 0
            while len(connected_component) != cur_size:
                cur_size = len(connected_component)
                for part in connected_component:
                    for adj in part.adj:
                        if not adj in connected_component:
                            connected_component.append(adj)

            if len(connected_component) != len(parts):
                return

            for group_idx, (group, sym_type) in enumerate(sym_groupings):
                if sym_type == 'trans': 
                    first_order_parent = False
                    for i in range(1, len(group)):
                        for j in range(i):
                            assert group[j] not in group[i].sym_parts
                            group[i].sym_parts.append((group[j], ('trans')))
                            assert group[i] not in group[j].sym_parts
                            group[j].sym_parts.append((group[i], ('trans')))

                            for adj in group[i].adj:
                                if group[j] in adj.adj:
                                    first_order_parent = True
                                    adj.sym_pairs.append((group[i], group[j], ('trans')))
                    
                    if not first_order_parent:
                        if secondary_target[group_idx] is not None:
                            sidx = secondary_target[group_idx]
                            parent_group = sym_groupings[sidx][0]
                            for parent_part in parent_group:
                                sec_child = [adj for adj in parent_part.adj if adj in group]
                                parent_part.secondary_sym_children.append(sec_child)
                elif sym_type == 'ref':
                    for i in range(1, len(group)):
                        for j in range(i):
                            iidx = parts.index(group[i])
                            jidx = parts.index(group[j])
                            assert group[j] not in group[i].sym_parts
                            assert group[i] not in group[j].sym_parts

                            if ref_params[iidx][jidx] is not None:
                                params = ref_params[iidx][jidx]
                                assert ref_params[jidx][iidx] is None
                                group[i].sym_parts.append((group[j], ('ref', params)))
                                group[j].sym_parts.append((group[i], ('refr', params)))

                                for adj in group[i].adj:
                                    if group[j] in adj.adj:
                                        adj.sym_pairs.append((group[i], group[j], ('ref', params)))
                            if ref_params[jidx][iidx] is not None:
                                params = ref_params[jidx][iidx]
                                assert ref_params[iidx][jidx] is None
                                group[j].sym_parts.append((group[i], ('ref', params)))
                                group[i].sym_parts.append((group[j], ('refr', params)))

                                for adj in group[i].adj:
                                    if group[j] in adj.adj:
                                        adj.sym_pairs.append((group[j], group[i], ('ref', params)))
                elif sym_type == 'rot':
                    for i in range(1, len(group)):
                        for j in range(i):
                            iidx = parts.index(group[i])
                            jidx = parts.index(group[j])
                            assert group[j] not in group[i].sym_parts
                            assert group[i] not in group[j].sym_parts

                            if rot_params[iidx][jidx] is not None:
                                params = rot_params[iidx][jidx]
                                assert rot_params[jidx][iidx] is None
                                group[i].sym_parts.append((group[j], ('rot', params)))
                                group[j].sym_parts.append((group[i], ('rotr', params)))

                                for adj in group[i].adj:
                                    if group[j] in adj.adj:
                                        adj.sym_pairs.append((group[i], group[j], ('rot', params)))
                            if rot_params[jidx][iidx] is not None:
                                params = rot_params[jidx][iidx]
                                assert rot_params[iidx][jidx] is None
                                group[j].sym_parts.append((group[i], ('rot', params)))
                                group[i].sym_parts.append((group[j], ('rotr', params)))

                                for adj in group[i].adj:
                                    if group[j] in adj.adj:
                                        adj.sym_pairs.append((group[j], group[i], ('rot', params)))
                else:
                    raise NotImplementedError


            for part in parts:
                assert len(part.adj_box) == len(part.adj_dists) == len(part.adj_idxs)
                part.adj_box = torch.stack(part.adj_box)
                part.adj_normals = None
                part.adj_descriptor = part.adj_box

            part_to_idx = {part:i for (i, part) in enumerate(parts)}

            for part in parts:
                u = part_to_idx[part]
                for adj_part in part.adj:
                    v = part_to_idx[adj_part]
                    assert u != v
                    if u < v:
                        graph_edges.append((u,v))

            num_adjs = [len(part.adj) for part in parts]

            if save_data:
                with open(f"{save_dir}/{shape_id}_full.pkl", 'wb') as f:
                    pickle.dump((parts, graph_edges), f, pickle.HIGHEST_PROTOCOL)
            if save_simple:
                for part in parts:
                    part.points = None
                    part.centered_pc = None
                    part.adj_idxs = None
                    part.adj_dists = None
                with open(f"{save_dir}/{shape_id}.pkl", 'wb') as f:
                    pickle.dump((parts, graph_edges), f, pickle.HIGHEST_PROTOCOL)

        def tryparse(i):
            try:
                parse(i)
            except:
                pass
        p = Pool(10)

        for _ in tqdm(p.imap_unordered(tryparse, range(len(files))), total=len(files)):
            pass