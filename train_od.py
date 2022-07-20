import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import *
import random
import time
import utils
import argparse
import torch.optim as optim
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pickle
from model.gnn import GraphNet
from model.pointnet import PointNetCls
import pyod
from pyod.models.ocsvm import OCSVM

torch.multiprocessing.set_sharing_strategy('file_system')

class Part():
    def __init__(self):
        pass

def get_features(nodes, degrees, degrees2, dists):
    n = len(nodes)
    features = torch.zeros(n * 3)
    for dist in dists:
        features[dist] += 1
    for degree in degrees:
        features[n + degree-1] += 1
    for degree in degrees2:
        features[n*2 + degree-1] += 1
    return features


class ODDataset():
    def __init__(self, data_dir=None, indices=None, data_root_dir='data', len_nodes = 20):
        self.indices = indices
        self.data_dir = data_dir
        self.data_root_dir = data_root_dir
        idxs = []
        for (i,filename) in enumerate(Path(f"{data_root_dir}/{data_dir}").glob('*.pkl')):
            if "full" not in str(filename):
                idxs.append(int(str(filename).split("/")[-1][:-4]))
        self.idxs = sorted(idxs)
        if indices is None:
            self.indices = (0, len(idxs))
        
        self.len_dict = {}
        for i in range(2, 20):
            self.len_dict[i] = []
        for idx in idxs:
            with open(f"{self.data_root_dir}/{self.data_dir}/{idx}.pkl", 'rb') as f:
                nodes, edges = pickle.load(f)
            if len(nodes) < 20:
                self.len_dict[len(nodes)].append(idx)

    def __len__(self):
        return 1000

    def __getitem__(self, index):
        return self.get_data(index, False)

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

    def get_data(self, index, eval=False):
        target = self.len_dict[self.len_nodes]
        i = random.randint(0, len(target)-1)
        with open(f"{self.data_root_dir}/{self.data_dir}/{target[i]}.pkl", 'rb') as f:
            nodes, edges = pickle.load(f)

        dists, degrees, degrees2 = self.check_longest_path(nodes)

        features = get_features(nodes, degrees, degrees2, dists)

        return features


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import os
    import torch.optim as optim

    batch_size = 32
    training_size = 1000
    
    parser = argparse.ArgumentParser(description='retrieval')
    parser.add_argument('--save-dir', type=str, default=None, metavar='N')
    parser.add_argument('--data-root-dir', type=str, default='/data_hdd/part-data', metavar='N')
    parser.add_argument('--category', type=str, default='chair', metavar='N')
    args = parser.parse_args()

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = f"checkpoints/{args.category}_od"
    utils.ensuredir(save_dir)

    data_root_dir = args.data_root_dir
    data_dir = f"graph_{args.category}_train_final"


    train_dataset = ODDataset(indices=None, data_dir = data_dir, data_root_dir = data_root_dir, len_nodes = 0)
    for num_nodes in range(18,20):
        print(f"Fitting SVM for {num_nodes} parts...")
        if len(train_dataset.len_dict[num_nodes]) == 0:
            continue
        train_dataset.len_nodes = num_nodes
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = batch_size,
            num_workers = 8,
            shuffle = True,
        )

        hs = []
        with torch.no_grad():
            for i in range(50-num_nodes*2):
                print(f"{i}/{50-num_nodes*2}", end="\r")
                for batch_idx, (features) in enumerate(train_loader):
                    h = features
                    hs.append(h.detach().cpu())

        hs = torch.cat(hs, dim=0)
        
        oc = OCSVM(contamination=0.01)
        hs = hs.numpy()
        t0 = time.time()
        oc.fit(hs)
        print(time.time() - t0)
        with open(f"{save_dir}/svm_{num_nodes}.pkl", 'wb') as f:
            pickle.dump(oc, f, pickle.HIGHEST_PROTOCOL)
            

