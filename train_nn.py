import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import time
import utils
import argparse
import torch.optim as optim
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from PIL import Image
import pickle
from data.dataset import *
import model
import matplotlib.pyplot as plt


class Part():
    def __init__(self):
        pass

def LOG(msg):
    print(msg)
    logfile.write(msg + '\n')
    logfile.flush()

def build_model(task, model_params):
    if task == 'contact':
        from model.contact import Model
        feature_size = 13
        edge_size = 1
    elif task == 'retrieval':
        from model.retrieval import Model
        feature_size = 13
        edge_size = 1
    elif task == 'connect':
        from model.connect import Model
        feature_size = 13
        edge_size = 1
    else:
        raise NotImplementedError
    
    model = Model(
        feature_size=feature_size,
        edge_size=edge_size,
        **model_params,
    )

    return model

def get_collate_fn(task):
    if task == 'contact':
        return collate_data_contact
    elif task == 'retrieval':
        return collate_data_retrieval
    elif task == 'connect':
        return collate_data_connect
    else:
        raise NotImplementedError


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import os
    import yaml
    import shutil

    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--save-dir', type=str, default=None, metavar='N')
    parser.add_argument('--config-dir', type=str, default=None, metavar='N')
    parser.add_argument('--data-root-dir', type=str, default='/data_hdd/part-data', metavar='N')
    parser.add_argument('--task', type=str, default='connect', metavar='N')
    parser.add_argument('--category', type=str, default='chair', metavar='N')
    args = parser.parse_args()

    data_root_dir = args.data_root_dir

    cat = args.category
    task = args.task
    save_dir = args.save_dir
    if save_dir is None:
        save_dir = f"checkpoints/{cat}_{task}"
    utils.ensuredir(save_dir)

    data_dir = f"graph_{cat}_train_final"
    valid_dir = f"graph_{cat}_val_final"
    valid_dir = data_dir.replace("train", "val")

    config_dir = args.config_dir
    if config_dir is None:
        config_dir = f"config/{task}.yaml"

    with open(config_dir, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    shutil.copyfile(config_dir, f'{save_dir}/config.yaml')
    collate_fn = get_collate_fn(task)

    train_dataset = SPSMDataset(indices=None, data_dir = data_dir, data_root_dir = data_root_dir, task=task, preload=False)
    training_size = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config['batch_size'],
        num_workers = config['num_workers'],
        shuffle = True,
        collate_fn = collate_fn
    )

    valid_dataset = SPSMDataset(indices=None, data_dir = valid_dir, data_root_dir = data_root_dir, task=task, preload=False)
    valid_size = len(valid_dataset)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = config['batch_size'],
        num_workers = config['num_workers'],
        shuffle = False,
        collate_fn = collate_fn
    )

    if task == 'retrieval':
        train_dataset_neg = SPSMDataset(indices=None, data_dir = data_dir, data_root_dir = data_root_dir, task='neg', preload=False)
        train_loader_neg = torch.utils.data.DataLoader(
            train_dataset_neg,
            batch_size = config['batch_size'],
            num_workers = config['num_workers'],
            shuffle = True,
            collate_fn = collate_data_neg
        )

        valid_dataset_neg = SPSMDataset(indices=None, data_dir = valid_dir, data_root_dir = data_root_dir, task='neg', preload=False)
        valid_loader_neg = torch.utils.data.DataLoader(
            valid_dataset_neg,
            batch_size = config['batch_size'],
            num_workers = config['num_workers'],
            shuffle = True,
            collate_fn = collate_data_neg
        )

    logfile = open(f"{save_dir}/log.txt", 'w')
    device = config['device']

    model_params = config['model_params']
    model = build_model(task = args.task, model_params = model_params)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    batch_size = config['batch_size']

    def train(epoch):
        model.train()
        likelihoods = None

        total_stats = model.get_stats_dict()

        if task == 'retrieval':
            neg_iterator = iter(train_loader_neg)

        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()

            if task == 'retrieval':
                loss, stats = model(data, device, neg_iterator)
            else:
                loss, stats = model(data, device)

            if (loss.isnan() or loss > 100):
                return None

            loss.backward()

            optimizer.step()
            for key, value in stats.items():
                total_stats[key] += value

            print(f'{batch_idx}/{training_size/batch_size}', end="\r")

        for key in total_stats.keys():
            total_stats[key] = total_stats[key] / training_size

        for key, value in total_stats.items():
            LOG(f"Average {key}: {value}")

        return total_stats

    def valid(epoch):
        model.eval()
        likelihoods = None

        total_stats = model.get_stats_dict()

        if task == 'retrieval':
            neg_iterator = iter(valid_loader_neg)

        for batch_idx, data in enumerate(valid_loader):
            with torch.no_grad():
                if task == 'retrieval':
                    _, stats = model(data, device, neg_iterator)
                else:
                    _, stats = model(data, device)
                for key, value in stats.items():
                    total_stats[key] += value

        LOG("=====Validation=====")
        for key in total_stats.keys():
            total_stats[key] = total_stats[key] / valid_size

        for key, value in total_stats.items():
            LOG(f"Average {key}: {value}")
        LOG("====================")
        return stats
   
    def plot_loss(losses, out_dir):
        plt.clf()
        plt.plot([l[0] for l in losses], [l[1] for l in losses])
        plt.grid()
        plt.savefig(out_dir)

    def update_all_stats(all_stats, new_stats, epoch):
        for key, value in new_stats.items():
            if key in all_stats:
                all_stats[key].append((epoch,value))
            else:
                all_stats[key] = [(epoch, value)]

    all_train_stats = {}
    all_val_stats = {}
    best_val_criterion = 10000000
    for i in range(0, config['num_epochs']):
        LOG(f'=========================== Epoch {i} ===========================')
        t = time.time()
        train_stats = train(i)
        if train_stats is None: #Somehow sometimes something somewhat explodes somewhere
            model.load_state_dict(torch.load(f"{save_dir}/backup.pt"))
            i -= (i % 5)
            if i % 5 == 0:
                i -= 5
            LOG(f"EXPLODED, resume from epoch {i}")
        else:
            if i % 5 == 0:
                torch.save(model.state_dict(), f"{save_dir}/backup.pt")
            if i % config['eval_every'] == 0:
                val_stats = valid(i)

                val_criterion = val_stats[model.get_model_selection_crition()]
                if val_criterion < best_val_criterion:
                    best_val_criterion = val_criterion
                    torch.save(model.state_dict(), f"{save_dir}/best.pt")

                update_all_stats(all_train_stats, train_stats, i)
                update_all_stats(all_val_stats, val_stats, i)

                for key, value in all_train_stats.items():
                    plot_loss(value, f"{save_dir}/train_{key}.png")

                for key, value in all_val_stats.items():
                    plot_loss(value, f"{save_dir}/val_{key}.png")

        LOG(f"Time taken: {str(time.time()-t)}")
        if i % config['save_every'] == 0:
            torch.save(model.state_dict(), f"{save_dir}/{i}.pt")