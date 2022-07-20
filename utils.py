import gzip
import math
import os
import os.path
import sys
import pickle
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np


def show_img(array):
    try:
        img = Image.fromarray(array.astype('uint8'))
    except:
        img = array
    img.show()

def show_img_color(array):
    try:
        img = Image.fromarray(np.rollaxis(np.rollaxis(array,2), 2).astype('uint8'))
    except:
        img = array
    img.show()

def save_img(array, dirr):
    img = Image.fromarray(array.astype('uint8'))
    img.save(dirr)

def save_img_color(array, dirr):
    img = Image.fromarray(np.rollaxis(np.rollaxis(array,2), 2).astype('uint8'))
    img.save(dirr)

'''
Ensure a directory exists
'''
def ensuredir(dirname):
    """
    Ensure a directory exists
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def write_obj(verts, faces, outfile):
    with open(outfile, 'w') as f:
        for a, b, c in verts.tolist():
            f.write(f'v {a} {b} {c}\n')

        for a, b, c in faces.tolist():
            f.write(f"f {a} {b} {c}\n")

def load_obj(fn):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []; faces = [];
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

    f = np.vstack(faces)
    v = np.vstack(vertices)
    return v, f

def get_pc_scale(pc):
    return torch.sqrt(torch.max(torch.sum((pc - torch.mean(pc, axis=0))**2, axis=1)))

def get_chamfer_distance(pc1, pc2): #normalized
    dist = pairwise_dist(pc1, pc2)
    error = torch.mean(torch.min(dist, axis=1)[0]) + torch.mean(torch.min(dist, axis=0)[0])
    scale = get_pc_scale(pc1) + get_pc_scale(pc2)
    return error / scale

def pairwise_dist_batch(p1, p2):
    n = p1.shape[0]
    a = (p1**2).sum(dim=2).view(n,-1,1)
    b = (p2**2).sum(dim=2).view(n,1,-1)
    c = torch.bmm(p1, p2.transpose(1,2))
    return a + b - 2 * c

def pairwise_dist(p1, p2):
    return (p1**2).sum(dim=1).view(-1,1) + (p2**2).sum(dim=1).view(1,-1) - 2 * p1@p2.t()