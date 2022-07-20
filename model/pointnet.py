from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class PointNetfeat(nn.Module):
    def __init__(self, input_k = 19):
        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.Identity()
        self.bn2 = nn.Identity()
        self.bn3 = nn.Identity()
        self.global_feat = global_feat
        self.feature_transform = feature_transform

    def forward(self, x):
        n_pts = x.size()[1]
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        trans_feat = None
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x

class PointNetCls(nn.Module):
    def __init__(self, input_k=19, k=2):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(input_k = input_k)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.Identity()
        self.bn2 = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x

    def get_feature(self, x):
        x = self.feat(x)
        return x

