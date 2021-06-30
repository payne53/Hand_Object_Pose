# -*- coding: utf-8 -*-
# @Time    : 2020/11/4 9:50
# @Author  : Shark
# @Site    : 
# @File    : interactionnet_v0.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from models.resnet import *
from models.graphunet import *
from pdb import set_trace

d_model = 128

class Graph():
    def __init__(self, num_hand=21, num_obj=8):
        self.num_node_hand = num_hand
        self.num_node_obj = num_obj
        self.num_node = self.num_node_hand + self.num_node_obj

        self.neighbour_hand = [(0, 1), (1, 2), (2, 3), (3, 4),
                          (0, 5), (5, 6), (6, 7), (7, 8),
                          (0, 9), (9, 10), (10, 11), (11, 12),
                          (0, 13), (13, 14), (14, 15), (15, 16),
                          (0, 17), (17, 18), (18, 19), (19, 20)]

        self.neighbour_object = [(0, 1), (1, 2), (2, 3), (3, 0),
                                 (4, 5), (5, 6), (6, 7), (7, 4),
                                 (0, 4), (1, 5), (2, 6), (3, 7)]

        self.sym_link = [(4, 8, 12, 16, 20), (3, 7, 11, 15, 19), (2, 6, 10, 14, 18), (1, 5, 9, 13, 17)]
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.get_adjacency()


    def get_adjacency(self):
        self.A_hand = np.zeros((self.num_node_hand, self.num_node_hand))
        self.A_obj = np.zeros((self.num_node_obj, self.num_node_obj))
        self.A = np.zeros((self.num_node, self.num_node))

        for i in range(self.num_node_hand):
            self.A_hand[i, i] = 1

        for i in range(self.num_node_obj):
            self.A_obj[i, i] = 1

        for i in range(self.num_node):
            self.A[i, i] = 1

        for (i, j) in self.neighbour_hand:
            self.A_hand[i, j] = self.A_hand[j, i] = 1
            self.A[i, j] = self.A[j, i] = 1

        for (i, j) in self.neighbour_object:
            self.A_obj[i, j] = self.A_obj[j, i] = 1
            self.A[i + self.num_node_hand, j + self.num_node_hand] = self.A[j + self.num_node_hand, i + self.num_node_hand] = 1

class HandNet(nn.Module):
    def __init__(self, in_features, out_features, out_dim=3):
        super(HandNet, self).__init__()
        self.gconv1 = GraphConv(in_features=in_features, out_features=d_model)
        self.gconv2 = GraphConv(in_features=d_model, out_features=d_model)
        self.gconv3 = GraphConv(in_features=d_model, out_features=out_features)
        self.graph = Graph()
        self.A1 = Parameter(torch.from_numpy(self.graph.A_hand).float().cuda(), requires_grad=True)
        self.A2 = Parameter(torch.from_numpy(self.graph.A_hand).float().cuda(), requires_grad=True)
        self.A3 = Parameter(torch.from_numpy(self.graph.A_hand).float().cuda(), requires_grad=True)
        self.fc = nn.Linear(out_features, out_dim)

    def forward(self, x):
        x1 = self.gconv1(x, self.A1)
        x2 = self.gconv2(x1, self.A2)
        x3 = self.gconv3(x2, self.A3)
        out = self.fc(x3)
        return x3, out

class ObjectNet(nn.Module):
    def __init__(self, in_features, out_features, out_dim=3):
        super(ObjectNet, self).__init__()
        self.gconv1 = GraphConv(in_features=in_features, out_features=d_model)
        self.gconv2 = GraphConv(in_features=d_model, out_features=d_model)
        self.gconv3 = GraphConv(in_features=d_model, out_features=out_features)
        self.graph = Graph()
        self.A1 = Parameter(torch.from_numpy(self.graph.A_obj).float().cuda(), requires_grad=True)
        self.A2 = Parameter(torch.from_numpy(self.graph.A_obj).float().cuda(), requires_grad=True)
        self.A3 = Parameter(torch.from_numpy(self.graph.A_obj).float().cuda(), requires_grad=True)
        self.fc = nn.Linear(out_features, out_dim)

    def forward(self, x):
        x1 = self.gconv1(x, self.A1)
        x2 = self.gconv2(x1, self.A2)
        x3 = self.gconv3(x2, self.A3)
        out = self.fc(x3)
        return x3, out


class InteractionNet(nn.Module):
    def __init__(self):
        super(InteractionNet, self).__init__()
        self.fusion1 = GraphConv(in_features=3, out_features=d_model)
        self.fusion2 = GraphConv(in_features=d_model, out_features=d_model)
        self.fusion3 = GraphConv(in_features=d_model, out_features=d_model)
        self.fusion4 = GraphConv(in_features=d_model, out_features=d_model)
        self.fusion5 = GraphConv(in_features=d_model, out_features=d_model)
        self.fusion6 = GraphConv(in_features=d_model, out_features=d_model)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)
        self.dropout3 = nn.Dropout(p=0.3)

        self.A_1 = Parameter(torch.eye(29).float().cuda(), requires_grad=True)
        self.A_2 = Parameter(torch.eye(29).float().cuda(), requires_grad=True)
        self.A_3 = Parameter(torch.eye(29).float().cuda(), requires_grad=True)
        self.A_4 = Parameter(torch.eye(29).float().cuda(), requires_grad=True)
        self.A_5 = Parameter(torch.eye(29).float().cuda(), requires_grad=True)
        self.A_6 = Parameter(torch.eye(29).float().cuda(), requires_grad=True)

        self.linear_trans = nn.Linear(d_model, 3)


    def forward(self, init_pose):
        feat = self.get_feat(init_pose)
        trans = self.linear_trans(feat)     # bs, 29, 128
        res = init_pose + trans
        return res

    def get_feat(self, init_pose):
        x = init_pose
        x = self.fusion1(x, self.A_1)
        x = self.fusion2(x, self.A_2)
        x = self.dropout1(x)
        x = self.fusion3(x, self.A_3)
        x = self.fusion4(x, self.A_4)
        x = self.dropout2(x)
        x = self.fusion5(x, self.A_5)
        x = self.fusion6(x, self.A_6)
        x = self.dropout3(x)
        return x

class ContextModule(nn.Module):
    def __init__(self, in_features, out_features, refine_only=True):
        super(ContextModule, self).__init__()
        self.handnet = HandNet(in_features=in_features, out_features=d_model)
        self.objnet = ObjectNet(in_features=in_features, out_features=d_model)
        self.fc_hand = nn.Linear(d_model, out_features)
        self.fc_obj = nn.Linear(d_model, out_features)
        self.reg = InteractionNet()

        if refine_only:
            for p in self.handnet.parameters():
                p.requires_grad = False

            for p in self.objnet.parameters():
                p.requires_grad = False

    def forward(self, x, argu=None):
        hand_feat, hand_points_0 = self.handnet(x[:, :21, :])
        obj_feat, obj_points_0 = self.objnet(x[:, 21:, :])

        init_pose = torch.cat([hand_points_0, obj_points_0], dim=1)
        res1 = self.reg(init_pose)
        return init_pose, res1


class ContextNet(nn.Module):
    def __init__(self):
        super(ContextNet, self).__init__()
        self.resnet = resnet50(pretrained=False, num_classes=29*2)
        self.graphnet = GraphNet(in_features=2050, out_features=2)
        self.interactionnet = ContextModule(in_features=2, out_features=3)

    def forward(self, x):
        points2D_init, features = self.resnet(x)
        features = features.unsqueeze(1).repeat(1, 29, 1)
        in_features = torch.cat([points2D_init, features], dim=2)
        points2D = self.graphnet(in_features)

        points3D_init, points3D = self.interactionnet(points2D)
        return points2D_init, points2D, points3D_init, points3D

