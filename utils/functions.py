# -*- coding: utf-8 -*-
# @Time    : 2020/12/28 15:30
# @Author  : Shark
# @Site    : 
# @File    : functions.py
# @Software: PyCharm

import os
import numpy as np
import torch
import torch.nn as nn
import math
import random

def cal_err(pred, gt):
    """
    :param pred:    bs, num_actions, 29, 3
    :param gt:      bs, 1, 29, 3
    :return:
    """
    diff = gt - pred
    dist = torch.sqrt(torch.sum(diff.pow(2), dim=-1))
    return dist

def get_rewards(init, preds, gts):
    """
    :param preds:   bs, num_actions, 29, 3
    :param gts:     bs, 29, 3
    :return:        rewards
    """
    init = init.unsqueeze(1)    # bs, 1, 29, 3
    gts = gts.unsqueeze(1)      # bs, 1, 29, 3

    init_dist = cal_err(init, gts)  # bs, num_actions, 29
    init_dist_hand, init_dist_obj = init_dist.split([21, 8], dim=2)     # bs, num_actions, 21 / 8
    init_dist_hand, init_dist_obj = init_dist_hand.mean(2), init_dist_obj.mean(2)   # bs, num_actions

    pred_dist = cal_err(preds, gts) # bs, num_actions, 29
    pred_dist_hand, pred_dist_obj = pred_dist.split([21, 8], dim=2)     # bs, num_actions, 21 / 8
    pred_dist_hand, pred_dist_obj = pred_dist_hand.mean(2), pred_dist_obj.mean(2)   # bs, num_actions

    delta_hand = init_dist_hand - pred_dist_hand
    delta_obj = init_dist_obj - pred_dist_obj
    return delta_hand, delta_obj


def data_argumentation(labels2d, labels3d):
    """
    :param labels2d: bs, 29, 2
    :param labels3d: bs, 29, 3
    :return:
    """
    bs = labels2d.shape[0]
    param_offset = torch.randn((bs, 1, 3)).type_as(labels3d)
    param_zoom = torch.randn((bs, 3)) * 0.01 + 1
    param_rotat = torch.randn((bs, 3)) * np.pi / 32

    center = labels3d.mean(1, keepdim=True)
    labels3d_ = labels3d - center

    H_z = torch.eye(3).repeat(bs, 1, 1).type_as(labels3d)
    H_z[:, 0, 0] = param_zoom[:, 0]
    H_z[:, 1, 1] = param_zoom[:, 1]
    H_z[:, 2, 2] = param_zoom[:, 2]

    thetax, thetay, thetaz = param_rotat[:, 0], param_rotat[:, 1], param_rotat[:, 2]
    H_rx = torch.eye(3).repeat(bs, 1, 1).type_as(labels3d)
    H_rx[:, 1, 1] = torch.cos(thetax)
    H_rx[:, 1, 2] = -torch.sin(thetax)
    H_rx[:, 2, 1] = torch.sin(thetax)
    H_rx[:, 2, 2] = torch.cos(thetax)
    H_ry = torch.eye(3).repeat(bs, 1, 1).type_as(labels3d)
    H_ry[:, 0, 0] = torch.cos(thetay)
    H_ry[:, 0, 2] = torch.sin(thetay)
    H_ry[:, 2, 0] = -torch.sin(thetay)
    H_ry[:, 2, 2] = torch.cos(thetay)
    H_rz = torch.eye(3).repeat(bs, 1, 1).type_as(labels3d)
    H_rz[:, 0, 0] = torch.cos(thetaz)
    H_rz[:, 0, 1] = -torch.sin(thetaz)
    H_rz[:, 1, 0] = torch.sin(thetaz)
    H_rz[:, 1, 1] = torch.cos(thetaz)
    H_r = torch.bmm(H_rx, H_ry).bmm(H_rz)

    H = torch.bmm(H_z, H_r)
    labels3d_ = torch.bmm(H, labels3d_.transpose(1, 2)).transpose(1, 2)
    labels3d = labels3d_ + center + param_offset

    noise = torch.randn_like(labels2d).type_as(labels2d)
    labels2d = labels2d + noise
    return labels2d, labels3d

if __name__ == '__main__':
    labels2d = torch.randn((2, 29, 2))
    labels3d = torch.randn((2, 29, 3))
    a, b = data_argumentation(labels2d, labels3d)
    print(torch.norm(a - labels2d), torch.norm(b - labels3d))
