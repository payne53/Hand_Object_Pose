# -*- coding: utf-8 -*-
# @Time    : 2020/12/2 14:20
# @Author  : Shark
# @Site    : 
# @File    : losses.py
# @Software: PyCharm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace

cam_intr = np.array([[1395.749023, 0, 935.732544],
                     [0, 1395.749268, 540.681030],
                     [0, 0, 1]])

class ReProjectionLoss(nn.Module):
    def __init__(self):
        super(ReProjectionLoss, self).__init__()
        self.cam_intr = torch.tensor(cam_intr, requires_grad=False).float().cuda()
        self.criterion = nn.MSELoss()

    def forward(self, pred, gt):
        return self.criterion(self.convert_3D_to_2d(pred), gt)

    def convert_3D_to_2d(self, points3d):
        batch_size = points3d.shape[0]
        cam_intr = self.cam_intr.expand(batch_size, 3, 3)
        points2d = cam_intr.bmm(points3d.transpose(1, 2)).transpose(1, 2)
        points2d = (points2d / points2d[:, :, 2:])[:, :, :2]
        return points2d


def batch_pairwise_dist(x, y, use_cuda=True):
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    if use_cuda:
        dtype = torch.cuda.LongTensor
    else:
        dtype = torch.LongTensor
    diag_ind_x = torch.arange(0, num_points_x).type(dtype)
    diag_ind_y = torch.arange(0, num_points_y).type(dtype)
    rx = (
        xx[:, diag_ind_x, diag_ind_x]
        .unsqueeze(1)
        .expand_as(zz.transpose(2, 1))
    )
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = rx.transpose(2, 1) + ry - 2 * zz
    return torch.sqrt(P + 1e-6)

def batch_dist(x, y):
    """
    shape: bs, kps, 3
    """
    diff = x - y
    return diff.norm(dim=-1)


def bilinear_sample(u, v):
    return [v * (1 - u), v * u, (1 - v) * (1 - u), u * (1 - v)]


class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.n_kp_hand = 21
        self.n_kp_obj = 8
        self.eta = 1

    def forward(self, poses):
        hand_poses = poses[:, :self.n_kp_hand, :]
        obj_poses = poses[:, self.n_kp_hand:, :]

        length = self.get_obj_info(obj_poses)  # bs, 3

        hand_centers = hand_poses.mean(1)   # bs, 3
        obj_centers = obj_poses.mean(1)     # bs, 3

        dist_center = batch_dist(hand_centers, obj_centers)
        loss_ctr = dist_center / length
        mask = loss_ctr < self.eta
        mask = mask.float()
        loss_ctr = (loss_ctr * mask).sum() / (mask.sum() + 1)
        return loss_ctr

    def get_obj_info(self, obj_poses):
        """
        obj_poses: bs, 8, 3
        """
        ind1 = torch.tensor([0, 0, 0, 7, 7, 7]).type_as(obj_poses).long()
        ind2 = torch.tensor([2, 1, 4, 3, 5, 6]).type_as(obj_poses).long()
        length = batch_dist(obj_poses[:, ind1, :], obj_poses[:, ind2, :]).mean(1, keepdim=True)
        return length


class AffinityLoss(nn.Module):
    def __init__(self):
        super(AffinityLoss, self).__init__()
        self.face_inds = [[0, 1, 2, 3], [0, 4, 2, 6], [0, 1, 4, 5], [1, 3, 5, 7], [2, 3, 6, 7], [4, 5, 6, 7]]
        self.n_grid = 11
        self.n_contacts = 10
        self.n_face = 6
        self.n_kp_hand = 21
        self.n_kp_obj = 8
        self.ita = 0.2
        self.n_points = self.n_grid * self.n_grid
        self.grid = [bilinear_sample(u, v) for u in np.linspace(0, 1, self.n_grid) for v in np.linspace(0, 1, self.n_grid)]
        self.grid = torch.tensor(self.grid)


    def forward(self, poses):
        hand_poses = poses[:, :self.n_kp_hand, :]
        obj_poses = poses[:, self.n_kp_hand:, :]

        face_points = self.sample_face_points(obj_poses)            # bs, 6, 121, 3
        hand_contacts, obj_contacts, dist = self.sample_contact_points(hand_poses, face_points)  # bs, 5, 3

        points1, points2, length = self.get_obj_info(obj_poses)     # bs, 3

        mask = (dist < length * self.ita).unsqueeze(2).float()      # bs, 5, 1
        mask = torch.bmm(mask, mask.transpose(1, 2))                # bs, 5, 5

        direct_vec = points2 - points1                              # bs, 3
        normed_direct_vec = direct_vec / (direct_vec.norm(dim=1, keepdim=True) + 1e-5)      # bs, 3
        vec_contacts = obj_contacts - points1.unsqueeze(1)
        inner = (direct_vec.unsqueeze(1) * vec_contacts).sum(-1)    # bs, 5
        t = inner / (direct_vec.norm(dim=1, keepdim=True) + 1e-5)   # bs, 5
        r = points1.unsqueeze(1) + normed_direct_vec.unsqueeze(1) * t.unsqueeze(2)          # bs, 5, 3
        normal_vec = obj_contacts - r                               # bs, 5, 3
        normal_vec = normal_vec / (normal_vec.norm(dim=2, keepdim=True) + 1e-5)             # bs, 5, 3
        cos_similarity = torch.bmm(normal_vec, normal_vec.transpose(1, 2))                  # bs, 5, 5
        loss_div = (cos_similarity * mask).sum() / (mask.sum() + 1)
        return loss_div

    def get_obj_info(self, obj_poses):
        """
        obj_poses: bs, 8, 3
        """
        points1 = obj_poses[:, :4, :].mean(dim=1)   # bs, 3
        points2 = obj_poses[:, 4:, :].mean(dim=1)   # bs, 3
        ind1 = torch.tensor([0, 1, 2, 3]).type_as(obj_poses).long()
        ind2 = torch.tensor([1, 2, 3, 0]).type_as(obj_poses).long()
        length1 = batch_dist(obj_poses[:, ind1, :], obj_poses[:, ind2, :]).mean(1, keepdim=True)
        length2 = batch_dist(obj_poses[:, ind1 + 4, :], obj_poses[:, ind2 + 4, :]).mean(1, keepdim=True)
        length = (length1 + length2) * 0.5
        return points1, points2, length

    def sample_face_points(self, obj_poses):
        """
        Sample points in each face of obj bbox.
        obj_poses: bs, 8, 3
        """
        bs = obj_poses.shape[0]
        faces = []
        for ind in self.face_inds:
            face = obj_poses[:, ind, :]
            faces.append(face)
        faces = torch.stack(faces, dim=1)               # bs, 6, 4, 3
        faces = faces.transpose(2, 3).view(-1, 3, 4)    # bs * 6, 3, 4
        grid = self.grid.transpose(0, 1).unsqueeze(0).type_as(obj_poses)   # 1, 4, 121
        grid = grid.expand(bs * self.n_face, -1, -1)    # bs * 6, 4, 121
        points = torch.bmm(faces, grid)                 # bs * 6, 3, 121
        points = points.view(bs, self.n_face, 3, self.n_points).transpose(2, 3)
        return points.contiguous()                      # bs, 6, 121, 3

    def sample_contact_points(self, hand_points, obj_points):
        """
        Sample contact localization between hand points and object points.
        hand_points: bs, 21, 3
        obj_points: bs, 6, 121, 3
        """
        bs = hand_points.shape[0]
        obj_points = obj_points.view(bs, self.n_face * self.n_points, 3)            # bs, 6 * 121, 3
        pdist = batch_pairwise_dist(hand_points, obj_points, use_cuda=hand_points.is_cuda)        # bs, 21, 6 * 121
        pdist = pdist.view(bs, self.n_kp_hand, self.n_face, self.n_points)          # bs, 21, 6, 121
        dist_hand2face, idxs_hand2face = torch.min(pdist, 3)                        # bs, 21, 6
        dist_hand2face = dist_hand2face.view(bs, -1)                                # bs, 21 * 6

        dist, inds = torch.sort(dist_hand2face, dim=1, descending=False)            # bs, 21 * 6
        inds_hand, inds_face = inds // self.n_face, inds % self.n_face              # bs, 21 * 6
        inds_hand, inds_face = inds_hand[:, :self.n_contacts], inds_face[:, :self.n_contacts]   # bs, 5
        hand_contacts, obj_contacts = [], []
        for i in range(bs):
            ind_obj = idxs_hand2face[i, inds_hand[i], inds_face[i]]
            hand_contacts.append(hand_points[i, inds_hand[i]])
            obj_contacts.append(obj_points[i, inds_face[i] * self.n_points + ind_obj])

        hand_contacts = torch.stack(hand_contacts, 0)       # bs, 5, 3
        obj_contacts = torch.stack(obj_contacts, 0)         # bs, 5, 3
        dist = dist[:, :self.n_contacts]
        return hand_contacts, obj_contacts, dist


if __name__ == '__main__':
    # criterion = AffinityLoss()
    criterion = CenterLoss()

    labels3d = torch.tensor([
        [  84.3087,  -13.0808,  122.6638],
        [  71.1635,  -13.7556,  118.8352],
        [  75.7617,  -44.4343,  162.8187],
        [  79.3344,  -68.2711,  196.9932],
        [  70.3700,  -76.7703,  223.2307],
        [  66.0437,  -80.8889,  145.9971],
        [  66.8462, -102.7894,  185.1309],
        [  67.4175, -118.3817,  212.9927],
        [  67.8121, -109.9173,  233.2953],
        [  86.0517,  -73.5516,  151.4603],
        [  90.2891, -107.8372,  187.8208],
        [  93.3347, -132.4798,  213.9548],
        [  92.5167, -116.8393,  227.9442],
        [ 101.2401,  -66.8223,  155.0702],
        [ 107.9332,  -87.4588,  195.5296],
        [ 113.0058, -103.0991,  226.1935],
        [ 105.2252,  -89.0046,  239.6772],
        [ 115.3957,  -52.9872,  154.9294],
        [ 120.3458,  -67.4650,  191.7527],
        [ 124.1539,  -78.6031,  220.0813],
        [ 117.2371,  -75.5441,  239.6721],
        [  47.6871, -109.6977,  307.3289],
        [  97.2942,   30.4298,  528.3614],
        [ -15.3903,  -18.4679,  263.6489],
        [  34.2168,  121.6595,  484.6814],
        [ 126.5146,  -74.3847,  267.2502],
        [ 176.1217,   65.7427,  488.2827],
        [  63.4372,   16.8450,  223.5701],
        [ 113.0444,  156.9725,  444.6026]])
    poses = torch.randn((2, 29, 3))
    labels3d = labels3d.unsqueeze(0)

    loss = criterion(labels3d)
    # loss = criterion(labels3d)
    print(loss)
