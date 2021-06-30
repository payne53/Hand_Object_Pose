# -*- coding: utf-8 -*-
# @Time    : 2020/10/22 13:48
# @Author  : Shark
# @Site    : 
# @File    : visual_example.py
# @Software: PyCharm

import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable

from utils.model import select_model
from utils.options import parse_args_function
from utils.dataset import Dataset
from utils.evalutils import EvalUtil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from pdb import set_trace

import argparse

cam_extr = np.array([[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
                     [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
                     [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
                     [0, 0, 0, 1]])
cam_intr = np.array([[1395.749023, 0, 935.732544],
                     [0, 1395.749268, 540.681030],
                     [0, 0, 1]])


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def draw_3d_skeleton(ax, pose_cam_xyz, is_gt = False):
    """
    :param pose_cam_xyz: 21 x 3
    :param image_size: H, W
    :return:
    """
    #     assert pose_cam_xyz.shape[0] == 21

    #     fig = plt.figure()
    #     fig.set_size_inches(float(image_size[0]) / fig.dpi, float(image_size[1]) / fig.dpi, forward=True)

    #     ax = plt.subplot(111, projection='3d')
    marker_sz = 15
    line_wd = 2
    if is_gt:
        color_hand_joints = np.zeros((21, 3))
    else:
        # color_hand_joints = [[1.0, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],
        #                      # thumb
        #                      [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
        #                      [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
        #                      [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
        #                      [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little
        color_hand_joints = ['r', 'r', 'r', 'r', 'r',
                             'm', 'm', 'm', 'm',
                             'b', 'b', 'b', 'b',
                             'c', 'c', 'c', 'c',
                             'g', 'g', 'g', 'g']

    for joint_ind in range(pose_cam_xyz.shape[0]):
        ax.plot(pose_cam_xyz[joint_ind:joint_ind + 1, 0], pose_cam_xyz[joint_ind:joint_ind + 1, 1],
                pose_cam_xyz[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)

        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            ax.plot(pose_cam_xyz[[0, joint_ind], 0], pose_cam_xyz[[0, joint_ind], 1], pose_cam_xyz[[0, joint_ind], 2],
                    color=color_hand_joints[joint_ind], lineWidth=line_wd)
        else:
            ax.plot(pose_cam_xyz[[joint_ind - 1, joint_ind], 0], pose_cam_xyz[[joint_ind - 1, joint_ind], 1],
                    pose_cam_xyz[[joint_ind - 1, joint_ind], 2], color=color_hand_joints[joint_ind], linewidth=line_wd)

    ax.axis('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def draw_bbox(ax, joints, is_gt = False):
    marker_sz = 15
    line_wd = 2
    links = [(0, 1, 3, 2, 0), (4, 5, 7, 6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    if is_gt:
        color = 'black'
    else:
        color = 'cyan'

    for i in range(joints.shape[0]):
        ax.plot(joints[i:i + 1, 0], joints[i:i + 1, 1], joints[i:i + 1, 2], '.', c=color, markersize=marker_sz)

    for link in links:
        for j in range(len(link) - 1):
            ax.plot(joints[[link[j], link[j + 1]], 0], joints[[link[j], link[j + 1]], 1],
                    joints[[link[j], link[j + 1]], 2], color=color, lineWidth=line_wd)


# Display utilities
def visualize_joints_2d(ax, joints, joint_idxs=True, links=None, alpha=1, is_gt=False):
    """Draw 2d skeleton on matplotlib axis"""
    if links is None:
        links = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
    # Scatter hand joints on image
    x = joints[:, 0]
    y = joints[:, 1]
    ax.scatter(x, y, 1, 'r')

    # Add idx labels to joints
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            plt.annotate(str(row_idx), (row[0], row[1]))
    _draw2djoints(ax, joints, links, alpha=alpha, is_gt=is_gt)


def _draw2djoints(ax, annots, links, alpha = 1, is_gt = False):
    """Draw segments, one color per link"""
    if is_gt:
        colors = ['black'] * len(links)
    elif len(links) == 5:
        colors = ['r', 'm', 'b', 'c', 'g']
    else:
        colors = ['cyan'] * len(links)

    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            _draw2dseg(ax, annots, finger_links[idx], finger_links[idx + 1], c=colors[finger_idx], alpha=alpha)


def _draw2dseg(ax, annot, idx1, idx2, c = 'r', alpha = 1):
    """Draw segment of given color"""
    ax.plot([annot[idx1, 0], annot[idx2, 0]], [annot[idx1, 1], annot[idx2, 1]], c=c, alpha=alpha)


def convert_3D_to_2d(verts_camcoords):
    verts_hom2d = np.array(cam_intr).dot(verts_camcoords.transpose()).transpose()
    verts_proj = (verts_hom2d / verts_hom2d[:, 2:])[:, :2]
    return verts_proj


parser = argparse.ArgumentParser()
parser.add_argument('--model_def', type=str, default='hopenet')
parser.add_argument('--ckpt', type=str, default='./checkpoints/exp1/ckpt-7000.pkl')
parser.add_argument('--i', type=int, default=100)
parser.add_argument('--type', type=str, default='full')
args = parser.parse_args()

model = select_model(args.model_def)
model = nn.DataParallel(model, device_ids=[0])
model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
model.eval()

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])
testset = Dataset(root='datasets/fhad/', load_set='test', transform=transform)

i = args.i
print(testset.images[i])
inputs, labels2d, labels3d = testset.__getitem__(i)
labels2d = torch.from_numpy(labels2d).type_as(inputs)
# labels3d = torch.from_numpy(labels3d).type_as(inputs)

if args.type == 'full':
    inputs = inputs.unsqueeze(0)

    if args.model_def.lower() == 'hopenet':
        outputs2d_init, outputs2d, outputs3d_3 = model(inputs)
    else:
        outputs2d_init, outputs2d, outputs3d_init, outputs3d_3 = model(inputs)

    outputs3d = outputs3d_3.detach().cpu().numpy()[0]
    outputs2d = outputs2d.detach().cpu().numpy()[0]
else:
    labels2d = labels2d.unsqueeze(0)
    outputs3d = model(labels2d)
    outputs3d = outputs3d.detach().cpu().numpy()[0]
    outputs2d = labels2d.detach().cpu().numpy()[0]
    labels2d = labels2d.detach().cpu().numpy()[0]

image_size = (800, 800)
# fig = plt.figure()

fig = plt.gcf()
fig.set_size_inches(float(image_size[0]) / fig.dpi, float(image_size[1]) / fig.dpi, forward=True)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
ax = plt.subplot(111, projection='3d')
# ax.invert_zaxis()
# ax.invert_yaxis()
# ax.invert_xaxis()
draw_3d_skeleton(ax, labels3d[:21], is_gt=True)
draw_3d_skeleton(ax, outputs3d[:21])
draw_bbox(ax, labels3d[21:], is_gt=True)
draw_bbox(ax, outputs3d[21:])

# draw_3d_skeleton(ax, labels3d[:21], is_gt=False)
# draw_bbox(ax, labels3d[21:], is_gt=False)

res = testset.images[i].split('/')
name = 'visuals/{}-{}-{}-{}-3d.png'.format(res[-5], res[-4], res[-3], res[-1].split('.')[0].split('_')[1])
plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
plt.margins(0,0,0)
plt.savefig(name)
plt.show()

# # Plot everything
plt.cla()
plt.clf()
plt.close()
fig = plt.gcf()
# Load image and display
ax = fig.add_subplot(111)
ax.set_aspect('equal')
img_path = testset.images[i]
print('Loading image from {}'.format(img_path))
img = Image.open(img_path)
width, height = img.size
fig.set_size_inches(width/100.0/3.0, height/100.0/3.0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
ax.imshow(img, aspect='equal')
links_bbox = [(0, 1, 3, 2, 0), (4, 5, 7, 6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

visualize_joints_2d(ax, labels2d[:21], joint_idxs=False, is_gt=True)
visualize_joints_2d(ax, convert_3D_to_2d(outputs3d[:21]), joint_idxs=False, is_gt=False)
visualize_joints_2d(ax, labels2d[21:], links=links_bbox, joint_idxs=False, is_gt=True)
visualize_joints_2d(ax, convert_3D_to_2d(outputs3d[21:]), links=links_bbox, joint_idxs=False, is_gt=False)

# visualize_joints_2d(ax, labels2d[:21], joint_idxs=False, is_gt=False)
# visualize_joints_2d(ax, labels2d[21:], links=links_bbox, joint_idxs=False, is_gt=False)

res = testset.images[i].split('/')
name = 'visuals/{}-{}-{}-{}-2d.png'.format(res[-5], res[-4], res[-3], res[-1].split('.')[0].split('_')[1])
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
plt.margins(0,0)
plt.savefig(name)
plt.show()