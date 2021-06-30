# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

"""# Import Libraries"""
import os
import time
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from utils.model import select_model
from utils.options import parse_args_function
from utils.dataset import Dataset
from utils.evalutils import EvalUtil
from utils.functions import *

from tqdm import tqdm
from pdb import set_trace

args = parse_args_function()

"""# Load Dataset"""

root = args.input_file

# mean = np.array([120.46480086, 107.89070987, 103.00262132])
# std = np.array([5.9113948 , 5.22646725, 5.47829601])

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

if args.train:
    trainset = Dataset(root=root, load_set='train', transform=transform, no_image=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    print('Train files loaded')

if args.val:
    valset = Dataset(root=root, load_set='val', transform=transform, no_image=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    print('Validation files loaded')

if args.test:
    testset = Dataset(root=root, load_set='test', transform=transform, no_image=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    print('Test files loaded')

"""# Model"""

use_cuda = False
if args.gpu:
    use_cuda = True

model = select_model(args.model_def)

if use_cuda and torch.cuda.is_available():
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=args.gpu_number)

"""# Load Snapshot"""

if args.pretrained_model != '':
    model_dict = model.state_dict()
    params = torch.load(args.pretrained_model, map_location='cpu')
    state_dict = {k: v for k, v in params.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    print("{} loaded.".format(args.pretrained_model))

if args.resume != '':
    model.load_state_dict(torch.load(args.resume))
    print("{} loaded.".format(args.resume))
    losses = np.load(args.resume[:-4].replace('ckpt', 'losses') + '.npy').tolist()
    start = len(losses)
else:
    losses = []
    start = 0

"""# Optimizer"""

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_step_gamma)
scheduler.last_epoch = start

if args.resume != '':
    optimizer.load_state_dict(torch.load(args.resume.replace('ckpt', 'optim')))

lambda_1 = 0.01
lambda_2 = 1

"""# Train"""
best_err, best_epoch = 100, 0

if args.train:
    log_path = os.path.join(args.logs, args.exp)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    save_folder = os.path.join(args.output_file, args.exp)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print('Begin training the network...')

    for epoch in range(start, args.num_iterations):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        train_loss = 0.0
        for i, tr_data in enumerate(trainloader):
            # get the inputs
            # inputs, labels2d, labels3d = tr_data
            labels2d, labels3d = tr_data

            # wrap them in Variable
            # inputs = Variable(inputs)
            labels2d = Variable(labels2d)
            labels3d = Variable(labels3d)

            if use_cuda and torch.cuda.is_available():
                # inputs = inputs.float().cuda(device=args.gpu_number[0])
                labels2d = labels2d.float().cuda(device=args.gpu_number[0])
                labels3d = labels3d.float().cuda(device=args.gpu_number[0])

            optimizer.zero_grad()

            res0, res1 = model(labels2d)
            loss0 = criterion(res0, labels3d)
            loss1 = criterion(res1, labels3d)
            loss = loss0 + loss1

            loss.backward()

            optimizer.step()

            # print statistics
            running_loss += loss.data
            train_loss += loss.data
            if (i + 1) % args.log_batch == 0:  # print every log_iter mini-batches
                print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss / args.log_batch))
                running_loss = 0.0

        losses.append((train_loss / (i + 1)).cpu().numpy())
        writer.add_scalar('Train/loss', (train_loss / (i + 1)).cpu().numpy(), epoch + 1)

        if args.val and (epoch + 1) % args.val_epoch == 0:
            model.eval()
            num_res = 2
            hand_utils = [EvalUtil() for _ in range(num_res)]
            obj_utils = [EvalUtil(num_kp=8) for _ in range(num_res)]
            val_loss = 0.0

            with torch.no_grad():
                for v, val_data in tqdm(enumerate(valloader)):
                    # get the inputs
                    # inputs, labels2d, labels3d = val_data
                    labels2d, labels3d = val_data

                    # wrap them in Variable
                    # inputs = Variable(inputs)
                    labels2d = Variable(labels2d)
                    labels3d = Variable(labels3d)

                    if use_cuda and torch.cuda.is_available():
                        # inputs = inputs.float().cuda(device=args.gpu_number[0])
                        labels2d = labels2d.float().cuda(device=args.gpu_number[0])
                        labels3d = labels3d.float().cuda(device=args.gpu_number[0])

                    outputs3d_init, outputs3d = model(labels2d)
                    loss0 = criterion(outputs3d_init, labels3d)
                    loss1 = criterion(outputs3d, labels3d)
                    loss = loss0 + loss1

                    val_loss += loss.data

                    results = [outputs3d_init, outputs3d]

                    for r in range(num_res):
                        for j in range(len(outputs3d)):
                            hand_utils[r].feed(labels3d[j, :21], results[r][j, :21])
                            obj_utils[r].feed(labels3d[j, 21:], results[r][j, 21:])

            print('val error: %.5f' % (val_loss / (v + 1)))
            hand_errors = [np.mean([util._get_epe(i)[0] for i in range(21)]) for util in hand_utils]
            obj_errors = [np.mean([util._get_epe(i)[0] for i in range(8)]) for util in obj_utils]
            print('mean hand error: {:.5f} {:.5f} mm'.format(*hand_errors))
            print('mean object error: {:.5f} {:.5f} mm'.format(*obj_errors))
            writer.add_scalar('Val/loss', val_loss / (v + 1), epoch + 1)
            writer.add_scalar('Val/Mean hand error', hand_errors[-1], epoch + 1)
            writer.add_scalar('Val/Mean object error', obj_errors[-1], epoch + 1)

        if (epoch + 1) % args.snapshot_epoch == 0:
            filename1 = 'ckpt-{}.pkl'.format(epoch + 1)
            filename2 = 'losses-{}.npy'.format(epoch + 1)
            filename3 = 'optim-{}.pkl'.format(epoch + 1)
            torch.save(model.state_dict(), os.path.join(save_folder, filename1))
            np.save(os.path.join(save_folder, filename2), np.array(losses))
            torch.save(optimizer.state_dict(), os.path.join(save_folder, filename3))

        # Decay Learning Rate
        scheduler.step()

    print('Finished Training')

"""# Test"""

if args.test:
    print('Begin testing the network...')
    model.eval()
    running_loss = 0.0
    num_res = 2
    hand_utils = [EvalUtil() for _ in range(num_res)]
    obj_utils = [EvalUtil(num_kp=8) for _ in range(num_res)]

    with torch.no_grad():
        for i, ts_data in enumerate(testloader):
            # get the inputs
            # inputs, labels2d, labels3d = ts_data
            labels2d, labels3d = ts_data

            # wrap them in Variable
            # inputs = Variable(inputs)
            labels2d = Variable(labels2d)
            labels3d = Variable(labels3d)

            if use_cuda and torch.cuda.is_available():
                # inputs = inputs.float().cuda(device=args.gpu_number[0])
                labels2d = labels2d.float().cuda(device=args.gpu_number[0])
                labels3d = labels3d.float().cuda(device=args.gpu_number[0])

            outputs3d_init, outputs3d = model(labels2d)
            loss0 = criterion(outputs3d_init, labels3d)
            loss1 = criterion(outputs3d, labels3d)
            loss = loss1

            running_loss += loss.data

            results = [outputs3d_init, outputs3d]

            for r in range(num_res):
                for j in range(len(outputs3d)):
                    hand_utils[r].feed(labels3d[j, :21], results[r][j, :21])
                    obj_utils[r].feed(labels3d[j, 21:], results[r][j, 21:])

    print('test error: %.5f' % (running_loss / (i + 1)))
    hand_errors = [np.mean([util._get_epe(i)[0] for i in range(21)]) for util in hand_utils]
    obj_errors = [np.mean([util._get_epe(i)[0] for i in range(8)]) for util in obj_utils]
    print('mean hand error: {:.5f} {:.5f} mm'.format(*hand_errors))
    print('mean object error: {:.5f} {:.5f} mm'.format(*obj_errors))