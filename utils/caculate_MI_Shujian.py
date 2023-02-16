# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 21:08:10 2021
@author: Xi Yu, Shujian Yu
"""

from scipy.spatial.distance import pdist, squareform
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from utils_mae.misc import MetricLogger
from utils.utils import logger, accuracy, sync_weights
import time
from pdb import set_trace


def pairwise_distances(x):
    # x should be two dimensional
    b = x.shape[0]
    x = x.view(b, -1)
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def calculate_gram_mat(x, sigma):
    dist = pairwise_distances(x)
    # dist = dist/torch.max(dist)
    return torch.exp(-dist / sigma)


def reyi_entropy(x, sigma):
    alpha = 1.01
    print("pairwise distance")
    k = calculate_gram_mat(x, sigma)
    k = k / torch.trace(k)
    print("compute eigv")
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy(x, y, s_x, s_y):
    alpha = 1.01
    # print("x shape is {}, y shape is {}".format(x.shape, y.shape))
    x = calculate_gram_mat(x, s_x)
    y = calculate_gram_mat(y, s_y)
    k = torch.mul(x, y)
    k = k / torch.trace(k)
    print("pairwise distance joint entropy")
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))

    return entropy


def calculate_MI(x, y, s_x, s_y):
    Hx = reyi_entropy(x, sigma=s_x)
    Hy = reyi_entropy(y, sigma=s_y)
    Hxy = joint_entropy(x, y, s_x, s_y)
    Ixy = Hx + Hy - Hxy
    # normlize = Ixy/(torch.max(Hx,Hy))

    return Ixy


def patchify(imgs, patch_size=None):
    """
    imgs: (N, C, H, W)
    x: (N, L, patch_size**2 *c)
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    c = imgs.shape[1]
    x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
    return x


def measure_MI(args, test_loader, model, linear_classifiers, criterion, log, best_acc_hidx, local=False):

    model.eval()
    linear_classifiers.eval()

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

    # apply the CAM
    linear_classifier = linear_classifiers[best_acc_hidx]

    # switch to evaluate mode
    model.eval()
    model.requires_grad_()
    linear_classifier.eval()

    metric_logger = MetricLogger(delimiter="  ", log=log)
    header = 'Test:'

    end = time.time()
    for i, (images, target) in enumerate(metric_logger.log_every(test_loader, 10, header)):
        if args.local_rank is not None:
            images = images.cuda(args.local_rank, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.local_rank, non_blocking=True)

        with torch.no_grad():
            pred = linear_classifier(model(images, record_feat=True))
            features = model.recorded_feature
            model.recorded_feature = None

            acc1, acc5 = accuracy(pred, target, topk=(1, 5))
            metric_logger.meters['acc@1'].update(acc1.item(), n=images.shape[0])

        # compute MI on 3, 6, 9, 12 layers
        sampled_layers = [2, 5, 8, 11]

        with torch.no_grad():
            target_numpy = target.float().cpu().detach().numpy()
            target_numpy = target_numpy.reshape(target.shape[0], -1)
            k_target = squareform(pdist(target_numpy, 'euclidean'))  # Calculate Euclidiean distance between all samples.
            sigma_target = np.mean(np.mean(np.sort(k_target[:, :10], 1)))

            inputs_numpy = images.cpu().detach().numpy()
            inputs_numpy = inputs_numpy.reshape(images.shape[0], -1)
            k_input = squareform(pdist(inputs_numpy, 'euclidean'))
            sigma_input = np.mean(np.mean(np.sort(k_input[:, :10], 1)))

        if local:
            # patchify
            images = patchify(images, patch_size=16)
            batch_size = images.shape[0]
            L = images.shape[1]
            images = images.reshape(batch_size * L, -1)
            features = [f[:, 1:].reshape(batch_size * L, -1) for f in features]
            pred = pred.unsqueeze(1).expand(-1, L, -1).reshape(batch_size * L, -1)
            target = target.float().unsqueeze(1).expand(-1, L).reshape(batch_size * L, -1)

        feature_measure = [f for f in torch.stack(features)[sampled_layers]] + [pred, ]

        for cnt_layer, feature in zip(sampled_layers + ["Pred", ], feature_measure):
            with torch.no_grad():
                first_dim = images.shape[0]

                print("compute Z")
                Z_numpy = feature.cpu().detach().numpy()
                Z_numpy = Z_numpy.reshape(first_dim, -1)
                k = squareform(pdist(Z_numpy, 'euclidean'))  # Calculate Euclidiean distance between all samples.
                print("k shape is {}".format(k.shape))
                sigma_z = np.mean(np.mean(np.sort(k[:, :10], 1)))

                print("compute MI")
                Ixf = calculate_MI(feature, images, sigma_z, sigma_input)
                Iyf = calculate_MI(feature, target.float(), sigma_z, sigma_target)

                metric_logger.meters['Ixf@{}'.format(cnt_layer)].update(Ixf.item(), n=images.shape[0])
                metric_logger.meters['Iyf@{}'.format(cnt_layer)].update(Iyf.item(), n=images.shape[0])



