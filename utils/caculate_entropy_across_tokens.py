# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 21:08:10 2021
@author: Xi Yu, Shujian Yu
"""

import os
from scipy.spatial.distance import pdist, squareform
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from utils_mae.misc import MetricLogger
from utils.utils import logger, accuracy
from pdb import set_trace

import torchvision.transforms as transforms
import PIL
# def pairwise_distances(x):
#     # x should be two dimensional
#     return


def cos_sim_entropy_measure(x):
    # remove cls head
    x = x[:, 1:]
    # remove the mean vector of each sample size: (B, N, C)
    x = x - x.mean(dim=1, keepdims=True)
    # normalize the channel dim
    x = F.normalize(x, dim=-1)

    pair_cos_sims = torch.bmm(x, x.permute(0, 2, 1))

    # compute pair_dist
    sample_pts = (torch.ones_like(pair_cos_sims) - torch.eye(pair_cos_sims.shape[1]).to(pair_cos_sims.device).unsqueeze(0)).bool()
    b = pair_cos_sims.shape[0]
    pair_cos_sims = pair_cos_sims[sample_pts].reshape(b, -1)

    # compute the distribution
    entropies = []
    distributions = []
    cos_sim_means = []
    for pair_cos_sim in pair_cos_sims:
        pair_dist = torch.histc(pair_cos_sim, 100, min=-1, max=1)
        pair_dist = pair_dist / pair_dist.sum()
        # compute the entropy
        entropy = Categorical(probs=pair_dist).entropy()
        entropies.append(entropy.item())
        distributions.append(pair_dist.detach().cpu().numpy())
        cos_sim_means.append(pair_cos_sim.mean().detach().cpu().numpy())

    return entropies, distributions, cos_sim_means


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


def measure_cos_sim_dist_entropy(args, test_loader, model, linear_classifiers, criterion, log, best_acc_hidx):

    model.eval()
    linear_classifiers.eval()

    # apply the CAM
    linear_classifier = linear_classifiers[best_acc_hidx]

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

    # switch to evaluate mode
    model.eval()
    model.requires_grad_()
    linear_classifier.eval()

    metric_logger = MetricLogger(delimiter="  ", log=log)
    header = 'Test:'

    distributions_all = {}
    for i, (images, target) in enumerate(metric_logger.log_every(test_loader, 10, header)):
        if args.local_rank is not None:
            images = images.cuda(args.local_rank, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.local_rank, non_blocking=True)

        with torch.no_grad():
            pred = linear_classifier(model(images, record_feat=True, record_feat_attn=True))
            features = model.recorded_feature
            model.recorded_feature = None

            acc1, acc5 = accuracy(pred, target, topk=(1, 5))
            metric_logger.meters['acc@1'].update(acc1.item(), n=images.shape[0])

        # compute MI on 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 layers
        sampled_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        entropies_save = {}

        # compute imgs
        with torch.no_grad():
            feature_img = patchify(images, patch_size=16)
            entropies, distributions, cos_sim_means = cos_sim_entropy_measure(feature_img)
            metric_logger.meters['Entropy@{}'.format("img")].update(np.mean(entropies), n=images.shape[0])
            # metric_logger.meters['Mean@{}'.format("img")].update(np.mean(cos_sim_means), n=images.shape[0])
            entropies_save["layer_{}".format("img")] = distributions[0]

            key_distributions_all = "layer_{}".format("img")
            if key_distributions_all in distributions_all:
                distributions_all[key_distributions_all] += np.array(distributions).sum(0)
            else:
                distributions_all[key_distributions_all] = np.array(distributions).sum(0)

        for cnt_layer, feature in zip(sampled_layers, torch.stack(features)[sampled_layers]):
            with torch.no_grad():
                entropies, distributions, cos_sim_means = cos_sim_entropy_measure(feature)
                metric_logger.meters['Entropy@{}'.format(cnt_layer)].update(np.mean(entropies), n=images.shape[0])
                # metric_logger.meters['Mean@{}'.format(cnt_layer)].update(np.mean(cos_sim_means), n=images.shape[0])
                entropies_save["layer_{}".format(cnt_layer)] = distributions[0]

                key_distributions_all = "layer_{}".format(cnt_layer)
                if key_distributions_all in distributions_all:
                    distributions_all[key_distributions_all] += np.array(distributions).sum(0)
                else:
                    distributions_all[key_distributions_all] = np.array(distributions).sum(0)

        # save 1 evey 10 iters
        if torch.distributed.get_rank() == 0 and i % 10 == 0:
            save_path = os.path.join(log.path, "save")
            os.system("mkdir -p {}".format(save_path))

            np.save(os.path.join(save_path, "img_{}_dist.npy".format(i)), entropies_save)

            # save img
            rgb_img = unnormalize(images[0])
            rgb_img = rgb_img.permute(1, 2, 0)
            rgb_img = rgb_img.cpu().numpy()

            rgb_img = (np.clip(rgb_img, 0, 1) * 255).astype(np.uint8)
            image = PIL.Image.fromarray(rgb_img)
            image.save(os.path.join(save_path, "img_{}.jpg".format(i)))

    # save the avg distributions
    normalize = np.sum(list(distributions_all.values()))
    distributions_all = {key: item/normalize for key, item in distributions_all.items()}
    log.info("distribution_sum is {}".format(str(distributions_all)))
    save_path = os.path.join(log.path, "save")
    np.save(os.path.join(save_path, "dist_all.npy"), distributions_all)
