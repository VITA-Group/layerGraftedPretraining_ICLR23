from scipy.spatial.distance import pdist, squareform
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pdb import set_trace
from utils_mae.misc import MetricLogger
from utils.utils import logger, accuracy


class Mine(nn.Module):
    def __init__(self, feat1_size=3, feat2_size=2, output_size=1, hidden_size=128):
        super().__init__()
        self.fc1_noise = nn.Linear(feat1_size, hidden_size, bias=False)
        self.fc1_sample = nn.Linear(feat2_size, hidden_size, bias=False)
        self.fc1_bias = nn.Parameter(torch.zeros(hidden_size))
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.ma_et = None

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, noise, sample):
        x_noise = self.fc1_noise(noise)
        x_sample = self.fc1_sample(sample)
        # set_trace()
        x = F.leaky_relu(x_noise + x_sample + self.fc1_bias.unsqueeze(0), negative_slope=2e-1)
        x = F.leaky_relu(self.fc2(x), negative_slope=2e-1)
        x = F.leaky_relu(self.fc3(x), negative_slope=2e-1)
        return x


def measure_MINE(args, train_loader, model, linear_classifiers, criterion, log, best_acc_hidx, local=False):

    model.eval()
    linear_classifiers.eval()

    # apply the CAM
    linear_classifier = linear_classifiers[best_acc_hidx]

    # switch to evaluate mode
    model.eval()
    model.requires_grad_()
    linear_classifier.eval()

    # hyper-parameters
    ## compute MI on 3, 6, 9, 12 layers
    sampled_layers = [2, 5, 8, 11]
    lr_candidates = [0.01, 0.001, 0.0001, 0.00001]

    print("local is {}".format(local))
    if local:
        epochs = 10
        image_size = 3 * 16 * 16
        label_size = 1000
        feat_size = linear_classifier.module.linear.in_features
    else:
        epochs = 10
        image_size = 3 * 224 * 224
        label_size = 1000
        feat_size = linear_classifier.module.linear.in_features * (model.patch_embed.num_patches + 1)

    # init linear classifiers
    M_layers = {}
    for sampled_layer in sampled_layers:
        name_image = "layer{}_image".format(sampled_layer)
        name_label = "layer{}_label".format(sampled_layer)

        for lr in lr_candidates:
            M_layers["{}_lr{}".format(name_image, lr)] = Mine(feat1_size=image_size, feat2_size=feat_size, hidden_size=256)
            M_layers["{}_lr{}".format(name_label, lr)] = Mine(feat1_size=label_size, feat2_size=feat_size, hidden_size=256)

    name_image = "layerPred_image"
    name_label = "layerPred_label"

    for lr in lr_candidates:
        M_layers["{}_lr{}".format(name_image, lr)] = Mine(feat1_size=image_size, feat2_size=label_size, hidden_size=256)
        M_layers["{}_lr{}".format(name_label, lr)] = Mine(feat1_size=label_size, feat2_size=label_size, hidden_size=256)

    for layer_name in list(M_layers.keys()):
        layer = M_layers[layer_name].to(args.local_rank)
        layer = torch.nn.parallel.DistributedDataParallel(layer, device_ids=[args.local_rank])
        M_layers[layer_name] = layer

    # init optimizers & schedulers
    optimizers = {}
    schedulers = {}

    for layer_name, layer in M_layers.items():
        # set optimizer
        parameters = layer.parameters()
        lr = float(layer_name.split("_lr")[-1])
        log.info("employ lr {} for layer {}".format(lr, layer_name))
        optimizer = torch.optim.SGD(
            parameters,
            lr * torch.distributed.get_world_size() / 256.,
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)

        optimizers[layer_name] = optimizer
        schedulers[layer_name] = scheduler

    for epoch in range(epochs):
        metric_logger = MetricLogger(delimiter="  ", log=log)
        header = 'MI epoch {}/{}:'.format(epoch, epochs)

        for i, (images, target) in enumerate(metric_logger.log_every(train_loader, 10, header)):
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

            if not local:
                sampled_layers = sampled_layers + ["Pred", ]
                feature_measure = [f for f in torch.stack(features)[sampled_layers]] + [pred, ]
            else:
                feature_measure = [f for f in torch.stack(features)[sampled_layers]]

            # set_trace()
            for cnt_layer, feature in zip(sampled_layers, feature_measure):
                Ixf_list = []
                Iyf_list = []
                for lr in lr_candidates:
                    # optimize the MI function
                    name_image = "layer{}_image_lr{}".format(cnt_layer, lr)
                    name_label = "layer{}_label_lr{}".format(cnt_layer, lr)

                    Ixf, Iyf = learn_mine(images, target, feature, M_layers[name_image], M_layers[name_label],
                                          M_opt=optimizers[name_image], M_target_opt=optimizers[name_label],
                                          local=local)

                    Ixf_list.append(Ixf)
                    Iyf_list.append(Iyf)

                metric_logger.meters['Ixf@{}'.format(cnt_layer)].update(np.nanmax(Ixf_list), n=images.shape[0])
                metric_logger.meters['Iyf@{}'.format(cnt_layer)].update(np.nanmax(Iyf_list), n=images.shape[0])

        for key, sch in schedulers.items():
            sch.step()


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


def learn_mine(images, target, feature, M, M_target, M_opt, M_target_opt, ma_rate=0.001, local=False):
    '''
    Mine is learning for MI of (input, output) of Generator.
    '''

    if local:
        # patchify
        images = patchify(images, patch_size=16)
        batch_size = images.shape[0]
        L = images.shape[1]
        images = images.reshape(batch_size*L, -1)
        feature = feature[:, 1:].reshape(batch_size*L, -1)
    else:
        batch_size = images.shape[0]
        images = images.reshape(batch_size, -1)
        feature = feature.reshape(batch_size, -1)

    feature_bar = feature[torch.randperm(len(feature))]

    # optimize the M
    et = torch.mean(torch.exp(M(images, feature_bar)))
    if M.module.ma_et is None:
        M.module.ma_et = et.detach().item()
    M.module.ma_et += ma_rate * (et.detach().item() - M.module.ma_et)

    mutual_information = torch.mean(M(images, feature)) - torch.log(et) * et.detach() / M.module.ma_et
    loss = - mutual_information

    M_opt.zero_grad()
    loss.backward()
    M_opt.step()

    # optimize the M_target
    target_oneHot = F.one_hot(target, num_classes=1000).float()

    if local:
        target_oneHot = target_oneHot.unsqueeze(1).expand(-1, L, -1)
        target_oneHot = target_oneHot.reshape(batch_size * L, -1)

    et_target = torch.mean(torch.exp(M_target(target_oneHot, feature_bar)))
    if M_target.module.ma_et is None:
        M_target.module.ma_et = et_target.detach().item()
    M_target.module.ma_et += ma_rate * (et_target.detach().item() - M_target.module.ma_et)

    mutual_information_target = torch.mean(M_target(target_oneHot, feature)) - \
                                torch.log(et_target) * et_target.detach() / M_target.module.ma_et
    loss = - mutual_information_target

    M_target_opt.zero_grad()
    loss.backward()
    M_target_opt.step()

    return mutual_information.item(), mutual_information_target.item()
