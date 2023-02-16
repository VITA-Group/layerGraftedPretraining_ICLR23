# References:
# Moco-v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

import argparse
import math
import os
import random
import shutil
import time
import warnings
import copy

import PIL.Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as torchvision_models

from models import vits

from utils.utils import logger, accuracy, sync_weights
from utils.init_datasets import init_datasets
from utils.speed_test import speed_test

from utils.caculate_MINE import measure_MINE
from utils.caculate_entropy_across_tokens import measure_cos_sim_dist_entropy

from thop_modified import profile

from utils_mae.misc import MetricLogger
import itertools

import cyanure as cyan


torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_large', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names
model_names += ['mmseg_vit_base']
model_names += ['beit_base_patch16_224']

gate_names = ['', 'vit_gate_small', 'vit_gate_base', 'vit_gate_large']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('experiment', type=str)
parser.add_argument('--save_dir', type=str, default="checkpoints_moco")

parser.add_argument('--data', metavar='DIR', default="", help='path to dataset')
parser.add_argument('--dataset', default="imagenet", help='dataset')
parser.add_argument('--customSplit', type=str, default='')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1024, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1024), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='GPU id to use.')

parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# additional configs:
parser.add_argument('--pretrained', default='', type=str, help='path to moco pretrained checkpoint')

# vit sep configs:
parser.add_argument('--sep-path', default=0, type=int, help="select sep patch")

parser.add_argument('--fine-tune', action='store_true')
parser.add_argument('--test-interval', type=int, default=1)

# options for distillation
parser.add_argument('--distillation', action='store_true', help='if use distillation')
parser.add_argument('--distillation_checkpoint', default="", type=str)
parser.add_argument('--distillation_temp', default=0.1, type=float)

parser.add_argument('--mae_aug', action="store_true")
parser.add_argument('--CAM', action="store_true")
parser.add_argument('--CAM_block', default=0, type=int, help='path to moco pretrained checkpoint')

parser.add_argument('--MI', action="store_true", help='measure the MI')
parser.add_argument('--cosine_sim_entropy', action="store_true", help='measure the cosine_sim_entropy')

parser.add_argument('--add_batch_norm', action='store_true', help='if add the batch norm')

# options for ablation study
parser.add_argument('--feature_from', default=-1, type=int, help='the layer index for extracting the feature, -1 denotes the last layer')

# options for speed testing
parser.add_argument('--speed_test', action='store_true', help='if test the speed')
parser.add_argument('--profile_model', action='store_true', help='if profile the model')

parser.add_argument('--logstic_reg', action='store_true', help='if employ logstic reg (for few-shot eval)')

parser.add_argument('--norm_beit', action='store_true', help='if employ normalization of beit')

best_acc1 = 0


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', save_dir="checkpoints", only_best=False):
    if not only_best:
        torch.save(state, os.path.join(save_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_dir, filename), os.path.join(save_dir, 'model_best.pth.tar'))


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    logName = "log.txt"
    save_dir = os.path.join(args.save_dir, args.experiment)
    if not os.path.exists(save_dir):
        os.system("mkdir -p {}".format(save_dir))
    log = logger(path=save_dir, log_name=logName)

    main_worker(args.local_rank, args, log)


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000, add_batch_norm=False):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

        self.add_batch_norm = add_batch_norm
        if self.add_batch_norm:
            self.bn = torch.nn.BatchNorm1d(dim, affine=False, eps=1e-6)

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        if self.add_batch_norm:
            x = self.bn(x)

        # linear layer
        return self.linear(x)


def logistic_reg(features_train, features_val, features_test, labels_train, labels_val, labels_test, log):
    def reg(normalize, lambd, features_train, features_val, labels_train, labels_val):
        cyan.preprocess(features_train, normalize=normalize, columns=False, centering=True)
        classifier = cyan.MultiClassifier(loss='multiclass-logistic', penalty='l2', fit_intercept=False)

        lambd /= len(features_train)
        classifier.fit(
            features_train,
            labels_train,
            it0=10,
            lambd=lambd,
            lambd2=lambd,
            nthreads=-1,
            tol=1e-3,
            solver='auto',
            seed=0,
            max_epochs=300)

        train_score = classifier.score(features_train, labels_train)
        cyan.preprocess(features_val, normalize=normalize, columns=False, centering=True)
        val_score = classifier.score(features_val, labels_val)

        return train_score, val_score

    # normSearchList = [True, False]
    # lambdSearchList = [0.0025, 0.00025, 0.000025]

    normSearchList = [True]
    lambdSearchList = [0.00025]

    search_list = []
    for norm in normSearchList:
        search_list += [{"normalize": norm, "lambd": lambd} for lambd in lambdSearchList]

    features_train = features_train.cpu().numpy()
    features_val = features_val.cpu().numpy()
    features_test = features_test.cpu().numpy()
    labels_train = labels_train.cpu().numpy()
    labels_val = labels_val.cpu().numpy()
    labels_test = labels_test.cpu().numpy()

    for search_dict in search_list:
        normalize, lambd = search_dict["normalize"], search_dict["lambd"]
        train_score, val_score = reg(normalize, lambd, features_train, features_val, labels_train, labels_val)
        log.info('for norm: {}, lambd: {}, train score: {}'.format(normalize, lambd, train_score))
        log.info('for norm: {}, lambd: {}, val score: {}'.format(normalize, lambd, val_score))
        search_dict["score"] = val_score

    search_list = sorted(search_list, key=lambda x: x["score"])
    best_search = search_list[-1]
    normalize, lambd = best_search["normalize"], best_search["lambd"]
    # train_score, test_score = reg(normalize, lambd, features_train, features_test, labels_train, labels_test)

    test_score = best_search["score"]

    return test_score

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def CAM(args, test_loader, model, linear_classifiers, criterion, log, best_acc_hidx):
    from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

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

        model_cam = nn.Sequential(model, linear_classifier)
        target_layers = [model_cam[0].blocks[args.CAM_block].mlp.fc1]
        cam = GradCAM(model=model_cam, target_layers=target_layers, use_cuda=True,
                      reshape_transform=reshape_transform)
        targets = [ClassifierOutputTarget(category.cpu().data.numpy()) for category in target]
        grayscale_cams = cam(input_tensor=images, targets=targets)

        with torch.no_grad():
            pred = model_cam(images)
            acc1, acc5 = accuracy(pred, target, topk=(1, 5))
            metric_logger.meters['acc@1'].update(acc1.item(), n=images.shape[0])

        # In this example grayscale_cam has only one image in the batch:
        os.system("mkdir -p {}".format("{}/images/block{}").format(log.path, args.CAM_block))
        for cnt, (image, grayscale_cam) in enumerate(zip(images, grayscale_cams)):
            rgb_img = unnormalize(image)
            rgb_img = rgb_img.permute(1, 2, 0)
            rgb_img = rgb_img.cpu().numpy()

            rgb_img = np.clip(rgb_img, 0, 1)
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            image = PIL.Image.fromarray(visualization)
            image.save("{}/images/block{}/{}.png".format(log.path, args.CAM_block, cnt + i * images.shape[0]))

        if (i + 1) * images.shape[0] > 500:
            break


def main_worker(local_rank, args, log):
    global best_acc1
    args.local_rank = local_rank

    # suppress printing if not master
    if args.multiprocessing_distributed and args.local_rank != 0:
        # def print_pass(*args):
        #     pass
        # builtins.print = print_pass
        log.local_rank = 1

    log.info(str(args))

    if args.local_rank is not None:
        log.info("Use GPU: {} for training".format(args.local_rank))

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method="env://")
        torch.distributed.barrier()

    # Data loading code
    if args.norm_beit:
        log.info("norm_beit")
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    if args.logstic_reg:
        train_datasets, val_datasets, test_datasets = init_datasets(args, transform_test, transform_test)
    else:
        train_datasets, val_datasets, test_datasets = init_datasets(args, transform_train, transform_test)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_datasets)
    else:
        train_sampler = None

    # print("args.batch_size is {}".format(args.batch_size))
    batch_size = int(args.batch_size / torch.distributed.get_world_size())
    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    # Enabling distributed evaluation with an eval dataset not divisible by process number. '
    # 'This will slightly alter validation results as extra duplicate entries are added to achieve '
    # 'equal num of samples per-process, so we only do this for validation
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_datasets)
    val_loader = torch.utils.data.DataLoader(
        val_datasets,
        batch_size=64, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    test_loader = torch.utils.data.DataLoader(
        test_datasets,
        batch_size=16, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    nb_classes = len(np.unique(train_datasets.targets))
    # print("nb_classes is {}".format(nb_classes))

    args.rank = torch.distributed.get_rank()
    # create model
    log.info("=> creating model '{}'".format(args.arch))
    if args.arch.startswith('vit'):
        model = vits.__dict__[args.arch](num_classes=nb_classes, feature_from=args.feature_from)
        linear_keyword = 'head'
    elif args.arch.startswith('mmseg'):
        from models import mmseg_vit
        model = mmseg_vit.__dict__[args.arch](num_classes=nb_classes)
        linear_keyword = 'head'
    elif args.arch.startswith('beit'):
        from models import beit_vit
        model = beit_vit.__dict__[args.arch](num_classes=nb_classes)
        linear_keyword = 'head'
    else:
        assert args.feature_from < 0
        model = torchvision_models.__dict__[args.arch](num_classes=nb_classes)
        linear_keyword = 'fc'

    assert not args.distillation
    model_teacher = None
    if args.distillation:
        model_teacher = copy.deepcopy(model)

    assert not args.fine_tune
    log.info("Conduct linear evaluation")
    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
            param.requires_grad = False

    # init the fc layer
    getattr(model, linear_keyword).weight.data.normal_(mean=0.0, std=0.01)
    getattr(model, linear_keyword).bias.data.zero_()

    log.info(str(model))

    # summary the flops, params
    if args.profile_model:
        flops, params = profile(model.cuda(), inputs=(torch.randn(1, 3, 224, 224, device="cuda"),), verbose=False)
        flops /= 10 ** 9
        params /= 10 ** 6
        log.info("base encoder flops: {:.04}G, params {:.04}M".format(flops, params))
        return

    feat_dim = model.head.in_features
    model.head = torch.nn.Identity()

    args.lrs = [base * scale for scale in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10] for base in [1, 3, 5, 7, 9]]
    # args.lrs = [base * scale for scale in [ 1e-1, ] for base in [1,]]
    args.wds = [0, 1e-6]
    args.optims = ['sgd']

    args.permutes = list(itertools.product(args.lrs, args.wds, args.optims))


    linear_classifiers = nn.ModuleList()
    optimizers = []
    schedulers = []
    # print("barrier")
    # torch.distributed.barrier()
    # print("after barrier")
    # print("args.local_rank is {}".format(args.local_rank))
    for pm in args.permutes:
        lr, wd, _ = pm
        linear_classifier = LinearClassifier(feat_dim, num_labels=nb_classes, add_batch_norm=args.add_batch_norm)
        linear_classifier = linear_classifier.to(args.local_rank)
        linear_classifier = torch.nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.local_rank])
        linear_classifiers.append(linear_classifier)

        # set optimizer
        parameters = linear_classifier.parameters()
        optimizer = torch.optim.SGD(
            parameters,
            lr * args.batch_size * torch.distributed.get_world_size() / 256.,
            weight_decay=wd,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

        optimizers.append(optimizer)
        schedulers.append(scheduler)

    print("set linear classifiers")

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained) or os.path.isdir(args.pretrained):
            log.info("=> loading checkpoint '{}'".format(args.pretrained))
            if os.path.isfile(args.pretrained):
                checkpoint = torch.load(args.pretrained, map_location="cpu")
            else:
                raise ValueError("Model {} do not exist".format(args.pretrained))

            # for mimco pretrain
            if "mimco" in args.pretrained:
                state_dict = checkpoint["model"]
                for key in list(state_dict.keys()):
                    if key.startswith("encoder."):
                        state_dict[key.replace("encoder.", "")] = state_dict[key]
                    del state_dict[key]

                args.start_epoch = 0
                msg = model.load_state_dict(state_dict, strict=False)
                log.info(str(msg))
            elif "mmseg" in args.arch:
                args.start_epoch = 0

                for key in list(checkpoint.keys()):
                    if key.startswith("norm."):
                        checkpoint[key.replace("norm.", "ln1.")] = checkpoint[key]
                        del checkpoint[key]

                msg = model.backbone.load_state_dict(checkpoint, strict=False)
                log.info(str(msg))
            elif "mae" in args.pretrained and "model" in checkpoint:
                state_dict = checkpoint["model"]
                args.start_epoch = 0
                msg = model.load_state_dict(state_dict, strict=False)
                log.info(str(msg))
                # assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                model = cvt_state_dict(state_dict, model, args, linear_keyword)
            else:
                state_dict = checkpoint
                msg = model.load_state_dict(state_dict, strict=False)
                log.info(str(msg))

            log.info("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(args.pretrained))


    if not torch.cuda.is_available():
        raise NotImplementedError()
        log.info('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.local_rank is not None:
            torch.cuda.set_device(args.local_rank)
            model.cuda(args.local_rank)
            if model_teacher is not None:
                model_teacher.cuda(args.local_rank)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / torch.distributed.get_world_size())
            args.workers = args.workers
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)

            if model_teacher is not None:
                model_teacher.cuda()
                model_teacher = torch.nn.parallel.DistributedDataParallel(model_teacher)
    elif args.local_rank is not None:
        raise NotImplementedError()
        torch.cuda.set_device(args.local_rank)
        model = model.cuda(args.local_rank)
    else:
        raise NotImplementedError()
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.local_rank)

    cudnn.benchmark = True

    # optionally resume from a checkpoint
    # assert not args.resume
    best_acc_hidx = -1
    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cpu")

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc']

            if "backbone_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["backbone_state_dict"])

            linear_classifiers.load_state_dict(checkpoint['state_dict'])
            for cnt, opt in enumerate(optimizers):
                opt.load_state_dict(checkpoint['optimizers'][cnt])
            for cnt, sch in enumerate(schedulers):
                sch.load_state_dict(checkpoint['schedulers'][cnt])
            log.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            best_acc_hidx = checkpoint['best_acc_hidx']
        else:
            for _ in range(3):
                print("=> no checkpoint found at '{}', train from scratch !!!!!!!!!!!!".format(args.resume))

    if args.CAM:
        CAM(args, test_loader, model, linear_classifiers, criterion, log, best_acc_hidx)
        return

    if args.MI:
        # measure_MI(args, train_loader, model, linear_classifiers, criterion, log, best_acc_hidx, local=True)
        measure_MINE(args, train_loader, model, linear_classifiers, criterion, log, best_acc_hidx, local=True)
        return

    if args.cosine_sim_entropy:
        measure_cos_sim_dist_entropy(args, val_loader, model, linear_classifiers, criterion, log, best_acc_hidx)
        return

    if model_teacher is not None:
        assert args.distillation_checkpoint != ""
        checkpoint_distill = torch.load(args.distillation_checkpoint, map_location="cpu")

        model_teacher.load_state_dict(checkpoint_distill["state_dict"])
        model_teacher = model_teacher.cuda()
        for param in model_teacher.parameters():
            param.requires_grad = False

    if args.evaluate:
        test_stats = validate(test_loader, model, linear_classifiers, args.permutes, criterion, args, log, prefix="Test: ")
        group_best_acc = 0
        group_best_acc_hidx = 0
        group_best_acc_sweep_lr_only = 0
        for group, pm in enumerate(args.permutes):
            if group % (len(args.wds) * len(args.optims)) == 0:
                group_best_acc_sweep_lr_only = max(group_best_acc_sweep_lr_only, test_stats['acc{}@1'.format(group)])
            # group_best_acc = max(group_best_acc, test_stats['acc{}@1'.format(group)])
            if test_stats['acc{}@1'.format(group)] >= group_best_acc:
                group_best_acc_hidx = group
                group_best_acc = test_stats['acc{}@1'.format(group)]

        log.info(f"Accuracy of the network on the {len(test_loader)} test images: {group_best_acc:.1f}%")
        return

    if model_teacher is not None:
        top_1_avg = validate(test_loader, model_teacher, criterion, args, log, prefix="Test Teacher: ")
        log.info("Top 1 acc for teacher model is {}".format(top_1_avg))

    if args.logstic_reg:
        if not os.path.isfile(os.path.join(log.path, 'save_features.pth')):
            features_train, labels_train = inferFeature(train_loader, model, args, log, gather=True)
            features_val, labels_val = inferFeature(val_loader, model, args, log, gather=True)
            features_test, labels_test = inferFeature(test_loader, model, args, log, gather=False)

            if torch.distributed.get_rank() == 0:
                torch.save({'features_train': features_train, 'labels_train': labels_train,
                            'features_val': features_val, 'labels_val': labels_val,
                            'features_test': features_test, 'labels_test': labels_test},
                           os.path.join(log.path, 'save_features.pth'))
        else:
            features = torch.load(os.path.join(log.path, 'save_features.pth'), map_location='cpu')
            features_train, labels_train = features['features_train'], features['labels_train']
            features_val, labels_val = features['features_val'], features['labels_val']
            features_test, labels_test = features['features_test'], features['labels_test']

        if torch.distributed.get_rank() == 0:
            test_score = logistic_reg(features_train, features_val, features_test, labels_train,
                                      labels_val, labels_test, log)
        torch.distributed.barrier()

        log.info("Best test acc is {}".format(test_score))
        return

    torch.cuda.empty_cache()

    if args.speed_test:
        speed_test(train_loader, model, args, log)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, init_lr, epoch, args)
        # log.info("current lr is {}".format(optimizer.state_dict()['param_groups'][0]['lr']))

        # train for one epoch
        train(train_loader, model, linear_classifiers, criterion, optimizers, epoch, args, log, model_teacher=model_teacher)

        for scheduler in schedulers:
            scheduler.step()

        if epoch % args.test_interval == 0:
            # evaluate on validation set
            test_stats = validate(val_loader, model, linear_classifiers, args.permutes, criterion, args, log)
            group_best_acc = 0
            group_best_acc_hidx = 0
            group_best_acc_sweep_lr_only = 0
            for group, pm in enumerate(args.permutes):
                if group % (len(args.wds) * len(args.optims)) == 0:
                    group_best_acc_sweep_lr_only = max(group_best_acc_sweep_lr_only, test_stats['acc{}@1'.format(group)])
                # group_best_acc = max(group_best_acc, test_stats['acc{}@1'.format(group)])
                if test_stats['acc{}@1'.format(group)] >= group_best_acc:
                    group_best_acc_hidx = group
                    group_best_acc = test_stats['acc{}@1'.format(group)]

            log.info(f"Accuracy of the network on the {len(val_datasets)} val images: {group_best_acc:.1f}%")

            # remember best acc@1 and save checkpoint
            is_best = group_best_acc > best_acc1
            best_acc1 = max(group_best_acc, best_acc1)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank == 0): # only the first GPU saves checkpoint
                save_dict = {
                    "epoch": epoch + 1,
                    "backbone_state_dict": model.state_dict(),
                    "state_dict": linear_classifiers.state_dict(),
                    "optimizers": [optimizer.state_dict() for optimizer in optimizers],
                    "schedulers": [scheduler.state_dict() for scheduler in schedulers],
                    "best_acc": best_acc1,
                    'best_acc_hidx': group_best_acc_hidx,
                    "best_acc_sweep_lr_only": group_best_acc_sweep_lr_only,
                }
                save_checkpoint(save_dict, is_best, save_dir=log.path)

        torch.distributed.barrier()

    torch.cuda.empty_cache()
    # load best model for testing
    checkpoint = torch.load(os.path.join(log.path, 'model_best.pth.tar'), map_location="cpu")
    state_dict = checkpoint['state_dict']
    linear_classifiers.load_state_dict(state_dict)

    if args.mae_aug:
        test_stats_all = []
        for seed in range(3):
            test_stats = validate(test_loader, model, linear_classifiers, args.permutes, criterion, args, log)
            test_stats_all.append(test_stats)

        test_stats_new = {}
        for key in test_stats_all[0]:
            all = []
            for stat in test_stats_all:
                all.append(stat[key])
            test_stats_new[key] = np.mean(all)
            test_stats_new[key+'_var'] = np.std(all)
        test_stats = test_stats_new
    else:
        test_stats = validate(test_loader, model, linear_classifiers, args.permutes, criterion, args, log)

    group_best_acc = 0
    group_best_acc_hidx = 0
    group_best_acc_sweep_lr_only = 0
    for group, pm in enumerate(args.permutes):
        if group % (len(args.wds) * len(args.optims)) == 0:
            group_best_acc_sweep_lr_only = max(group_best_acc_sweep_lr_only, test_stats['acc{}@1'.format(group)])
        # group_best_acc = max(group_best_acc, test_stats['acc{}@1'.format(group)])
        if test_stats['acc{}@1'.format(group)] >= group_best_acc:
            group_best_acc_hidx = group
            group_best_acc = test_stats['acc{}@1'.format(group)]

    if args.mae_aug:
        group_best_acc_var = test_stats['acc{}@1_var'.format(group_best_acc_hidx)]
        log.info(f"Final Test Accuracy of the network on the {len(test_datasets)} test images: {group_best_acc:.1f}+-{group_best_acc_var:.2f}%")
    else:
        log.info(f"Final Test Accuracy of the network on the {len(test_datasets)} test images: {group_best_acc:.1f}%")
    log.info("Best test acc is in lr {} wd {} optim {}".format(*args.permutes[group_best_acc_hidx]))


def random_masking_gene_id(N, L, device, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]

    return ids_restore, ids_keep


def train(train_loader, model, linear_classifiers, criterion, optimizers, epoch, args, log, model_teacher=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    metric_logger = MetricLogger(delimiter="  ", log=log)
    # losses = AverageMeter('Loss', ':.4e')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, top1, top5],
    #     prefix="Epoch: [{}]".format(epoch),
    #     log=log)

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    if not args.fine_tune:
        model.eval()
    else:
        model.train()
    linear_classifiers.train()

    end = time.time()
    header = 'Epoch: [{}]'.format(epoch)
    for i, (images, target) in enumerate(metric_logger.log_every(train_loader, args.print_freq, header)):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.local_rank is not None:
            images = images.cuda(args.local_rank, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.local_rank, non_blocking=True)

        if model_teacher is not None:
            with torch.no_grad():
                teacher_logits = model_teacher.eval()(images)

        # compute output
        if args.mae_aug:
            _, mask_ids_keep = random_masking_gene_id(images.shape[0], model.patch_embed.num_patches,
                                                      images.device, mask_ratio=0.75)
        else:
            mask_ids_keep = None

        if "mmseg" in args.arch or "beit" in args.arch:
            features = model(images)
        else:
            features = model(images, mask_ids_keep=mask_ids_keep)

        losses = []
        for linear_classifier, optimizer in zip(linear_classifiers, optimizers):
            pred = linear_classifier(features)
            # compute cross entropy loss
            loss = nn.CrossEntropyLoss()(pred, target)
            optimizer.zero_grad()
            loss.backward()
            # step
            optimizer.step()
            losses.append(loss.item())

        torch.cuda.synchronize()

        metric_logger.update(loss_min=np.min(losses))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    log.info("Averaged stats: {}".format(metric_logger))


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def validate(val_loader, model, linear_classifiers, permutes, criterion, args, log, prefix="Validation: "):
    # switch to evaluate mode
    model.eval()
    linear_classifiers.eval()

    metric_logger = MetricLogger(delimiter="  ", log=log)
    header = 'Test:'

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(metric_logger.log_every(val_loader, 50, header)):
            if args.local_rank is not None:
                images = images.cuda(args.local_rank, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.local_rank, non_blocking=True)

            # compute output
            if args.mae_aug:
                _, mask_ids_keep = random_masking_gene_id(images.shape[0], model.patch_embed.num_patches,
                                                          images.device, mask_ratio=0.75)
            else:
                mask_ids_keep = None

            if "mmseg" in args.arch or "beit" in args.arch:
                output = model(images)
            else:
                output = model(images, mask_ids_keep=mask_ids_keep)

            losses = []
            acc1s = []
            acc5s = []
            for group, linear_classifier in enumerate(linear_classifiers):

                pred = linear_classifier(output)
                loss = nn.CrossEntropyLoss()(pred, target)
                losses.append(loss)

                acc1, acc5 = accuracy(pred, target, topk=(1, 5))
                acc1s.append(acc1)
                acc5s.append(acc5)

                batch_size = images.shape[0]
                metric_logger.update(**{'loss{}'.format(group): loss.item()})
                metric_logger.meters['acc{}@1'.format(group)].update(acc1.item(), n=batch_size)
                if linear_classifier.module.num_labels >= 5:
                    metric_logger.meters['acc{}@5'.format(group)].update(acc5.item(), n=batch_size)

        metric_logger.synchronize_between_processes()
        log_msg = ""
        for group, (pm, linear_classifier) in enumerate(zip(permutes, linear_classifiers)):
            lr, wd, optim = pm
            if linear_classifier.module.num_labels >= 5:
                log_msg += '* [Lr {lr:.5f} Wd {wd:.0e} Optim {optim:4}] Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}\n'\
                           .format(lr=lr, wd=wd, optim=optim,
                        top1=metric_logger.meters['acc{}@1'.format(group)],
                        top5=metric_logger.meters['acc{}@5'.format(group)],
                        losses=metric_logger.meters['loss{}'.format(group)])
            else:
                log_msg += '* [Lr {lr:.5f} Wd {wd:.0e} Optim {optim:4}] Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}\n'\
                           .format(lr=lr, wd=wd, optim=optim,
                                top1=metric_logger.meters['acc{}@1'.format(group)],
                                losses=metric_logger.meters['loss{}'.format(group)])

        log.info(log_msg)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def inferFeature(loader, model, args, log, gather=True):
    # switch to evaluate mode
    model.eval()

    metric_logger = MetricLogger(delimiter="  ", log=log)
    header = 'Infer:'

    features = []
    labels = []

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(metric_logger.log_every(loader, 50, header)):
            if args.local_rank is not None:
                images = images.cuda(args.local_rank, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.local_rank, non_blocking=True)

            # compute output
            if args.mae_aug:
                _, mask_ids_keep = random_masking_gene_id(images.shape[0], model.patch_embed.num_patches,
                                                          images.device, mask_ratio=0.75)
            else:
                mask_ids_keep = None

            if "mmseg" in args.arch or "beit" in args.arch:
                output = model(images)
            else:
                output = model(images, mask_ids_keep=mask_ids_keep)

            if gather:
                output = concat_all_gather(output.contiguous())
                target = concat_all_gather(target.contiguous())

            features.append(output.detach().cpu())
            labels.append(target.detach().cpu())

    metric_logger.synchronize_between_processes()

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def sanity_check(state_dict, pretrained_weights, linear_keyword, log):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    log.info("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore linear layer
        if '%s.weight' % linear_keyword in k or '%s.bias' % linear_keyword in k:
            continue

        # name in pretrained model
        k_pre = 'module.base_encoder.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.base_encoder.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    log.info("=> sanity check passed.")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log = log

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.log.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def cvt_state_dict(state_dict, model, args, linear_keyword):
    # rename moco pre-trained keys
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
            if "_aux" in k:
                # print("skip k is {}".format(k))
                continue
            # remove prefix
            state_dict[k[len("module.base_encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    args.start_epoch = 0
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    # assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}

    return model



if __name__ == '__main__':
    main()
