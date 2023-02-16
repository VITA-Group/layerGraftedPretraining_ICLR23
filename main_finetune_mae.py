# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import math
import sys
from typing import Iterable, Optional

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

# assert timm.__version__ == "0.3.2"  # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy

import utils_mae.lr_decay as lrd
import utils_mae.misc as misc
from utils_mae.datasets import build_transform
from utils_mae.pos_embed import interpolate_pos_embed
from utils_mae.misc import NativeScalerWithGradNormCount as NativeScaler
import utils_mae.lr_sched as lr_sched

from utils.init_datasets import init_datasets

from models import vits
from utils.eval_gradient import evaluate_gradient
from utils.eval_attn_dist import eval_avg_attn_dist, avg_attn_dist_reg_fun, compute_mean_attention_dist
from utils.utils import AverageMeter
from pdb import set_trace

from torch import nn

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--eval_interval', default=1, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--arch', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--norm_beit', action='store_true',
                        help='Use norm beit')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data', default='', type=str,
                        help='dataset path')
    parser.add_argument('--customSplit', default='', type=str)
    parser.add_argument('--dataset', default='imagenet', type=str)

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--eval_gradient', action='store_true', help='gradient evaluation')
    parser.add_argument('--eval_avg_attn_dist', action='store_true', help='avg attn distance evaluation')
    parser.add_argument('--eval_gradient_eval_aug', action='store_true', help='gradient evaluation')
    parser.add_argument('--tuneFromFirstFC', action='store_true')
    parser.add_argument('--fc_scale', default=1, type=int)

    parser.add_argument('--no_aug', action='store_true')

    parser.add_argument('--fixTo', default=-1, type=int)
    parser.add_argument('--fixBnStat', action="store_true")
    parser.add_argument('--add1BlockTo', default=-1, type=int)
    parser.add_argument('--add1BlockToNumLayers', default=1, type=int)
    parser.add_argument('--reset_layers', default=-1, type=int)
    parser.add_argument('--add_batch_norm', action="store_true")

    # add regularization
    parser.add_argument('--avg_attn_dist_reg', default=-1, type=float, help='the avg attn dist reg')
    parser.add_argument('--avg_attn_dist_reg_target', default="", type=str, help='the avg attn dist reg target')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    if args.local_rank == -1:
        args.local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

    misc.init_distributed_mode(args)

    if args.batch_size % misc.get_world_size() != 0:
        raise ValueError(
            "batch size of {} is not divisible by world size of {}".format(args.batch_size, misc.get_world_size()))
    args.batch_size = args.batch_size // misc.get_world_size()

    log = misc.logger(args.output_dir, local_rank=misc.get_rank())

    log.info('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    log.info("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)
    if (args.eval_gradient_eval_aug and args.eval_gradient) or args.no_aug:
        dataset_train, dataset_val, dataset_test = init_datasets(args, transform_val, transform_val)
    else:
        dataset_train, dataset_val, dataset_test = init_datasets(args, transform_train, transform_val)
    nb_classes = len(np.unique(dataset_train.targets))

    log.info(str(transform_train))
    log.info(str(dataset_train))
    log.info(str(dataset_val))
    log.info(str(dataset_test))

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        log.info("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                log.info(
                    'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias

            if len(dataset_test) % num_tasks != 0:
                log.info(
                    'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if args.no_aug:
        mixup_active = False
    if mixup_active:
        log.info("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=nb_classes)

    if args.arch.startswith('beit'):
        from models import beit_vit
        model = beit_vit.__dict__[args.arch](num_classes=nb_classes)
    else:
        model = vits.__dict__[args.arch](
            num_classes=nb_classes,
            drop_path_rate=args.drop_path,
            add1BlockTo=args.add1BlockTo,
            add1BlockToNumLayers=args.add1BlockToNumLayers,
        )

    if args.tuneFromFirstFC:
        middle_dim = 4096
        ch = model.head.in_features
        num_class = model.head.out_features
        model.head = nn.Sequential(nn.Linear(ch, middle_dim, bias=False), nn.Linear(middle_dim, num_class))

    if args.finetune:
        if os.path.isfile(args.finetune):
            checkpoint = torch.load(args.finetune, map_location='cpu')
        else:
            raise ValueError("No such file or dir: {}".format(args.finetune))

        log.info("Load pre-trained checkpoint from: %s" % args.finetune)
        if "model" in checkpoint:
            checkpoint_model = checkpoint['model']
        elif 'state_dict' in checkpoint:
            checkpoint_model = checkpoint['state_dict']
        else:
            checkpoint_model = checkpoint

        if list(checkpoint_model.keys())[0].startswith('module.'):
            checkpoint_model = {k[7:]: v for k, v in checkpoint_model.items()}

        if sorted(list(checkpoint_model.keys()))[0].startswith('base_encoder.'):
            checkpoint_model = cvt_state_dict(checkpoint_model, model, args, "head")
        else:
            assert not args.tuneFromFirstFC

        if "mimco" in args.finetune:
            for key in list(checkpoint_model.keys()):
                if key.startswith("encoder."):
                    checkpoint_model[key.replace("encoder.", "")] = checkpoint_model[key]
                del checkpoint_model[key]

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                log.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model

        # set_trace()
        if args.reset_layers > 0:
            checkpoint_model = reset_state_dict(checkpoint_model, args.reset_layers, log)

        msg = model.load_state_dict(checkpoint_model, strict=False)
        log.info(str(msg))

        if set(msg.missing_keys) != {'head.weight', 'head.bias'}:
            print("msg load are {}".format(msg))
            print("msg load are {}".format(msg))
            print("msg load are {}".format(msg))

        # manually initialize fc layer
        if args.tuneFromFirstFC:
            trunc_normal_(model.head[1].weight, std=2e-5)
        else:
            trunc_normal_(model.head.weight, std=2e-5)

    if args.add_batch_norm:
        feat_dim = model.head.in_features
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(feat_dim, affine=False, eps=1e-6), model.head)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log.info("Model = %s" % str(model_without_ddp))
    log.info('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    log.info("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    log.info("actual lr: %.2e" % args.lr)

    log.info("accumulate grad iterations: %d" % args.accum_iter)
    log.info("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        find_unused_parameters = True if args.fixTo > 0 or args.reset_layers > 0 or args.add1BlockTo > 0 else False
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=find_unused_parameters)
        model_without_ddp = model.module

    if args.fixTo > 0:
        assert args.reset_layers < 0
        fixBackBone(model, args.fixTo, log)

    if args.reset_layers > 0:
        fixBackBone(model, 100, log, fixStart=args.reset_layers)

    if args.add1BlockTo > 0:
        fixBackBone(model, args.add1BlockTo, log)

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                        no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                        layer_decay=args.layer_decay,
                                        fc_scale=args.fc_scale,
                                        log=log
                                        )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    log.info("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, log)
        log.info(f"Accuracy of the network on the {len(dataset_val)} val images: {test_stats['acc1']:.1f}%")
        exit(0)

    if args.eval_gradient:
        evaluate_gradient(data_loader_train, data_loader_test, model, optimizer, args, log)
        return

    if args.eval_avg_attn_dist:
        eval_avg_attn_dist(data_loader_test, model, args, log)
        return

    log.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = -1.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args,
            log=log
        )

        if epoch % args.eval_interval == 0:
            test_stats = evaluate(data_loader_val, model, device, log)
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, epoch_name="last")
            log.info(f"Accuracy of the network on the {len(dataset_val)} val images: {test_stats['acc1']:.1f}%")

            if test_stats["acc1"] > max_accuracy:
                if args.output_dir:
                    misc.save_model_best(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch)

            max_accuracy = max(max_accuracy, test_stats["acc1"])
            log.info(f'Max accuracy: {max_accuracy:.2f}%')

            if log_writer is not None:
                log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
                log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
                log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    log.info('Training time {}'.format(total_time_str))

    # read best model and eval on test dataset
    torch.cuda.empty_cache()
    time.sleep(5) # sleep 5 secs for models to finish the saving
    misc.load_model_best(args, model)

    test_stats = evaluate(data_loader_test, model, device, log)
    log.info(f"Final Test Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc1']:.1f}%")


def fixBackBone(model, fixTo, log, fixStart=0):
    if fixTo > 0:
        if fixStart <= 0:
            # fix the embedding
            model.module.pos_embed.requires_grad = False
            model.module.cls_token.requires_grad = False

            for param in model.module.patch_embed.parameters():
                param.requires_grad = False
            model.module.patch_embed.eval()

        for cnt in range(fixStart, fixTo):
            if cnt < len(model.module.blocks):
                log.info("fix block {}".format(cnt))
                for param in model.module.blocks[cnt].parameters():
                    param.requires_grad = False
                model.module.blocks[cnt].eval()


def reset_state_dict(state_dict, resetTo, log):
    if resetTo > 0:
        reset_key_words = ["pos_embed", "cls_token", "patch_embed"]
        for cnt_block in range(resetTo):
            reset_key_words.append("blocks.{}.".format(cnt_block))
        log.info("reset_key_words are {}".format(reset_key_words))

        for key in list(state_dict.keys()):
            for reset_key_word in reset_key_words:
                if reset_key_word in key:
                    del state_dict[key]

    return state_dict


def cvt_state_dict(state_dict, model, args, linear_keyword):
    # rename moco pre-trained keys
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('base_encoder') and not k.startswith('base_encoder.%s' % linear_keyword):
            # remove prefix
            state_dict[k[len("base_encoder."):]] = state_dict[k]

        if k.startswith('base_encoder') and k.startswith('base_encoder.%s' % linear_keyword) and args.tuneFromFirstFC:
            if k.startswith('base_encoder.%s.0' % linear_keyword):
                state_dict[k[len("base_encoder."):]] = state_dict[k]

        # delete renamed or unused k
        del state_dict[k]

    args.start_epoch = 0

    return state_dict


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None, log=None):
    model.train(True)

    if args.add1BlockTo > 0:
        assert args.fixBnStat

    if args.fixBnStat:
        if args.fixTo > 0:
            assert args.reset_layers < 0
            fixBackBone(model, args.fixTo, log)

        if args.reset_layers > 0:
            fixBackBone(model, 100, log, fixStart=args.reset_layers)

        if args.add1BlockTo > 0:
            fixBackBone(model, args.add1BlockTo, log)

    metric_logger = misc.MetricLogger(delimiter="  ", log=log)
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        log.info('log_dir: {}'.format(log_writer.log_dir))

    Attn_dist_rec = [AverageMeter() for _ in range(len(model.module.blocks))]

    if args.avg_attn_dist_reg > 0:
        attention_dist_metric = compute_mean_attention_dist(patch_size=model.module.patch_embed.patch_size[0],
                                                            num_patches=model.module.patch_embed.num_patches)
        attention_dist_metric.distance_matrix = attention_dist_metric.distance_matrix.cuda()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.avg_attn_dist_reg < 0:
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)
        else:
            # compute output
            with torch.cuda.amp.autocast(True):
                outputs, attns = model(samples, return_attn=True)
                loss = criterion(outputs, targets)

            # aggregate features, compute VIC for each layer
            mean_distance_list, loss_attn = avg_attn_dist_reg_fun(attns, args.avg_attn_dist_reg_target,
                                                                  args.avg_attn_dist_reg, attention_dist_metric)
            for cnt_block in range(len(mean_distance_list)):
                Attn_dist_rec[cnt_block].update(mean_distance_list[cnt_block])

        loss_value = loss.item()

        if args.avg_attn_dist_reg > 0:
            loss += loss_attn
            metric_logger.update(loss_attn=loss_attn)

        if not math.isfinite(loss_value):
            log.info("Loss is {}, stopping training".format(loss_value))
            torch.distributed.barrier()
            raise ValueError("Loss is {}, stopping training".format(loss_value))

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

        if data_iter_step % 100 == 0 and args.avg_attn_dist_reg > 0:
            mean_distance_result = [attn_dist.avg for attn_dist in Attn_dist_rec]
            msg = "{}/{}, mean dist result is {}".format(data_iter_step, len(data_loader), str(mean_distance_result))
            log.info(msg)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    log.info("Averaged stats: {}".format(metric_logger))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, log):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ", log=log)
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    log.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
