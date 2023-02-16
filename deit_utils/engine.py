# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch
from  torch import nn

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

# from utils.moe_utils import collect_noisy_gating_loss, collect_moe_activation
from utils.lr_sched import adjust_learning_rate

from moco.builder import concat_all_gather

from pdb import set_trace

from .losses import DistillationLoss

def contrastive_loss(q, k, temp=0.2):
    # normalize
    q = nn.functional.normalize(q, dim=1)
    k = nn.functional.normalize(k, dim=1)
    # gather all targets
    k = concat_all_gather(k)
    # Einstein sum is more intuitive
    logits = torch.einsum('nc,mc->nm', [q, k]) / temp
    # print("logits mean is {}".format(logits.mean()))
    N = logits.shape[0]  # batch size per GPU
    labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
    # print("labels is {}".format(labels))
    return nn.CrossEntropyLoss()(logits, labels) * (2 * temp)


def supervised_contrastive_loss(q, k, label, temp=0.2):
    # exclude the data with the same label

    # normalize
    q = nn.functional.normalize(q, dim=1)
    k = nn.functional.normalize(k, dim=1)
    # gather all targets
    if torch.distributed.is_initialized():
        k = concat_all_gather(k)

    supervised_label_q = label
    supervised_label_k = label
    if torch.distributed.is_initialized():
        supervised_label_k = concat_all_gather(supervised_label_k)

    # only save the different data
    label_mask = (torch.abs(supervised_label_q.unsqueeze(1) - supervised_label_k.unsqueeze(0)) > 0)
    # include the logits of the same sample

    N = label_mask.shape[0]
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    label_mask[torch.arange(N, dtype=torch.long), torch.arange(N, dtype=torch.long) + N * rank] = True

    # Einstein sum is more intuitive
    logits = torch.einsum('nc,mc->nm', [q, k]) / temp

    # mask out the pair with the same label
    logits.masked_fill_(~label_mask.bool().detach(), float('-inf'))

    # print("logits mean is {}".format(logits.mean()))
    N = logits.shape[0]  # batch size per GPU
    labels = (torch.arange(N, dtype=torch.long) + N * rank).cuda()
    # print("labels is {}".format(labels))

    return nn.CrossEntropyLoss()(logits, labels) * (2 * temp)


@torch.no_grad()
def evaluate(data_loader, model, device, log):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ", log=log)
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
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



import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t", log=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.log = log

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    self.log.info(log_msg.format(
                                  i, len(iterable), eta=eta_string,
                                  meters=str(self),
                                  time=str(iter_time), data=str(data_time),
                                  memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    self.log.info(log_msg.format(
                                  i, len(iterable), eta=eta_string,
                                  meters=str(self),
                                  time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.log.info('{} Total time: {} ({:.4f} s / it)'.format(
                      header, total_time_str, total_time / len(iterable)))