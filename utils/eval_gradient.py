import os
import torch

from .utils import AverageMeter
import numpy as np

from collections import OrderedDict
import math

from utils_mae.misc import NativeScalerWithGradNormCount as NativeScaler


def evaluate_gradient(train_loader, test_loader, model, optimizer, args, log):
    losses = AverageMeter()

    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # switch to train mode
    model.eval()

    loss_scaler = NativeScaler()
    criterion = torch.nn.CrossEntropyLoss()

    # for loader, save_name in zip([train_loader, test_loader], ["train", "test"]):
    for loader, save_name in zip([test_loader], ["test"]):
        iters_per_epoch = len(loader)
        for i, (samples, targets) in enumerate(loader):
            device = model.device
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(True):
                outputs = model(samples)
                loss = criterion(outputs, targets)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                log.info("Loss is {}, stopping training".format(loss_value))
                torch.distributed.barrier()
                raise ValueError("Loss is {}, stopping training".format(loss_value))

            loss_scaler(loss, optimizer,
                        parameters=model.parameters(), create_graph=False,
                        update_grad=False)

            torch.cuda.synchronize()

            if i % 10 == 0:
                msg = "{}/{}, loss avg is {:.3f}".format(i, iters_per_epoch, losses.avg)
                log.info(msg)

        grads = OrderedDict()
        params = OrderedDict()
        for name, param in model.named_parameters():
            if param.grad is None:
                grads[name] = None
                params[name] = None
            else:
                grads[name] = param.grad.cpu().detach() / len(loader)
                params[name] = param.cpu().detach()

        model.zero_grad()

        if local_rank == 0:
            torch.save(grads, os.path.join(log.path, "save_grad_{}.pth".format(save_name)))
            torch.save(params, os.path.join(log.path, "save_param_{}.pth".format(save_name)))

