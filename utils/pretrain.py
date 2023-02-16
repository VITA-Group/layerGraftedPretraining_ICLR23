import torchvision.transforms as transforms
import moco.loader

import os
import torch

from .utils import AverageMeter
from torchvision.transforms import functional as F
import numpy as np

from pdb import set_trace


def gather_features(features, local_rank, world_size):
    features_list = [torch.zeros_like(features) for _ in range(world_size)]
    torch.distributed.all_gather(features_list, features)
    features_list[local_rank] = features
    features = torch.cat(features_list)
    return features


def pretrain_transform(crop_min, with_gate_aug=False, local_crops_number=0):
    if local_crops_number > 0:
        raise ValueError("local_crops_number > 0 is not supported for standard transform")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    augmentation1 = [
        transforms.RandomResizedCrop(224, scale=(crop_min, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    augmentation2 = [
        transforms.RandomResizedCrop(224, scale=(crop_min, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.1),
        transforms.RandomApply([moco.loader.Solarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    if not with_gate_aug:
        train_transform = moco.loader.TwoCropsTransform(transforms.Compose(augmentation1),
                                                        transforms.Compose(augmentation2))
        return train_transform
    else:
        transform_gate = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        train_transform = moco.loader.ThreeCropsTransform(transforms.Compose(augmentation1),
                                                          transforms.Compose(augmentation2),
                                                          transform_gate)
        return train_transform


def evaluate_pretrain(train_loader, model, args, log):
    losses = AverageMeter()
    rank_calculate = RankCalculator()

    # switch to train mode
    model.train()

    iters_per_epoch = len(train_loader)
    moco_m = args.moco_m
    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
            if args.local_rank is not None:
                images[0] = images[0].cuda(args.local_rank, non_blocking=True)
                images[1] = images[1].cuda(args.local_rank, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(True):
                loss, q1, q2, k1, k2 = model(images[0], images[1], moco_m)

            losses.update(loss.item(), images[0].size(0))
            rank_calculate.update(q1, q2)
            print("q1 shape is {}".format(q1.shape))

            if i % 10 == 0:
                msg = "{}/{}, loss avg is {:.3f}".format(i, iters_per_epoch, losses.avg)
                log.info(msg)

    rank_calculate.cal_rank_save(log)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def evaluate_VIC(train_loader, model, args, log):
    losses = AverageMeter()
    rank_calculate = RankCalculator()

    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # switch to train mode
    model.train()

    iters_per_epoch = len(train_loader)
    moco_m = args.moco_m

    VIC_rec = [{"Var": AverageMeter(), "InVar": AverageMeter(), "CoVar": AverageMeter(), "Norm": AverageMeter()} for _ in range(len(model.module.base_encoder.blocks))]

    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
            if args.local_rank is not None:
                images[0] = images[0].cuda(args.local_rank, non_blocking=True)
                images[1] = images[1].cuda(args.local_rank, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(True):
                model(images[0], images[1], moco_m, record_feature=True)
            features = model.module.features

            # aggregate features, compute VIC for each layer
            for key, item in features.items():
                for cnt in range(len(item)):
                    item[cnt] = gather_features(item[cnt], local_rank, world_size)

            if torch.distributed.get_rank() == 0:
                for layer_cnt in range(len(features["x1"])):
                    # cal VIC layer by layer
                    x = features["x1"][layer_cnt].mean(1)
                    y = features["x2"][layer_cnt].mean(1)

                    # compute norm
                    norm_x = torch.norm(x, p=2, dim=-1)
                    norm_y = torch.norm(y, p=2, dim=-1)
                    VIC_rec[layer_cnt]["Norm"].update(((norm_x + norm_y) / 2).mean().item())

                    # normalize
                    x_norm = x / norm_x.unsqueeze(-1)
                    y_norm = y / norm_y.unsqueeze(-1)

                    mse_loss = torch.nn.functional.mse_loss(x_norm, y_norm)
                    VIC_rec[layer_cnt]["InVar"].update(mse_loss.item())

                    x = x - x.mean(dim=0)
                    y = y - y.mean(dim=0)
                    norm_x = torch.norm(x, p=2, dim=-1)
                    norm_y = torch.norm(y, p=2, dim=-1)
                    # calculate the norm after remove mean
                    x = x / norm_x.unsqueeze(-1)
                    y = y / norm_y.unsqueeze(-1)

                    std_x = torch.sqrt(x.var(dim=0) + 0.0001).mean()
                    std_y = torch.sqrt(y.var(dim=0) + 0.0001).mean()
                    VIC_rec[layer_cnt]["Var"].update(((std_x + std_y) / 2).item())

                    batch_size = x.shape[0]
                    cov_x = (x.T @ x) / (batch_size - 1)
                    cov_y = (y.T @ y) / (batch_size - 1)
                    cov_loss = (off_diagonal(cov_x).pow_(2).sum() + off_diagonal(cov_y).pow_(2).sum()) / 2
                    VIC_rec[layer_cnt]["CoVar"].update(cov_loss.item())

            if i % 10 == 0:
                msg = "{}/{}, loss avg is {:.3f}".format(i, iters_per_epoch, losses.avg)
                log.info(msg)

    if torch.distributed.get_rank() == 0:
        VIC_result = [{key: rec[key].avg for key in rec} for rec in VIC_rec]
        np.save(os.path.join(log.path, "VIC_result"), VIC_result)


def evaluate_pretrain_simRank(train_loader, model, args, log):
    rank_losses = AverageMeter()
    similarity_losses = AverageMeter()
    rank_calculate = RankCalculator()

    # switch to train mode
    model.train()

    iters_per_epoch = len(train_loader)
    moco_m = args.moco_m
    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
            if args.local_rank is not None:
                images[0] = images[0].cuda(args.local_rank, non_blocking=True)
                images[1] = images[1].cuda(args.local_rank, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(True):
                rank_l, similarity_l, q1, q2 = model(images[0], images[1], moco_m)

            rank_losses.update(rank_l.item(), images[0].size(0))
            similarity_losses.update(similarity_l.item(), images[0].size(0))
            rank_calculate.update(q1, q2)

            if i % 10 == 0:
                msg = "{}/{}, rank loss avg is {:.3f}, similarity loss avg is {:.3f}".format(i, iters_per_epoch, rank_losses.avg, similarity_losses.avg)
                log.info(msg)

    rank_calculate.cal_rank_save(log)


class RankCalculator(object):
    def __init__(self):
        self.q_comb = None
        self.update_cnt = 0
        pass

    def update(self, q1, q2):
        q_comb = torch.mm(q1.permute(1,0), q2)
        if self.q_comb is None:
            self.q_comb = q_comb / q1.shape[0]
            self.update_cnt += q1.shape[0]
        else:
            self.q_comb = self.q_comb * (self.update_cnt / (self.update_cnt + q1.shape[0])) + \
                          q_comb      * (q1.shape[0] / (self.update_cnt + q1.shape[0]))
            self.update_cnt += q1.shape[0]


    def cal_rank_save(self, log):

        tol_thresholds = [0, 1e-5, 1e-4, 1e-3, 0.001, 0.01, 0.1, 0.2]

        log.info("q_comb norm is {}".format(torch.norm(self.q_comb)))
        for tol_thre in tol_thresholds:
            log.info("For tol_thre of {}, q_comb rank is {}".format(tol_thre, torch.linalg.matrix_rank(self.q_comb.float(), tol=tol_thre)))

        if torch.distributed.get_rank() == 0:
            torch.save(self.q_comb, os.path.join(log.path, "q_comb.pth"))


############################## CMAE augmentations ##############################################
class PixelShiftRandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(self, *args, min_shift=0, max_shift=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_shift = min_shift
        self.max_shift = max_shift

    def forward(self, img, pos=None):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
            bbox: (x, y, width, height)
        """
        assert pos is None
        # print("resize crop")
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        shift_h = torch.randint(self.min_shift, self.max_shift, size=(1,)).item()
        shift_h = ((-1) ** (torch.randint(0, 2, size=(1,)).item())) *  shift_h
        shift_w = torch.randint(self.min_shift, self.max_shift, size=(1,)).item()
        shift_w = ((-1) ** (torch.randint(0, 2, size=(1,)).item())) *  shift_w

        width, height = F._get_image_size(img)
        shift_h = np.clip(shift_h, a_min=-i, a_max=height - h - i + 1)
        shift_w = np.clip(shift_w, a_min=-j, a_max=width - w - j + 1)

        # print("before {}, after {}".format((i, j, h, w), (i+shift_h, j+shift_w, h, w)))

        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), \
               F.resized_crop(img, i+shift_h, j+shift_w, h, w, self.size, self.interpolation),

    def __repr__(self):
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', shift={0}'.format(tuple(round(shift, 4) for shift in [self.min_shift, self.max_shift]))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class TwoCropsPixShiftTransform:
    """Take two random crops of one image"""

    def __init__(self, pixel_shift, base_transform1, base_transform2):
        self.pixel_shift = pixel_shift
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1, im2 = self.pixel_shift(x)
        im1 = self.base_transform1(im1)
        im2 = self.base_transform2(im2)
        return [im1, im2]


def pretrain_transform_cmae(input_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    augmentation1 = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    augmentation2 = [
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.1),
        transforms.RandomApply([moco.loader.Solarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    pixel_shift = PixelShiftRandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=3)

    train_transform = TwoCropsPixShiftTransform(pixel_shift,
                                                transforms.Compose(augmentation1),
                                                transforms.Compose(augmentation2))
    return train_transform
