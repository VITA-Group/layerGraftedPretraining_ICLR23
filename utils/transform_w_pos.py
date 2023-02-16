import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import Tuple, List, Optional

import numpy as np
import torch
import torchvision
from torch import Tensor
from pdb import set_trace
from PIL import Image


import moco
import moco.loader

try:
    import accimage
except ImportError:
    accimage = None

from torchvision.transforms import functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode, _interpolation_modes_from_int


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class RandomResizedCropWpos(torch.nn.Module):
    """Crop the given image to random size and aspect ratio.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size (int or sequence): expected output size of each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
            In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        scale (tuple of float): scale range of the cropped image before resizing, relatively to the origin image.
        ratio (tuple of float): aspect ratio range of the cropped image before resizing.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` and
            ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.

    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(
            img: Tensor, scale: List[float], ratio: List[float]
    ) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = F._get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

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

        bbox = torch.Tensor([j, i, w, h])
        width, height = torchvision.transforms.functional._get_image_size(img)
        bbox[[0, 2]] = bbox[[0, 2]] / width
        bbox[[1, 3]] = bbox[[1, 3]] / height

        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), bbox

    def __repr__(self):
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class RandomHorizontalFlipWpos(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, pos):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
            bbox: (x, y, width, height, flip_flag)
        """
        if torch.rand(1) < self.p:
            pos = torch.cat([pos, torch.ones_like(pos[0:1])], dim=0)
            return F.hflip(img), pos
        pos = torch.cat([pos, torch.zeros_like(pos[0:1])], dim=0)
        return img, pos

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ComposeWpos(transforms.Compose):
    def __call__(self, img, pos=None):
        for t in self.transforms:
            # print(t)
            img, pos = t(img, pos)
        return img, pos


class RandomApplyWpos(transforms.RandomApply):
    def forward(self, img, pos):
        if self.p < torch.rand(1):
            return img, pos
        for t in self.transforms:
            img, pos = t(img, pos)
        return img, pos

class ColorJitterWpos(transforms.ColorJitter):
    def forward(self, img, pos):
        # print("color jitter")
        img = super(ColorJitterWpos, self).forward(img)
        return img, pos

class RandomGrayscaleWpos(transforms.RandomGrayscale):
    def forward(self, img, pos):
        # print("gray scale")
        img = super(RandomGrayscaleWpos, self).forward(img)
        return img, pos



class GaussianBlurWpos(moco.loader.GaussianBlur):
    def __call__(self, img, pos):
        # print("GaussianBlurWpos")
        img = super(GaussianBlurWpos, self).__call__(img)
        return img, pos


class ToTensorWpos(transforms.ToTensor):
    def __call__(self, img, pos):
        img = super(ToTensorWpos, self).__call__(img)
        return img, pos


class normalizeWpos(transforms.Normalize):
    def forward(self, img, pos):
        img = super(normalizeWpos, self).forward(img)
        return img, pos


class SolarizeWpos(moco.loader.Solarize):
    def __call__(self, img, pos):
        img = super(SolarizeWpos, self).__call__(img)
        return img, pos


class TwoCropsTransformWpos(moco.loader.TwoCropsTransform):
    def __call__(self, x):
        im1, pos1 = self.base_transform1(x)
        im2, pos2 = self.base_transform2(x)
        return [im1, im2], [pos1, pos2]


class TwoCropsPlusLocalCropsTransformWpos(moco.loader.TwoCropsPlusLocalCropsTransform):
    def __call__(self, x):
        im1, pos1 = self.base_transform1(x)
        im2, pos2 = self.base_transform2(x)

        ims_small = []
        poses_small = []
        for _ in range(self.local_crop_num):
            im_small, pos_small = self.local_transform(x)
            ims_small.append(im_small)
            poses_small.append(pos_small)

        return [im1, im2, ims_small], [pos1, pos2, poses_small]


class ThreeCropsTransformWpos(moco.loader.ThreeCropsTransform):
    """Take two random crops of one image"""
    def __call__(self, x):
        im1, pos1 = self.base_transform1(x)
        im2, pos2 = self.base_transform2(x)
        im3 = self.base_transform3(x)

        # print("im1 shape is {}, im2 shape is {}, im3 shape is {}".format(
        #     im1.size(), im2.size(), im3.size()
        # ))

        return [im1, im2, im3], [pos1, pos2]


def pretrain_transform_w_pos(crop_min, with_gate_aug=False, debug_mode=False, local_crops_number=0):
    normalize = normalizeWpos(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])

    if debug_mode:
        normalize = normalizeWpos(mean=[0, 0, 0],
                                  std=[1.0, 1.0, 1.0])

    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    augmentation1 = [
        RandomResizedCropWpos(224, scale=(crop_min, 1.)),
        RandomApplyWpos([
            ColorJitterWpos(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        RandomGrayscaleWpos(p=0.2),
        RandomApplyWpos([GaussianBlurWpos([.1, 2.])], p=1.0),
        RandomHorizontalFlipWpos(),
        ToTensorWpos(),
        normalize
    ]

    augmentation2 = [
        RandomResizedCropWpos(224, scale=(crop_min, 1.)),
        RandomApplyWpos([
            ColorJitterWpos(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        RandomGrayscaleWpos(p=0.2),
        RandomApplyWpos([GaussianBlurWpos([.1, 2.])], p=0.1),
        RandomApplyWpos([SolarizeWpos()], p=0.2),
        RandomHorizontalFlipWpos(),
        ToTensorWpos(),
        normalize
    ]

    assert not with_gate_aug
    train_transform = TwoCropsTransformWpos(ComposeWpos(augmentation1),
                                            ComposeWpos(augmentation2))

    if debug_mode:
        transform_test = [
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
        ]
        train_transform = ThreeCropsTransformWpos(ComposeWpos(augmentation1),
                                                  ComposeWpos(augmentation2),
                                                  transforms.Compose(transform_test))

    if local_crops_number > 0:
        local_transfo = ComposeWpos([
            RandomResizedCropWpos(96, scale=(0.05, 0.25), interpolation=Image.BICUBIC),
            RandomApplyWpos([
                ColorJitterWpos(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            RandomGrayscaleWpos(p=0.2),
            RandomApplyWpos([GaussianBlurWpos([.1, 2.])], p=0.5),
            RandomHorizontalFlipWpos(),
            ToTensorWpos(),
            normalize,
        ])
        return TwoCropsPlusLocalCropsTransformWpos(ComposeWpos(augmentation1),
                                                   ComposeWpos(augmentation2),
                                                   local_transfo,
                                                   local_crops_number)

    return train_transform
