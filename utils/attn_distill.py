import torch
from torch.nn import functional as F

import numpy as np
from PIL import Image

from pdb import set_trace

from torchvision import transforms


def xywh2x1y1x2y2(bboxes):
    '''
    inplace
    :param bboxes: [batch, 5] the last dim: [x, y, w, h, flip_flag]
    :return: bboxes: [batch, 5] the last dim: [x1, y1, x2, y2, flip_flag]
    '''

    bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]
    bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0]
    return bboxes


def x1y1x2y22xywh(bboxes):
    '''
    inplace
    :param bboxes: [batch, 5] the last dim: [x, y, w, h, flip_flag]
    :return: bboxes: [batch, 5] the last dim: [x1, y1, x2, y2, flip_flag]
    '''

    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    return bboxes


def overlap_region(x1, x2):
    '''
    :param x1, x2: [batch, 5] the last dim: [x, y, w, h, flip_flag]
    :return: x1, x2: [batch, token_num, 4]
    '''
    # generate the overlap region without flip
    # keep x1y1x2y2 format
    # x1 = x1.clone()
    # x2 = x2.clone()
    x1 = xywh2x1y1x2y2(x1)
    x2 = xywh2x1y1x2y2(x2)

    bboxes_overlap_x1y1, _ = torch.max(torch.stack([x1[:, :2], x2[:, :2]], dim=0), dim=0)
    bboxes_overlap_x2y2, _ = torch.min(torch.stack([x1[:, 2:], x2[:, 2:]], dim=0), dim=0)
    bboxes_overlap = torch.cat([bboxes_overlap_x1y1, bboxes_overlap_x2y2], dim=1)

    bboxes_overlap = x1y1x2y22xywh(bboxes_overlap)
    valid = torch.logical_and(bboxes_overlap[:, 2] > 0, bboxes_overlap[:, 3] > 0)
    bboxes_overlap = xywh2x1y1x2y2(bboxes_overlap)

    #
    bboxes_overlap_x1 = bboxes_overlap.clone()
    bboxes_overlap_x1[:, 4] = x1[:, 4]
    x1 = x1y1x2y22xywh(x1)
    bboxes_overlap_x1[:, [0, 2]] = (bboxes_overlap_x1[:, [0, 2]] - x1[:, [0, ]]) / x1[:, [2, ]]
    bboxes_overlap_x1[:, [1, 3]] = (bboxes_overlap_x1[:, [1, 3]] - x1[:, [1, ]]) / x1[:, [3, ]]
    x1 = xywh2x1y1x2y2(x1)

    # add flip
    flip_flag_samples = x1[:, 4] > 0
    x1_flip_flag_samples = bboxes_overlap_x1[flip_flag_samples]
    x1_flip_flag_samples[:, 0], x1_flip_flag_samples[:, 2] = 1.0 - x1_flip_flag_samples[:, 2], 1.0 - x1_flip_flag_samples[:, 0]
    bboxes_overlap_x1[flip_flag_samples] = x1_flip_flag_samples

    bboxes_overlap_x2 = bboxes_overlap.clone()
    bboxes_overlap_x2[:, 4] = x2[:, 4]
    x2 = x1y1x2y22xywh(x2)
    bboxes_overlap_x2[:, [0, 2]] = (bboxes_overlap_x2[:, [0, 2]] - x2[:, [0, ]]) / x2[:, [2, ]]
    bboxes_overlap_x2[:, [1, 3]] = (bboxes_overlap_x2[:, [1, 3]] - x2[:, [1, ]]) / x2[:, [3, ]]
    x2 = xywh2x1y1x2y2(x2)

    # add flip
    flip_flag_samples = x2[:, 4] > 0
    x2_flip_flag_samples = bboxes_overlap_x2[flip_flag_samples]
    x2_flip_flag_samples[:, 0], x2_flip_flag_samples[:, 2] = 1.0 - x2_flip_flag_samples[:, 2], 1.0 - x2_flip_flag_samples[:, 0]
    bboxes_overlap_x2[flip_flag_samples] = x2_flip_flag_samples

    return x1y1x2y22xywh(bboxes_overlap_x1), x1y1x2y22xywh(bboxes_overlap_x2), valid


def bbox2gridPts(bboxes, sample_width=14, sample_height=14):
    '''
    :param bboxes: [batch, 5] the last dim: [x, y, w, h, flip_flag]
    :return: bboxes: [batch, h, w, 2], the last dim: [x, y]
    '''
    # bbox from [0,1] to [-1,1]
    bboxes = bboxes.clone()
    bboxes[:, :2] = bboxes[:, :2] * 2 - 1
    bboxes[:, 2:4] = bboxes[:, 2:4] * 2

    b, _ = bboxes.shape
    sample_pts_relative = torch.Tensor([[(x, y) for x in range(sample_width)] for y in range(sample_height)]).to(bboxes.device)
    sample_pts_relative[:, :, [0,]] /= max((sample_width - 1), 1e-6)   # x width
    sample_pts_relative[:, :, [1,]] /= max((sample_height - 1), 1e-6)  # y height
    sample_pts_relative = sample_pts_relative.unsqueeze(0).expand(b, -1, -1, -1).clone()

    # flip
    flip_flag_samples = bboxes[:, 4] > 0
    sample_pts_relative[flip_flag_samples] = torch.flip(sample_pts_relative[flip_flag_samples], dims=[2, ])

    # custom
    sample_pts_relative[:, :, :, [0,]] *= bboxes[:, 2].unsqueeze(1).unsqueeze(1).unsqueeze(1).clone()  # x width
    sample_pts_relative[:, :, :, [1,]] *= bboxes[:, 3].unsqueeze(1).unsqueeze(1).unsqueeze(1).clone()  # y height
    sample_pts_relative[:, :, :, 0] += bboxes[:, 0].unsqueeze(1).unsqueeze(1).clone()
    sample_pts_relative[:, :, :, 1] += bboxes[:, 1].unsqueeze(1).unsqueeze(1).clone()

    # flatten
    return sample_pts_relative


def attnGridSample(grid1, grid2, attn, height_attn, width_attn):
    '''
    :param grid1: [batch, h, w, 2] grid for the first h*w
    :param grid2: [batch, h, w, 2] grid for the second h*w
    :param attn: [batch, num_attn, h*w + 1, h*w + 1]
    :return:
    '''

    attn_cls_token_x1, attn_cls_token_x2, attn_patches = attn[:, :, 1:, 0], attn[:, :, 0, 1:], attn[:, :, 1:, 1:]

    grid1 = grid1.to(attn_cls_token_x1.dtype)
    grid2 = grid2.to(attn_cls_token_x1.dtype)

    # grid sample cls token
    b, c, l = attn_cls_token_x1.shape
    assert l == height_attn * width_attn
    attn_cls_token_x1_sampled = F.grid_sample(attn_cls_token_x1.reshape(b, c, height_attn, width_attn), grid1, padding_mode="border", align_corners=True)
    attn_cls_token_x2_sampled = F.grid_sample(attn_cls_token_x2.reshape(b, c, height_attn, width_attn), grid2, padding_mode="border", align_corners=True)

    # grid sample patch tokens
    attn_patches = attn_patches.reshape(b, c*l, height_attn, width_attn)
    attn_patches_first_sampled = F.grid_sample(attn_patches, grid2, padding_mode="border", align_corners=True)
    _, _, h_sample, w_sample = attn_patches_first_sampled.shape
    attn_patches_first_sampled = attn_patches_first_sampled.reshape(b, c, l, h_sample, w_sample).permute(0, 1, 3, 4, 2).reshape(b, c * h_sample * w_sample, height_attn, width_attn)
    attn_patches_sampled = F.grid_sample(attn_patches_first_sampled, grid1, padding_mode="border", align_corners=True).reshape(b, c, h_sample * w_sample, -1).permute(0, 1, 3, 2)

    return attn_cls_token_x1_sampled, attn_cls_token_x2_sampled, attn_patches_sampled


@torch.no_grad()
def test_attn_sample(x1_valid, x2_valid, bbox1_samplePts, bbox2_samplePts, x1_overlap, x2_overlap):
    # gene attn gt
    b, c, h, w = x1_overlap.shape
    x1_overlap = x1_overlap.reshape(b, c, h*w)
    x2_overlap = x2_overlap.reshape(b, c, h*w)
    attn_gt = torch.bmm(x1_overlap.permute(0, 2, 1), x2_overlap)
    attn_gt = torch.stack([attn_gt, attn_gt], dim=1)

    b, c, h, w = x1_valid.shape
    x1_valid = torch.cat([torch.zeros(b, c, 1, device=x1_valid.device), x1_valid.reshape(b, c, h*w)], dim=-1)
    x2_valid = torch.cat([torch.zeros(b, c, 1, device=x1_valid.device), x2_valid.reshape(b, c, h*w)], dim=-1)
    attn_full = torch.bmm(x1_valid.permute(0, 2, 1), x2_valid)
    attn_full = torch.stack([attn_full, attn_full], dim=1)
    _, _, attn_patches_sampled = attnGridSample(bbox1_samplePts, bbox2_samplePts, attn_full, h, w)

    print(torch.norm(attn_patches_sampled - attn_gt))
    set_trace()

    pass


@torch.no_grad()
def test_attn_sample2():
    x = torch.Tensor([[i for i in range(j, j+50)] for j in range(0, 50)]).unsqueeze(0).unsqueeze(0)
    # bbox1 = torch.Tensor([[0, 9, 30, 40, 0], ])
    # bbox2 = torch.Tensor([[19, 19, 30, 30, 0],])
    # bbox1[:, :4] = bbox1[:, :4] / 49
    # bbox2[:, :4] = bbox2[:, :4] / 49

    bbox1 = torch.Tensor([[0, 10, 30, 40, 0], ])
    bbox2 = torch.Tensor([[20, 20, 30, 30, 0],])
    bbox1[:, :4] = bbox1[:, :4] / 50
    bbox2[:, :4] = bbox2[:, :4] / 50

    x1 = F.grid_sample(x, bbox2gridPts(bbox1, sample_width=50, sample_height=50), padding_mode="border", align_corners=True)
    x2 = F.grid_sample(x, bbox2gridPts(bbox2, sample_width=50, sample_height=50), padding_mode="border", align_corners=True)

    bbox1_overlap, bbox2_overlap, valid = overlap_region(bbox1, bbox2)
    bbox1_samplePts = bbox2gridPts(bbox1_overlap[valid], sample_width=1, sample_height=1)
    bbox2_samplePts = bbox2gridPts(bbox2_overlap[valid], sample_width=1, sample_height=1)
    x1_overlap = F.grid_sample(x1, bbox1_samplePts, padding_mode="border", align_corners=True)
    x2_overlap = F.grid_sample(x2, bbox2_samplePts, padding_mode="border", align_corners=True)

    set_trace()
    # gene attn gt
    b, c, h, w = x1_overlap.shape
    x1_overlap = x1_overlap.reshape(b, c, h*w)
    x2_overlap = x2_overlap.reshape(b, c, h*w)
    attn_gt = torch.bmm(x1_overlap.permute(0, 2, 1), x2_overlap)
    attn_gt = torch.stack([attn_gt, attn_gt], dim=1)

    # gene attn full
    x1_valid = x1[valid]
    x2_valid = x2[valid]
    # x1_valid = F.interpolate(x1_valid, size=(64, 64))
    # x2_valid = F.interpolate(x2_valid, size=(64, 64))

    b, c, h, w = x1_valid.shape
    x1_valid = torch.cat([torch.zeros(b, c, 1, device=x1_valid.device), x1_valid.reshape(b, c, h*w)], dim=-1)
    x2_valid = torch.cat([torch.zeros(b, c, 1, device=x1_valid.device), x2_valid.reshape(b, c, h*w)], dim=-1)
    attn_full = torch.bmm(x1_valid.permute(0, 2, 1), x2_valid)
    attn_full = torch.stack([attn_full, attn_full], dim=1)
    _, _, attn_patches_sampled = attnGridSample(bbox1_samplePts, bbox2_samplePts, attn_full, h, w)

    print(torch.norm(attn_patches_sampled - attn_gt))
    set_trace()

    pass


def overlap_clip(x1, x2, bbox1, bbox2):
    def tensor2img(img):
        img = img.permute(1,2,0)
        img *= 255
        img = img.int()
        img = img.cpu().numpy().astype(np.uint8)
        img = Image.fromarray(img)
        return img

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

    # compute the overlap region
    bbox1_overlap, bbox2_overlap, valid = overlap_region(bbox1, bbox2)
    bbox1_samplePts = bbox2gridPts(bbox1_overlap[valid], sample_width=32, sample_height=32)
    bbox2_samplePts = bbox2gridPts(bbox2_overlap[valid], sample_width=32, sample_height=32)

    x1 = F.interpolate(x1, size=(64, 64))
    x2 = F.interpolate(x2, size=(64, 64))

    # interpolate the origin function
    x1_overlap = F.grid_sample(x1[valid], grid=bbox1_samplePts, padding_mode="border", align_corners=True)
    x2_overlap = F.grid_sample(x2[valid], grid=bbox2_samplePts, padding_mode="border", align_corners=True)

    # test attn sample
    # test_attn_sample(x1[valid].clone(), x2[valid].clone(), bbox1_samplePts, bbox2_samplePts, x1_overlap, x2_overlap)
    # test_attn_sample2()

    # normalize back and write
    for cnt, (img1, img2, img1_overlap, img2_overlap) in \
            enumerate(zip(x1[valid], x2[valid], x1_overlap, x2_overlap)):
        img1 = tensor2img(unnormalize(img1))
        img2 = tensor2img(unnormalize(img2))
        img1_overlap = tensor2img(unnormalize(img1_overlap))
        img2_overlap = tensor2img(unnormalize(img2_overlap))

        img1.save("{}_img1.png".format(cnt))
        img2.save("{}_img2.png".format(cnt))
        img1_overlap.save("{}_img1_overlap.png".format(cnt))
        img2_overlap.save("{}_img2_overlap.png".format(cnt))

    # test attn grid sample

    set_trace()


def cross_img_attn_dist_loss(attn1, attn2, bbox1, bbox2, h_attn, w_attn, sample_width=14, sample_height=14):
    bbox1_overlap, bbox2_overlap, valid = overlap_region(bbox1, bbox2)
    bbox1_samplePts = bbox2gridPts(bbox1_overlap[valid], sample_width=sample_width, sample_height=sample_height)
    bbox2_samplePts = bbox2gridPts(bbox2_overlap[valid], sample_width=sample_width, sample_height=sample_height)

    if valid.sum() > 0:
        sample_attn1_cls1, sample_attn1_cls2, sample_attn1_patches = attnGridSample(bbox1_samplePts, bbox1_samplePts, attn1[valid], h_attn, w_attn)
        sample_attn2_cls1, sample_attn2_cls2, sample_attn2_patches = attnGridSample(bbox2_samplePts, bbox2_samplePts, attn2[valid], h_attn, w_attn)

        loss = - (sample_attn1_cls1 * torch.log(sample_attn2_cls1)).sum(-1).mean() \
               - (sample_attn1_cls2 * torch.log(sample_attn2_cls2)).sum(-1).mean() \
               - (sample_attn1_patches * torch.log(sample_attn2_patches)).sum(-1).mean()
    else:
        return torch.Tensor([0,]).to(attn1.device).to(attn1.dtype)

    return loss
