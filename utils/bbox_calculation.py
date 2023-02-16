import sys
sys.path.append(".")

import numpy
import numpy as np
from PIL import Image, ImageDraw

import torch
import torchvision
from pdb import set_trace
from torch.utils.data import DataLoader

def bbox_img_2_bbox_patch(bboxes, patch_num_width=14, patch_num_height=14):
    '''
    :param bboxes: [batch, 5] the last dim: [x, y, w, h, flip_flag]
    :return: bboxes: [batch, token_num, 4]
    '''
    b, _ = bboxes.shape
    patch_bbox_relative = torch.Tensor(
        [[(x, y, 1, 1) for x in range(patch_num_width)] for y in range(patch_num_height)]).to(bboxes.device)
    patch_bbox_relative[:, :, [0, 2]] /= patch_num_width   # x width
    patch_bbox_relative[:, :, [1, 3]] /= patch_num_height  # y height
    patch_bbox_relative = patch_bbox_relative.unsqueeze(0).expand(b, -1, -1, -1).clone()

    # flip
    flip_flag_samples = bboxes[:, 4] > 0
    patch_bbox_relative[flip_flag_samples] = torch.flip(patch_bbox_relative[flip_flag_samples], dims=[2, ])

    # custom
    patch_bbox_relative[:, :, :, [0, 2]] *= bboxes[:, 2].unsqueeze(1).unsqueeze(1).unsqueeze(1).clone()  # x width
    patch_bbox_relative[:, :, :, [1, 3]] *= bboxes[:, 3].unsqueeze(1).unsqueeze(1).unsqueeze(1).clone()  # y height
    patch_bbox_relative[:, :, :, 0] += bboxes[:, 0].unsqueeze(1).unsqueeze(1).clone()
    patch_bbox_relative[:, :, :, 1] += bboxes[:, 1].unsqueeze(1).unsqueeze(1).clone()

    # flatten
    return patch_bbox_relative.view(b, patch_num_width * patch_num_height, 4)


def calculateAreaLossWmiou(q1_noCls, q2_noCls, k1_noCls, k2_noCls, bboxes, iou_gate_threshold=0.2,
                           patch_num_width=14, patch_num_height=14):
    bboxes1, bboxes2 = bboxes
    if q1_noCls is not None:
        bboxes1, bboxes2 = bboxes1.to(q1_noCls.device), bboxes2.to(q1_noCls.device)

    # cvt bbox to the bboxes of each patch
    patch_bboxes1 = bbox_img_2_bbox_patch(bboxes1, patch_num_width, patch_num_height)
    patch_bboxes2 = bbox_img_2_bbox_patch(bboxes2, patch_num_width, patch_num_height)

    # cvt [x y w h] to [x1 y1 x2 y2]
    patch_bboxes1[:, :, [2,3]] += patch_bboxes1[:, :, [0,1]]
    patch_bboxes2[:, :, [2,3]] += patch_bboxes2[:, :, [0,1]]

    # calculate the iou of each bbox
    box_ious = []
    for patch_bboxes1_img, patch_bboxes2_img in zip(patch_bboxes1, patch_bboxes2):
        # patch_bboxes1_img: N, 4
        # patch_bboxes2_img: M, 4
        box_ious.append(torchvision.ops.box_iou(patch_bboxes1_img, patch_bboxes2_img))
    box_ious = torch.stack(box_ious) # B, N, M

    max_iou_for_2, max_iou_idx_for_2 = box_ious.max(dim=1) # B, M
    max_iou_for_1, max_iou_idx_for_1 = box_ious.max(dim=2) # B, N

    if q1_noCls is None:
        # for debugging
        return patch_bboxes1, patch_bboxes2, box_ious

    contrast_pair1_mask = max_iou_for_1 > iou_gate_threshold
    contrast_pair1_k2 = torch.gather(k2_noCls, 1, max_iou_idx_for_1.unsqueeze(2).expand(-1, -1, k2_noCls.size(2)))

    contrast_pair2_mask = max_iou_for_2 > iou_gate_threshold
    contrast_pair2_k1 = torch.gather(k1_noCls, 1, max_iou_idx_for_2.unsqueeze(2).expand(-1, -1, k1_noCls.size(2)))

    # double check the gathering and incorporate the masking to contrastive loss caculation

    contrast_pair1 = (contrast_pair1_mask,
                      q1_noCls,
                      contrast_pair1_k2)

    contrast_pair2 = (contrast_pair2_mask,
                      q2_noCls,
                      contrast_pair2_k1)

    # TODO: finish miou coding
    return contrast_pair1, contrast_pair2


def testCalculateAreaLossWmiou():
    q1_noCls = torch.Tensor([[i, i] for i in range(2)])
    q2_noCls = torch.Tensor([[i+10, i+10] for i in range(3)])
    k1_noCls = torch.Tensor([[i, i] for i in range(2)]) * 10
    k2_noCls = torch.Tensor([[i+10, i+10] for i in range(3)]) * 10

    bboxes1 = torch.Tensor([[0, 0, 1, 1, 0],])
    bboxes2 = torch.Tensor([[0, 0, 0.5, 0.5, 1],])

    calculateAreaLossWmiou(q1_noCls, q2_noCls, k1_noCls, k2_noCls, [bboxes1, bboxes2], 0.2, patch_num_width=2, patch_num_height=2)


def testTranformAndIouCal():
    from transform_w_pos import pretrain_transform_w_pos
    from init_datasets import get_imagenet100_root_split
    from init_datasets import Custom_Dataset

    def tensor2img(img):
        img = img.permute(1,2,0)
        img *= 255
        img = img.int()
        img = img.cpu().numpy().astype(np.uint8)
        img = Image.fromarray(img)
        return img

    # init
    transform = pretrain_transform_w_pos(0.08, debug_mode=True)
    root, txt_train, txt_val, txt_test, pathReplaceDict = get_imagenet100_root_split("", "")
    dataset = Custom_Dataset(root=root, txt=txt_train, transform=transform, pathReplace=pathReplaceDict)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=0, pin_memory=True)

    patch_num_width = 3
    patch_num_height = 3
    colors = [(255, 0, 0, 125),
              (0, 255, 0, 125),
              (0, 0, 255, 125),
              ]

    for sample, _ in dataloader:
        imgs, bboxes = sample
        imgs1, imgs2, origin_imgs = imgs

        patch_bboxes1, patch_bboxes2, box_ious = calculateAreaLossWmiou(None, None, None, None, bboxes,
                                                                        iou_gate_threshold=0.2,
                                                                        patch_num_width=patch_num_width,
                                                                        patch_num_height=patch_num_height)

        # plot bboxes in image and save
        for cnt_img, (origin_img, img1, img2, patch_bbox1, patch_bbox2, box_iou) in \
                enumerate(zip(origin_imgs, imgs1, imgs2, patch_bboxes1, patch_bboxes2, box_ious)):
            origin_img = tensor2img(origin_img)
            img1 = tensor2img(img1)
            img2 = tensor2img(img2)

            origin_img1 = origin_img.copy()
            origin_img2 = origin_img.copy()
            origin_img3 = origin_img.copy()
            origin_img4 = origin_img.copy()

            width, height = origin_img1.size
            patch_bbox1[:, [0,2]] *= width
            patch_bbox1[:, [1,3]] *= height
            patch_bbox2[:, [0,2]] *= width
            patch_bbox2[:, [1,3]] *= height
            patch_bbox1 = patch_bbox1.cpu().numpy()
            patch_bbox2 = patch_bbox2.cpu().numpy()

            # plot diag bbox in origin img
            draw_bbox1 = patch_bbox1.reshape(patch_num_height, patch_num_width, 4)
            draw_bbox2 = patch_bbox2.reshape(patch_num_height, patch_num_width, 4)

            # pick the diagonal bboxes
            # draw_bbox1 = numpy.stack([draw_bbox1[i,i] for i in range(min(patch_num_height, patch_num_width))])
            # draw_bbox2 = numpy.stack([draw_bbox2[i,i] for i in range(min(patch_num_height, patch_num_width))])
            # draw all bboxes
            draw_bbox1 = draw_bbox1.reshape(-1, 4)
            draw_bbox2 = draw_bbox2.reshape(-1, 4)

            box_iou = box_iou.flatten()
            score, idx = box_iou.sort(dim=-1)
            idx_largest = idx[-3:]
            score_largest = score[-3:]
            idx1_largest = idx_largest // len(patch_bbox1)
            idx2_largest = idx_largest % len(patch_bbox1)

            print("score_largest for cnt_img {} is {}".format(cnt_img, score_largest))

            # plot the most match bbox in origin img
            draw1 = ImageDraw.Draw(origin_img1, "RGBA")
            for bbox in draw_bbox1:
                draw1.rectangle(bbox, fill=None, outline=colors[1], width=2)
            # for cnt, bbox in enumerate(patch_bbox1[idx1_largest]):
            #     if score_largest[cnt] > 0.2:
            #         draw1.rectangle(bbox, fill=colors[cnt], outline=None, width=0)

            draw2 = ImageDraw.Draw(origin_img2, "RGBA")
            for bbox in draw_bbox2:
                draw2.rectangle(bbox, fill=None, outline=colors[0], width=2)

            draw3 = ImageDraw.Draw(origin_img3, "RGBA")
            for bbox in draw_bbox1:
                draw3.rectangle(bbox, fill=None, outline=colors[1], width=2)
            for cnt, bbox in enumerate(patch_bbox1[idx1_largest]):
                if score_largest[cnt] > 0.2 and cnt == 2:
                    draw3.rectangle(bbox, fill=colors[1], outline=None, width=0)

            for bbox in draw_bbox2:
                draw3.rectangle(bbox, fill=None, outline=colors[0], width=2)
            for cnt, bbox in enumerate(patch_bbox2[idx2_largest]):
                if score_largest[cnt] > 0.2 and cnt == 2:
                    draw3.rectangle(bbox, fill=colors[0], outline=None, width=0)

            origin_img1.save("cnt{}_origin_img1.png".format(cnt_img))
            origin_img2.save("cnt{}_origin_img2.png".format(cnt_img))
            origin_img3.save("cnt{}_origin_img3.png".format(cnt_img))
            origin_img4.save("cnt{}_origin_img.png".format(cnt_img))
            img1.save("cnt{}_aug_img1.png".format(cnt_img))
            img2.save("cnt{}_aug_img2.png".format(cnt_img))

            set_trace()



def test_batch_contrastive_learning_with_mask():
    from torch import nn

    @torch.no_grad()
    def concat_all_gather(tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        return tensor

    def contrastive_loss_batch(q, k, T=0.2, mask=None, debug=False):
        # normalize
        q = nn.functional.normalize(q, dim=-1)
        k = nn.functional.normalize(k, dim=-1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nbc,mbc->nbm', [q, k]) / T
        # print("logits mean is {}".format(logits.mean()))
        N, B, _ = logits.shape  # batch size per GPU
        logits = logits.permute(0, 2, 1)
        labels = (torch.arange(N, dtype=torch.long) ).cuda().unsqueeze(-1).expand(N, B)
        if mask is not None:
            mask = mask.to(labels.device)
            labels = mask * labels + (~mask) * -1
            labels = labels.long()

        # print("labels is {}".format(labels))
        if debug:
            return nn.CrossEntropyLoss(ignore_index=-1, reduction="none")(logits, labels) * (2 * T)

        return nn.CrossEntropyLoss(ignore_index=-1)(logits, labels) * (2 * T)

    torch.manual_seed(0)
    feature1 = torch.rand((2, 3, 100)).cuda()
    mask = torch.Tensor([[0, 0, 0],
                         [0, 0, 0]]).cuda()
    feature2 = torch.rand((2, 3, 100)).cuda()

    loss = contrastive_loss_batch(feature1, feature2, mask=mask)
    print(loss)


if __name__ == "__main__":
    # test_batch_contrastive_learning_with_mask()
    testTranformAndIouCal()
