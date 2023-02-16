import torch
import torch.nn as nn
import os
import time
import numpy as np
import random
import torch.nn.functional as F
import re
from pdb import set_trace

import torch.distributed as dist


def gather_features(features, local_rank, world_size):
    features_list = [torch.zeros_like(features) for _ in range(world_size)]
    torch.distributed.all_gather(features_list, features)
    features_list[local_rank] = features
    features = torch.cat(features_list)
    return features


def sync_weights(model, except_key_words):
    state_dict = model.state_dict()
    for key, item  in state_dict.items():
        flag_sync = True
        for key_word in except_key_words:
            if key_word in key:
                flag_sync = False
                break

        if flag_sync:
            torch.distributed.broadcast(item, 0)

    model.load_state_dict(state_dict)
    return


# loss
def pair_cosine_similarity(x, eps=1e-8):
    n = x.norm(p=2, dim=1, keepdim=True)
    return (x @ x.t()) / (n * n.t()).clamp(min=eps)


def nt_xent(x, t=0.5):
    # print("device of x is {}".format(x.device))

    x = pair_cosine_similarity(x)
    x = torch.exp(x / t)
    idx = torch.arange(x.size()[0])
    # Put positive pairs on the diagonal
    idx[::2] += 1
    idx[1::2] -= 1
    x = x[idx]
    # subtract the similarity of 1 from the numerator
    x = x.diag() / (x.sum(0) - torch.exp(torch.tensor(1 / t)))

    return -torch.log(x).mean()


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


def pgd_attack(model, images, labels, device, eps=8. / 255., alpha=2. / 255., iters=20):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()

    # init
    delta = torch.rand_like(images) * eps * 2 - eps
    delta = torch.nn.Parameter(delta)

    for i in range(iters):
        outputs = model.eval()(images + delta)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        delta.data = delta.data + alpha * delta.grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)
        delta.data = torch.clamp(images + delta.data, min=0, max=1) - images

    return images + delta


def eval_adv_test(model, device, test_loader, epsilon, alpha, criterion, log, attack_iter=40):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # fix random seed for testing
    torch.manual_seed(1)

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        input, target = input.to(device), target.to(device)
        input_adv = pgd_attack(model, input, target, device, eps=epsilon, iters=attack_iter, alpha=alpha).data

        # compute output
        output = model.eval()(input_adv)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        top1.update(prec1, input.size(0))
        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % 10 == 0) or (i == len(test_loader) - 1):
            log.info(
                'Test: [{}/{}]\t'
                'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
                'Prec@1: {top1.val:.3f}({top1.avg:.3f})\t'.format(
                    i, len(test_loader), batch_time=batch_time,
                    loss=losses, top1=top1
                )
            )

    log.info(' * Adv Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


class logger(object):
    def __init__(self, path, log_name="log.txt", local_rank=0):
        self.path = path
        self.local_rank = local_rank
        self.log_name = log_name

    def info(self, msg):
        if self.local_rank == 0:
            print(msg)
            with open(os.path.join(self.path, self.log_name), 'a') as f:
                f.write(msg + "\n")


def fix_bn(model, fixto):
    if fixto == 'nothing':
        # fix none
        # fix previous three layers
        pass
    elif fixto == 'layer1':
        # fix the first layer
        for name, m in model.named_modules():
            if not ("layer2" in name or "layer3" in name or "layer4" in name or "fc" in name):
                m.eval()
    elif fixto == 'layer2':
        # fix the previous two layers
        for name, m in model.named_modules():
            if not ("layer3" in name or "layer4" in name or "fc" in name):
                m.eval()
    elif fixto == 'layer3':
        # fix every layer except fc
        # fix previous four layers
        for name, m in model.named_modules():
            if not ("layer4" in name or "fc" in name):
                m.eval()
    elif fixto == 'layer4':
        # fix every layer except fc
        # fix previous four layers
        for name, m in model.named_modules():
            if not ("fc" in name):
                m.eval()
    else:
        assert False


def change_batchnorm_momentum(module, value):
  if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
    module.momentum = value
  for name, child in module.named_children():
    change_batchnorm_momentum(child, value)


def get_negative_mask_to_another_branch(batch_size):
    negative_mask = torch.ones((batch_size, batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0

    return negative_mask


def nt_xent_only_compare_to_another_branch(x1, x2, t=0.5):
    out1 = F.normalize(x1, dim=-1)
    out2 = F.normalize(x2, dim=-1)
    d = out1.size()
    batch_size = d[0]

    neg = torch.exp(torch.mm(out1, out2.t().contiguous()) / t)

    mask = get_negative_mask_to_another_branch(batch_size).cuda()
    neg = neg.masked_select(mask).view(batch_size, -1)

    # pos score
    pos = torch.exp(torch.sum(out1 * out2, dim=-1) / t)

    # estimator g()
    Ng = neg.sum(dim=-1)

    # contrastive loss
    loss = (- torch.log(pos / (pos + Ng)))
    return loss.mean()


def nt_xent_compare_to_queue(out1, out2, queue, t=0.5, sampleWiseLoss=False):

    d = out1.size()
    batch_size = d[0]

    neg = torch.exp(torch.mm(out1, queue.clone().detach()) / t)

    # pos score
    pos = torch.exp(torch.sum(out1 * out2, dim=-1) / t)

    # estimator g()
    Ng = neg.sum(dim=-1)

    # contrastive loss
    loss = (- torch.log(pos / (pos + Ng)))

    if sampleWiseLoss:
        return loss
    else:
        return loss.mean()


def gatherFeatures(features, local_rank, world_size):
    features_list = [torch.zeros_like(features) for _ in range(world_size)]
    torch.distributed.all_gather(features_list, features)
    features_list[local_rank] = features
    features = torch.cat(features_list)
    return features


# loss
def pair_cosine_similarity(x, eps=1e-8):
    n = x.norm(p=2, dim=1, keepdim=True)
    return (x @ x.t()) / (n * n.t()).clamp(min=eps)


def nt_xent(x, t=0.5, sampleWiseLoss=False, return_prob=False):
    # print("device of x is {}".format(x.device))

    x = pair_cosine_similarity(x)
    x = torch.exp(x / t)
    idx = torch.arange(x.size()[0])
    # Put positive pairs on the diagonal
    idx[::2] += 1
    idx[1::2] -= 1
    x = x[idx]
    # subtract the similarity of 1 from the numerator
    x = x.diag() / (x.sum(0) - torch.exp(torch.tensor(1 / t)))

    if return_prob:
        return x.reshape(len(x) // 2, 2).mean(-1)

    sample_loss = -torch.log(x)

    if sampleWiseLoss:
        return sample_loss.reshape(len(sample_loss) // 2, 2).mean(-1)

    return sample_loss.mean()


def nt_xent_weak_compare(x, t=0.5, features2=None, easy_mining=0.9):
    if features2 is None:
        out = F.normalize(x, dim=-1)
        d = out.size()
        batch_size = d[0] // 2
        out = out.view(batch_size, 2, -1).contiguous()
        out_1 = out[:, 0]
        out_2 = out[:, 1]
    else:
        batch_size = x.shape[0]
        out_1 = F.normalize(x, dim=-1)
        out_2 = F.normalize(features2, dim=-1)

    # neg score
    out = torch.cat([out_1, out_2], dim=0)
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / t)

    mask = get_negative_mask(batch_size).cuda()
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    total_neg_num = neg.shape[1]
    hard_sample_num = max(int((1 - easy_mining) * total_neg_num), 1)
    score = neg
    threshold = score.kthvalue(hard_sample_num, dim=1, keepdim=True)[0]
    neg = (score <= threshold) * neg
    Ng = neg.sum(dim=-1)

    # pos score
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / t)
    pos = torch.cat([pos, pos], dim=0)

    # contrastive loss
    loss = (- torch.log(pos / (pos + Ng)))

    return loss.mean()


def focal_loss(prob, gamma):
    """Computes the focal loss"""
    loss = (1 - prob) ** gamma * (-torch.log(prob))
    return loss.mean()


def fix_focal_loss(prob, fix_prob, gamma):
    """Computes the focal loss"""
    loss = (1 - fix_prob) ** gamma * (-torch.log(prob))
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        # self.weight = weight

    def forward(self, prob):
        return focal_loss(prob, self.gamma)


class DistillCrossEntropy(nn.Module):
    def __init__(self, T):
        super(DistillCrossEntropy, self).__init__()
        self.T = T
        return

    def forward(self, inputs, target):
        """
        :param inputs: prediction logits
        :param target: target logits
        :return: loss
        """
        log_likelihood = - F.log_softmax(inputs / self.T, dim=1)
        sample_num, class_num = target.shape
        loss = torch.sum(torch.mul(log_likelihood, torch.softmax(target / self.T, dim=1)))/sample_num

        return loss


class FocalLoss_fix(nn.Module):
    def __init__(self, gamma=0., fix_probs=None):
        super(FocalLoss_fix, self).__init__()
        '''
        fix_prob: FloatTensor, the prob for each sample, it follows the order of index
        '''

        assert gamma >= 0
        self.gamma = gamma
        assert fix_probs is not None
        self.fix_probs = fix_probs
        # self.weight = weight

    def forward(self, prob, idxs):
        fix_prob = self.fix_probs[idxs]
        return fix_focal_loss(prob, fix_prob, self.gamma)


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def nt_xent_instance_large(x, t=0.5, return_porbs=False):
    out = F.normalize(x, dim=-1)
    d = out.size()
    batch_size = d[0] // 2
    out = out.view(batch_size, 2, -1).contiguous()
    out_1 = out[:, 0]
    out_2 = out[:, 1]
    out = torch.cat([out_1, out_2], dim=0)

    # doesn't give gradient
    losses = []
    probs = []

    with torch.no_grad():
        for cnt in range(batch_size):
            # pos score
            pos = torch.exp(torch.sum(out_1[cnt] * out_2[cnt]) / t)
            pos = torch.stack([pos, pos], dim=0)

            Ng1 = torch.exp((out_1[cnt].unsqueeze(0) * out).sum(1) / t).sum() - torch.exp(torch.Tensor([1 / t,])).to(out.device)
            Ng2 = torch.exp((out_2[cnt].unsqueeze(0) * out).sum(1) / t).sum() - torch.exp(torch.Tensor([1 / t,])).to(out.device)
            Ng = torch.cat([Ng1, Ng2], dim=0)

            if not return_porbs:
                # contrastive loss
                losses.append(- torch.log(pos / Ng).mean())
            else:
                # contrastive loss
                probs.append((pos / Ng).mean())

    if not return_porbs:
        losses = torch.stack(losses)
        return losses
    else:
        probs = torch.stack(probs)
        return probs


def nt_xent_inter_batch_multiple_time(x, t=0.5, batch_size=512, repeat_time=10, return_porbs=False):
    out = F.normalize(x, dim=-1)
    d = out.size()
    dataset_len = d[0] // 2
    dataset_features = out.view(dataset_len, 2, -1).contiguous()

    # doesn't give gradient
    losses_all = []

    with torch.no_grad():
        for cnt in range(repeat_time):
            losses_batch = []
            # order features
            random_order = torch.randperm(dataset_len, device=out.device)
            order_back = torch.argsort(random_order)
            dataset_features_1 = dataset_features[:, 0]
            dataset_features_2 = dataset_features[:, 1]
            # dataset_features_1 = dataset_features_1[random_order]
            # dataset_features_2 = dataset_features_2[random_order]

            # get the loss
            assert dataset_len >= batch_size
            for i in range(int(np.ceil(dataset_len / batch_size))):
                if (i + 1) * batch_size < dataset_len:
                    samplingIdx = random_order[i * batch_size: (i + 1) * batch_size]
                    offset = 0
                else:
                    samplingIdx = random_order[dataset_len - batch_size:]
                    offset = i * batch_size - (dataset_len - batch_size)

                # calculate loss
                out1 = dataset_features_1[samplingIdx]
                out2 = dataset_features_2[samplingIdx]

                out = torch.stack([out1, out2], dim=1).view((batch_size * 2, -1))
                losses_or_probs = nt_xent(out, t=t, sampleWiseLoss=True, return_prob=return_porbs)[offset:]
                losses_batch.append(losses_or_probs)

            # reset the order
            losses_batch = torch.cat(losses_batch, dim=0)
            losses_batch = losses_batch[order_back]
            losses_all.append(losses_batch)

        # average togather
        losses_all = torch.stack(losses_all).mean(0)

        return losses_all


def nt_xent_debiased(x, t=0.5, tau_plus=0.5, debiased=False, weightIns=None, distanceWeightingMode=False, sampleWiseLoss=False, features2=None, returnProb=False):
    if features2 is None:
        out = F.normalize(x, dim=-1)
        d = out.size()
        batch_size = d[0] // 2
        out = out.view(batch_size, 2, -1).contiguous()
        out_1 = out[:, 0]
        out_2 = out[:, 1]
    else:
        batch_size = x.shape[0]
        out_1 = F.normalize(x, dim=-1)
        out_2 = F.normalize(features2, dim=-1)

    # neg score
    out = torch.cat([out_1, out_2], dim=0)
    # print("temperature is {}".format(t))
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / t)

    if returnProb:
        assert not distanceWeightingMode

    if not distanceWeightingMode:
        mask = get_negative_mask(batch_size).cuda()
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / t)
        pos = torch.cat([pos, pos], dim=0)

        # estimator g()
        if debiased:
            N = batch_size * 2 - 2
            Ng = (-tau_plus * N * pos + neg.sum(dim=-1)) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min=N * np.e ** (-1 / t))
        else:
            Ng = neg.sum(dim=-1)

        # contrastive loss
        if returnProb:
            return pos / (pos + Ng)

        loss = (- torch.log(pos / (pos + Ng)))

        assert weightIns is None
        if sampleWiseLoss:
            return (loss[batch_size:] + loss[:batch_size]) / 2
        else:
            return loss.mean()
    else:
        assert not sampleWiseLoss

        mask = get_negative_mask(batch_size).cuda()
        neg = neg * mask
        negWeight = torch.cat([weightIns, weightIns], dim=0).unsqueeze(0) * mask
        negWeightNormalized = negWeight / negWeight.sum(0, keepdim=True) * mask.sum(0, keepdim=True)
        neg = neg * negWeightNormalized

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / t)
        pos = torch.cat([pos, pos], dim=0)

        # estimator g()
        assert not debiased
        Ng = neg.sum(dim=-1)

        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng)))

        return loss.mean()


def nt_xent_sigmoid_focal(x, t, gamma, features2=None):
    if features2 is None:
        out = F.normalize(x, dim=-1)
        d = out.size()
        batch_size = d[0] // 2
        out = out.view(batch_size, 2, -1).contiguous()
        out_1 = out[:, 0]
        out_2 = out[:, 1]
    else:
        batch_size = x.shape[0]
        out_1 = F.normalize(x, dim=-1)
        out_2 = F.normalize(features2, dim=-1)

    # neg score
    out = torch.cat([out_1, out_2], dim=0)
    neg = torch.mm(out, out.t().contiguous()) / t

    mask = get_negative_mask(batch_size).cuda()
    neg = neg.masked_select(mask)
    # pos score
    pos = torch.sum(out_1 * out_2, dim=-1) / t
    pos = torch.cat([pos, pos], dim=0)

    # print("pos is {}, neg is {}".format(pos, neg))

    # get sigmoid prob
    p_pos = torch.sigmoid(pos)
    p_neg = torch.sigmoid(-neg)
    p_all = torch.cat([p_pos, p_neg])

    # print("after sigmoid, pos is {}, neg is {}".format(p_pos, p_neg))

    # contrastive loss
    loss = - ((1 - p_all) ** gamma * torch.log(p_all)).sum() / (batch_size * 2)

    return loss


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def VIC_loss(x, y, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
    repr_loss = F.mse_loss(x, y)

    x = torch.cat(FullGatherLayer.apply(x), dim=0)
    y = torch.cat(FullGatherLayer.apply(y), dim=0)
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)

    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

    batch_size = x.shape[0]
    cov_x = (x.T @ x) / (batch_size - 1)
    cov_y = (y.T @ y) / (batch_size - 1)
    num_features = y.shape[-1]
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(num_features) + off_diagonal(cov_y).pow_(2).sum().div(num_features)

    loss = (
            sim_coeff * repr_loss
            + std_coeff * std_loss
            + cov_coeff * cov_loss
    )
    return loss


def getStatisticsFromTxt(txtName, num_class=1000):
      statistics = [0 for _ in range(num_class)]
      with open(txtName, 'r') as f:
        lines = f.readlines()
      for line in lines:
            s = re.search(r" ([0-9]+)$", line)
            if s is not None:
              statistics[int(s[1])] += 1
      return statistics


def getAllClassNamesFromTxt(txtName):
  names = []
  with open(txtName, 'r') as f:
    lines = f.readlines()
  for line in lines:
    s = re.search(r"train/(n[0-9]+)/n[0-9]+_[0-9]+\.JPEG ([0-9]+)", line)
    if (s is not None):
      names.append(str(s[1]))

  names = np.unique(names).tolist()
  return names


def getAllClassNamesFromTxtPlaces(txtName):
  names = []
  with open(txtName, 'r') as f:
    lines = f.readlines()
  for line in lines:
    s = re.search(r"data_256/([a-zA-Z/]+)/[0-9]+\.jpg ([0-9]+)", line)
    if (s is not None):
      names.append(str(s[1]))

  names = np.unique(names).tolist()
  return names


def gather_tensor(tensor, local_rank, world_size):
    # gather features
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor)
    tensor_list[local_rank] = tensor
    tensors = torch.cat(tensor_list)
    return tensors


def reOrderData(idxs, labels, features):
    # sort all losses and idxes
    labels_new = []
    features_new = []
    idxs_new = []

    # reorder
    for idx, label, feature in zip(idxs, labels, features):
        order = np.argsort(idx)
        idxs_new.append(idx[order])
        labels_new.append(label[order])
        features_new.append(feature[order])

    # check if equal
    for cnt in range(len(idxs_new) - 1):
        if not np.array_equal(idxs_new[cnt], idxs_new[cnt+1]):
            raise ValueError("idx for {} and {} should be the same".format(cnt, cnt+1))

    return idxs_new, labels_new, features_new


def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=0):
    assert warmup_steps >= 0

    if step < warmup_steps:
        lr = lr_max * step / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos((step - warmup_steps) / (total_steps - warmup_steps) * np.pi))

    return lr


def get_CUB200_data_split(root, customSplit):
    if os.path.isdir(root):
        root = root
    else:
        if os.path.isdir("../../data/CUB_200_2011/images"):
            root = "../../data/CUB_200_2011/images"
        elif os.path.isdir("/mnt/models/dataset/CUB_200_2011/images"):
            root = "/mnt/models/dataset/CUB_200_2011/images"
        else:
            assert False

    txt_train = "split/CUB_200/train_split.txt"
    txt_val = "split/CUB_200/val_split.txt"
    txt_test = "split/CUB_200/test_split.txt"

    if customSplit != '':
        txt_train = "split/CUB_200/{}.txt".format(customSplit)

    return root, txt_train, txt_val, txt_test


def remove_state_dict_module(state_dict):
    # rename pre-trained keys
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.'):
            # remove prefix
            state_dict[k.replace("module.", "")] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

    return state_dict


def fix_backbone(model, log):
    # fix every layer except fc
    # fix previous four layers
    log.info("fix backbone")
    for name, param in model.named_parameters():
        if not ("fc" in name):
            param.requires_grad = False

    for name, m in model.named_modules():
        if not ("fc" in name):
            m.eval()


def fix_agent(model, log):
    # fix every layer except fc
    # fix previous four layers
    log.info("fix agent")
    for name, param in model.named_parameters():
        param.requires_grad = False

    for name, m in model.named_modules():
        m.eval()


def modify_model_weight(model, random_prune_ratio):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module.weight.data.copy_(module.weight.data * (1 - random_prune_ratio))


if __name__ == "__main__":
    a = torch.rand([8, 128], device="cuda")

    loss_ori = nt_xent_debiased(a, t=0.5, sampleWiseLoss=True)
    # loss = nt_xent_inter_batch_multiple_time(a, t=0.5, batch_size=4, repeat_time=5)
    loss = nt_xent(a, t=0.5, sampleWiseLoss=True)

    print("loss_ori is {}, loss new is {}".format(loss_ori, loss))
