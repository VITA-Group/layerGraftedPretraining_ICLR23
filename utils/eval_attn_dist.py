import torch
from torch import nn

from .utils import AverageMeter
import numpy as np

from pdb import set_trace


def gather_features(features, local_rank, world_size):
    features_list = [torch.zeros_like(features) for _ in range(world_size)]
    torch.distributed.all_gather(features_list, features)
    features_list[local_rank] = features
    features = torch.cat(features_list)
    return features


def compute_distance_matrix(patch_size, num_patches, length):
    """Helper function to compute distance matrix."""

    distance_matrix = np.zeros((num_patches, num_patches))

    for i in range(num_patches):
        for j in range(num_patches):
            if i == j:  # zero distance
                continue

            xi, yi = (int(i / length)), (i % length)
            xj, yj = (int(j / length)), (j % length)

            distance_matrix[i, j] = patch_size * np.linalg.norm([xi - xj, yi - yj])

    return distance_matrix


class compute_mean_attention_dist(nn.Module):
    def __init__(self, patch_size, num_patches):
        super(compute_mean_attention_dist, self).__init__()
        
        length = int(np.sqrt(num_patches))
        assert (length ** 2 == num_patches), ("Num patches is not perfect square")

        distance_matrix = compute_distance_matrix(patch_size, num_patches, length)
        h, w = distance_matrix.shape

        distance_matrix = distance_matrix.reshape((1, 1, h, w))
        distance_matrix = torch.from_numpy(distance_matrix)
        self.distance_matrix = distance_matrix

    def forward(self, attention_weights):
        mean_distances = attention_weights * self.distance_matrix
        mean_distances = torch.sum(mean_distances, dim=-1)  # sum along last axis to get average distance per token
        mean_distances = torch.mean(mean_distances, dim=-1)  # now average across all the tokes

        return mean_distances


def eval_avg_attn_dist(loader, model, args, log):
    losses = AverageMeter()

    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # switch to train mode
    model.train()

    iters_per_epoch = len(loader)

    Attn_dist_rec = [AverageMeter() for _ in range(len(model.module.blocks))]

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if args.local_rank is not None:
                images = images.cuda(args.local_rank, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(True):
                output, attns = model(images, return_attn=True)

            # aggregate features, compute VIC for each layer
            attn_all = []
            for attn in attns:
                attn_all.append(gather_features(attn, local_rank, world_size))

            if torch.distributed.get_rank() == 0:
                for cnt_block, attn in enumerate(attn_all):
                    attn = attn[:, :, 1:, 1:]
                    mean_distances = compute_mean_attention_dist(patch_size=16, num_patches=attn.shape[2])(attn)
                    Attn_dist_rec[cnt_block].update(mean_distances.mean().cpu().numpy())

            if i % 10 == 0:
                mean_distance_result = [attn_dist.avg for attn_dist in Attn_dist_rec]
                msg = "{}/{}, loss avg is {:.3f}, mean dist result is {}".format(i, iters_per_epoch,
                                                                                 losses.avg, str(mean_distance_result))
                log.info(msg)

    if torch.distributed.get_rank() == 0:
        mean_distance_result = [attn_dist.avg for attn_dist in Attn_dist_rec]
        log.info("mean dist result is {}".format(str(mean_distance_result)))
        # np.save(os.path.join(log.path, "mean_distance_result"), mean_distance_result)


def avg_attn_dist_reg_fun(attn, attn_target, attn_target_reg, attention_dist_metric):
    loss_attn = 0
    mean_distance_list = []

    attn_target = attn_target.split(",")
    attn_target = [float(a) for a in attn_target]
    assert len(attn_target) == len(attn)

    for cnt_block, attn in enumerate(attn):
        attn = attn[:, :, 1:, 1:]
        mean_distance = attention_dist_metric(attn).mean()
        mean_distance_list.append(mean_distance.item())
        loss_attn += attn_target_reg * torch.abs(mean_distance - attn_target[cnt_block])

    return mean_distance_list, loss_attn
