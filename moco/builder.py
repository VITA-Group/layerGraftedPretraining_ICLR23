# References:
# Moco-v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

from pdb import set_trace

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import nt_xent_debiased
from utils.utils import VIC_loss

from utils.attn_distill import overlap_clip, cross_img_attn_dist_loss


def predictor_temp_scheduler(epoch):
    if epoch < 100:
        return - (30 - 1) / 100 * epoch + 30
    else:
        return 1


class Conditioned_Mlp(nn.Module):
    def __init__(self, dim, mlps):
        super(Conditioned_Mlp, self).__init__()

        self.mlps = mlps
        self.gate = nn.Linear(dim * 2, 4)

    def forward(self, q, k, temp=1.0):
        q_pred = []
        for mlp in self.mlps:
            q_pred.append(mlp(q))
        q_pred = torch.stack(q_pred, dim=1)

        # print("temp is {}".format(temp))
        gate = self.gate(torch.cat([q, k], dim=-1)) / temp
        gate = F.softmax(gate, dim=-1)

        return (q_pred * gate.unsqueeze(-1)).sum(1)


class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0, simclr_version=False, VIC_version=False, return_features=False,
                 return_representation=False, conditioned_predictor=False, conditioned_predictor_temp=False,
                 attn_distill=False, attn_distill_cross_view=False, attn_distill_layers_num=1, cmae=False, mae_aug=False):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T
        self.simclr_version = simclr_version
        self.VIC_version = VIC_version

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        if (not self.simclr_version) and (not self.VIC_version):
            self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        if (not simclr_version) and (not self.VIC_version):
            for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
                param_m.data.copy_(param_b.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

        self.return_features = return_features
        self.return_representation = return_representation
        self.conditioned_predictor = conditioned_predictor
        self.conditioned_predictor_temp = conditioned_predictor_temp

        if self.return_representation:
            self.base_encoder.head = nn.Identity()
            self.momentum_encoder.head = nn.Identity()
            self.predictor = nn.Identity()

        if self.conditioned_predictor:
            self.predictor = Conditioned_Mlp(dim, nn.ModuleList([self._build_mlp(2, dim, mlp_dim, dim)
                                                                 for _ in range(4)]))

        self.attn_distill = attn_distill
        self.attn_distill_cross_view = attn_distill_cross_view
        self.attn_distill_layers_num = attn_distill_layers_num
        self.cmae = cmae
        self.mae_aug = mae_aug
        if self.conditioned_predictor_temp:
            self.predictor_temp_scheduler = predictor_temp_scheduler
        else:
            self.predictor_temp_scheduler = None

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def simclr_loss(self, q1, q2):
        # normalize
        q1 = nn.functional.normalize(q1, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)

        q1 = concat_all_gather_wGrad(q1)
        q2 = concat_all_gather_wGrad(q2)

        return nt_xent_debiased(q1, features2=q2, t=self.T) * torch.distributed.get_world_size()

    def distill_attn(self, q1_attns, k1_attns):
        loss_all = 0
        for cnt_layer, (q1_attn, k1_attn) in enumerate(zip(q1_attns, k1_attns)):
            # distill the attn for last k layers
            if cnt_layer >= len(q1_attns) - self.attn_distill_layers_num:
                loss = - (k1_attn * torch.log(q1_attn)).sum(-1).mean()
                loss_all += loss

        return loss_all / self.attn_distill_layers_num

    def cross_distill_attn(self, attns1, attns2, bbox1, bbox2, h_attn, w_attn):
        loss_all = 0
        for cnt_layer, (attn1, attn2) in enumerate(zip(attns1, attns2)):
            # distill the attn for last k layers
            if cnt_layer >= len(attns1) - self.attn_distill_layers_num:
                loss = cross_img_attn_dist_loss(attn1, attn2, bbox1, bbox2, h_attn, w_attn)
                loss_all += loss

        return loss_all / self.attn_distill_layers_num

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def random_masking_gene_id(self, N, L, device, mask_ratio):
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

    def forward(self, x1, x2, m, epoch=-1, record_feature=False, bboxes=None):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features
        # print("q1 = self.predictor(self.base_encoder(x1))")
        if self.mae_aug:
            _, ids_keep_1 = self.random_masking_gene_id(x1.shape[0], self.base_encoder.patch_embed.num_patches,
                                                        x1.device, mask_ratio=0.75)
            _, ids_keep_2 = self.random_masking_gene_id(x2.shape[0], self.base_encoder.patch_embed.num_patches,
                                                        x1.device, mask_ratio=0.75)
        else:
            ids_keep_1, ids_keep_2 = None, None

        # print("x1.shape is {}".format(x1.shape))
        if self.cmae:
            assert not self.mae_aug
            assert not self.simclr_version
            assert not self.VIC_version
            assert not self.conditioned_predictor
            assert not self.attn_distill
            assert not self.return_features

            with torch.no_grad():  # no gradient
                self._update_momentum_encoder(m)  # update the momentum encoder
                # compute momentum features as targets
                k2 = self.momentum_encoder(x2)

            # mask augmentation for base encoder
            q1_feature = self.base_encoder.patch_embed(x1)
            q1_feature = q1_feature + self.base_encoder.pos_embed[:, 1:, :]
            q1_feature, mask, ids_restore, ids_keep = self.random_masking(q1_feature, 0.75)

            # append cls token
            cls_token = self.base_encoder.cls_token + self.base_encoder.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(q1_feature.shape[0], -1, -1)
            q1_feature = torch.cat((cls_tokens, q1_feature), dim=1)

            for cnt, blk in enumerate(self.base_encoder.blocks):
                q1_feature = blk(q1_feature)

            q1 = self.predictor(self.base_encoder.head(q1_feature[:, 0]))

            return self.contrastive_loss(q1, k2)

        if (not self.simclr_version) and (not self.VIC_version):
            with torch.no_grad():  # no gradient
                self._update_momentum_encoder(m)  # update the momentum encoder

            if self.attn_distill:
                assert not self.mae_aug
                # compute momentum features as targets
                k1, k1_attns = self.momentum_encoder(x1, return_attn=True)
                k2, k2_attns = self.momentum_encoder(x2, return_attn=True)
            else:
                # compute momentum features as targets
                k1 = self.momentum_encoder(x1, mask_ids_keep=ids_keep_1)
                k2 = self.momentum_encoder(x2, mask_ids_keep=ids_keep_2)

        if not self.conditioned_predictor:
            features = {}

            if self.attn_distill:
                assert not self.mae_aug
                assert not self.VIC_version

                q1_feat, q1_attns = self.base_encoder(x1, return_attn=True, record_feat=record_feature)
                features["x1"] = self.base_encoder.recorded_feature
                self.base_encoder.recorded_feature = None
                q2_feat, q2_attns = self.base_encoder(x2, return_attn=True, record_feat=record_feature)
                features["x2"] = self.base_encoder.recorded_feature
                self.base_encoder.recorded_feature = None
                self.features = features

                q1 = self.predictor(q1_feat)
                q2 = self.predictor(q2_feat)
            else:
                if self.VIC_version:
                    assert not self.mae_aug
                    q1 = self.base_encoder(x1, record_feat=record_feature)
                    features["x1"] = self.base_encoder.recorded_feature
                    self.base_encoder.recorded_feature = None
                    q2 = self.base_encoder(x2, record_feat=record_feature)
                    features["x2"] = self.base_encoder.recorded_feature
                    self.base_encoder.recorded_feature = None
                    self.features = features
                else:
                    q1 = self.predictor(self.base_encoder(x1, record_feat=record_feature, mask_ids_keep=ids_keep_1))
                    features["x1"] = self.base_encoder.recorded_feature
                    self.base_encoder.recorded_feature = None
                    q2 = self.predictor(self.base_encoder(x2, record_feat=record_feature, mask_ids_keep=ids_keep_2))
                    features["x2"] = self.base_encoder.recorded_feature
                    self.base_encoder.recorded_feature = None
                    self.features = features
        else:
            assert not self.mae_aug
            assert not self.attn_distill
            assert not record_feature
            predictor_temp = 1 if self.predictor_temp_scheduler is None else self.predictor_temp_scheduler(epoch)
            q1 = self.predictor(self.base_encoder(x1), k2, predictor_temp)
            q2 = self.predictor(self.base_encoder(x2), k1, predictor_temp)

        if self.simclr_version:
            assert not self.mae_aug
            assert not self.return_features
            return self.simclr_loss(q1, q2)

        if self.VIC_version:
            assert not self.return_features
            return VIC_loss(q1, q2)

        if self.return_features:
            assert not self.attn_distill
            q1 = concat_all_gather_wGrad(q1.contiguous())
            q2 = concat_all_gather_wGrad(q2.contiguous())
            k1 = concat_all_gather_wGrad(k1.contiguous())
            k2 = concat_all_gather_wGrad(k2.contiguous())
            return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1), q1, q2, k1, k2
        else:
            if self.attn_distill:
                if not self.attn_distill_cross_view:
                    return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1), \
                           self.distill_attn(q1_attns, k1_attns) + self.distill_attn(q2_attns, k2_attns)
                else:
                    bbox1, bbox2 = bboxes
                    # verify the overlap region clip code
                    # overlap_clip(x1, x2, bbox1, bbox2)
                    h_attn = self.base_encoder.patch_embed.grid_size[0]
                    w_attn = self.base_encoder.patch_embed.grid_size[1]
                    return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1), \
                           self.cross_distill_attn(q1_attns, k2_attns, bbox1, bbox2, h_attn, w_attn) + \
                           self.cross_distill_attn(q2_attns, k1_attns, bbox2, bbox1, h_attn, w_attn)
            else:
                return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)


class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head
        if (not self.simclr_version) and (not self.VIC_version):
            del self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        if (not self.simclr_version) and (not self.VIC_version):
            self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        if not self.VIC_version:
            self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# utils
@torch.no_grad()
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


def concat_all_gather_wGrad(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)
    tensors_gather[torch.distributed.get_rank()] = tensor
    output = torch.cat(tensors_gather, dim=0)
    return output
