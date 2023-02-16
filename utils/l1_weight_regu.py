import torch
from torch import nn

from collections import OrderedDict

class L1Regularizer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.record_weight = OrderedDict()
        for name, param in model.named_parameters():
            self.record_weight[name] = param.detach().clone()

    def forward(self, model, layer_smaller_than):
        loss = 0
        for name, param in model.named_parameters():
            id = get_layer_id_for_vit(name, len(model.blocks) + 1)
            if id <= layer_smaller_than:
                loss += torch.norm(self.record_weight[name] - param, p=1)

        return loss


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if 'cls_token' in name or 'pos_embed' in name:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers