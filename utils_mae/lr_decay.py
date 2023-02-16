# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
# --------------------------------------------------------
# References:
# ELECTRA https://github.com/google-research/electra
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import json
from pdb import set_trace

import torch.distributed


def param_groups_lrd_moco(model, weight_decay=0.05, no_weight_decay_list=[],
                          layer_decay=.75, lr_layer_wise=""):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    if lr_layer_wise != "":
        num_layers = len(model.base_encoder.blocks)
        lr_layer_wise = [float(lr) for lr in lr_layer_wise.split(",")]
        assert num_layers % len(lr_layer_wise) == 0
        block_size_each = num_layers // len(lr_layer_wise)
        # TODO: check if this work
        layer_scales = [lr_layer_wise[0], ] + [lr_layer_wise[i // block_size_each] for i in range(num_layers)]
    else:
        num_layers = len(model.base_encoder.blocks) + 1
        layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        if "base_encoder." in n:
            layer_id = get_layer_id_for_vit(n.replace("base_encoder.", ""), num_layers)
            # print("name of {} is in layer {}".format(n, layer_id))
        else:
            layer_id = num_layers
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            if torch.distributed.get_rank() == 0:
                print("lr scale of {} is {}".format(group_name, this_scale))

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75, fc_scale=1, log=None):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if "head" in n:
            group_name += "_head"

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            if "head" in group_name:
                this_scale *= fc_scale
                if log is not None:
                    print("scale of group {} is {}".format(group_name, this_scale))

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers