from collections import OrderedDict
# from models.gate_funs.noisy_gate import NoisyGate
# from models.gate_funs.noisy_gate_vmoe import NoisyGate_VMoE
from models.custom_moe_layer import FMoETransformerMLP
import torch.distributed
import os

from pdb import set_trace

import torch.nn.functional as F
import shutil

def gather_features(features, local_rank, world_size):
    features_list = [torch.zeros_like(features) for _ in range(world_size)]
    torch.distributed.all_gather(features_list, features)
    features_list[local_rank] = features
    features = torch.cat(features_list)
    return features


def collect_moe_model_state_dict(moe_state_dict):
    collect_moe_state_dict = OrderedDict()

    for key, item in moe_state_dict.items():
        if "mlp.experts.htoh4" in key or "mlp.experts.h4toh" in key:
            collect_moe_state_dict[key] = gather_features(item, torch.distributed.get_rank(), torch.distributed.get_world_size())
        else:
            collect_moe_state_dict[key] = item

    return collect_moe_state_dict

def filter_state(state):
    from collections import OrderedDict
    new_state = OrderedDict()
    for key, item in state.items():
        if "mlp.experts.htoh4" in key or "mlp.experts.h4toh" in key:
            new_state[key] = item
    return new_state

def save_moe_model_to_dir(state, filename, save_dir):
    rank = torch.distributed.get_rank()
    dirname = os.path.join(save_dir, filename)
    if rank == 0:
        if os.path.isfile(dirname):
            os.system("rm {}".format(dirname))
        os.system("mkdir -p {}".format(dirname))
    torch.distributed.barrier()

    save_name = os.path.join(dirname, "{}.pth".format(rank))
    if rank != 0:
        state["state_dict"] = filter_state(state["state_dict"])
    torch.save(state, save_name)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', save_dir="checkpoints", moe_save=False, only_best=False):
    if moe_save:
        if "optimizer" in state:
            del state["optimizer"]
        if not only_best:
            save_moe_model_to_dir(state, filename, save_dir)
        if is_best:
            save_moe_model_to_dir(state, "model_best.pth.tar", save_dir)
    else:
        if not only_best:
            torch.save(state, os.path.join(save_dir, filename))
        if is_best:
            shutil.copyfile(os.path.join(save_dir, filename), os.path.join(save_dir, 'model_best.pth.tar'))

def read_specific_group_experts(moe_state_dict, rank, num_experts):
    for key, item in moe_state_dict.items():
        if "mlp.experts.htoh4" in key or "mlp.experts.h4toh" in key:
            moe_state_dict[key] = item[rank * num_experts: (rank + 1) * num_experts]
        else:
            moe_state_dict[key] = item

    return moe_state_dict


def collect_noisy_gating_loss(model, weight):
    loss = 0
    for module in model.modules():
        if (isinstance(module, NoisyGate) or isinstance(module, NoisyGate_VMoE)) and module.has_loss:
            loss += module.get_loss()
    return loss * weight


def collect_moe_activation(model, batch_size, activation_suppress="pool", return_name=False):
    gate_activations = []
    names = []
    for name, module in model.named_modules():
        if (isinstance(module, NoisyGate) or isinstance(module, NoisyGate_VMoE)) and module.has_activation:
            activation = module.get_activation()
            _, c = activation.shape
            activation = activation.reshape(batch_size, -1, c)
            if activation_suppress == "pool":
                activation = activation.mean(dim=1)
            elif activation_suppress == "concat":
                activation = torch.reshape(activation.shape[0], -1)
            elif activation_suppress == "origin":
                pass
            else:
                raise ValueError("No activation_suppress of {}".format(activation_suppress))
            gate_activations.append(activation)
            names.append(name)

    if not return_name:
        return gate_activations
    else:
        return gate_activations, names


def set_moe_mask(model, select_idx_dict):
    for name, module in model.named_modules():
        if (isinstance(module, NoisyGate) or isinstance(module, NoisyGate_VMoE)):
            module.select_idx = select_idx_dict[name]

class feature_avger(object):
    def __init__(self):
        self.avg = None
        self.cnt = 0

    def update(self, features):
        if self.avg is None:
            self.avg = features.mean(0)
        else:
            self.avg = self.avg         * (self.cnt          / (self.cnt + features.shape[0])) + \
                       features.mean(0) * (features.shape[0] / (self.cnt + features.shape[0]))

        self.cnt += features.shape[0]

def prune_moe_experts(model, train_loader, log, moe_experts_prune_num):
    model.train()

    for cnt, (image, label) in enumerate(train_loader):
        image = image.cuda(non_blocking=True)
        pred = model(image)
        gate_activations, gate_names = collect_moe_activation(model, pred.shape[0], return_name=True)

        if cnt == 0:
            gate_activations_avger_dict = {name: feature_avger() for name in gate_names}

        gate_activations = [F.softmax(g, dim=1) for g in gate_activations]
        for gate_name, gate_activation in zip(gate_names, gate_activations):
            gate_activations_avger_dict[gate_name].update(gate_activation.detach())

        if cnt % 100 == 0:
            log.info("prune gate stat cal: [{}/{}]".format(cnt, len(train_loader)))


    save_gate_activations_avger_dict = {k: item.avg for k, item in gate_activations_avger_dict.items()}
    torch.save(save_gate_activations_avger_dict, os.path.join(log.path, "save_gate_activations_avger_dict.pth"))

    # prune
    select_idx_dict = {}
    for n, item in save_gate_activations_avger_dict.items():
        assert moe_experts_prune_num <= len(item)
        top_logits, top_indices = item.topk(moe_experts_prune_num)
        select_idx_dict[n] = top_indices
    set_moe_mask(model, select_idx_dict)

    return gate_activations_avger_dict


def set_moe_layer_train_mode(model):
    for module in model.modules():
        if isinstance(module, FMoETransformerMLP):
            module.train()


def get_parameter_group(args, model):
    if "moe" in args.arch:
        params_moe_mlps = []
        params_other = []
        for key, param in model.named_parameters():
            if "mlp.experts.htoh4" in key or "mlp.experts.h4toh" in key:
                params_moe_mlps.append(param)
            else:
                params_other.append(param)
        params = [{'name': "moe_mlps", 'params': params_moe_mlps, 'lr': args.lr * args.experts_lr_ratio},
                  {'name': "other", 'params': params_other, 'lr': args.lr}]
        return params
    else:
        return model.parameters()
