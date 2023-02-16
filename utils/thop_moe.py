import torch
from fmoe import FMoELinear
# from models.gate_funs.noisy_gate_vmoe import NoisyGate_VMoE
from pdb import set_trace


def count_fmoe_linear(m, x, y):
    # per output element
    total_mul = m.in_feat
    num_elements = y.numel()
    total_ops = total_mul * num_elements

    m.total_ops += torch.DoubleTensor([int(total_ops)])


def count_fmoe_vmoeGate(m, x, y):

    # per output element
    total_mul, tot_expert = m.w_gate.shape
    total_ops = x[0].numel() * tot_expert

    m.total_ops += torch.DoubleTensor([int(total_ops)])


THOP_DICT={FMoELinear: count_fmoe_linear,
           NoisyGate_VMoE: count_fmoe_vmoeGate}
