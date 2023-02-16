import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import grad
import torch.optim as optim
import random
from pdb import set_trace
import numpy as np

# different mlp for different samples
class MLP_D(nn.Module):
    def __init__(self, in_dim, hidden_dim, batch_size=2, layers=4):
        super(MLP_D, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.batch_size = batch_size

        if layers == 1:
            model = [nn.Conv1d(self.in_dim * batch_size, batch_size, kernel_size=1, groups=batch_size)]
        elif layers > 1:
            model = [nn.Conv1d(self.in_dim * batch_size, self.in_dim * batch_size, kernel_size=1, groups=batch_size)]
            for _ in range(layers - 2):
                model += [nn.ReLU(True),
                          nn.Conv1d(self.in_dim * batch_size, self.in_dim * batch_size, kernel_size=1, groups=batch_size),]
            model += [nn.ReLU(True),
                      nn.Conv1d(self.in_dim * batch_size, batch_size, kernel_size=1, groups=batch_size),]
        else:
            ValueError("invalid layers number of {}".format(layers))
        main = nn.ModuleList(model)
        self.main = main

    def forward(self, input):
        '''
        :param input: shape [n, patches, c]
        :return: output [n, 1]
        '''
        n, patches, c = input.shape
        # print("n is {}".format(n))
        assert n == self.batch_size
        input = input.permute(0, 2, 1).reshape(1, -1, patches)
        # n * patches
        for l in self.main:
            input = l(input)
        output = input.squeeze(0)
        # one distribution for each image
        output = output
        return output


def gradient_penalty(critic, h_s, h_t):
    # based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
    alpha = torch.rand(h_s.size(0), 1).to(h_s.device)
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = interpolates.requires_grad_()

    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    # print("gradients is {}".format(gradients))
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()
    return gradient_penalty


def gradient_penalty_batch(critic, h_s, h_t):
    # based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
    alpha = torch.rand(h_s.size(0), h_s.size(1), 1).to(h_s.device)
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = interpolates.requires_grad_()
    # print("interpolates shape is {}".format(interpolates.shape))

    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    # print("gradients shape is {}".format(gradients.shape))
    # print("gradients is {}".format(gradients))
    gradient_norm = gradients.norm(2, dim=-1)
    # print("gradient_norm shape is {}".format(gradient_norm.shape))
    gradient_penalty = ((gradient_norm - 1)**2).mean(-1)
    # print("gradient_penalty is {}".format(gradient_penalty))
    return gradient_penalty


class wassertein_distance(nn.Module):
    def __init__(self, proj_net_args, lr, opt_steps, gp_w=1000, log=None):
        super(wassertein_distance, self).__init__()
        self.wassertein_proj = MLP_D(**proj_net_args)
        self.lr = lr
        self.optimizer_proj = optim.RMSprop(self.wassertein_proj.parameters(), lr=lr)
        self.opt_steps = opt_steps
        self.distributed = False

        self.last_target = -1
        self.last_gp = -1

        self.gp_w = gp_w

        if log is not None:
            log.info("moe wasserstein proj_net_args is {}".format(proj_net_args))
            log.info("moe wasserstein lr is {}".format(lr))
            log.info("moe wasserstein opt_steps is {}".format(opt_steps))
            log.info("moe wasserstein gp_w is {}".format(gp_w))

        self.weight_init()

    def distribute(self, args, ):
        assert "moe" in args.arch and (not args.moe_data_distributed)
        # import fmoe
        self.wassertein_proj = fmoe.DistributedGroupedDataParallel(self.wassertein_proj, device_ids=[args.local_rank])
        comm = "none"
        for p in self.wassertein_proj.parameters():
            setattr(p, "dp_comm", comm)
        self.optimizer_proj = optim.RMSprop(self.wassertein_proj.parameters(), lr=self.lr)
        self.distributed = True

    def weight_init(self):
        for m in self.wassertein_proj.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0.0, 0.2)
                m.bias.data.zero_()

    def weight_sync(self):
        for m in self.wassertein_proj.modules():
            if isinstance(m, nn.Conv1d):
                torch.distributed.all_reduce(m.weight.data)
                m.weight.data = m.weight.data / torch.distributed.get_world_size()
                torch.distributed.all_reduce(m.bias.data)
                m.bias.data = m.bias.data / torch.distributed.get_world_size()

    # @torch.no_grad()
    # def update_parameters(self, ):
    #     print_flag = False
    #     for cnt, param in enumerate(self.wassertein_proj.parameters()):
    #         param.data -= self.lr * param.grad.data
    #         if param.grad.data is not None and (not print_flag):
    #             print_flag = True
    #             print("param.data.grad mean is {}".format(param.grad.data.mean().item()))
    #             print("param.data mean is {}".format(param.data.mean().item()))

    def forward(self, q1, q2):
        """
        :param q1: gating prob distribution, shape: [n, patches, c]
        :param q2:
        :return:
        """
        # print("shape of q is {}".format(q1.shape))
        q1_noGrad = q1.detach()
        q2_noGrad = q2.detach()

        q1_noGrad.requires_grad = True
        q2_noGrad.requires_grad = True

        # 1. maximize the distance w.r.t the wassertein_proj (use the parameter of last iter for next iter)
        for param in self.wassertein_proj.parameters():
            param.requires_grad = True
        self.wassertein_proj.train()

        for i in range(self.opt_steps):
            # clamp parameters to a cube
            # for p in self.wassertein_proj.parameters():
            #     p.data.clamp_(clamp_lower, clamp_upper)

            q1_proj = self.wassertein_proj(q1_noGrad)
            q2_proj = self.wassertein_proj(q2_noGrad)

            q1_proj = q1_proj.squeeze(-1)
            q2_proj = q2_proj.squeeze(-1)

            # use abs for the minus of the distribution proj
            target = torch.abs(q1_proj.mean(-1) - q2_proj.mean(-1)).mean()
            gp = gradient_penalty_batch(self.wassertein_proj, q1_noGrad, q2_noGrad).mean()
            # gp = torch.Tensor([0]).to(target.device)
            # max target
            self.optimizer_proj.zero_grad()
            (-target + self.gp_w * gp).backward()
            self.optimizer_proj.step()
            if self.distributed:
                self.wassertein_proj.allreduce_params()

            # print("for iter {}, target is {:.03f}, gp is {:.03f}".format(i, target.item(), gp.item()))

        for param in  self.wassertein_proj.parameters():
            param.requires_grad = False
        self.wassertein_proj.eval()

        # 2. calcuate the distance
        q1_proj = self.wassertein_proj(q1)
        q2_proj = self.wassertein_proj(q2)

        q1_proj = q1_proj.squeeze(-1)
        q2_proj = q2_proj.squeeze(-1)

        w_distance = torch.abs(q1_proj.mean(-1) - q2_proj.mean(-1))
        # print("w_distance shape is {}".format(w_distance.shape))
        self.last_target = w_distance.mean(0).item()
        self.last_gp = gp.item()

        return w_distance


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


def test_w_distance():
    input1 = torch.rand(2, 1000, 1).cuda()
    input2 = torch.rand(2, 1000, 1).cuda()

    input2[0] += 0.5
    input2[1] -= 0.5

    opt_args = {"in_dim": 1, "hidden_dim": 2, "batch_size": 2}
    w_distance_measure = wassertein_distance(opt_args, lr=1e-3, opt_steps=1000).cuda()

    w_distance = w_distance_measure(input1, input2)

    print("output w_distance is {}, expect ~{}".format(w_distance, 0.5))


def test_w_distance2():
    input1 = torch.load("q1_noGrad.pth").cuda()
    input2 = torch.load("q2_noGrad.pth").cuda()

    opt_args = {"in_dim": 12, "hidden_dim": 12, "batch_size": 32}
    w_distance_measure = wassertein_distance(opt_args, lr=1e-3, opt_steps=1000).cuda()

    w_distance = w_distance_measure(input1, input2)

    print("output w_distance is {}".format(w_distance))


def test_gradient_penalty():
    device = "cuda"

    # one distribution example
    critic = nn.Sequential(
        nn.Linear(4, 1)
    ).to(device)
    critic[0].weight.data = torch.ones_like(critic[0].weight.data)
    critic[0].bias.data = torch.ones_like(critic[0].bias.data)
    sample = torch.ones(2, 4).to(device)
    gp = gradient_penalty(critic, sample, sample)
    print("gp is {} for all ones mlp".format(gp))

    critic[0].weight.data = torch.ones_like(critic[0].weight.data) * 2
    critic[0].bias.data = torch.ones_like(critic[0].bias.data)
    sample = torch.ones(2, 4).to(device)
    gp = gradient_penalty(critic, sample, sample)
    print("gp is {} for all twos mlp".format(gp))

    # multiple distribution in batch example
    net = MLP_D(4, 13, batch_size=2, layers=1).to(device)
    net.main[0].weight.data = torch.ones_like(net.main[0].weight.data)
    net.main[0].bias.data = torch.ones_like(net.main[0].bias.data)
    net.main[0].weight.data[1] += 1
    sample = torch.ones(2, 2, 4).to(device)

    gp = gradient_penalty_batch(net, sample, sample)
    print("gp is {} for mix ones and twos mlp".format(gp))
    set_trace()


def test_independency_of_batch():
    device = "cuda"

    # multiple distribution in batch example
    for i in range(3):
        net = MLP_D(4, 4, batch_size=2, layers=3).to(device)
        for m in net.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0.0, 0.2)
                m.bias.data.zero_()

        sample = torch.ones(2, 2, 4).to(device)
        sample.requires_grad_()
        out = net(sample).sum(-1)
        loss = out[1]
        loss.backward()
        print("out grad is {}".format(sample.grad.mean(-1).mean(-1)))
    set_trace()


if __name__ == "__main__":
    # test_gradient_penalty()
    test_w_distance()
