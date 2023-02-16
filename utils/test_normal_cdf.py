import torch
from torch.distributions.normal import Normal

normal = Normal(torch.tensor([0.0], device="cpu"), torch.tensor([1.0], device="cpu"),)

prob_if_in = normal.cdf(1.0)
