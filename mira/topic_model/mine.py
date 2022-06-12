
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from torch.nn.utils import spectral_norm

EPS = 1e-6

class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / \
            (running_mean + EPS) / input.shape[0]
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)

    # Recalculate ema

    return t_log, running_mean


class ConcatLayer(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, self.dim)


class Mine(nn.Module):

    lr = 1e-4
    hidden = 64
    loss_beta = 1000

    @classmethod
    def get_statistics_network(cls, dim, hidden):

        return nn.Sequential(
            ConcatLayer(1),
            nn.Linear(dim,hidden), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(hidden, 1),
        )

    def __init__(self, T, alpha=0.01):
        super().__init__()
        self.running_mean = 0
        self.alpha = alpha
        self.T = T

    def forward(self, x, z, z_marg=None):
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]

        t = self.T((x, z)).mean()
        t_marg = self.T((x, z_marg))

        second_term, self.running_mean = ema_loss(
            t_marg, self.running_mean, self.alpha)

        return -t + second_term

    def mi(self, x, z, z_marg=None):

        with torch.no_grad():
            mi = -self.forward(x, z, z_marg)
        return mi


class Wasserstein(Mine):

    lr = 1e-4
    hidden = 64
    loss_beta = 1000

    @classmethod
    def get_statistics_network(cls, dim, hidden):

        return nn.Sequential(
            ConcatLayer(1),
            spectral_norm(nn.Linear(dim,hidden)), nn.ReLU(),
            spectral_norm(nn.Linear(hidden, hidden)), nn.ReLU(),
            spectral_norm(nn.Linear(hidden, 1)),
        )



def dual_ema_loss(x, running_mean, alpha):

    if running_mean == 0:
        running_mean = x
    else:
        running_mean = ema(x, alpha, running_mean.item())

    return running_mean


class WassersteinDual(Wasserstein):

    lr = 1e-4
    hidden = 64
    loss_beta = 5000

    def __init__(self, T, alpha=0.01):
        super().__init__(T, alpha = alpha)

    def forward(self, x, z, z_marg=None):
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]

        t = self.T((x, z)).mean()
        t_marg = self.T((x, z_marg)).mean()

        self.running_mean = dual_ema_loss(
            t_marg, self.running_mean, self.alpha)

        return -t + t_marg
        

class WassersteinDualFlat(Wasserstein):

    lr = 1e-4
    hidden = 64
    loss_beta = 5000

    def __init__(self, T, alpha=0.01):
        super().__init__(T, alpha = alpha)

    def forward(self, x, z, z_marg=None):
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]

        t = self.T((x, z)).mean()
        t_marg = self.T((x, z_marg)).mean()

        return -t + t_marg
