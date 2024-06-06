from abc import ABC, abstractmethod
import math

from einops import rearrange
import torch
from torch import nn


class TimeEmbedder(nn.Module, ABC):
    @property
    @abstractmethod
    def out_dim(self):
        pass


class SinusoidalPositionalEmbedding(TimeEmbedder):
    def __init__(self, dim, theta=10000):
        super().__init__()
        dim = dim + dim % 2  # make dim even
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    @property
    def out_dim(self):
        return self.dim


class RandomOrLearnedSinusoidalPositionalEmbedding(TimeEmbedder):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        self.dim = dim + dim % 2  # make dim even
        self.weights = nn.Parameter(torch.randn(self.dim // 2), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

    @property
    def out_dim(self):
        return self.dim + 1
