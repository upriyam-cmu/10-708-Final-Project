import math
from typing import List

import torch
from torch import nn

from einops import rearrange

from edge_rec.attend import Attention, RMSNorm


def build_feed_forward(in_dim, hidden_dims, out_dim, activation_fn):
    hidden_dims = list(hidden_dims)
    assert len(hidden_dims) == 0 or activation_fn is not None
    components = []
    for d_in, d_out in zip([in_dim] + hidden_dims, hidden_dims + [out_dim]):
        components.append(nn.Conv2d(d_in, d_out, 1))
        components.append(activation_fn)
    return nn.Sequential(*components[:-1])


def modulate(scale, shift=None):
    def _modify(x):
        x = x * (torch.cat(scale, dim=1) if type(scale) == tuple else scale)
        if shift is not None:
            x = x + (torch.cat(shift, dim=1) if type(shift) == tuple else shift)
        return x

    return _modify


class pipe:
    extract = None

    def __init__(self, x: any):
        self.value = x.value if type(x) == pipe else x

    def __or__(self, fn):
        return self.value if fn is None else pipe(fn(self.value))


def T(x):
    # transpose last 2 dims of x
    return x.transpose(-1, -2)


def divisible_by(numer, denom):
    return (numer % denom) == 0


def get_kwargs(kwargs, **defaults):
    if kwargs is None:
        kwargs = {}
    return {**defaults, **kwargs}


# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
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


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.dim = dim + 1
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class SubgraphAttnModel(nn.Module):
    def __init__(self, feature_dim: int, hidden_dims: List[int], time_embedder,
                 attn_kwargs: dict = None, feed_forward_kwargs: dict = None):
        super().__init__()
        attn_kwargs = get_kwargs(attn_kwargs, heads=4, dim_head=32, num_mem_kv=4, flash=False)
        feed_forward_kwargs = get_kwargs(feed_forward_kwargs, hidden_dims=[], activation_fn=None)
        hidden_dims = list(hidden_dims)

        self.time_embed_initial = nn.Sequential(
            time_embedder,
            nn.Linear(time_embedder.dim, 4 * feature_dim),
            nn.SiLU(),
            nn.Linear(4 * feature_dim, 4 * feature_dim)
        )

        self.blocks = nn.ModuleList()
        for in_dim, out_dim in zip([feature_dim] + hidden_dims, hidden_dims + [1]):
            modules = nn.ModuleDict()
            self.blocks.append(modules)

            modules["time_embed"] = nn.Sequential(
                nn.SiLU(),
                nn.Linear(4 * feature_dim, 9 * in_dim)
            )

            modules["layer_norm_1"] = RMSNorm(in_dim)

            # noinspection PyArgumentList
            modules["row_attn"] = Attention(**attn_kwargs, dim=in_dim)
            # noinspection PyArgumentList
            modules["col_attn"] = Attention(**attn_kwargs, dim=in_dim)

            modules["layer_norm_2"] = RMSNorm(2 * in_dim)
            modules["residual_transform"] = nn.Conv2d(2 * in_dim, out_dim, 1)

            # noinspection PyArgumentList
            modules["feed_forward"] = build_feed_forward(
                **feed_forward_kwargs,
                in_dim=2 * in_dim,
                out_dim=out_dim
            )

    def forward(self, subgraph, times):
        """
        subgraph: Tensor(shape=(b, f, n, m))
        """
        time_embeds = self.time_embed_initial(times)
        for block in self.blocks:
            block_time_embeds = rearrange(block["time_embed"](time_embeds), 'b c -> b c 1 1')
            t = tuple(block_time_embeds.chunk(9, dim=1))

            subgraph = pipe(subgraph) | block["layer_norm_1"] | modulate(t[0], t[1]) | pipe.extract
            row_attn = pipe(subgraph) | block["row_attn"] | modulate(t[2]) | pipe.extract
            col_attn = pipe(subgraph) | T | block["col_attn"] | T | modulate(t[3]) | pipe.extract
            merged = torch.cat([row_attn + 0.5 * subgraph, col_attn + 0.5 * subgraph], dim=1)
            normed = pipe(merged) | block["layer_norm_2"] | modulate(t[4:6], t[6:8]) | pipe.extract
            projected = pipe(normed) | block["feed_forward"] | modulate(t[8]) | pipe.extract
            residual = pipe(merged) | block["residual_transform"] | pipe.extract
            subgraph = projected + residual
        return subgraph


if __name__ == '__main__':
    model = SubgraphAttnModel(10, [], SinusoidalPosEmb(10))
    model(torch.rand(1, 10, 8, 9), torch.tensor([1]))
