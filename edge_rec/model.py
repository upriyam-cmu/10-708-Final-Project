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


def idx(spec, **lengths):
    return lambda x: rearrange(x, spec, **lengths)


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


class MovieLensFeatureEmb(nn.Module):
    MAX_N_GENRES = 6

    def __init__(self, age_dim=4, gender_dim=3, occupation_dim=8, genre_dim=16, add_genres=True):
        # NB: embed_dim should be even
        super().__init__()

        self.age_embedding = nn.Embedding(
            num_embeddings=6,
            embedding_dim=age_dim
        )
        self.gender_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=gender_dim
        )
        self.occupation_embedding = nn.Embedding(
            num_embeddings=21,
            embedding_dim=occupation_dim
        )
        self.genre_embedding = nn.Embedding(
            num_embeddings=19,
            embedding_dim=genre_dim
        )

        self.age_dim = age_dim
        self.gender_dim = gender_dim
        self.occupation_dim = occupation_dim
        self.genre_dim = genre_dim
        self.add_genres = add_genres

    @property
    def combined_genre_dim(self):
        return self.genre_dim * (1 if self.add_genres else self.MAX_N_GENRES)

    @property
    def embed_dim(self):
        return self.age_dim + self.gender_dim + self.occupation_dim + self.combined_genre_dim + 1

    def forward(self, x):
        # dims = [ft, user, movie]
        # ft = [1 rating, 6 genres, --> these are all bogus rn --> 1 age, 1 gender, 1 occupation]
        # x.shape = (b, f, n, m)
        assert x.shape[1] == 4 + self.MAX_N_GENRES

        if self.add_genres:
            collapse_genres = lambda z: z.swapdims(1, -1).sum(dim=-1)
        else:
            collapse_genres = idx('b f n m e -> b (f e) n m')

        rating_embeds = x[:, 0:1]
        genre_embeds = pipe(x[:, 1:7]) | self.genre_embedding | collapse_genres | pipe.extract
        age_embeds = pipe(x[:, 7]) | self.age_embedding | idx('b n m e -> b e n m') | pipe.extract
        gender_embeds = pipe(x[:, 8]) | self.gender_embedding | idx('b n m e -> b e n m') | pipe.extract
        occupation_embeds = pipe(x[:, 9]) | self.occupation_embedding | idx('b n m e -> b e n m') | pipe.extract

        full_embeds = torch.cat([rating_embeds, genre_embeds, age_embeds, gender_embeds, occupation_embeds], dim=1)
        assert full_embeds.shape[1] == self.embed_dim

        return full_embeds


# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        assert divisible_by(dim, 2)
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
                nn.Linear(4 * feature_dim, 8 * in_dim + 1)
            )

            modules["layer_norm_1"] = RMSNorm(in_dim)

            modules["row_attn"] = Attention(**attn_kwargs, dim=in_dim)
            modules["col_attn"] = Attention(**attn_kwargs, dim=in_dim)

            modules["layer_norm_2"] = RMSNorm(2 * in_dim)
            modules["residual_transform"] = nn.Conv2d(2 * in_dim, out_dim, 1)

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
            block_time_embeds = pipe(time_embeds) | block["time_embed"] | idx('b c -> b c 1 1') | pipe.extract
            t, t2 = tuple(block_time_embeds[:, :-1].chunk(8, dim=1)), block_time_embeds[:, -1:]

            subgraph = pipe(subgraph) | block["layer_norm_1"] | modulate(t[0], t[1]) | pipe.extract
            row_attn = pipe(subgraph) | block["row_attn"] | modulate(t[2]) | pipe.extract
            col_attn = pipe(subgraph) | T | block["col_attn"] | T | modulate(t[3]) | pipe.extract
            merged = torch.cat([row_attn + 0.5 * subgraph, col_attn + 0.5 * subgraph], dim=1)
            normed = pipe(merged) | block["layer_norm_2"] | modulate(t[4:6], t[6:8]) | pipe.extract
            projected = pipe(normed) | block["feed_forward"] | modulate(t2) | pipe.extract
            residual = pipe(merged) | block["residual_transform"] | pipe.extract
            subgraph = projected + residual
        return subgraph


class GraphReconstructionModel(nn.Module):
    def __init__(self, feature_embedding, subgraph_model):
        super().__init__()
        self.embedding = feature_embedding
        self.core_model = subgraph_model

    def forward(self, x, t):
        x = self.embedding(x)
        out = self.core_model(x, t)
        assert out.shape[1] == 1
        return out.squeeze(dim=1)

    @staticmethod
    def default():
        embed = MovieLensFeatureEmb()
        return GraphReconstructionModel(
            embed,
            SubgraphAttnModel(
                embed.embed_dim, [],
                SinusoidalPosEmb(embed.embed_dim)
            )
        )


if __name__ == '__main__':
    model = GraphReconstructionModel.default()
    model(torch.randint(0, 2, (1, 10, 8, 9)), torch.tensor([1]))
