import math
from typing import List

import torch
from torch import nn

from einops import rearrange

from edge_rec.attend import SelfAttention, CrossAttention, Stacked1DAttention, RMSNorm
from edge_rec.pipe import *


def build_feed_forward(in_dim, hidden_dims, out_dim, activation_fn):
    hidden_dims = list(hidden_dims)
    assert len(hidden_dims) == 0 or activation_fn is not None
    components = []
    for d_in, d_out in zip([in_dim] + hidden_dims, hidden_dims + [out_dim]):
        components.append(nn.Conv2d(d_in, d_out, 1))
        components.append(activation_fn)
    return nn.Sequential(*components[:-1])


def divisible_by(numer, denom):
    return (numer % denom) == 0


def get_kwargs(kwargs, **defaults):
    if kwargs is None:
        kwargs = {}
    return {**defaults, **kwargs}


class MovieLensFeatureEmb(nn.Module):
    MAX_N_GENRES = 6

    N_AGE_VALUES = 7
    N_GENDER_VALUES = 2
    N_OCCUPATION_VALUES = 21
    N_GENRE_VALUES = 19

    def __init__(self, age_dim=4, gender_dim=2, occupation_dim=8, genre_dim=8, add_genres=True):
        # NB: embed_dim should be even
        super().__init__()

        self.age_embedding = nn.Embedding(
            num_embeddings=self.N_AGE_VALUES,
            embedding_dim=age_dim
        )
        self.gender_embedding = nn.Embedding(
            num_embeddings=self.N_GENDER_VALUES,
            embedding_dim=gender_dim
        )
        self.occupation_embedding = nn.Embedding(
            num_embeddings=self.N_OCCUPATION_VALUES,
            embedding_dim=occupation_dim
        )
        self.genre_embedding = nn.Embedding(
            num_embeddings=self.N_GENRE_VALUES,
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
        return self.age_dim + self.gender_dim + self.occupation_dim + self.combined_genre_dim + 14

    def forward(self, x):
        # dims = [ft, user, movie]
        # ft = [1 rating, 6 genres, --> these are all bogus rn --> 1 age, 1 gender, 1 occupation, 1 movie_review_counts, 1 user_review_counts]
        # x.shape = (b, f, n, m)
        assert x.shape[1] == 26

        if self.add_genres:
            collapse_genres = lambda z: z.swapdims(1, -1).sum(dim=-1)
        else:
            collapse_genres = idx('b f n m e -> b (f e) n m')

        rating_embeds = x[:, 0:1]
        movie_review_embeds = x[:,19:21]
        user_review_embeds = x[:,24:26]
        genre_embeds = x[:, 1:18]
        #genre_embeds = pipe(x[:, 1:7]) | toi | assert_in(0, self.N_GENRE_VALUES) \
        #               | self.genre_embedding | collapse_genres | pipe.extract
        age_embeds = pipe(x[:, 21]) | toi | assert_in(0, self.N_AGE_VALUES) \
                     | self.age_embedding | idx('b n m e -> b e n m') | pipe.extract
        gender_embeds = pipe(x[:, 22]) | toi | assert_in(0, self.N_GENDER_VALUES) \
                        | self.gender_embedding | idx('b n m e -> b e n m') | pipe.extract
        occupation_embeds = pipe(x[:, 23]) | toi | assert_in(0, self.N_OCCUPATION_VALUES) \
                            | self.occupation_embedding | idx('b n m e -> b e n m') | pipe.extract
        
        full_embeds = torch.cat([rating_embeds, genre_embeds, movie_review_embeds, age_embeds, gender_embeds, occupation_embeds, user_review_embeds], dim=1)
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

            modules["time_embed_1"] = nn.Sequential(
                nn.SiLU(),
                nn.Linear(4 * feature_dim, 8 * in_dim)
            )
            modules["time_embed_2"] = nn.Sequential(
                nn.SiLU(),
                nn.Linear(4 * feature_dim, out_dim)
            )

            modules["layer_norm_1"] = RMSNorm(in_dim)

            modules["row_attn"] = SelfAttention(**attn_kwargs, dim=in_dim)
            modules["col_attn"] = SelfAttention(**attn_kwargs, dim=in_dim)

            modules["layer_norm_2"] = RMSNorm(2 * in_dim)
            modules["residual_transform"] = nn.Conv2d(2 * in_dim, out_dim, 1)

            modules["feed_forward"] = build_feed_forward(
                **feed_forward_kwargs,
                in_dim=2 * in_dim,
                out_dim=out_dim
            )

    @staticmethod
    def _row_attn(subgraph, mask, block, t):
        return pipe(subgraph, mask) | block["row_attn"] | modulate(t[2]) | pipe.extract

    @staticmethod
    def _col_attn(subgraph, mask, block, t):
        return pipe(T(subgraph), mask) | block["col_attn"] | T | modulate(t[3]) | pipe.extract

    def forward(self, subgraph, times, mask, batch_size=None):
        """
        subgraph: Tensor(shape=(b, f, n, m))
        """
        b, _, n, m = subgraph.shape
        assert n != 1 or m != 1

        time_embeds = self.time_embed_initial(times)
        for block in self.blocks:
            block_time_embeds_1 = pipe(time_embeds) | block["time_embed_1"] | idx('b c -> b c 1 1') | pipe.extract
            block_time_embeds_2 = pipe(time_embeds) | block["time_embed_2"] | idx('b c -> b c 1 1') | pipe.extract
            t, t2 = tuple(block_time_embeds_1.chunk(8, dim=1)), block_time_embeds_2

            subgraph = pipe(subgraph) | block["layer_norm_1"] | modulate(t[0], t[1]) | pipe.extract

            if batch_size is None:
                row_attn = col_attn = None
                if n != 1:
                    row_attn = pipe(subgraph, mask) | block["row_attn"] | modulate(t[2]) | pipe.extract
                if m != 1:
                    col_attn = pipe(T(subgraph), mask) | block["col_attn"] | T | modulate(t[3]) | pipe.extract
                if row_attn is None:
                    row_attn = col_attn
                if col_attn is None:
                    col_attn = row_attn
            else:
                assert b == 1
                chunk = lambda x: x.tensor_split(tuple(range(batch_size, len(x), batch_size)), dim=0)
                merge = lambda x: torch.cat(x, dim=0)

                batched_rows = pipe(subgraph) | idx('b f n m -> m f n b') | chunk | pipe.extract
                rows_mask = pipe(mask) | idx('b n m -> m n b') | chunk | pipe.extract
                processed_rows = tuple(
                    pipe(batch, batch_mask) | block["row_attn"] | modulate(t[2]) | pipe.extract
                    for batch, batch_mask in zip(batched_rows, rows_mask)
                )
                batched_cols = pipe(subgraph) | idx('b f n m -> n f m b') | chunk | pipe.extract
                cols_mask = pipe(mask) | idx('b n m -> n m b') | chunk | pipe.extract
                processed_cols = tuple(
                    pipe(batch, batch_mask) | block["col_attn"] | modulate(t[3]) | pipe.extract
                    for batch, batch_mask in zip(batched_cols, cols_mask)
                )

                row_attn = pipe(processed_rows) | merge | idx('m f n b -> b f n m') | pipe.extract
                col_attn = pipe(processed_cols) | merge | idx('n f m b -> b f n m') | pipe.extract

            merged = torch.cat([row_attn + 0.5 * subgraph, col_attn + 0.5 * subgraph], dim=1)

            normed = pipe(merged) | block["layer_norm_2"] | modulate(t[4:6], t[6:8]) | pipe.extract
            projected = pipe(normed) | block["feed_forward"] | modulate(t2) | pipe.extract

            residual = pipe(merged) | block["residual_transform"] | pipe.extract

            subgraph = projected + residual

        return subgraph


class SubgraphTransformer(nn.Module):
    def __init__(self, feature_dim: int, n_blocks: int, time_embedder,
                 attn_kwargs: dict = None, feed_forward_kwargs: dict = None):
        super().__init__()
        attn_kwargs = get_kwargs(attn_kwargs, heads=4, dim_head=32, num_mem_kv=4, flash=False)
        feed_forward_kwargs = get_kwargs(feed_forward_kwargs, hidden_dims=[], activation_fn=None)

        self.feature_dim = feature_dim = feature_dim - 1
        self.time_embed_initial = nn.Sequential(
            time_embedder,
            nn.Linear(time_embedder.dim, 4),
            nn.SiLU(),
            nn.Linear(4, 4)
        )

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for _ in range(n_blocks):
            # encoder
            encoder_modules = nn.ModuleDict()
            self.encoder_blocks.append(encoder_modules)

            encoder_modules["self_attn"] = Stacked1DAttention(SelfAttention, **attn_kwargs, dim=feature_dim)
            encoder_modules["layer_norm_1"] = RMSNorm(feature_dim)
            encoder_modules["feed_forward"] = build_feed_forward(
                **feed_forward_kwargs,
                in_dim=feature_dim,
                out_dim=feature_dim
            )
            encoder_modules["layer_norm_2"] = RMSNorm(feature_dim)

            # decoder
            decoder_modules = nn.ModuleDict()
            self.decoder_blocks.append(decoder_modules)

            decoder_modules["time_embed"] = nn.Sequential(
                nn.SiLU(),
                nn.Linear(4, 9)
            )

            decoder_modules["layer_norm_1"] = RMSNorm(1)
            decoder_modules["self_attn"] = Stacked1DAttention(SelfAttention, **attn_kwargs, dim=1)
            decoder_modules["layer_norm_2"] = RMSNorm(1)
            decoder_modules["cross_attn"] = Stacked1DAttention(CrossAttention, **attn_kwargs, d_qk=feature_dim, d_v=1)
            decoder_modules["layer_norm_3"] = RMSNorm(1)
            decoder_modules["feed_forward"] = build_feed_forward(**feed_forward_kwargs, in_dim=1, out_dim=1)

        self.final_linear = nn.Conv2d(1, 1, 1)

    def forward(self, subgraph, times, mask):
        """
        subgraph: Tensor(shape=(b, f, n, m))
        mask: Tensor(shape=(b, n, m))
        """
        ratings, fts = subgraph[:, :1, :, :], subgraph[:, 1:, :, :]
        assert ratings.shape[1] == 1 and fts.shape[1] == self.feature_dim

        time_embeds = self.time_embed_initial(times)
        maybe_attach_mask = (lambda x: pipe(x, mask)) if mask is not None else pipe

        for enc_block, dec_block in zip(self.encoder_blocks, self.decoder_blocks):
            # encoder
            fts = pipe(fts) | enc_block["self_attn"] | add(fts) | enc_block["layer_norm_1"] | pipe.extract
            fts = pipe(fts) | enc_block["feed_forward"] | add(fts) | enc_block["layer_norm_2"] | pipe.extract

            # decoder
            block_time_embeds = pipe(time_embeds) | dec_block["time_embed"] | idx('b f -> b f 1 1') | pipe.extract
            te = tuple(block_time_embeds.chunk(9, dim=1))

            ratings = pipe(ratings) | dec_block["layer_norm_1"] | modulate(te[0], te[1]) | maybe_attach_mask \
                      | dec_block["self_attn"] | modulate(te[2]) | add(ratings) | pipe.extract
            maybe_attach_mask = pipe  # stop masking after first block

            attach_fts = lambda x: pipe(fts, x)
            ratings = pipe(ratings) | dec_block["layer_norm_2"] | modulate(te[3], te[4]) | attach_fts \
                      | dec_block["cross_attn"] | modulate(te[5]) | add(ratings) | pipe.extract

            ratings = pipe(ratings) | dec_block["layer_norm_3"] | modulate(te[6], te[7]) \
                      | dec_block["feed_forward"] | modulate(te[8]) | add(ratings) | pipe.extract

        return self.final_linear(ratings)


class GraphReconstructionModel(nn.Module):
    def __init__(self, feature_embedding, subgraph_model):
        super().__init__()
        self.embedding = feature_embedding
        self.core_model = subgraph_model

    def forward(self, x, t, mask=None):
        x = self.embedding(x)
        out = self.core_model(x, t, mask=mask)
        assert out.shape[1] == 1
        return out.squeeze(dim=1)

    @staticmethod
    def default(transformer=True):
        embed = MovieLensFeatureEmb()  # (3, 2, 4, 8)
        if transformer:
            core = SubgraphTransformer(
                feature_dim=embed.embed_dim,
                n_blocks=12,
                time_embedder=SinusoidalPosEmb(embed.embed_dim + embed.embed_dim % 2),
                # attn_kwargs=dict(heads=1, dim_head=8, num_mem_kv=0)
            )
        else:
            core = SubgraphAttnModel(
                embed.embed_dim, [],
                RandomOrLearnedSinusoidalPosEmb(embed.embed_dim + embed.embed_dim % 2),
                # attn_kwargs=dict(heads=1, dim_head=8, num_mem_kv=0)
            )
        return GraphReconstructionModel(embed, core)


if __name__ == '__main__':
    model = GraphReconstructionModel.default(transformer=True)
    print("num params:", sum(param.numel() for param in model.parameters()))
    model(torch.rand(1, 26, 8, 9), torch.tensor([1]), None)