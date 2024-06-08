from .attend import SelfAttention, Stacked1DSelfAttention as RCSSelfAttn, SeparableCrossAttention as RCSCrossAttn
from .embed import TimeEmbedder

from ..utils import Model, get_kwargs

from math import sqrt
from typing import Optional, Tuple, Union

from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F


def build_feed_forward(in_dim, hidden_dims, out_dim, activation_fn):
    avg_dim = sqrt(in_dim * out_dim)
    hidden_dims = [int(avg_dim * h_dim) for h_dim in hidden_dims]
    if len(hidden_dims) != 0 and activation_fn is None:
        raise ValueError("Must specify activation function for feed-forward networks if num hidden layers > 0")

    components = []
    for d_in, d_out in zip([in_dim] + hidden_dims, hidden_dims + [out_dim]):
        components.append(nn.Conv2d(d_in, d_out, 1))
        components.append(activation_fn)

    return nn.Sequential(*components[:-1])


def update_default_kwargs(kwargs, **defaults):
    if kwargs is None:
        kwargs = {}
    return {**defaults, **kwargs}


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class FeatureEncoderBlock(nn.Module):
    def __init__(self, feature_dim_size: int, attn_kwargs: dict, feed_forward_kwargs: dict):
        super().__init__()

        self.self_attn = SelfAttention(**attn_kwargs, dim=feature_dim_size)
        self.layer_norm_1 = RMSNorm(feature_dim_size)
        self.feed_forward = build_feed_forward(
            **feed_forward_kwargs,
            in_dim=feature_dim_size,
            out_dim=feature_dim_size,
        )
        self.layer_norm_2 = RMSNorm(feature_dim_size)

    def forward(self, features):
        """
        features = Tensor(shape=(b, n, f))

        key:
        - b = batch size
        - n = num users/products in subgraph
        - f = feature embed dim size
        """
        # reshape features to match format used in attn/MLP layers
        features = rearrange(features, 'b n f -> b f n 1')

        # apply self-attention layer, residual connection, and first layer norm
        features = self.self_attn(features) + features  # residual
        features = self.layer_norm_1(features)

        # apply MLP layer, residual connection, and second layer norm
        features = self.feed_forward(features) + features  # residual
        features = self.layer_norm_2(features)

        # return features in original shape
        return rearrange(features, 'b f n 1 -> b n f')


class RatingDecoderBlock(nn.Module):
    def __init__(self, channel_dim_size: int, core_block: nn.Module):
        super().__init__()

        self.layer_norm = RMSNorm(channel_dim_size)
        self.core_block = core_block

    def forward(
            self,
            noise_map: torch.Tensor,
            *args: torch.Tensor,
            scale: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        noise_map: Tensor(shape=(b, c, n, m))
        args -- mask, or features, or nothing
        scale: tuple of 3 tensors with scaling constants
        """
        orig_noise_map = noise_map
        assert len(scale) == 3

        # apply layer norm & modulate
        noise_map = self.layer_norm(noise_map)
        noise_map = scale[0] * noise_map + scale[1]

        # apply core block (self attn, cross attn, or MLP) & modulate
        noise_map = self.core_block(noise_map, *args)
        noise_map = scale[2] * noise_map

        # apply residual connection & return
        return noise_map + orig_noise_map  # residual


class GraphTransformer(Model):
    __DEFAULT_ATTN_KWARGS = dict(heads=4, dim_head=32, num_mem_kv=4, flash=False, share_weights=True, dropout=0.)
    __DEFAULT_FEED_FORWARD_KWARGS = dict(hidden_dims=(), activation_fn=None)

    def __init__(
            self,
            n_blocks: int,
            n_channels: int,
            n_features: Union[int, Tuple[int, int]],
            time_embedder: TimeEmbedder,
            n_channels_internal: Optional[int] = None,
            attn_kwargs: dict = None,
            feed_forward_kwargs: dict = None,
    ):
        super().__init__(model_spec=get_kwargs())
        attn_kwargs = {
            **self.__DEFAULT_ATTN_KWARGS,
            **(attn_kwargs or {}),
        }
        feed_forward_kwargs = {
            **self.__DEFAULT_FEED_FORWARD_KWARGS,
            **(feed_forward_kwargs or {}),
        }

        if isinstance(n_features, int):
            user_feature_dim_size = product_feature_dim_size = n_features
        elif isinstance(n_features, tuple):
            user_feature_dim_size, product_feature_dim_size = n_features
        else:
            raise ValueError(f"n_features must be an int or a tuple of 2 ints. Got {n_features}.")

        n_channels_internal = n_channels_internal or n_channels

        self.time_embed_initial = nn.Sequential(
            time_embedder,
            nn.Linear(time_embedder.out_dim, 4 * n_channels),
            nn.SiLU(),
            nn.Linear(4 * n_channels, 4 * n_channels_internal)
        )

        self.initial_linear = nn.Conv2d(n_channels, n_channels_internal, 1)

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for _ in range(n_blocks):
            # encoder
            encoder_modules = nn.ModuleDict()
            self.encoder_blocks.append(encoder_modules)

            encoder_modules["user_features_block"] = FeatureEncoderBlock(
                feature_dim_size=user_feature_dim_size,
                attn_kwargs=attn_kwargs,
                feed_forward_kwargs=feed_forward_kwargs,
            )
            encoder_modules["product_features_block"] = FeatureEncoderBlock(
                feature_dim_size=product_feature_dim_size,
                attn_kwargs=attn_kwargs,
                feed_forward_kwargs=feed_forward_kwargs,
            )

            # decoder
            decoder_modules = nn.ModuleDict()
            self.decoder_blocks.append(decoder_modules)

            decoder_modules["time_embed"] = nn.Sequential(
                nn.SiLU(),
                nn.Linear(4 * n_channels_internal, 9 * n_channels_internal)
            )

            decoder_modules["self_attn_block"] = RatingDecoderBlock(
                channel_dim_size=n_channels_internal,
                core_block=RCSSelfAttn(**attn_kwargs, dim=n_channels_internal),
            )
            decoder_modules["cross_attn_block"] = RatingDecoderBlock(
                channel_dim_size=n_channels_internal,
                core_block=RCSCrossAttn(
                    **attn_kwargs,
                    d_row_ft=user_feature_dim_size,
                    d_col_ft=product_feature_dim_size,
                    d_channel=n_channels_internal,
                ),
            )
            decoder_modules["feed_forward_block"] = RatingDecoderBlock(
                channel_dim_size=n_channels_internal,
                core_block=build_feed_forward(
                    **feed_forward_kwargs,
                    in_dim=n_channels_internal,
                    out_dim=n_channels_internal,
                ),
            )

        self.final_linear = nn.Conv2d(n_channels_internal, n_channels, 1)

    def forward(self, noise_map, user_features, product_features, time_steps, known_mask=None):
        """
        noise_map: Tensor(shape=(b, c, n, m))
        user_features: Tensor(shape=(b, n, f1))
        product_features: Tensor(shape=(b, m, f2))
        time_steps: Tensor(shape=(b,))
        known_mask: Tensor(shape=(b, n, m))

        key:
        - b = batch size
        - c = number of data channels
        - n = num users in subgraph
        - m = num products in subgraph
        - f1 = user feature embedding dim size
        - f2 = product feature embedding dim size
        """
        # initialize time embeddings
        time_embeds = self.time_embed_initial(time_steps)

        # loop through all blocks
        for enc_block, dec_block in zip(self.encoder_blocks, self.decoder_blocks):
            # encoder
            user_features = enc_block["user_features_block"](user_features)
            product_features = enc_block["product_features_block"](product_features)

            # time embedding constants
            block_time_embeds = rearrange(dec_block["time_embed"](time_embeds), 'b f -> b f 1 1')
            assert block_time_embeds.shape[1] % 9 == 0
            te = tuple(block_time_embeds.tensor_split(9, dim=1))

            # decoder
            noise_map = dec_block["self_attn_block"](noise_map, known_mask, scale=te[:3])
            known_mask = None  # drop mask after first block
            noise_map = dec_block["cross_attn_block"](noise_map, user_features, product_features, scale=te[3:6])
            noise_map = dec_block["self_attn_block"](noise_map, scale=te[6:])

        # final linear layer & return
        return self.final_linear(noise_map)
