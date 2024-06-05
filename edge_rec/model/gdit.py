from .attend import SelfAttention, CrossAttention, Stacked1DAttention, RMSNorm
from .embed import TimeEmbedder
from .utils import get_kwargs, build_feed_forward

from ..utils.pipe import pipe, add, modulate, idx

from torch import nn


class GraphTransformer(nn.Module):
    def __init__(self, feature_dim: int, n_blocks: int, time_embedder: TimeEmbedder,
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

    def forward(self, noise_matrix, user_features, product_features, time_steps, known_mask=None):
        """
        noise_matrix: Tensor(shape=(b, n, m))
        user_features: Tensor(shape=(b, n, f1))
        product_features: Tensor(shape=(b, m, f2))
        time_steps: Tensor(shape=(b,))
        known_mask: Tensor(shape=(b, n, m))

        key:
        - b = batch-size
        - n = num users in subgraph
        - m = num products in subgraph
        - f1 = user feature embedding dim size
        - f2 = product feature embedding dim size
        """
        # TODO fix GDiT implementation for new argument structure
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
