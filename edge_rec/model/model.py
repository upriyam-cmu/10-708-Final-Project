from .graph_transformer import GraphTransformer
from .embed import FeatureEmbedder, MovieLensFeatureEmbedder
from .embed import SinusoidalPositionalEmbedding, RandomOrLearnedSinusoidalPositionalEmbedding

import torch
from torch import nn


class GraphReconstructionModel(nn.Module):
    def __init__(self, feature_embedding: FeatureEmbedder, subgraph_model: nn.Module, feature_dim_size: int = 16):
        super().__init__()
        self.embedding = feature_embedding
        self.core_model = subgraph_model

        user_ft_size, product_ft_size = feature_embedding.output_sizes
        self.user_feature_transform = nn.Linear(user_ft_size, feature_dim_size)
        self.product_feature_transform = nn.Linear(product_ft_size, feature_dim_size)

    def forward(self, noise_map, user_features, product_features, time_steps, known_mask=None):
        # embed features
        user_features, product_features = self.embedding(user_features, product_features)

        # transform features to matching dim sizes
        user_features = self.user_feature_transform(user_features)
        product_features = self.product_feature_transform(product_features)

        # compute model results
        out = self.core_model(
            noise_map=noise_map,
            user_features=user_features,
            product_features=product_features,
            time_steps=time_steps,
            known_mask=known_mask,
        )

        # # collapse time dim & return
        # assert out.shape[1] == 1
        # return out.squeeze(dim=1)
        return out

    @staticmethod
    def default(feature_dim_size: int = 16):
        embed = MovieLensFeatureEmbedder()
        core = GraphTransformer(
            n_blocks=1,
            n_channels=1,
            n_features=feature_dim_size,
            time_embedder=SinusoidalPositionalEmbedding(feature_dim_size),
            # attn_kwargs=dict(heads=1, dim_head=8, num_mem_kv=0)
        )
        return GraphReconstructionModel(embed, core, feature_dim_size=feature_dim_size)
