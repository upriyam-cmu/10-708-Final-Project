from .gdit import GraphTransformer
from .embed import FeatureEmbedder, MovieLensFeatureEmbedder
from .embed import SinusoidalPositionalEmbedding, RandomOrLearnedSinusoidalPositionalEmbedding

import torch
from torch import nn


class GraphReconstructionModel(nn.Module):
    def __init__(self, feature_embedding: FeatureEmbedder, subgraph_model: nn.Module):
        super().__init__()
        self.embedding = feature_embedding
        self.core_model = subgraph_model

    def forward(self, noise_matrix, user_features, product_features, time_steps, known_mask=None):
        user_features, product_features = self.embedding(user_features, product_features)
        out = self.core_model(noise_matrix, user_features, product_features, time_steps, known_mask=known_mask)
        assert out.shape[1] == 1
        return out.squeeze(dim=1)

    @staticmethod
    def default():
        # TODO check default impl (need to fix lack of `embed.embed_dim`)
        embed = MovieLensFeatureEmbedder()
        core = GraphTransformer(
            feature_dim=embed.embed_dim,
            n_blocks=1,
            time_embedder=SinusoidalPositionalEmbedding(embed.embed_dim + embed.embed_dim % 2),
            # attn_kwargs=dict(heads=1, dim_head=8, num_mem_kv=0)
        )
        return GraphReconstructionModel(embed, core)


if __name__ == '__main__':
    model = GraphReconstructionModel.default()
    print("num params:", sum(param.numel() for param in model.parameters()))
    model(torch.rand(1, 26, 8, 9), torch.tensor([1]), None)
