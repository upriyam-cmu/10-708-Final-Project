from .embed import FeatureEmbedder, MovieLensFeatureEmbedder, SinusoidalPositionalEmbedding
from .graph_transformer import GraphTransformer

from ..datasets import RatingSubgraphData
from ..diffusion import RatingDenoisingModel

from torch import nn


class GraphReconstructionModel(nn.Module, RatingDenoisingModel):
    def __init__(self, feature_embedding: FeatureEmbedder, subgraph_model: nn.Module, feature_dim_size: int = None):
        super().__init__()
        self.embedding = feature_embedding
        self.core_model = subgraph_model

        if feature_dim_size is not None:
            user_ft_size, product_ft_size = feature_embedding.output_sizes
            user_feature_transform = nn.Linear(user_ft_size, feature_dim_size)
            product_feature_transform = nn.Linear(product_ft_size, feature_dim_size)
            self.feature_transforms = user_feature_transform, product_feature_transform
        else:
            self.feature_transforms = None

    def forward(self, rating_data: RatingSubgraphData, time_steps):
        # unpack arguments
        noise_map, known_mask = rating_data.ratings, rating_data.known_mask
        user_features, product_features = rating_data.user_features, rating_data.product_features

        # embed features
        user_features, product_features = self.embedding(user_features, product_features)

        # transform features to matching dim sizes
        if self.feature_transforms is not None:
            user_feature_transform, product_feature_transform = self.feature_transforms
            user_features = user_feature_transform(user_features)
            product_features = product_feature_transform(product_features)

        # add batch dims as necessary
        if len(noise_map.shape) not in (3, 4):
            raise ValueError(
                f"Unexpected shape of rating data. Expected 3- or 4-dimensional tensor. Got shape={noise_map.shape}."
            )

        if len(noise_map.shape) == 3:
            added_batch_dim = True

            noise_map = noise_map.unsqueeze(dim=0)
            if known_mask is not None:
                known_mask = known_mask.unsqueeze(dim=0)
            user_features = user_features.unsqueeze(dim=0)
            product_features = product_features.unsqueeze(dim=0)
        else:
            added_batch_dim = False

        # compute model results
        out = self.core_model(
            noise_map=noise_map,
            user_features=user_features,
            product_features=product_features,
            time_steps=time_steps,
            known_mask=known_mask,
        )

        # maybe drop batch dim & return prediction
        if added_batch_dim:
            assert out.shape[0] == 1
            out = out.squeeze(dim=0)
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
