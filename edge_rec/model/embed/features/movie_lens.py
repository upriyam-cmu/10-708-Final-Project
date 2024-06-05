from ..feature_embed import EmbedderConfigurationSchema as ECS, FeatureEmbedder

from typing import Optional


def _maybe_default(config: Optional[ECS]):
    if config is not None:
        return config

    return ECS(
        user_config={
            'age': (
                ECS.EnumEmbedding(enum_size=7, embedding_dim=4),
                True,  # required
            ),
            'gender': (
                ECS.EnumEmbedding(enum_size=2, embedding_dim=2),
                True,  # required
            ),
            'occupation': (
                ECS.EnumEmbedding(enum_size=21, embedding_dim=8),
                True,  # required
            ),
            'rating_counts': (
                ECS.IdentityEmbedding,
                False,  # optional
            ),
        },
        product_config={
            'genre_ids': (
                ECS.BatchedEnumEmbedding(enum_size=19, embedding_dim=8, collapse_dims=-1, collapse_method='add'),
                True,  # required
            ),
            # ignore 'genre_multihot' feature
            'rating_counts': (
                ECS.IdentityEmbedding,
                False,  # optional
            ),
        },
    )


class MovieLensFeatureEmbedder(FeatureEmbedder):
    def __init__(self, config: Optional[ECS] = None):
        # TODO make the arguments for this class better than just loading a single default config
        super().__init__(config=_maybe_default(config))
