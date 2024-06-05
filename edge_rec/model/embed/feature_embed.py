from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import nn


# noinspection PyPep8Naming
class EmbedderConfigurationSchema:
    ConfigItem = Tuple[nn.Module, Callable]
    # bool indicates whether feature is required, default True ........................ vvvv
    EmbeddingConfig = Dict[str, Union[Optional[ConfigItem], Tuple[Optional[ConfigItem], bool]]]

    def __init__(self, user_config: EmbeddingConfig, product_config: EmbeddingConfig):
        self.user_config = self._sanitize_config(user_config)
        self.product_config = self._sanitize_config(product_config)

    @property
    def modules(self) -> nn.Module:
        return nn.ModuleList(modules=[
            cfg[0]  # first element of ConfigItem
            for config in (self.user_config, self.product_config)
            for (cfg, _) in config.values()
            if cfg is not None
        ])

    @staticmethod
    def _sanitize_config(config: EmbeddingConfig) -> EmbeddingConfig:
        # converts all single ConfigItem entries to carry bool w/ required=True (default)
        return {
            feature_key: (cfg if cfg is not None and isinstance(cfg[1], bool) else (cfg, True))
            for feature_key, cfg in config.items()
        }

    IdentityEmbedding = None

    @staticmethod
    def EnumEmbedding(enum_size: int, embedding_dim: int) -> ConfigItem:
        embedding_module = nn.Embedding(
            num_embeddings=enum_size,
            embedding_dim=embedding_dim,
        )
        return embedding_module, embedding_module

    @staticmethod
    def BatchedEnumEmbedding(
            enum_size: int,
            embedding_dim: int,
            collapse_dims: Union[int, Tuple[int, ...]] = -1,
            collapse_method: str = 'add',  # 'add' or 'merge'
    ) -> ConfigItem:
        collapse_options = ('add', 'merge')
        if collapse_method not in collapse_options:
            raise ValueError(f"collapse_method must be one of {collapse_options}. Got '{collapse_method}'.")

        if isinstance(collapse_dims, int):
            collapse_dims = (collapse_dims,)
        target_dims = tuple(-i - 1 for i in range(len(collapse_dims)))
        if set(collapse_dims) == set(target_dims):
            collapse_dims = None

        embedding_module = nn.Embedding(
            num_embeddings=enum_size,
            embedding_dim=embedding_dim,
        )

        def _embed(data: torch.Tensor):
            # prepare data for collapsing dims
            if collapse_dims is not None:
                data = torch.movedim(data, collapse_dims, target_dims)
            data = torch.flatten(data, start_dim=target_dims[-1], end_dim=target_dims[0])

            # embed data
            data = embedding_module(data)

            # collapse dims
            if collapse_method == 'add':
                data = data.sum(dim=-2)
            elif collapse_method == 'merge':
                data = torch.flatten(data, start_dim=-2, end_dim=-1)
            else:
                assert False, f"Forgot to handle collapse_method='{collapse_method}'"

            # return embedded data
            return data

        return embedding_module, _embed

    @staticmethod
    def LinearEmbedding(in_dim: int, embedding_dim: int, bias: bool = True) -> ConfigItem:
        embedding_layer = nn.Linear(in_dim, embedding_dim, bias=bias)
        return embedding_layer, embedding_layer


class FeatureEmbedder(nn.Module):
    def __init__(self, config: EmbedderConfigurationSchema):
        super().__init__()
        self.embedding_config = config
        self._cfg_modules = config.modules  # so that all modules are registered by pytorch

    def _embed(self, features: dict, embedding_config: EmbedderConfigurationSchema.EmbeddingConfig) -> torch.Tensor:
        embedded_features = []
        for feature_key, (cfg, required) in embedding_config.items():
            if feature_key in features:
                feature = features[feature_key]
                if cfg is not None:
                    embedder = cfg[1]  # second element of EmbedderConfigurationSchema.ConfigItem
                    assert callable(embedder), "Cannot call embedder function"
                    feature = embedder(feature)
                embedded_features.append(feature)
            elif required:
                raise ValueError(
                    f"Cannot embed {features.keys()} with {self.__class__}. Missing required key '{feature_key}'."
                )
        assert len(embedded_features) > 0, "No features were embedded"
        return torch.cat(embedded_features, dim=-1)

    def forward(self, user_features: dict, product_features: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        user_features = self._embed(user_features, self.embedding_config.user_config)
        product_features = self._embed(product_features, self.embedding_config.product_config)

        # return features
        return user_features, product_features
