from ...utils import Model, merge_dicts

from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import nn


# noinspection PyPep8Naming
class EmbedderConfigurationSchema:
    ConfigItem = Tuple[Optional[Tuple[nn.Module, Callable]], int]  # (internal_modules, embedder_fn), output_dim_size
    EmbeddingConfig = Dict[str, ConfigItem]

    def __init__(self, user_config: EmbeddingConfig, product_config: EmbeddingConfig):
        if len(user_config) == 0:
            raise ValueError("No user features were selected for embedding")
        if len(product_config) == 0:
            raise ValueError("No product features were selected for embedding")

        self.user_config = user_config
        self.product_config = product_config

    @staticmethod
    def _generate_output_size_dicts(config: EmbeddingConfig, prefix: Optional[str] = None) -> Dict[str, int]:
        prefix = f"{prefix}_" if prefix is not None else ""
        return {
            f"{prefix}{feature_key}": output_dim_size
            for feature_key, (_, output_dim_size) in config.items()
        }

    @property
    def config_spec(self) -> dict:
        return merge_dicts(
            {},
            self._generate_output_size_dicts(self.user_config, prefix="user"),
            self._generate_output_size_dicts(self.product_config, prefix="product"),
            error_message="EmbedderConfigurationSchema has shared features between user/product cfg",
        )

    @property
    def output_sizes(self) -> Tuple[int, int]:
        return (
            sum(self._generate_output_size_dicts(self.user_config).values()),
            sum(self._generate_output_size_dicts(self.product_config).values()),
        )

    @property
    def modules(self) -> nn.Module:
        return nn.ModuleList(modules=[
            cfg[0]  # first element of ConfigItem
            for config in (self.user_config, self.product_config)
            for cfg, _ in config.values()
            if cfg is not None
        ])

    @staticmethod
    def IdentityEmbedding(dim_size: int) -> ConfigItem:
        return None, dim_size

    @staticmethod
    def EnumEmbedding(enum_size: int, embedding_dim: int) -> ConfigItem:
        embedding_module = nn.Embedding(
            num_embeddings=enum_size,
            embedding_dim=embedding_dim,
        )
        return (embedding_module, embedding_module), embedding_dim

    @staticmethod
    def BatchedEnumEmbedding(
            enum_size: int,
            embedding_dim: int,
            collapse_dims: Union[int, Tuple[int, ...]] = -1,
            collapse_method: str = 'add',  # 'add' or 'merge'
            collapsed_dim_size: Optional[int] = None,  # only required for collapse_method='merge'
    ) -> ConfigItem:
        collapse_options = ('add', 'merge')
        if collapse_method not in collapse_options:
            raise ValueError(f"collapse_method must be one of {collapse_options}. Got '{collapse_method}'.")

        if collapse_method == 'merge' and (collapsed_dim_size is None or collapsed_dim_size <= 0):
            raise ValueError(
                f"collapsed_dim_size must be a positive integer when collapse_method='merge'. "
                f"Got {collapsed_dim_size}."
            )

        if isinstance(collapse_dims, int):
            collapse_dims = (collapse_dims,)
        target_dims = tuple(-i - 1 for i in range(len(collapse_dims)))
        if set(collapse_dims) == set(target_dims):
            collapse_dims = None

        assert target_dims[0] == -1

        embedding_module = nn.Embedding(
            num_embeddings=enum_size,
            embedding_dim=embedding_dim,
        )

        def _embed(data: torch.Tensor):
            # prepare data for collapsing dims
            if collapse_dims is not None:
                data = torch.movedim(data, collapse_dims, target_dims)
            data = torch.flatten(data, start_dim=target_dims[-1], end_dim=-1)  # target_dims[0]
            if collapsed_dim_size is not None and data.shape[-1] != collapsed_dim_size:
                raise ValueError(
                    f"Given data does not conform to expected collapsed_dim_size of {collapsed_dim_size}. "
                    f"Got {data.shape[-1]}."
                )

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

        return (embedding_module, _embed), embedding_dim * (collapsed_dim_size or 1)

    @staticmethod
    def LinearEmbedding(in_dim: int, embedding_dim: int, bias: bool = True) -> ConfigItem:
        embedding_layer = nn.Linear(in_dim, embedding_dim, bias=bias)
        return (embedding_layer, embedding_layer), embedding_dim


class FeatureEmbedder(Model):
    def __init__(self, config: EmbedderConfigurationSchema, config_spec: Optional[dict] = None):
        super().__init__(config_spec=(config_spec or config.config_spec))
        self.embedding_config = config
        self._cfg_modules = config.modules  # so that all modules are registered by pytorch

    @property
    def output_sizes(self) -> Tuple[int, int]:
        return self.embedding_config.output_sizes

    def _embed(self, features: dict, embedding_config: EmbedderConfigurationSchema.EmbeddingConfig) -> torch.Tensor:
        embedded_features = []
        for feature_key, (cfg, out_dim_size) in embedding_config.items():
            if feature_key in features:
                feature = features[feature_key]
                if cfg is not None:
                    embedder = cfg[1]  # second element of EmbedderConfigurationSchema.ConfigItem
                    assert callable(embedder), "Cannot call embedder function"
                    feature = embedder(feature)
                assert feature.shape[-1] == out_dim_size, f"Output size mismatch on feature key {feature_key}"
                embedded_features.append(feature)
            else:
                raise ValueError(
                    f"Cannot embed {features.keys()} with {self.__class__}. Missing key '{feature_key}'."
                )
        assert len(embedded_features) > 0, "No features were embedded"
        return torch.cat(embedded_features, dim=-1)

    def forward(self, user_features: dict, product_features: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        user_features = self._embed(user_features, self.embedding_config.user_config)
        product_features = self._embed(product_features, self.embedding_config.product_config)

        # return features
        return user_features, product_features
