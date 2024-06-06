from .transforms import Transform

from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, Optional

import torch
from torch.utils.data import IterableDataset


class RatingSubgraphData(dict):
    def __init__(
            self,
            *,
            ratings: Optional[torch.Tensor],
            known_mask: Optional[torch.Tensor],
            user_features: Dict[str, torch.Tensor],
            product_features: Dict[str, torch.Tensor]
    ):
        super().__init__(
            ratings=ratings,
            known_mask=known_mask,
            user_features=user_features,
            product_features=product_features,
        )

    @property
    def ratings(self) -> Optional[torch.Tensor]:
        return self['ratings']

    @ratings.setter
    def ratings(self, ratings: Optional[torch.Tensor]):
        self['ratings'] = ratings

    @property
    def known_mask(self) -> Optional[torch.Tensor]:
        return self['known_mask']

    @known_mask.setter
    def known_mask(self, known_mask: Optional[torch.Tensor]):
        self['known_mask'] = known_mask

    @property
    def user_features(self) -> Dict[str, torch.Tensor]:
        return self['user_features']

    @user_features.setter
    def user_features(self, user_features: Dict[str, torch.Tensor]):
        self['user_features'] = user_features

    @property
    def product_features(self) -> Dict[str, torch.Tensor]:
        return self['product_features']

    @product_features.setter
    def product_features(self, product_features: Dict[str, torch.Tensor]):
        self['product_features'] = product_features

    @property
    def device(self):
        return self.ratings.device

    @property
    def dtype(self):
        return self.ratings.dtype

    def to(self, device) -> 'RatingSubgraphData':
        # edge data
        ratings = self.ratings.to(device) if self.ratings is not None else None
        known_mask = self.known_mask.to(device) if self.known_mask is not None else None

        # features
        user_features = {feature_key: value.to(device) for feature_key, value in self.user_features.items()}
        product_features = {feature_key: value.to(device) for feature_key, value in self.product_features.items()}

        # return new data object
        return RatingSubgraphData(
            # edge data
            ratings=ratings,
            known_mask=known_mask,
            # features
            user_features=user_features,
            product_features=product_features,
        )


class _DatasetWrapper(IterableDataset):
    def __init__(self, data_generator):
        self.data_generator = data_generator

    def __getitem__(self, idx=None):
        return self.data_generator()

    def __iter__(self):
        return self

    def __next__(self):
        return self.data_generator()


class DataHolder(ABC):
    def __init__(self, *, data_root, dataset_class, test_split_ratio=0.1, force_download=False):
        self.test_split_ratio = test_split_ratio

        processed_data_path = data_root / "processed/data.pt"
        if force_download or not processed_data_path.exists():
            dataset_class(str(data_root), force_reload=True).process()

        assert processed_data_path.exists(), f"Failed to load {dataset_class} dataset"
        self.processed_data_path = str(processed_data_path)

    @staticmethod
    def _get_transform(possible_augmentations=None, transform_key=None) -> Optional[Transform]:
        if possible_augmentations is not None and transform_key in possible_augmentations:
            return possible_augmentations[transform_key] or Transform.Identity()
        else:
            return None

    @abstractmethod
    def get_subgraph(
            self,
            subgraph_size,
            target_density,
            *,
            return_train_edges: bool = True,
            return_test_edges: bool = True,
    ) -> RatingSubgraphData:
        pass

    def get_dataset(self, subgraph_size, target_density, train: bool):
        return _DatasetWrapper(
            partial(
                self.get_subgraph,
                subgraph_size=subgraph_size,
                target_density=target_density,
                return_train_edges=train,
                return_test_edges=not train,
            )
        )
