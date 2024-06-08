from .data import RatingSubgraphData
from .transforms import Transform

from abc import ABC, abstractmethod
from functools import partial
from typing import Optional, Tuple, Union

import numpy as np
from torch.utils.data import IterableDataset


class _DatasetWrapper(IterableDataset):
    def __init__(self, data_generator):
        self.data_generator = data_generator

    def _get_subgraph(self):
        # loop until you get a subgraph with data
        ratings_data: RatingSubgraphData = self.data_generator()
        while ratings_data.known_mask.sum() == 0:
            ratings_data: RatingSubgraphData = self.data_generator()

        # strip away wrapping RatingSubgraphData class
        return dict(ratings_data)

    def __getitem__(self, idx=None):
        return self._get_subgraph()

    def __iter__(self):
        return self

    def __next__(self):
        return self._get_subgraph()


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
    def get_subgraph_indices(
            self,
            subgraph_size: Union[Optional[int], Tuple[Optional[int], Optional[int]]],
            target_density: Optional[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def slice_subgraph(
            self,
            user_indices: np.ndarray,
            product_indices: np.ndarray,
            *,
            return_train_edges: bool = True,
            return_test_edges: bool = True,
    ) -> RatingSubgraphData:
        pass

    def get_subgraph(
            self,
            subgraph_size: Union[Optional[int], Tuple[Optional[int], Optional[int]]],
            target_density: Optional[float],
            *,
            return_train_edges: bool = True,
            return_test_edges: bool = True,
    ) -> RatingSubgraphData:
        user_indices, product_indices = self.get_subgraph_indices(
            subgraph_size=subgraph_size,
            target_density=target_density,
        )
        return self.slice_subgraph(
            user_indices=user_indices,
            product_indices=product_indices,
            return_train_edges=return_train_edges,
            return_test_edges=return_test_edges,
        )

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
