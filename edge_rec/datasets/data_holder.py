import numpy as np

from .transforms import Transform

from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import IterableDataset


def _stack_dicts(items):
    if len(items) == 0:
        raise ValueError("Cannot process empty list")

    if not isinstance(items[0], dict):
        return items

    return_dict = {key: [] for key in items[0]}

    for _dict in items:
        if not isinstance(_dict, dict):
            raise ValueError("Not all given items were dicts")
        if len(return_dict) != len(_dict):
            raise ValueError("Given dicts do not contain same keys")

        for key, value in _dict.items():
            if key not in return_dict:
                raise ValueError("Given dicts do not contain same keys")

            return_dict[key].append(value)

    return {key: _stack_dicts(values) for key, values in return_dict.items()}


def _merge_nullable_tensors(
        tensors: Union[List[None], List[torch.Tensor]],
        merge_fn: Callable[[List[torch.Tensor]], torch.Tensor],
) -> Optional[torch.Tensor]:
    if len(tensors) == 0 or tensors[0] is None:
        assert not any(tensor is not None for tensor in tensors)
        return None

    return merge_fn(tensors)


class RatingSubgraphData(dict):
    def __init__(
            self,
            *,
            ratings: Optional[torch.Tensor],
            known_mask: Optional[torch.Tensor],
            user_features: Dict[str, torch.Tensor],
            product_features: Dict[str, torch.Tensor]
    ):
        if user_features is None:
            raise ValueError("user_features cannot be None")
        if product_features is None:
            raise ValueError("product_features cannot be None")

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

    @property
    def has_batch_dims(self):
        return len(self.ratings.shape) > 3

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> 'RatingSubgraphData':
        # edge data
        ratings = fn(self.ratings) if self.ratings is not None else None
        known_mask = fn(self.known_mask) if self.known_mask is not None else None

        # features
        user_features = {feature_key: fn(value) for feature_key, value in self.user_features.items()}
        product_features = {feature_key: fn(value) for feature_key, value in self.product_features.items()}

        # return new data object
        return RatingSubgraphData(
            # edge data
            ratings=ratings,
            known_mask=known_mask,
            # features
            user_features=user_features,
            product_features=product_features,
        )

    def to(self, device: torch.device) -> 'RatingSubgraphData':
        return self._apply(lambda tensor: tensor.to(device))

    def slice(
            self,
            user_indices: Union[torch.Tensor, np.ndarray],
            product_indices: Union[torch.Tensor, np.ndarray],
            batched: Optional[bool] = None,
            numpy_indices: bool = True,
    ) -> 'RatingSubgraphData':
        assert len(user_indices.shape) == len(product_indices.shape) == 1
        meshgrid = np.meshgrid if numpy_indices else torch.meshgrid
        i, j = meshgrid(user_indices, product_indices, indexing='ij')

        if batched is None:
            if self.ratings is None:
                raise ValueError("Must specify batched argument if self.ratings is None.")
            batched = len(self.ratings.shape) > 3

        # edge data
        ratings = self.ratings[..., i, j] if self.ratings is not None else None
        known_mask = self.known_mask[..., i, j] if self.known_mask is not None else None

        # features
        user_features = {
            feature_key: (value[:, user_indices] if batched else value[user_indices])
            for feature_key, value in self.user_features.items()
        }
        product_features = {
            feature_key: (value[:, product_indices] if batched else value[product_indices])
            for feature_key, value in self.product_features.items()
        }

        # return new data object
        return RatingSubgraphData(
            # edge data
            ratings=ratings,
            known_mask=known_mask,
            # features
            user_features=user_features,
            product_features=product_features,
        )

    def clone(self, deep: bool = True) -> 'RatingSubgraphData':
        if deep:
            return self._apply(lambda tensor: tensor.clone())
        else:
            return RatingSubgraphData(**self)

    def detach(self) -> 'RatingSubgraphData':
        return self._apply(lambda tensor: tensor.detach())

    def numpy(self):
        # this is technically no longer a RatingSubgraphData, because the contents are ndarrays, not tensors
        return self._apply(lambda tensor: tensor.detach().cpu().numpy())

    @staticmethod
    def stack(rating_data_list: List['RatingSubgraphData'], create_new_dim: bool = False) -> 'RatingSubgraphData':
        # hacky, but gets the conversion done
        lists_of_data = RatingSubgraphData(**_stack_dicts(rating_data_list))
        return lists_of_data._apply(
            partial(
                _merge_nullable_tensors,
                merge_fn=partial(
                    torch.stack if create_new_dim else torch.cat,
                    dim=0,
                ),
            )
        )

    def with_batching(
            self,
            batch_specs: Optional[Union[torch.Size, int]] = None,
    ) -> Tuple['RatingSubgraphData', Union[torch.Size, int]]:
        """
        If batch_specs is None, adjusts batch to include just a single dimension.
        If batch_specs is an int, collapses that many leading dimensions.
        If batch_specs is torch.Size, reshapes leading dimension to specified shape.

        Returns the argument (to this same function) required to undo this batching operation.
        """
        if batch_specs is None:
            n_batch_dims = len(self.ratings.shape) - 3
            assert n_batch_dims >= 0
            batch_specs = n_batch_dims - 1

        if isinstance(batch_specs, int):
            if batch_specs == 0:
                return self.clone(deep=False), 0

            if batch_specs < 0:
                n_dims_to_add = -batch_specs
                return self._apply(lambda tensor: tensor.reshape((1,) * n_dims_to_add + tensor.shape)), n_dims_to_add

            # else batch_specs > 0
            n_dims_to_remove = batch_specs
            orig_dim_shape = self.ratings.shape[:n_dims_to_remove + 1]
            return self._apply(lambda tensor: tensor.flatten(end_dim=n_dims_to_remove)), orig_dim_shape

        if isinstance(batch_specs, tuple):  # or torch.Size
            return self._apply(lambda tensor: tensor.reshape(batch_specs + tensor.shape[1:])), len(batch_specs) - 1

        assert False, f"unreachable: did not handle {type(batch_specs)}"


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
