from ..utils import merge_nullable_tensors, stack_dicts

from contextlib import contextmanager
from functools import partial
from typing import Callable, ContextManager, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


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
    def shape(self):
        return self.ratings.shape

    @property
    def device(self):
        return self.ratings.device

    @property
    def dtype(self):
        return self.ratings.dtype

    @property
    def has_batch_dims(self):
        return len(self.shape) > 3

    def _apply(self, fn: Optional[Callable[[torch.Tensor], torch.Tensor]]) -> 'RatingSubgraphData':
        if fn is None:
            # default fn == identity fn == shallow copy
            return RatingSubgraphData(**self)

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
            batched = len(self.shape) > 3

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
    def stack(rating_data_list: List['RatingSubgraphData'], create_new_dim: bool = True) -> 'RatingSubgraphData':
        # hacky, but gets the conversion done
        lists_of_data = RatingSubgraphData(**stack_dicts(rating_data_list))
        return lists_of_data._apply(
            partial(
                merge_nullable_tensors,
                merge_fn=partial(
                    torch.stack if create_new_dim else torch.cat,
                    dim=0,
                ),
            )
        )

    @staticmethod
    def __compute_batching_transformation(
            shape: torch.Size,
            batch_specs: Optional[Union[torch.Size, int]] = None,
    ) -> Tuple[Optional[Callable[[torch.Tensor], torch.Tensor]], int]:
        """
        If batch_specs is None, adjusts batch to include just a single dimension.
        If batch_specs is an int, collapses that many leading dimensions.
        If batch_specs is torch.Size, reshapes leading dimension to specified shape.

        Returns the argument (to this same function) required to undo this batching operation.
        """
        if batch_specs is None:
            n_batch_dims = len(shape) - 3
            assert n_batch_dims >= 0
            batch_specs = n_batch_dims - 1

        if isinstance(batch_specs, int):
            if batch_specs == 0:
                return None, 0

            if batch_specs < 0:
                n_dims_to_add = -batch_specs
                return lambda tensor: tensor.reshape((1,) * n_dims_to_add + tensor.shape), n_dims_to_add

            # else batch_specs > 0
            n_dims_to_remove = batch_specs
            orig_dim_shape = shape[:n_dims_to_remove + 1]
            return lambda tensor: tensor.flatten(end_dim=n_dims_to_remove), orig_dim_shape

        if isinstance(batch_specs, tuple):  # or torch.Size
            return lambda tensor: tensor.reshape(batch_specs + tensor.shape[1:]), len(batch_specs) - 1

        assert False, f"unreachable: did not handle {type(batch_specs)}"

    def with_batching(
            self,
            batch_specs: Optional[Union[torch.Size, int]] = None,
    ) -> Tuple['RatingSubgraphData', Union[torch.Size, int]]:
        # see doc for self.__compute_batching_transformation
        transformer_fn, inversion_spec = self.__compute_batching_transformation(self.shape, batch_specs)
        return self._apply(transformer_fn), inversion_spec

    @contextmanager
    def batched(self) -> ContextManager[Tuple[
        'RatingSubgraphData',
        Optional[Callable[[torch.Tensor], torch.Tensor]],
        Optional[Callable[[torch.Tensor], torch.Tensor]],
    ]]:
        """
        Returns 3 elements:
        1) the batched object
        2) function to batch other tensors (None == identity)
        3) function to undo batching (None == identity)
        """
        # see doc for self.__compute_batching_transformation
        forward_fn, inversion_spec = self.__compute_batching_transformation(self.shape, None)
        batched_data = self._apply(forward_fn)
        inversion_fn, _ = self.__compute_batching_transformation(batched_data.shape, inversion_spec)
        yield batched_data, forward_fn, inversion_fn
