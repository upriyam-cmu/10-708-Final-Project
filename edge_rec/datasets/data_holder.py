from .transforms import Transform

from abc import ABC, abstractmethod
from collections import namedtuple
from functools import partial
from typing import Optional

from torch.utils.data import IterableDataset

RatingSubGraphData = namedtuple(
    'RatingSubGraphData',
    ['train_ratings', 'test_ratings', 'train_mask', 'test_mask', 'user_features', 'product_features']
)


class DataHolder(ABC):
    class __DatasetWrapper(IterableDataset):
        def __init__(self, data_generator):
            self.data_generator = data_generator

        def __getitem__(self, idx=None):
            return self.data_generator()

        def __iter__(self):
            return self

        def __next__(self):
            return self.data_generator()

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
    ) -> RatingSubGraphData:
        pass

    def get_dataset(self, subgraph_size, target_density: float, train: bool):
        return DataHolder.__DatasetWrapper(
            partial(
                self.get_subgraph,
                subgraph_size=subgraph_size,
                target_density=target_density,
                return_train_edges=train,
                return_test_edges=not train,
            )
        )
