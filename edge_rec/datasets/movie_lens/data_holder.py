from .ml1m import RawMovieLens1M
from .ml100k import RawMovieLens100K

from ..data import RatingSubgraphData
from ..data_holder import DataHolder
from ..transforms import Transform

from ...utils import get_kwargs

from functools import partial
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pathlib import Path
import torch


class MovieLensDataHolder(DataHolder):
    def __init__(
            self,
            root="./data",
            ml100k: bool = False,
            ml1m: bool = False,
            test_split=0.1,
            time_aware_split=True,
            force_download=False,
            augmentations: Dict[str, Transform] = None,
    ):
        if np.sum([ml100k, ml1m]) != 1:
            raise ValueError("Must specify dataset as exactly one of ML-100k or ML-1M.")

        super().__init__(
            data_root=(Path(root) / ("ml100k" if ml100k else "ml1m")),
            dataset_class=(RawMovieLens100K if ml100k else RawMovieLens1M),
            test_split_ratio=test_split,
            force_download=force_download,
            config_spec=get_kwargs(),
        )

        # load rating data
        data = torch.load(self.processed_data_path)[0]
        self.train_edges, self.test_edges, self.all_edges = self._split_edges(
            data[('user', 'rates', 'movie')]['edge_index'],
            data[('user', 'rates', 'movie')]['rating'],
            test_split=test_split,
            edge_timestamps=data[('user', 'rates', 'movie')]['time'] if time_aware_split else None
        )

        # initialize data transforms
        _get_transform = partial(self._get_transform, possible_augmentations=augmentations)

        self.ratings_transform = _get_transform(transform_key='ratings') or Transform.Identity()
        self.ratings_transform.fit(self.train_edges[1], numpy=True)

        self.rating_counts_transform = _get_transform(transform_key='rating_counts')

        # load user/movie features
        self.user_data, self.movie_data, self.n_users, self.n_movies = self._get_features(
            data['user']['x'],
            data['movie']['x'],
        )
        self._augment_features(self.train_edges)

        # additional processing
        self.top_users, self.top_movies = self._build_density_scores(self.train_edges, self.n_users, self.n_movies)

    @staticmethod
    def _split_edges(edge_indices: torch.Tensor, edge_ratings: torch.Tensor, test_split: float, edge_timestamps: Optional[torch.Tensor] = None):
        n_edges = len(edge_ratings)

        edge_indices, edge_ratings = edge_indices.numpy(), edge_ratings.numpy()
        
        indices_to_sort, sort_by = [edge_indices[0]], [0]
        if edge_timestamps is not None:
            edge_timestamps = edge_timestamps.numpy()
            indices_to_sort.append(edge_timestamps)
            sort_by.append(1)
        indices_df = pd.DataFrame(np.stack(indices_to_sort).T)
        sort_indices = indices_df.sort_values(sort_by).reset_index()["index"].to_numpy()

        edge_indices, edge_ratings = edge_indices[:, sort_indices], edge_ratings[sort_indices]
        train_group, test_group = [], []
        _, split_indices = np.unique(edge_indices[0], return_index=True)
        for edge_group in np.split(np.arange(n_edges), split_indices[1:]):
            if edge_timestamps is None:
                np.random.shuffle(edge_group)   
            n_test = int(test_split * len(edge_group))
            train_group.append(edge_group[:-n_test])
            test_group.append(edge_group[-n_test:])

        train_group = np.concatenate(train_group)
        test_group = np.concatenate(test_group)

        train_edges = edge_indices.T[train_group], edge_ratings[train_group]
        test_edges = edge_indices.T[test_group], edge_ratings[test_group]
        all_edges = edge_indices.T, edge_ratings

        return train_edges, test_edges, all_edges

    @staticmethod
    def _get_features(user_data_tensor, movie_data_tensor):
        assert len(user_data_tensor.shape) == 2 and user_data_tensor.shape[-1] == 3
        assert len(movie_data_tensor.shape) == 2 and movie_data_tensor.shape[-1] == 6
        n_users, n_movies = len(user_data_tensor), len(movie_data_tensor)

        user_data = {
            'id': torch.arange(n_users).int(),
            'age': user_data_tensor[:, 0].int(),
            'gender': user_data_tensor[:, 1].int(),
            'occupation': user_data_tensor[:, 2].int(),
        }

        genre_multihot = torch.zeros(len(movie_data_tensor), 19, device=movie_data_tensor.device)
        genre_multihot.scatter_(1, movie_data_tensor.long(), 1.)
        movie_data = {
            'id': torch.arange(n_movies).int(),
            'genre_ids': movie_data_tensor.int(),
            'genre_multihot': genre_multihot[:, 1:].int(),  # drop genre idx=0 (i.e. null genre?)
        }

        return user_data, movie_data, len(user_data_tensor), len(movie_data_tensor)

    @staticmethod
    def _count_unique(edges, idx, n_unique):
        ids, counts = np.unique(edges[0][:, idx], return_counts=True)
        out = np.zeros(n_unique)
        out[ids] = counts
        return torch.tensor(out)

    def _augment_features(self, edges):
        if self.rating_counts_transform is not None:
            # get counts
            user_counts = self._count_unique(edges, 0, self.n_users).reshape((self.n_users, 1)).float()
            movie_counts = self._count_unique(edges, 1, self.n_movies).reshape((self.n_movies, 1)).float()

            # transform counts
            user_count_fts = self.rating_counts_transform(user_counts)
            movie_count_fts = self.rating_counts_transform(movie_counts)

            # append counts
            self.user_data['rating_counts'] = user_count_fts
            self.movie_data['rating_counts'] = movie_count_fts

    @staticmethod
    def _build_density_scores(edges, n_users, n_movies):
        movie_stats = [set() for _ in range(n_movies)]
        user_stats = [set() for _ in range(n_users)]
        for user, movie in edges[0]:
            movie_stats[movie].add(user)
            user_stats[user].add(movie)
        assert len(user_stats) == n_users and len(movie_stats) == n_movies

        user_scores = np.array([
            sum(len(movie_stats[movie]) for movie in movies)
            for user, movies in enumerate(user_stats)
        ])
        movie_scores = np.array([
            sum(len(user_stats[user]) for user in users)
            for movie, users in enumerate(movie_stats)
        ])

        top_users = np.argsort(user_scores)[::-1]
        top_movies = np.argsort(movie_scores)[::-1]

        return top_users, top_movies

    def get_subgraph_indices(
            self,
            subgraph_size: Union[Optional[int], Tuple[Optional[int], Optional[int]]],
            target_density: Optional[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if subgraph_size is None:
            subgraph_size = (self.n_users, self.n_movies)
        else:
            sz1, sz2 = (subgraph_size, subgraph_size) if isinstance(subgraph_size, int) else subgraph_size
            if sz1 is None:
                sz1 = self.n_users
            if sz2 is None:
                sz2 = self.n_movies
            subgraph_size = (sz1, sz2)
        assert len(subgraph_size) == 2 and all(type(sz) == int for sz in subgraph_size), f"size={subgraph_size}"
        n_users_sampled, n_movies_sampled = subgraph_size

        if target_density is None:
            user_indices = np.random.choice(self.n_users, n_users_sampled, replace=False)
            movie_indices = np.random.choice(self.n_movies, n_movies_sampled, replace=False)
        else:
            slice_point = round((target_density ** (-1 / 2.25) - 1) * 500)
            assert slice_point >= n_users_sampled and slice_point >= n_movies_sampled, \
                "Desired density too high for desired subgraph size"

            random_weights = ((np.arange(slice_point)[::-1] + 1) / 500 + 1) ** -2.25
            random_weights = random_weights / random_weights.sum()
            user_indices = np.random.choice(
                self.top_users[:slice_point],
                size=n_users_sampled,
                replace=False,
                p=random_weights,
            )
            movie_indices = np.random.choice(
                self.top_movies[:slice_point],
                size=n_movies_sampled,
                replace=False,
                p=random_weights,
            )

        return user_indices, movie_indices

    @staticmethod
    def slice_edges(edges, user_indices, movie_indices) -> torch.Tensor:
        n_users_sampled, n_movies_sampled = len(user_indices), len(movie_indices)
        user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_indices)}
        assert len(user_id_to_idx) == n_users_sampled
        movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_indices)}
        assert len(movie_id_to_idx) == n_movies_sampled

        sliced = torch.zeros(n_users_sampled, n_movies_sampled)
        for (user_id, movie_id), rating in zip(*edges):
            if user_id in user_id_to_idx and movie_id in movie_id_to_idx:
                sliced[user_id_to_idx[user_id], movie_id_to_idx[movie_id]] = rating
        return sliced

    @staticmethod
    def get_ratings_and_mask(*all_edges, user_indices, movie_indices, ratings_transform: Transform = None):
        if len(all_edges) == 0:
            # no edges to select
            return None, None

        # merge all edge groups together
        all_ids, all_ratings = [], []
        for (ids, ratings) in all_edges:
            all_ids.append(ids)
            all_ratings.append(ratings)
        edges = np.concatenate(all_ids, axis=0), np.concatenate(all_ratings, axis=0)

        # get ratings
        ratings = MovieLensDataHolder.slice_edges(edges, user_indices, movie_indices)  # shape = (n, m)
        mask = (ratings != 0)

        # transform ratings
        int_ratings = ratings
        ratings = ratings.float()
        if ratings_transform is not None:
            ratings[mask] = ratings_transform(int_ratings[mask]).float()

        # return
        return ratings.unsqueeze(dim=0), mask.unsqueeze(dim=0)

    @staticmethod
    def slice_features(features, indices):
        if isinstance(features, dict):
            return {k: MovieLensDataHolder.slice_features(v, indices) for k, v in features.items()}
        else:
            return features[indices]

    def slice_subgraph(
            self,
            user_indices: np.ndarray,
            product_indices: np.ndarray,
            *,
            return_train_edges: bool = True,
            return_test_edges: bool = True,
    ) -> RatingSubgraphData:
        edges_to_include = []
        if return_train_edges:
            edges_to_include.append(self.train_edges)
        if return_test_edges:
            edges_to_include.append(self.test_edges)

        ratings, known_mask = self.get_ratings_and_mask(
            *edges_to_include,
            user_indices=user_indices,
            movie_indices=product_indices,
            ratings_transform=self.ratings_transform,
        )

        user_features = self.slice_features(self.user_data, user_indices)
        movie_features = self.slice_features(self.movie_data, product_indices)

        return RatingSubgraphData(
            # edge data
            ratings=ratings,
            known_mask=known_mask,
            # features
            user_features=user_features,
            product_features=movie_features,
        )
