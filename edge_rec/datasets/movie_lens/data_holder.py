from .ml1m import RawMovieLens1M
from .ml100k import RawMovieLens100K

from ..data_holder import DataHolder, RatingSubGraphData
from ..transforms import Transform

from functools import partial
from typing import Dict

import numpy as np
from pathlib import Path
import torch


class MovieLensDataHolder(DataHolder):
    def __init__(self, root="./data", ml_100k=True, test_split=0.1, force_download=False,
                 augmentations: Dict[str, Transform] = None):
        super().__init__(
            data_root=(Path(root) / ("ml100k" if ml_100k else "ml1m")),
            dataset_class=(RawMovieLens100K if ml_100k else RawMovieLens1M),
            test_split_ratio=test_split,
            force_download=force_download,
        )

        # load rating data
        data = torch.load(self.processed_data_path)[0]
        self.train_edges, self.test_edges, self.all_edges = self._split_edges(
            data[('user', 'rates', 'movie')]['edge_index'],
            data[('user', 'rates', 'movie')]['rating'],
            test_split=test_split,
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
    def _split_edges(edge_inds: torch.Tensor, edge_ratings: torch.Tensor, test_split: float):
        n_edges = len(edge_ratings)

        edge_inds, edge_ratings = edge_inds.numpy(), edge_ratings.numpy()
        sort_inds = np.argsort(edge_inds[0])
        edge_inds, edge_ratings = edge_inds[:, sort_inds], edge_ratings[sort_inds]

        train_group, test_group = [], []
        _, split_inds = np.unique(edge_inds[0], return_index=True)
        for edge_group in np.split(np.arange(n_edges), split_inds[1:]):
            np.random.shuffle(edge_group)
            n_test = int(test_split * len(edge_group))
            train_group.append(edge_group[:-n_test])
            test_group.append(edge_group[-n_test:])

        train_group = np.concatenate(train_group)
        test_group = np.concatenate(test_group)

        train_edges = edge_inds.T[train_group], edge_ratings[train_group]
        test_edges = edge_inds.T[test_group], edge_ratings[test_group]
        all_edges = edge_inds.T, edge_ratings

        return train_edges, test_edges, all_edges

    @staticmethod
    def _get_features(user_data_tensor, movie_data_tensor):
        assert len(user_data_tensor.shape) == 2 and user_data_tensor.shape[-1] == 3
        user_data = {
            'age': user_data_tensor[:, 0].int(),
            'gender': user_data_tensor[:, 1].int(),
            'occupation': user_data_tensor[:, 2].int(),
        }

        assert len(movie_data_tensor.shape) == 2 and movie_data_tensor.shape[-1] == 6
        genre_multihot = torch.zeros(len(movie_data_tensor), 19, device=movie_data_tensor.device)
        genre_multihot.scatter_(1, movie_data_tensor.long(), 1.)
        movie_data = {
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
            movie_counts = self._count_unique(edges, 0, self.n_movies).reshape((self.n_movies, 1)).float()

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

    @staticmethod
    def slice_edges(edges, user_inds, movie_inds) -> torch.Tensor:
        n_users_sampled, n_movies_sampled = len(user_inds), len(movie_inds)
        user_id_to_ind = {user_id: idx for idx, user_id in enumerate(user_inds)}
        assert len(user_id_to_ind) == n_users_sampled
        movie_id_to_ind = {movie_id: idx for idx, movie_id in enumerate(movie_inds)}
        assert len(movie_id_to_ind) == n_movies_sampled

        sliced = torch.zeros(n_users_sampled, n_movies_sampled)
        for (user_id, movie_id), rating in zip(*edges):
            if user_id in user_id_to_ind and movie_id in movie_id_to_ind:
                sliced[user_id_to_ind[user_id], movie_id_to_ind[movie_id]] = rating
        return sliced

    @staticmethod
    def get_ratings_and_mask(edges, user_inds, movie_inds, ratings_transform: Transform = None):
        # get ratings
        ratings = MovieLensDataHolder.slice_edges(edges, user_inds, movie_inds)  # shape = (n, m)
        mask = (ratings != 0)
        # transform ratings
        int_ratings = ratings
        ratings = ratings.float()
        if ratings_transform is not None:
            ratings[mask] = ratings_transform(int_ratings[mask]).float()
        # return
        return ratings.unsqueeze(dim=0), mask

    @staticmethod
    def slice_features(features, indices):
        if isinstance(features, dict):
            return {k: MovieLensDataHolder.slice_features(v, indices) for k, v in features.items()}
        else:
            return features[indices]

    def get_subgraph(
            self,
            subgraph_size,
            target_density,
            *,
            return_train_edges: bool = True,
            return_test_edges: bool = True,
    ) -> RatingSubGraphData:
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
            user_inds = np.random.choice(self.n_users, n_users_sampled, replace=False)
            movie_inds = np.random.choice(self.n_movies, n_movies_sampled, replace=False)
        else:
            slice_point = round((target_density ** (-1 / 2.25) - 1) * 500)
            assert slice_point >= n_users_sampled and slice_point >= n_movies_sampled, \
                "Desired density too high for desired subgraph size"

            random_weights = ((np.arange(slice_point)[::-1] + 1) / 500 + 1) ** -2.25
            random_weights = random_weights / random_weights.sum()
            user_inds = np.random.choice(
                self.top_users[:slice_point],
                size=n_users_sampled,
                replace=False,
                p=random_weights,
            )
            movie_inds = np.random.choice(
                self.top_movies[:slice_point],
                size=n_movies_sampled,
                replace=False,
                p=random_weights,
            )

        if return_train_edges:
            train_ratings, train_mask = self.get_ratings_and_mask(
                edges=self.train_edges,
                user_inds=user_inds,
                movie_inds=movie_inds,
                ratings_transform=self.ratings_transform,
            )
        else:
            train_ratings, train_mask = None, None

        if return_test_edges:
            test_ratings, test_mask = self.get_ratings_and_mask(
                edges=self.test_edges,
                user_inds=user_inds,
                movie_inds=movie_inds,
                ratings_transform=self.ratings_transform,
            )
        else:
            test_ratings, test_mask = None, None

        user_features = self.slice_features(self.user_data, user_inds)
        movie_features = self.slice_features(self.movie_data, movie_inds)

        return RatingSubGraphData(
            # ratings
            train_ratings=train_ratings,
            test_ratings=test_ratings,
            # masks
            train_mask=train_mask,
            test_mask=test_mask,
            # features
            user_features=user_features,
            product_features=movie_features,
        )
