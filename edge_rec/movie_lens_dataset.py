import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import MovieLens1M, MovieLens100K

from torch_geometric.data import HeteroData

from einops import repeat


class RawMovieLens100K(MovieLens100K):
    MOVIE_HEADERS = [
        "movieId", "title", "releaseDate", "videoReleaseDate", "IMDb URL",
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    USER_HEADERS = ["userId", "age", "gender", "occupation", "zipCode"]
    RATING_HEADERS = ["userId", "movieId", "rating", "timestamp"]

    def __init__(self, root, transform=None, pre_transform=None, force_reload=False):
        super(RawMovieLens100K, self).__init__(root, transform, pre_transform, force_reload)

    def __init__(self, root, transform=None, pre_transform=None, force_reload=False):
        super(RawMovieLens100K, self).__init__(root, transform, pre_transform, force_reload)

    def _bucket_ages(self, df):
        bins = [0, 18, 25, 35, 45, 50, 56, 200]
        labels = [0, 1, 2, 3, 4, 5, 6]
        df["age"] = pd.cut(df["age"], bins=bins, labels=labels)
        return df

    def _process_genres(self, df):
        l = df[self.MOVIE_HEADERS[6:]].values
        max_genres = l.sum(axis=1).max()
        idx_list = []
        for i in range(l.shape[0]):
            idxs = np.where(l[i, :] == 1)[0] + 1
            missing = max_genres - len(idxs)
            if missing > 0:
                idxs = np.array(list(idxs) + missing * [0])
            idx_list.append(idxs)
        return np.stack(idx_list)

    def process(self) -> None:
        import pandas as pd

        data = HeteroData()

        # Process movie data:
        df = pd.read_csv(
            self.raw_paths[0],
            sep='|',
            header=None,
            names=self.MOVIE_HEADERS,
            index_col='movieId',
            encoding='ISO-8859-1',
        )
        movie_mapping = {idx: i for i, idx in enumerate(df.index)}

        x = self._process_genres(df)
        data['movie'].x = torch.from_numpy(x).to(torch.float)

        self.df = x

        # Process user data:
        df = pd.read_csv(
            self.raw_paths[1],
            sep='|',
            header=None,
            names=self.USER_HEADERS,
            index_col='userId',
            encoding='ISO-8859-1',
        )
        user_mapping = {idx: i for i, idx in enumerate(df.index)}

        age = self._bucket_ages(df)["age"].to_numpy()[:, None]
        age = torch.from_numpy(age).to(torch.float)

        gender = df['gender'].str.get_dummies().values[:, 0][:, None]
        gender = torch.from_numpy(gender).to(torch.float)

        occupation = df['occupation'].str.get_dummies().values.argmax(axis=1)[:, None]
        occupation = torch.from_numpy(occupation).to(torch.float)

        data['user'].x = torch.cat([age, gender, occupation], dim=-1)

        # Process rating data for training:
        df = pd.read_csv(
            self.raw_paths[2],
            sep='\t',
            header=None,
            names=self.RATING_HEADERS,
        )

        src = [user_mapping[idx] for idx in df['userId']]
        dst = [movie_mapping[idx] for idx in df['movieId']]
        edge_index = torch.tensor([src, dst])
        data['user', 'rates', 'movie'].edge_index = edge_index

        rating = torch.from_numpy(df['rating'].values).to(torch.long)
        data['user', 'rates', 'movie'].rating = rating

        time = torch.from_numpy(df['timestamp'].values)
        data['user', 'rates', 'movie'].time = time

        data['movie', 'rated_by', 'user'].edge_index = edge_index.flip([0])
        data['movie', 'rated_by', 'user'].rating = rating
        data['movie', 'rated_by', 'user'].time = time

        # Process rating data for testing:
        df = pd.read_csv(
            self.raw_paths[3],
            sep='\t',
            header=None,
            names=self.RATING_HEADERS,
        )

        src = [user_mapping[idx] for idx in df['userId']]
        dst = [movie_mapping[idx] for idx in df['movieId']]
        edge_label_index = torch.tensor([src, dst])
        data['user', 'rates', 'movie'].edge_label_index = edge_label_index

        edge_label = torch.from_numpy(df['rating'].values).to(torch.float)
        data['user', 'rates', 'movie'].edge_label = edge_label

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])


class RawMovieLens1M(MovieLens1M):
    MOVIE_HEADERS = ["movieId", "title", "genres"]
    USER_HEADERS = ["userId", "gender", "age", "occupation", "zipCode"]
    RATING_HEADERS = ['userId', 'movieId', 'rating', 'timestamp']

    def __init__(self, root, transform=None, pre_transform=None, force_reload=False):
        super(RawMovieLens1M, self).__init__(root, transform, pre_transform, force_reload)

    def _bucket_ages(self, df):
        bins = [0, 18, 25, 35, 45, 50, 56, 200]
        labels = [0, 1, 2, 3, 4, 5, 6]
        df["age"] = pd.cut(df["age"], bins=bins, labels=labels)
        return df

    def _process_genres(self, df):
        l = df[self.MOVIE_HEADERS[6:]].values
        max_genres = l.sum(axis=1).max()
        idx_list = []
        for i in range(l.shape[0]):
            idxs = np.where(l[i, :] == 1)[0] + 1
            missing = max_genres - len(idxs)
            if missing > 0:
                idxs = np.array(list(idxs) + missing * [0])
            idx_list.append(idxs)
        return np.stack(idx_list)

    def process(self) -> None:
        import pandas as pd

        data = HeteroData()

        # Process movie data:
        df = pd.read_csv(
            self.raw_paths[0],
            sep='|',
            header=None,
            names=self.MOVIE_HEADERS,
            index_col='movieId',
            encoding='ISO-8859-1',
        )
        movie_mapping = {idx: i for i, idx in enumerate(df.index)}

        x = self._process_genres(df)
        data['movie'].x = torch.from_numpy(x).to(torch.float)

        self.df = x

        # Process user data:
        df = pd.read_csv(
            self.raw_paths[1],
            sep='|',
            header=None,
            names=self.USER_HEADERS,
            index_col='userId',
            encoding='ISO-8859-1',
        )
        user_mapping = {idx: i for i, idx in enumerate(df.index)}

        age = self._bucket_ages(df)["age"].to_numpy()[:, None]
        age = torch.from_numpy(age).to(torch.float)

        gender = df['gender'].str.get_dummies().values[:, 0][:, None]
        gender = torch.from_numpy(gender).to(torch.float)

        occupation = df['occupation'].str.get_dummies().values.argmax(axis=1)[:, None]
        occupation = torch.from_numpy(occupation).to(torch.float)

        data['user'].x = torch.cat([age, gender, occupation], dim=-1)

        # Process rating data for training:
        df = pd.read_csv(
            self.raw_paths[2],
            sep='\t',
            header=None,
            names=self.RATING_HEADERS,
        )

        src = [user_mapping[idx] for idx in df['userId']]
        dst = [movie_mapping[idx] for idx in df['movieId']]
        edge_index = torch.tensor([src, dst])
        data['user', 'rates', 'movie'].edge_index = edge_index

        rating = torch.from_numpy(df['rating'].values).to(torch.long)
        data['user', 'rates', 'movie'].rating = rating

        time = torch.from_numpy(df['timestamp'].values)
        data['user', 'rates', 'movie'].time = time

        data['movie', 'rated_by', 'user'].edge_index = edge_index.flip([0])
        data['movie', 'rated_by', 'user'].rating = rating
        data['movie', 'rated_by', 'user'].time = time

        # Process rating data for testing:
        df = pd.read_csv(
            self.raw_paths[3],
            sep='\t',
            header=None,
            names=self.RATING_HEADERS,
        )

        src = [user_mapping[idx] for idx in df['userId']]
        dst = [movie_mapping[idx] for idx in df['movieId']]
        edge_label_index = torch.tensor([src, dst])
        data['user', 'rates', 'movie'].edge_label_index = edge_label_index

        edge_label = torch.from_numpy(df['rating'].values).to(torch.float)
        data['user', 'rates', 'movie'].edge_label = edge_label

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])


class CoreMovieLensDataset:
    PROCESSED_DSET_PATH = "processed/data.pt"

    def __init__(self, root, ml_100k=True, test_split=0.1, force_download=False):
        root = Path(root) / ("ml100k" if ml_100k else "ml1m")
        processed_data_path = root / self.PROCESSED_DSET_PATH

        if force_download or not processed_data_path.exists():
            dataset_class = RawMovieLens100K if ml_100k else RawMovieLens1M
            dataset_class(str(root), force_reload=True).process()

        data = torch.load(processed_data_path)[0]
        self.user_data = data['user']['x'].int()
        self.movie_data = self._preprocess_movie_genres(data['movie']['x'].int())
        self.n_users = len(self.user_data)
        self.n_movies = len(self.movie_data)

        self.train_edges, self.test_edges, self.all_edges = self._split_edges(
            data[('user', 'rates', 'movie')]['edge_index'],
            data[('user', 'rates', 'movie')]['rating'],
            test_split=test_split
        )

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
    def _preprocess_movie_genres(movie_data, pad_len=6):
        movie_genres = []
        for one_hot in movie_data:
            nonzeros = torch.nonzero(one_hot, as_tuple=True)
            assert len(nonzeros) == 1
            nonzeros = nonzeros[0] + 1
            genre_list = torch.zeros(pad_len)
            genre_list[:len(nonzeros)] = nonzeros
            movie_genres.append(genre_list)
        return torch.stack(movie_genres)

    @staticmethod
    def _slice_edges(edges, user_inds, movie_inds) -> torch.Tensor:
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

    def get_subgraph(self, subgraph_size, assert_density, include_train_edges=True, include_test_edges=True):
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

        assert include_train_edges or include_test_edges
        if include_train_edges and include_test_edges:
            edges = self.all_edges
        else:
            edges = self.train_edges if include_train_edges else self.test_edges

        if not assert_density:
            user_inds = np.random.choice(self.n_users, n_users_sampled, replace=False)
            movie_inds = np.random.choice(self.n_movies, n_movies_sampled, replace=False)
        else:
            raise NotImplementedError("currently cannot assert density of subgraph")

            # indices = edges[0]
            #
            # user_inds, movie_inds = set(), set()
            # n_users_remaining, n_movies_remaining = n_users_sampled, n_movies_sampled
            #
            # while min(n_users_remaining, n_movies_remaining) > 0:
            #     assert indices[:, 0].unique() >= n_users_remaining and indices[:, 1].unique() >= n_movies_remaining, \
            #         "Subgraph is too large to guarantee that minimum density is preserved"
            #
            #     mask_inds = np.random.choice(len(indices), min(n_users_remaining, n_movies_remaining), replace=False)
            #     chosen_mask = np.ones(len(indices), dtype=bool)[mask_inds]
            #     new_users, new_movies = set(indices[chosen_mask, 0]), set(indices[chosen_mask, 1])
            #     user_inds.update(new_users)
            #     movie_inds.update(new_movies)
            #     indices = indices[~chosen_mask]
            #
            # assert indices[:, 0].unique() >= n_users_remaining and indices[:, 1].unique() >= n_movies_remaining, \
            #     "Subgraph is too large to guarantee that minimum density is preserved"
            #
            # assert len(user_inds) == n_users_sampled and len(movie_inds) == n_movies_sampled

        users, movies = self.user_data[user_inds], self.movie_data[movie_inds]
        ratings = self._slice_edges(edges, user_inds, movie_inds)

        ratings = (2 * (ratings.float() - 3) / 5).unsqueeze(dim=0)  # shape = (1, n, m)
        users = repeat(users, 'n f -> f n m', m=n_movies_sampled).float()
        movies = repeat(movies, 'm f -> f n m', n=n_users_sampled).float()

        return torch.cat([ratings, movies, users], dim=0)


class ProcessedMovieLens(Dataset):
    PROCESSED_ML_SUBPATH = "/processed/data.pt"

    def __init__(self, root, ml_100k=True, n_subsamples=10000, n_unique_per_sample=10,
                 transform=None, train=True, test_split=0.1, download=True):
        dataset_class = RawMovieLens100K if ml_100k else RawMovieLens1M

        self.train = train

        if download:
            self.raw = dataset_class(root, force_reload=True)
            self.raw.process()

        self.n_unique_per_sample = n_unique_per_sample
        self.n_subsamples = n_subsamples
        self.transform = transform
        print(root + self.PROCESSED_ML_SUBPATH)
        self.processed_data = torch.load(root + self.PROCESSED_ML_SUBPATH)
        self.user_feats = self.processed_data[0]['user']['x']
        self.movie_feats = self.processed_data[0]['movie']['x']
        self.num_users = self.user_feats.shape[0]
        self.num_movies = self.movie_feats.shape[0]

        self.processed_ratings = self._preprocess_ratings(self.processed_data)
        self.train_idxs, self.test_idxs = self._split_ratings(test_split)

    def build_feat_graph(self, transform_ratings=False, include_mask=False):
        movie_feats, user_feats = self.movie_feats, self.user_feats
        ratings = self.processed_ratings

        ratings_graph = torch.zeros(self.num_users, self.num_movies)
        ratings_graph[ratings[0], ratings[1]] = ratings[2].float()
        ratings_graph = ratings_graph.unsqueeze(-1)
        is_known_mask = ratings_graph != 0
        if transform_ratings:
            ratings_graph[is_known_mask] += 3
            ratings_graph = self._rating_transform(ratings_graph)

        user_ids, movie_ids = torch.meshgrid(
            torch.arange(self.num_users),
            torch.arange(self.num_movies),
            indexing='ij'
        )

        user_feat_graph = user_feats[user_ids]
        movie_feat_graph = movie_feats[movie_ids]

        components = [ratings_graph, movie_feat_graph, user_feat_graph]
        if include_mask:
            components += [is_known_mask]
        full_graph = torch.cat(components, dim=-1)
        return full_graph.permute(2, 0, 1)

    def _split_ratings(self, test_split):
        train_rating_idxs = []
        test_rating_idxs = []

        for i in range(self.num_users):
            rating_idxs = torch.where(self.processed_ratings[0, :] == i)[0]

            split_idx = int(len(rating_idxs) * (1 - test_split))
            train_rating_idxs.append(rating_idxs[:split_idx])
            test_rating_idxs.append(rating_idxs[split_idx:])

        train_rating_idxs = torch.cat(train_rating_idxs, dim=0)
        test_rating_idxs = torch.cat(test_rating_idxs, dim=0)

        return train_rating_idxs, test_rating_idxs

    def _preprocess_ratings(self, data):
        edges = data[0][('user', 'rates', 'movie')]
        edge_ratings = torch.concatenate([edges["edge_index"], edges["rating"].reshape((1, -1))])
        return edge_ratings

    def _rating_transform(self, data):
        data = data.float()
        ratings = data[0, :, :]
        data[0, :, :] = 2 * (ratings - 3) / 5
        return data

    def from_edges(self, indices=None):
        edge_ratings = self.processed_ratings
        movie_feats, user_feats = self.movie_feats, self.user_feats

        if indices is None:  # if indices is not passed, take the whole graph
            _, m = edge_ratings.shape
            indices = np.arange(m)

        xs, ys = self.get_unique_users(indices), self.get_unique_movies(indices)

        subsample_edges = edge_ratings[:, indices].T

        subsample_movie_feats = movie_feats[subsample_edges[:, 1], :]
        subsample_user_feats = user_feats[subsample_edges[:, 0], :]
        subsample_user_movie_feats = torch.cat([subsample_movie_feats, subsample_user_feats], dim=1)

        broadcasted_movie_feats, broadcasted_user_feats = (
            torch.broadcast_to(
                movie_feats[ys.sort().values, :].T.reshape((-1, 1, len(ys))).swapaxes(1, 2),
                (-1, len(ys), len(xs))
            ).swapaxes(1, 2),
            torch.broadcast_to(
                user_feats[xs.sort().values, :].T.reshape((-1, 1, len(xs))).swapaxes(1, 2),
                (-1, len(xs), len(ys))
            )
        )

        rating_matrix = torch.Tensor(
            pd.DataFrame(subsample_edges)
            .pivot(columns=[1], index=[0])
            .fillna(-10)
            .to_numpy()
        )

        edge_mask = (rating_matrix != -10)
        rating_matrix[~edge_mask] = 3

        item = torch.cat([
            rating_matrix.reshape((1, len(xs), len(ys))),
            broadcasted_movie_feats,
            broadcasted_user_feats
        ], dim=0)

        out = torch.cat([
            item,
            edge_mask[None, :, :]
        ], dim=0)

        out = self._rating_transform(out)

        return out

    def get_unique_users(self, indices=None):
        edge_ratings = self.processed_ratings
        _, m = edge_ratings.shape
        if indices is None:
            indices = np.arange(m)
        return edge_ratings[0, indices].unique()

    def get_unique_movies(self, indices=None):
        edge_ratings = self.processed_ratings
        _, m = edge_ratings.shape
        if indices is None:
            indices = np.arange(m)
        return edge_ratings[1, indices].unique()

    def __getitem__(self, idx):
        n_unique = self.n_unique_per_sample
        edge_ratings = self.processed_ratings
        movie_feats, user_feats = self.processed_data[0]["movie"]["x"], self.processed_data[0]["user"]["x"]

        if self.train:
            indices = np.random.choice(self.train_idxs, size=(n_unique,), replace=False)
        else:
            indices = np.random.choice(self.test_idxs, size=(n_unique,), replace=False)

        xs, ys = self.get_unique_users(indices), self.get_unique_movies(indices)

        unique_xs, unique_ys = len(xs), len(ys)
        users_missing, movies_missing = n_unique - unique_xs, n_unique - unique_ys

        while users_missing > 0:
            candidate_users = np.where(
                ~np.isin(edge_ratings[0, :], xs.unique()) & np.isin(edge_ratings[1, :], ys.unique())
            )[0]
            n_candidates = len(candidate_users)
            new_indices = np.random.choice(candidate_users, size=(min(users_missing, n_candidates),), replace=False)
            indices = np.concatenate([indices, new_indices])
            xs = edge_ratings[0, indices].unique()
            users_missing = n_unique - len(xs)

        while movies_missing > 0:
            candidate_movies = np.where(
                np.isin(edge_ratings[0, :], xs.unique()) & ~np.isin(edge_ratings[1, :], ys.unique())
            )[0]
            n_candidates = len(candidate_movies)
            new_indices = np.random.choice(candidate_movies, size=(min(movies_missing, n_candidates),), replace=False)
            indices = np.concatenate([indices, new_indices])
            ys = edge_ratings[1, indices].unique()
            movies_missing = n_unique - len(ys)

        xs, ys = self.get_unique_users(indices), self.get_unique_movies(indices)

        indices_xs = torch.where(torch.isin(edge_ratings[0, :], xs))[0]
        indices_ys = torch.where(torch.isin(edge_ratings[1, :], ys))[0]
        edge_subsample_indices = np.intersect1d(indices_xs, indices_ys)

        out = self.from_edges(edge_subsample_indices)

        return out

    def __len__(self):
        return self.n_subsamples


class FullGraphSampler(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, idx):
        return self.ds.build_feat_graph(transform_ratings=True, include_mask=True)

    def __len__(self):
        return 1
