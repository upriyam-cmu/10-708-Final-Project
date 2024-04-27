import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.datasets import MovieLens1M
from sklearn.preprocessing import QuantileTransformer
import os
import os.path as osp
from typing import Callable, List, Optional

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import fs

MOVIE_HEADERS = ["movieId", "title", "genres"]
USER_HEADERS = ["userId", "gender", "age", "occupation", "zipCode"]
RATING_HEADERS = ['userId', 'movieId', 'rating', 'timestamp']


def rating_transform(data):
    ratings = data[2, :]
    ratings[ratings == 1] = -5
    ratings[ratings == 2] = -2
    ratings[ratings == 3] = 0
    ratings[ratings == 4] = 2
    data[2, :] = ratings
    return data


class RawMovieLens1M(MovieLens1M):
    def __init__(self, root, transform=None, pre_transform=None, force_reload=False):
        super(RawMovieLens1M, self).__init__(root, transform, pre_transform, force_reload)

    def _process_genres(self, df):
        l = df["genres"].str.get_dummies('|').values
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
            sep='::',
            header=None,
            index_col='movieId',
            names=MOVIE_HEADERS,
            encoding='ISO-8859-1',
            engine='python',
        )
        movie_mapping = {idx: i for i, idx in enumerate(df.index)}

        genres = self._process_genres(df)
        genres = torch.from_numpy(genres).to(torch.float)

        data['movie'].x = genres

        # Process user data:
        df = pd.read_csv(
            self.raw_paths[1],
            sep='::',
            header=None,
            index_col='userId',
            names=USER_HEADERS,
            dtype='str',
            encoding='ISO-8859-1',
            engine='python',
        )
        user_mapping = {idx: i for i, idx in enumerate(df.index)}

        age = df['age'].str.get_dummies().values.argmax(axis=1)[:, None]
        age = torch.from_numpy(age).to(torch.float)

        gender = df['gender'].str.get_dummies().values[:, 0][:, None]
        gender = torch.from_numpy(gender).to(torch.float)

        occupation = df['occupation'].str.get_dummies().values.argmax(axis=1)[:, None]
        occupation = torch.from_numpy(occupation).to(torch.float)

        data['user'].x = torch.cat([age, gender, occupation], dim=-1)

        self.int_user_data = df

        # Process rating data:
        df = pd.read_csv(
            self.raw_paths[2],
            sep='::',
            header=None,
            names=RATING_HEADERS,
            encoding='ISO-8859-1',
            engine='python',
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

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])


class RatingQuantileTransform(object):
    def __init__(self):
        self.qt_transformer = QuantileTransformer(n_quantiles=5, output_distribution="normal")

    def __call__(self, data):
        ratings = data[2, :]
        data[2, :] = torch.Tensor(self.qt_transformer.fit_transform(ratings.reshape(-1, 1))).T
        return data


class ProcessedMovieLens(Dataset):
    PROCESSED_ML_SUBPATH = "/processed/data.pt"

    def __init__(self, root, n_subsamples=10000, n_unique_per_sample=10, dataset_transform=None, transform=None,
                 download=True):
        if download:
            self.ml_1m = RawMovieLens1M(root, force_reload=True)
            self.ml_1m.process()

        self.n_unique_per_sample = n_unique_per_sample
        self.n_subsamples = n_subsamples
        self.transform = transform
        self.dataset_transform = dataset_transform
        print(root + self.PROCESSED_ML_SUBPATH)
        self.processed_data = torch.load(root + self.PROCESSED_ML_SUBPATH)
        self.processed_ratings = self._preprocess_ratings(self.processed_data)

    def _preprocess_ratings(self, data):
        edges = data[0][('user', 'rates', 'movie')]
        edge_ratings = torch.concatenate([edges["edge_index"], edges["rating"].reshape((1, -1))])
        return self.dataset_transform(edge_ratings)

    def __getitem__(self, idx):
        n_unique = self.n_unique_per_sample
        edge_ratings = self.processed_ratings
        movie_feats, user_feats = self.processed_data[0]["movie"]["x"], self.processed_data[0]["user"]["x"]

        _, edge_size = edge_ratings.shape
        indices = torch.randint(0, edge_size, (n_unique,))
        sampled_edges = torch.ones((1, 1))
        while len(sampled_edges[0, :].unique()) < n_unique or len(sampled_edges[1, :].unique()) < n_unique:
            indices = torch.randint(0, edge_size, (n_unique,))
            sampled_edges = edge_ratings[:, indices]

        xs = edge_ratings[0, indices]
        ys = edge_ratings[1, indices]

        indices_xs = torch.where(torch.isin(edge_ratings[0, :], xs))[0]
        indices_ys = torch.where(torch.isin(edge_ratings[1, :], ys))[0]
        subsample_edges = edge_ratings[:, np.intersect1d(indices_xs, indices_ys)].T

        subsample_movie_feats = movie_feats[subsample_edges[:, 1], :]
        subsample_user_feats = user_feats[subsample_edges[:, 0], :]
        subsample_user_movie_feats = torch.cat([subsample_movie_feats, subsample_user_feats], dim=1)

        broadcasted_movie_feats, broadcasted_user_feats = (
            torch.broadcast_to(
                movie_feats[ys.unique().sort().values, :].T.reshape((-1, 1, n_unique)).swapaxes(1, 2),
                (-1, n_unique, n_unique)
            ).swapaxes(1, 2),
            torch.broadcast_to(
                user_feats[xs.unique().sort().values, :].T.reshape((-1, 1, n_unique)).swapaxes(1, 2),
                (-1, n_unique, n_unique)
            )
        )

        rating_matrix = torch.Tensor(
            pd.DataFrame(subsample_edges)
              .pivot(columns=[1], index=[0])
              .fillna(-10)
              .to_numpy()
            )
        
        edge_mask = (rating_matrix != -10)
        rating_matrix[edge_mask] = 0

        item = torch.cat([
            rating_matrix.reshape((1, n_unique, n_unique)),
            broadcasted_movie_feats,
            broadcasted_user_feats
        ], dim=0)

        return (item, edge_mask)

    def __len__(self):
        return self.n_subsamples
