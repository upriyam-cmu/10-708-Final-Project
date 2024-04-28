import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.datasets import MovieLens1M, MovieLens100K
from sklearn.preprocessing import QuantileTransformer

from torch_geometric.data import HeteroData


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

        x = df[MOVIE_HEADERS[6:]].values
        data['movie'].x = torch.from_numpy(x).to(torch.float)

        self.df = df

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

        age = df['age'].apply(str).str.get_dummies().values.argmax(axis=1)[:, None]
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
            names=self.MOVIE_HEADERS,
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
            names=self.USER_HEADERS,
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
            names=self.RATING_HEADERS,
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

    def __init__(self, root, ml_100k=True, n_subsamples=10000, n_unique_per_sample=10, dataset_transform=None,
                 transform=None, test_split=0.1,
                 download=True):
        dataset_class = RawMovieLens100K if ml_100k else RawMovieLens1M

        if download:
            self.raw = dataset_class(root, force_reload=True)
            self.raw.process()

        self.n_unique_per_sample = n_unique_per_sample
        self.n_subsamples = n_subsamples
        self.transform = transform
        self.dataset_transform = dataset_transform
        print(root + self.PROCESSED_ML_SUBPATH)
        self.processed_data = torch.load(root + self.PROCESSED_ML_SUBPATH)
        self.ratings = self._preprocess_ratings(self.processed_data)
        self.processed_ratings, self.test_ratings = self._split_ratings(self.ratings, test_split)
        
    def _split_ratings(self, ratings, test_split, rand=False):
        train_rating_idxs = []
        test_rating_idxs = []
        
        users = self.raw.data['user']['x'].shape[0]
        for i in range(users):
            rating_idxs = torch.where(ratings[0, :] == i)[0]
            if rand:
                ratings_idxs = rating_idxs[torch.randperm(len(rating_idxs))]
                
            split_idx = int(len(rating_idxs) * (1 - test_split))
            train_rating_idxs.append(rating_idxs[:split_idx])
            test_rating_idxs.append(rating_idxs[split_idx:])
        
        train_rating_idxs = torch.cat(train_rating_idxs, dim=0)
        test_rating_idxs = torch.cat(test_rating_idxs, dim=0)
        
        return ratings[:, train_rating_idxs], ratings[:, test_rating_idxs]

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
        movie_feats, user_feats = self.processed_data[0]["movie"]["x"], self.processed_data[0]["user"]["x"]

        _, m = edge_ratings.shape

        if indices is None:  # if indices is not passed, take the whole graph
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

        _, edge_size = edge_ratings.shape

        indices = np.random.choice(np.arange(edge_size), size=(n_unique,), replace=False)

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
