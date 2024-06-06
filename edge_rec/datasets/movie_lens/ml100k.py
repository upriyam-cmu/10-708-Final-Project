import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.datasets import MovieLens100K


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

    def _bucket_ages(self, df):
        bins = [0, 18, 25, 35, 45, 50, 56, 200]
        labels = [0, 1, 2, 3, 4, 5, 6]
        df["age"] = pd.cut(df["age"], bins=bins, labels=labels)
        return df

    def _process_genres(self, df, one_hot=True):
        l_df = df[self.MOVIE_HEADERS[6:]].values

        if one_hot:
            return l_df

        max_genres = l_df.sum(axis=1).max()
        idx_list = []
        for i in range(l_df.shape[0]):
            idxs = np.where(l_df[i, :] == 1)[0] + 1
            missing = max_genres - len(idxs)
            if missing > 0:
                idxs = np.array(list(idxs) + missing * [0])
            idx_list.append(idxs)
        out = np.stack(idx_list)
        return out

    def process(self) -> None:
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
