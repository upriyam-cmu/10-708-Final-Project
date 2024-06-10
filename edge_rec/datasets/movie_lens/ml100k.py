from .preprocessing import MovieLensPreprocessingMixin

import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.datasets import MovieLens100K


class RawMovieLens100K(MovieLens100K, MovieLensPreprocessingMixin):
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

    @staticmethod
    def _bucket_ages(df):
        bins = [0, 18, 25, 35, 45, 50, 56, 200]
        labels = [0, 1, 2, 3, 4, 5, 6]
        df["age"] = pd.cut(df["age"], bins=bins, labels=labels)
        return df

    def _load_ratings(self):
        dfs = [
            pd.read_csv(
                self.raw_paths[i],
                sep='\t',
                header=None,
                names=self.RATING_HEADERS,
            ) for i in (2, 3)
        ]
        return pd.concat(dfs, ignore_index=True)

    def process(self) -> None:
        data = HeteroData()
        ratings_df = self._load_ratings()

        # Process movie data:
        full_df = pd.read_csv(
            self.raw_paths[0],
            sep='|',
            header=None,
            names=self.MOVIE_HEADERS,
            index_col='movieId',
            encoding='ISO-8859-1',
        )
        df = self._remove_low_occurrence(ratings_df, full_df, "movieId")
        movie_mapping = {idx: i for i, idx in enumerate(df.index)}

        x = self._process_genres(df[self.MOVIE_HEADERS[6:]].values, one_hot=False)
        data['movie'].x = torch.from_numpy(x).to(torch.float)

        self.df = x

        # Process user data:
        full_df = pd.read_csv(
            self.raw_paths[1],
            sep='|',
            header=None,
            names=self.USER_HEADERS,
            index_col='userId',
            encoding='ISO-8859-1',
        )
        df = self._remove_low_occurrence(ratings_df, full_df, "userId")
        user_mapping = {idx: i for i, idx in enumerate(df.index)}

        age = self._bucket_ages(df)["age"].to_numpy()[:, None]
        age = torch.from_numpy(age).to(torch.float)

        gender = df['gender'].str.get_dummies().values[:, 0][:, None]
        gender = torch.from_numpy(gender).to(torch.float)

        occupation = df['occupation'].str.get_dummies().values.argmax(axis=1)[:, None]
        occupation = torch.from_numpy(occupation).to(torch.float)

        data['user'].x = torch.cat([age, gender, occupation], dim=-1)

        # Process rating data for training:
        df = self._remove_low_occurrence(ratings_df, ratings_df, ["userId", "movieId"])

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

        ratings_df["movieId"] = ratings_df["movieId"].apply(lambda x: movie_mapping[x])
        user_history = self._generate_user_history(ratings_df, 30)

        data['user', 'rates', 'movie'].history = torch.from_numpy(user_history)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
