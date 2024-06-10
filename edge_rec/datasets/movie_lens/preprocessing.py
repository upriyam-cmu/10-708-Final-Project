import numpy as np
import pandas as pd
from tqdm import tqdm


class MovieLensPreprocessingMixin:
    def _process_genres(self, genres, one_hot=True):
        if one_hot:
            return genres

        max_genres = genres.sum(axis=1).max()
        idx_list = []
        for i in range(genres.shape[0]):
            idxs = np.where(genres[i, :] == 1)[0] + 1
            missing = max_genres - len(idxs)
            if missing > 0:
                idxs = np.array(list(idxs) + missing * [0])
            idx_list.append(idxs)
        out = np.stack(idx_list)
        return out

    def _remove_low_occurrence(self, source_df, target_df, index_col):
        if isinstance(index_col, str):
            index_col = [index_col]
        out = target_df.copy()
        for col in index_col:
            count = source_df.groupby(col).agg(ratingCnt=("rating", "count"))
            high_occ = count[count["ratingCnt"] >= 5]
            out = out.merge(high_occ, on=col).drop(columns=["ratingCnt"])
        return out

    def _generate_user_history(self, ratings_df, history_len=30):
        print("Starting User History Generation...")
        orig_order = ratings_df.reset_index()["index"]
        # Sort the DataFrame by userId and timestamp
        ratings_df = ratings_df.sort_values(by=['userId', 'timestamp'])
        unique_users = ratings_df["userId"]
    
        shards = np.split(unique_users.unique(), 10)
        shards_filtered_df = []
        print("Batching User Ratings Dataframe...")
        for shard in tqdm(shards):
            shard_ratings_df = ratings_df[ratings_df["userId"].isin(shard)]
            # Merge with itself on userId, filter the rows where timestamp_x is less than timestamp_y and rating_y is greater than 3
            merged_df = shard_ratings_df.merge(shard_ratings_df, on='userId', suffixes=('_x', '_y'))
            shard_filtered_df = merged_df[(merged_df['timestamp_x'] > merged_df['timestamp_y']) & (merged_df['rating_y'] > 3)]
            shards_filtered_df.append(shard_filtered_df)
        filtered_df = pd.concat(shards_filtered_df)

        # Group by userId and timestamp_x, aggregate the movieId values as a comma-separated string
        history_df = filtered_df.groupby(['userId', 'timestamp_x'])['movieId_y'].apply(lambda x: np.unique(list(x)[:history_len])).reset_index()

        # Rename columns and merge with the original DataFrame
        result_df = ratings_df.merge(history_df, left_on=['userId', 'timestamp'], right_on=['userId', 'timestamp_x'], how='left')
        result_df = result_df.rename(columns={'movieId_y': 'history'}).drop(columns=['timestamp_x'])

        # Make sure we are preserving the original ordering
        result_df = result_df.loc[orig_order]

        history = result_df.fillna(-1)["history"].apply(lambda x: [x] if isinstance(x, int) else x).to_numpy()
        history_padded = [np.pad(x, pad_width=(0, history_len - len(x)), constant_values=-1) for x in history]
        out = np.stack(history_padded)
        print("Finished User History Generation...")
        import pdb; pdb.set_trace()
        return out
