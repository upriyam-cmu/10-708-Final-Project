from ..datasets import RatingSubgraphData, Transform

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import ndcg_score
import torch


def _reformat(tensor: torch.Tensor, name: str, dtype=None) -> np.ndarray:
    batch_size = np.prod(tensor.shape[:-2], dtype=int)
    if batch_size > 1:
        raise ValueError(f"{name} has batch size > 1. Cannot compute metrics.")

    tensor = tensor.reshape(tensor.shape[-2:])
    assert len(tensor.shape) == 2

    ndarray = tensor.detach().cpu().numpy()
    if dtype is not None:
        ndarray = ndarray.astype(dtype=dtype)

    return ndarray


def compute_metrics(
        predicted_ratings_graph: torch.Tensor,
        train_rating_data: RatingSubgraphData,
        test_rating_data: RatingSubgraphData,
        rating_transform: Optional[Transform] = None,
        top_ks: Tuple[int, ...] = (1, 5, 10, 20, 30, 40, 50),
) -> Dict[str, np.ndarray]:
    predicted_ratings_graph = _reformat(predicted_ratings_graph, 'predicted_ratings_graph', dtype=float)
    train_known_mask = _reformat(train_rating_data.known_mask, 'train_known_mask', dtype=bool)
    test_ratings_graph = _reformat(test_rating_data.ratings, 'test_ratings_graph', dtype=float)
    test_known_mask = _reformat(test_rating_data.known_mask, 'test_known_mask', dtype=bool)

    if rating_transform is not None:
        predicted_ratings_graph = rating_transform.invert(
            predicted_ratings_graph,
            numpy=True,
        )
        test_ratings_graph[test_known_mask] = rating_transform.invert(
            test_ratings_graph[test_known_mask],
            numpy=True,
        )

    return _compute_metrics(
        predicted_ratings_graph=predicted_ratings_graph,
        train_known_mask=train_known_mask,
        test_ratings_graph=test_ratings_graph,
        test_known_mask=test_known_mask,
        top_ks=top_ks,
    )


def _compute_metrics(
        predicted_ratings_graph: np.ndarray,
        train_known_mask: np.ndarray,
        test_ratings_graph: np.ndarray,
        test_known_mask: np.ndarray,
        top_ks: Tuple[int, ...] = (1, 5, 10, 20, 30, 40, 50),
) -> Dict[str, np.ndarray]:
    assert len(predicted_ratings_graph.shape) == 2
    assert len(train_known_mask.shape) == 2
    assert len(test_ratings_graph.shape) == 2
    assert len(test_known_mask.shape) == 2

    precision = np.zeros(len(top_ks))
    recall = np.zeros(len(top_ks))
    mean_reciprocal_rank = np.zeros(len(top_ks))
    hit_rate = np.zeros(len(top_ks))
    ndcg = np.zeros(len(top_ks))

    min_rating = test_ratings_graph[test_known_mask].min()
    if min_rating > 0:
        min_rating = 0
    elif min_rating > -1:
        min_rating = -1
    else:
        min_rating -= 1
    test_ratings_graph[~test_known_mask] = min_rating
    for i, k in enumerate(top_ks):
        ndcg[i] = ndcg_score(test_ratings_graph, predicted_ratings_graph, k=k)

    predicted_ratings_graph[train_known_mask] = float('-inf')
    ranked = np.argsort(predicted_ratings_graph, axis=1)[:, ::-1]

    n_users = predicted_ratings_graph.shape[0]
    for user in range(n_users):
        pred_rankings = ranked[user]
        test_movies = np.flatnonzero(test_known_mask[user])
        assert np.allclose(test_movies, sorted(test_movies))
        test_ratings = test_ratings_graph[user, test_movies]
        assert np.allclose(test_ratings, test_ratings_graph[user, test_known_mask[user]])
        true_rankings = test_movies[test_ratings.argsort()[::-1]]

        for i, k in enumerate(top_ks):
            isin = np.isin(pred_rankings[:k], true_rankings)
            hits = np.sum(isin)
            precision[i] += hits / k
            recall[i] += hits / len(true_rankings) if len(true_rankings) > 0 else 0
            first_hit = np.flatnonzero(isin)
            first_hit = first_hit[0] if len(first_hit) > 0 else float('inf')
            mean_reciprocal_rank[i] += 1 / (first_hit + 1)
            hit_rate[i] += 1 if hits > 0 else 0

    return {
        'precision': precision / n_users,
        'recall': recall / n_users,
        'mean_reciprocal_rank': mean_reciprocal_rank / n_users,
        'hit_rate': hit_rate / n_users,
        'ndcg': ndcg,
    }
