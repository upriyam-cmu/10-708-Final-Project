import torch
from sklearn.metrics import ndcg_score


def get_metrics(train_edges, test_edges, pred_graph, top_ks=(1, 5, 10, 20, 30, 40, 50)):
    precision = torch.zeros(len(top_ks))
    recall = torch.zeros_like(precision)
    mean_reciprocal_rank = torch.zeros_like(precision)
    hit_rate = torch.zeros_like(precision)
    ndcg = torch.zeros_like(precision)

    test_graph = torch.zeros_like(pred_graph)
    test_graph[test_edges[0], test_edges[1]] = test_edges[2].float()
    for i, k in enumerate(top_ks):
        ndcg[i] = ndcg_score(test_graph, pred_graph, k=k)

    pred_graph = pred_graph.detach().cpu()
    pred_graph[train_edges[0], train_edges[1]] = float('-inf')
    ranked = torch.argsort(pred_graph, dim=1, descending=True)

    for user in range(pred_graph.shape[0]):
        pred_rankings = ranked[user]
        test_movies = test_edges[1][test_edges[0] == user]
        test_ratings = test_edges[2][test_edges[0] == user]
        true_rankings = test_movies[test_ratings.argsort(descending=True)]

        for i, k in enumerate(top_ks):
            isin = torch.isin(pred_rankings[:k], true_rankings)
            hits = torch.sum(isin)
            precision[i] += hits / k
            recall[i] += hits / len(true_rankings)
            first_hit = torch.where(isin)[0]
            first_hit = first_hit[0] if len(first_hit) > 0 else float('inf')
            mean_reciprocal_rank[i] += 1 / (first_hit + 1)
            hit_rate[i] += 1 if hits > 0 else 0

    metrics = {
        'precision': precision / pred_graph.shape[0],
        'recall': recall / pred_graph.shape[0],
        'mean_reciprocal_rank': mean_reciprocal_rank / pred_graph.shape[0],
        'hit_rate': hit_rate / pred_graph.shape[0],
        'ndcg': ndcg
    }

    return metrics
