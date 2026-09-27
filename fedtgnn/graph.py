"""Local patient-similarity graphs (one per silo, training patients only)
and inductive insertion of evaluation patients.

Evaluation patients receive directed edges *from* their k nearest training
patients only (train -> eval). With PyG's source-to-target message flow and
target-side degree normalisation, adding these edges leaves every training
node's representation unchanged, so an evaluation patient can never
influence training nodes, and evaluation patients never see each other.
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def _gaussian(d, sigma):
    return np.exp(-(d ** 2) / (2 * sigma ** 2))


def knn_graph(Z, k):
    """Symmetric k-NN graph over training nodes Z (n x d).
    Returns edge_index (2 x E), edge_weight (E,), sigma (median k-NN distance)."""
    n = len(Z)
    k = min(k, n - 1)
    nn_ = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    dist, ind = nn_.kneighbors(Z)
    dist, ind = dist[:, 1:], ind[:, 1:]
    sigma = float(np.median(dist)) + 1e-8
    src = np.repeat(np.arange(n), k)
    dst = ind.reshape(-1)
    w = _gaussian(dist.reshape(-1), sigma)
    # symmetrise, keep one weight per undirected pair
    pairs = {}
    for a, b, ww in zip(src, dst, w):
        key = (a, b) if a < b else (b, a)
        pairs[key] = ww
    keys = np.array(list(pairs.keys()))
    ws = np.array(list(pairs.values()))
    ei = np.concatenate([keys.T, keys.T[::-1]], axis=1)
    ew = np.concatenate([ws, ws])
    return (torch.tensor(ei, dtype=torch.long),
            torch.tensor(ew, dtype=torch.float), sigma)


def insert_eval_nodes(Z_train, Z_eval, k, sigma, train_edge_index,
                      train_edge_weight):
    """Graph over [train ; eval] nodes. Eval node j gets edges from its k
    nearest training nodes (search space = Z_train). Returns edge_index,
    edge_weight with eval nodes indexed n_train + j."""
    n_tr = len(Z_train)
    if len(Z_eval) == 0:
        return train_edge_index, train_edge_weight
    k = min(k, n_tr)
    nn_ = NearestNeighbors(n_neighbors=k).fit(Z_train)
    dist, ind = nn_.kneighbors(Z_eval)
    src = ind.reshape(-1)
    dst = n_tr + np.repeat(np.arange(len(Z_eval)), k)
    w = _gaussian(dist.reshape(-1), sigma)
    ei = torch.cat([train_edge_index,
                    torch.tensor(np.stack([src, dst]), dtype=torch.long)], 1)
    ew = torch.cat([train_edge_weight, torch.tensor(w, dtype=torch.float)])
    return ei, ew
