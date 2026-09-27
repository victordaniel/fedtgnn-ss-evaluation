"""Per-fold federated context: what each silo holds locally.

Unlabeled patients' outcomes are replaced by -1 in the training tensors, so
no training code path can read them. True labels of unlabeled patients are
kept separately (``y_hidden``) and used only for pseudo-label diagnostics.
"""

from dataclasses import dataclass

import numpy as np
import torch

from .graph import knn_graph
from .preprocess import FederatedStandardizer, continuous_mask


@dataclass
class Silo:
    sid: int
    idx: np.ndarray            # global indices of training patients
    X: torch.Tensor            # standardised features (n x d)
    y: torch.Tensor            # labels, -1 for unlabeled
    lab: torch.Tensor          # bool mask
    unl: torch.Tensor          # bool mask
    y_hidden: torch.Tensor     # true labels (diagnostics ONLY)
    ei: torch.Tensor
    ew: torch.Tensor
    sigma: float
    eval_idx: dict             # name -> global indices of eval patients here
    eval_X: dict               # name -> standardised eval features


@dataclass
class FoldContext:
    silos: list
    Xs: np.ndarray             # standardised full matrix (for sklearn baselines)
    y: np.ndarray              # full labels (only indexed by labeled/eval ids)
    split: object
    cont_mask: torch.Tensor
    d: int


def build_context(X_raw, y, split, k=10):
    scaler = FederatedStandardizer().fit(X_raw, split.silos)
    Xs = scaler.transform(X_raw)
    cm = torch.tensor(continuous_mask(X_raw, split.silos), dtype=torch.float)
    silos = []
    for s, (idx, lab) in enumerate(zip(split.silos, split.labeled)):
        Xt = torch.tensor(Xs[idx], dtype=torch.float)
        lab_mask = torch.tensor(np.isin(idx, lab))
        y_true = torch.tensor(y[idx], dtype=torch.long)
        y_vis = torch.where(lab_mask, y_true, torch.full_like(y_true, -1))
        ei, ew, sigma = knn_graph(Xs[idx], k)
        ev_idx = {'val': split.val[split.val_silo == s],
                  'test': split.test[split.test_silo == s]}
        ev_X = {n: torch.tensor(Xs[i], dtype=torch.float) for n, i in ev_idx.items()}
        silos.append(Silo(s, idx, Xt, y_vis, lab_mask, ~lab_mask, y_true,
                          ei, ew, sigma, ev_idx, ev_X))
    return FoldContext(silos, Xs, y, split, cm, Xs.shape[1])
