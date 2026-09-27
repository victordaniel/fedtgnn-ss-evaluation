"""Leakage, count and metric checks. Run: python -m pytest tests -q"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fedtgnn import splits                               # noqa: E402
from fedtgnn.context import build_context                # noqa: E402
from fedtgnn.graph import insert_eval_nodes, knn_graph   # noqa: E402
from fedtgnn.metrics import calibration, evaluate        # noqa: E402
from fedtgnn.models import FedTGNNEncoder                # noqa: E402


def _toy(n=400, d=6, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.35).astype(int)
    X = rng.normal(size=(n, d)) + y[:, None]
    X[rng.random((n, d)) < 0.1] = np.nan
    return X, y


def test_splits_partition_and_counts():
    X, y = _toy()
    for rho in (0.1, 0.8):
        for sp in splits.make_splits(y, rho, 3, 0.5, n_repeats=2, n_folds=5):
            splits.check_no_leakage(sp)
            for s, lab in zip(sp.silos, sp.labeled):
                for c in (0, 1):
                    n_c = int((y[s] == c).sum())
                    assert int((y[lab] == c).sum()) == max(1, int(round((1 - rho) * n_c)))


def test_splits_paired_across_rho():
    _, y = _toy()
    a = list(splits.make_splits(y, 0.1, 3, 0.5, n_repeats=1))
    b = list(splits.make_splits(y, 0.8, 3, 0.5, n_repeats=1))
    for sa, sb in zip(a, b):
        assert np.array_equal(sa.test, sb.test)
        assert all(np.array_equal(u, v) for u, v in zip(sa.silos, sb.silos))


def test_unlabeled_targets_hidden():
    X, y = _toy()
    sp = next(splits.make_splits(y, 0.8, 3, 0.5, n_repeats=1))
    ctx = build_context(X, y, sp)
    for s in ctx.silos:
        assert (s.y[s.unl] == -1).all()
        assert (s.y[s.lab] >= 0).all()


def test_scaler_ignores_eval_patients():
    X, y = _toy()
    sp = next(splits.make_splits(y, 0.5, 3, 0.5, n_repeats=1))
    X2 = X.copy()
    X2[sp.test] = 1e6            # corrupt test features only
    X2[sp.val] = -1e6
    a = build_context(X, y, sp)
    b = build_context(X2, y, sp)
    for sa, sb in zip(a.silos, b.silos):
        assert torch.allclose(sa.X, sb.X)
        assert torch.equal(sa.ei, sb.ei)


def test_eval_insertion_does_not_change_training_nodes():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(60, 5)).astype(np.float32)
    E = rng.normal(size=(15, 5)).astype(np.float32)
    ei, ew, sig = knn_graph(Z, 5)
    m = FedTGNNEncoder(5).eval()
    with torch.no_grad():
        lg_tr, _ = m(torch.tensor(Z), ei, ew)
        ei2, ew2 = insert_eval_nodes(Z, E, 5, sig, ei, ew)
        lg_all, _ = m(torch.tensor(np.vstack([Z, E])), ei2, ew2)
    assert torch.allclose(lg_tr, lg_all[:60], atol=1e-5)
    # eval nodes never send messages
    assert (ei2[0] < 60).all()


def test_calibration_perfect_model():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.02, 0.98, 20000)
    y = (rng.random(20000) < p).astype(int)
    ci, cs = calibration(y, p)
    assert abs(ci) < 0.05 and abs(cs - 1) < 0.05


def test_confusion_counts_sum():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.8, 0.9])
    m = evaluate(y, p, 0.5)
    assert m['tp'] + m['tn'] + m['fp'] + m['fn'] == 5
    assert (m['tp'], m['fp'], m['fn'], m['tn']) == (2, 1, 1, 1)
