"""Baselines with a small validation-only tuning budget (sensitivity
analysis; PROTOCOL.md, Amendment 5). For each method the configuration with
the highest VALIDATION AUROC is selected from a fixed grid; the test fold is
never consulted. The selected configuration's predictions are returned."""

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

from .baselines import _collect, _lr_parts, _proba
from .heads import logreg
from .models import fedavg, load_avg

GRIDS = {
    'LR': [dict(C=c) for c in (0.01, 0.1, 1.0, 10.0)],
    'MLP': [dict(hidden=h, lr=lr) for h in (32, 64, 128) for lr in (0.001, 0.005, 0.01)],
    'RF': [dict(max_depth=d, min_samples_leaf=m) for d in (None, 5, 10) for m in (1, 5)],
    'XGB': [dict(max_depth=d, n_estimators=n, learning_rate=lr)
            for d in (3, 6) for n in (100, 300) for lr in (0.05, 0.3)],
    'SVM': [dict(C=c, gamma=g) for c in (0.1, 1.0, 10.0) for g in ('scale', 0.01, 0.1)],
}


def _val_auc(ctx, pr):
    yv = np.concatenate([ctx.y[s.eval_idx['val']] for s in ctx.silos])
    pv = np.concatenate(pr['val'])
    return roc_auc_score(yv, pv) if len(np.unique(yv)) > 1 else 0.5


def _lr(ctx, seed, C):
    pr, _ = logreg(_lr_parts(ctx), 'fedavg', C=C, seed=seed)
    return pr


def _mlp(ctx, seed, hidden, lr, rounds=10, local_epochs=5):
    torch.manual_seed(seed)
    net = lambda: torch.nn.Sequential(torch.nn.Linear(ctx.d, hidden), torch.nn.ReLU(),
                                      torch.nn.Dropout(0.3), torch.nn.Linear(hidden, 2))
    g = net()
    for _ in range(rounds):
        states, sizes = [], []
        for s in ctx.silos:
            m = copy.deepcopy(g)
            opt = torch.optim.Adam(m.parameters(), lr=lr)
            for _ in range(local_epochs):
                m.train()
                opt.zero_grad()
                F.cross_entropy(m(s.X[s.lab]), s.y[s.lab]).backward()
                opt.step()
            states.append(m.state_dict())
            sizes.append(int(s.lab.sum()))
        load_avg(g, fedavg(states, sizes))
    g.eval()
    with torch.no_grad():
        return {n: [F.softmax(g(s.eval_X[n]), 1)[:, 1].numpy() for s in ctx.silos]
                for n in ('val', 'test')}


def _ens(ctx, seed, kind, **kw):
    models, sizes = [], []
    for s in ctx.silos:
        Xl, yl = s.X[s.lab].numpy(), s.y[s.lab].numpy()
        if len(np.unique(yl)) < 2:
            continue
        if kind == 'RF':
            clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=seed,
                                         n_jobs=1, **kw)
        elif kind == 'XGB':
            clf = xgb.XGBClassifier(eval_metric='logloss', random_state=seed, verbosity=0, n_jobs=1, **kw)
        else:
            clf = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=seed, **kw)
        models.append(clf.fit(Xl, yl))
        sizes.append(len(yl))
    w = np.array(sizes) / sum(sizes)
    return {n: [sum(wi * _proba(m, s.eval_X[n]) for wi, m in zip(w, models)) for s in ctx.silos]
            for n in ('val', 'test')}


def _tuned(kind, fn):
    def run(ctx, seed=0):
        t0 = time.time()
        best, best_auc, best_cfg = None, -1, None
        for cfg in GRIDS[kind]:
            pr = fn(ctx, seed, **cfg)
            a = _val_auc(ctx, pr)
            if a > best_auc:
                best, best_auc, best_cfg = pr, a, cfg
        out = _collect(ctx, best, dict(train_time_s=time.time() - t0, comm_bytes=float('nan')))
        out['_meta']['selected'] = str(best_cfg)
        return out
    return run


TUNED = {
    'FedAvg-LR (tuned)': _tuned('LR', _lr),
    'FedAvg-MLP (tuned)': _tuned('MLP', _mlp),
    'FedEns-RF (tuned)': _tuned('RF', lambda c, s, **k: _ens(c, s, 'RF', **k)),
    'FedEns-XGB (tuned)': _tuned('XGB', lambda c, s, **k: _ens(c, s, 'XGB', **k)),
    'FedEns-SVM (tuned)': _tuned('SVM', lambda c, s, **k: _ens(c, s, 'SVM', **k)),
}
