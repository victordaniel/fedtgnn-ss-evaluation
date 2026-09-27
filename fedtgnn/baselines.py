"""Baselines. Every method sees exactly the same fold context: the same
silos, labeled/unlabeled masks, local training graphs, and inductive
evaluation patients. Each returns {'val': (idx, prob), 'test': (idx, prob),
'_meta': {...}}.

Naming follows what each method actually does:
  FedAvg-LR / FedProx-LR / FedAvg-MLP : parameter averaging over rounds
  FedEns-RF / FedEns-XGB / FedEns-SVM : one-shot size-weighted soft-vote
                                        ensemble of locally trained models
                                        (tree ensembles cannot be FedAvg-ed)
  FedST-XGB                            : FedEns-XGB with local self-training
  FedMatch-tab / RSCFed-tab            : tabular adaptations (MLP backbone)
"""

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC

from .fedtgnn import FedTGNNConfig, train_fedtgnn
from .graph import insert_eval_nodes
from .heads import logreg
from .models import GCN, MLP, SAGE, fedavg, load_avg, n_params_bytes


def _collect(ctx, probs, meta):
    out = {}
    for n in ('val', 'test'):
        idx = np.concatenate([s.eval_idx[n] for s in ctx.silos])
        out[n] = (idx, np.concatenate(probs[n]))
    out['_meta'] = meta
    return out


def _lab_np(ctx, s):
    return s.X[s.lab].numpy(), s.y[s.lab].numpy()


# ------------------------------------------------------- logistic regression
def _lr_parts(ctx):
    return [(s.X[s.lab], s.y[s.lab], dict(s.eval_X)) for s in ctx.silos]


def fedavg_lr(ctx, seed=0):
    t0 = time.time()
    parts = [(Xl, yl, dict(ev, members=Xl)) for Xl, yl, ev in _lr_parts(ctx)]
    pr, comm = logreg(parts, 'fedavg', seed=seed)
    out = _collect(ctx, pr, dict(train_time_s=time.time() - t0, comm_bytes=comm))
    out['members'] = (np.concatenate([s.idx[s.lab.numpy()] for s in ctx.silos]),
                      np.concatenate(pr['members']))
    return out


def fedprox_lr(ctx, seed=0):
    t0 = time.time()
    pr, comm = logreg(_lr_parts(ctx), 'fedavg', prox_mu=0.01, seed=seed)
    return _collect(ctx, pr, dict(train_time_s=time.time() - t0, comm_bytes=comm))


def local_lr(ctx, seed=0):
    t0 = time.time()
    pr, _ = logreg(_lr_parts(ctx), 'local', seed=seed)
    return _collect(ctx, pr, dict(train_time_s=time.time() - t0, comm_bytes=0))


def cent_lr(ctx, seed=0):
    t0 = time.time()
    pr, _ = logreg(_lr_parts(ctx), 'central', seed=seed)
    return _collect(ctx, pr, dict(train_time_s=time.time() - t0, comm_bytes=0))


# ------------------------------------------------------------------- MLPs
def _mlp_fed(ctx, seed, ssl=None, rounds=10, local_epochs=5, tau=0.95, subsample=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    g = MLP(ctx.d)
    comm, pb = 0, n_params_bytes(g)
    for _ in range(rounds):
        chosen = list(range(len(ctx.silos)))
        if subsample and len(chosen) > 2:
            chosen = sorted(rng.choice(chosen, max(2, len(chosen) // 2), replace=False))
        states, sizes = [], []
        for i in chosen:
            s = ctx.silos[i]
            m = copy.deepcopy(g)
            opt = torch.optim.Adam(m.parameters(), lr=0.005)
            Xl, yl, Xu = s.X[s.lab], s.y[s.lab], s.X[s.unl]
            with torch.no_grad():
                pu = F.softmax((g if ssl == 'fedmatch' else m)(Xu), 1) if len(Xu) else None
            for _ in range(local_epochs):
                m.train()
                opt.zero_grad()
                loss = F.cross_entropy(m(Xl), yl)
                if ssl and pu is not None:
                    conf, pc = pu.max(1)
                    keep = conf >= tau
                    if keep.any():
                        if ssl == 'fedmatch':   # consistency with global model
                            loss = loss + 0.5 * F.kl_div(F.log_softmax(m(Xu[keep]), 1),
                                                         pu[keep], reduction='batchmean')
                        else:                   # RSCFed-style local pseudo-labels
                            loss = loss + 0.3 * F.cross_entropy(m(Xu[keep]), pc[keep])
                loss.backward()
                opt.step()
            states.append(m.state_dict())
            sizes.append(int(s.lab.sum()))
        load_avg(g, fedavg(states, sizes))
        comm += 2 * pb * len(chosen)
    g.eval()
    with torch.no_grad():
        pr = {n: [F.softmax(g(s.eval_X[n]), 1)[:, 1].numpy() for s in ctx.silos]
              for n in ('val', 'test')}
    return pr, comm


def fedavg_mlp(ctx, seed=0):
    t0 = time.time()
    pr, comm = _mlp_fed(ctx, seed)
    return _collect(ctx, pr, dict(train_time_s=time.time() - t0, comm_bytes=comm))


def fedmatch_tab(ctx, seed=0):
    t0 = time.time()
    pr, comm = _mlp_fed(ctx, seed, ssl='fedmatch')
    return _collect(ctx, pr, dict(train_time_s=time.time() - t0, comm_bytes=comm))


def rscfed_tab(ctx, seed=0):
    t0 = time.time()
    pr, comm = _mlp_fed(ctx, seed, ssl='rscfed', subsample=True)
    return _collect(ctx, pr, dict(train_time_s=time.time() - t0, comm_bytes=comm))


# ------------------------------------------------ one-shot soft-vote ensembles
def _proba(clf, X):
    X = X.numpy() if torch.is_tensor(X) else X
    return clf.predict_proba(X)[:, 1] if len(X) else np.zeros(0)


def _make(name, seed):
    if name == 'RF':
        return RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                      random_state=seed, n_jobs=1)
    if name == 'XGB':
        return xgb.XGBClassifier(eval_metric='logloss', random_state=seed,
                                 verbosity=0, n_jobs=1)
    if name == 'SVM':
        return SVC(kernel='rbf', probability=True, class_weight='balanced',
                   random_state=seed)
    raise ValueError(name)


def _ensemble(ctx, name, seed, self_train=False):
    t0 = time.time()
    models, sizes = [], []
    for s in ctx.silos:
        Xl, yl = _lab_np(ctx, s)
        if len(np.unique(yl)) < 2:
            continue
        clf = _make(name, seed)
        if self_train:
            y_st = s.y.numpy().copy()          # -1 marks unlabeled
            clf = SelfTrainingClassifier(clf, threshold=0.95, max_iter=10)
            clf.fit(s.X.numpy(), y_st)
        else:
            clf.fit(Xl, yl)
        models.append(clf)
        sizes.append(len(yl))
    w = np.array(sizes) / sum(sizes)
    pr = {n: [sum(wi * _proba(m, s.eval_X[n]) for wi, m in zip(w, models)) for s in ctx.silos]
          for n in ('val', 'test')}
    return _collect(ctx, pr, dict(train_time_s=time.time() - t0, comm_bytes=float('nan')))


def fedens_rf(ctx, seed=0):
    return _ensemble(ctx, 'RF', seed)


def fedens_xgb(ctx, seed=0):
    return _ensemble(ctx, 'XGB', seed)


def fedens_svm(ctx, seed=0):
    return _ensemble(ctx, 'SVM', seed)


def fedst_xgb(ctx, seed=0):
    return _ensemble(ctx, 'XGB', seed, self_train=True)


def cent_xgb(ctx, seed=0):
    t0 = time.time()
    Xl = np.concatenate([_lab_np(ctx, s)[0] for s in ctx.silos])
    yl = np.concatenate([_lab_np(ctx, s)[1] for s in ctx.silos])
    clf = _make('XGB', seed).fit(Xl, yl)
    pr = {n: [_proba(clf, s.eval_X[n]) for s in ctx.silos] for n in ('val', 'test')}
    return _collect(ctx, pr, dict(train_time_s=time.time() - t0, comm_bytes=0))


# ----------------------------------------------------------------- GNNs
def _gnn_eval(model, s):
    model.eval()
    n_tr = len(s.X)
    res = {}
    with torch.no_grad():
        for n in ('val', 'test'):
            ei, ew = insert_eval_nodes(s.X.numpy(), s.eval_X[n].numpy(), 10, s.sigma, s.ei, s.ew)
            lg, _ = model(torch.cat([s.X, s.eval_X[n]]), ei, ew)
            res[n] = F.softmax(lg[n_tr:], 1)[:, 1].numpy()
    return res


def _gnn(ctx, cls, seed, federated=True, rounds=10, local_epochs=3, epochs=30):
    torch.manual_seed(seed)
    t0 = time.time()
    g = cls(ctx.d)
    comm, pb = 0, n_params_bytes(g)
    models = [copy.deepcopy(g) for _ in ctx.silos]

    def step(m, s, n_ep):
        opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
        for _ in range(n_ep):
            m.train()
            opt.zero_grad()
            lg, _ = m(s.X, s.ei, s.ew)
            F.cross_entropy(lg[s.lab], s.y[s.lab]).backward()
            opt.step()

    if federated:
        for _ in range(rounds):
            states = []
            for m, s in zip(models, ctx.silos):
                m.load_state_dict(g.state_dict())
                step(m, s, local_epochs)
                states.append(m.state_dict())
            load_avg(g, fedavg(states, [len(s.X) for s in ctx.silos]))
            comm += 2 * pb * len(ctx.silos)
        models = [g] * len(ctx.silos)
    else:
        for m, s in zip(models, ctx.silos):
            step(m, s, epochs)
    evs = [_gnn_eval(m, s) for m, s in zip(models, ctx.silos)]
    pr = {n: [e[n] for e in evs] for n in ('val', 'test')}
    return _collect(ctx, pr, dict(train_time_s=time.time() - t0, comm_bytes=comm))


def fedavg_gcn(ctx, seed=0):
    return _gnn(ctx, GCN, seed)


def fedavg_sage(ctx, seed=0):
    return _gnn(ctx, SAGE, seed)


def local_gcn(ctx, seed=0):
    return _gnn(ctx, GCN, seed, federated=False)


def cent_grand(ctx, seed=0, epochs=100, S=2, drop=0.2, temp=0.5, lam=1.0):
    """Compact GRAND-style reference: one GCN over the union of all silos'
    training graphs (disconnected components, no cross-silo edges), random
    feature-dropout views and sharpened consistency on unlabeled nodes.
    Non-private reference: pools all silos' training data centrally."""
    torch.manual_seed(seed)
    t0 = time.time()
    offs = np.cumsum([0] + [len(s.X) for s in ctx.silos])
    X = torch.cat([s.X for s in ctx.silos])
    y = torch.cat([s.y for s in ctx.silos])
    lab = torch.cat([s.lab for s in ctx.silos])
    unl = ~lab
    ei = torch.cat([s.ei + int(o) for s, o in zip(ctx.silos, offs)], 1)
    ew = torch.cat([s.ew for s in ctx.silos])
    m = GCN(ctx.d)
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(epochs):
        m.train()
        opt.zero_grad()
        views = [m(X * (torch.rand_like(X) > drop) / (1 - drop), ei, ew)[0] for _ in range(S)]
        loss = sum(F.cross_entropy(v[lab], y[lab]) for v in views) / S
        with torch.no_grad():
            pbar = sum(F.softmax(v, 1) for v in views) / S
            sharp = pbar ** (1 / temp)
            sharp = sharp / sharp.sum(1, keepdim=True)
        loss = loss + lam * sum((F.softmax(v, 1)[unl] - sharp[unl]).pow(2).sum(1).mean()
                                for v in views) / S
        loss.backward()
        opt.step()
    evs = [_gnn_eval(m, s) for s in ctx.silos]
    pr = {n: [e[n] for e in evs] for n in ('val', 'test')}
    return _collect(ctx, pr, dict(train_time_s=time.time() - t0, comm_bytes=0))


# ------------------------------------------------ variants of the proposed arch
def local_tgnn(ctx, seed=0):
    """Full FedTGNN-SS objective (incl. pseudo-labelling) without federation."""
    return train_fedtgnn(ctx, FedTGNNConfig(federated=False), seed=seed)


def fedtgnn_ss(ctx, seed=0, diag=None):
    return train_fedtgnn(ctx, FedTGNNConfig(), seed=seed, diag=diag)


METHODS = {
    'FedTGNN-SS': fedtgnn_ss,
    'FedAvg-LR': fedavg_lr,
    'FedProx-LR': fedprox_lr,
    'FedAvg-MLP': fedavg_mlp,
    'FedEns-RF': fedens_rf,
    'FedEns-XGB': fedens_xgb,
    'FedEns-SVM': fedens_svm,
    'FedST-XGB': fedst_xgb,
    'FedMatch-tab': fedmatch_tab,
    'RSCFed-tab': rscfed_tab,
    'FedAvg-GCN': fedavg_gcn,
    'FedAvg-SAGE': fedavg_sage,
    'Local-LR': local_lr,
    'Local-GCN': local_gcn,
    'Local-TGNN': local_tgnn,
    'Cent-LR': cent_lr,
    'Cent-XGB': cent_xgb,
    'Cent-GRAND': cent_grand,
}
