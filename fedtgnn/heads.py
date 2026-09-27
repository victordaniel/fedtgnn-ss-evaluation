"""Logistic regression trained by federated averaging (FedAvg / FedProx),
locally, or centrally. Used both as baselines and as the FedTGNN-SS
calibrated head on [X || H]. No patient-level data leave a silo in the
federated modes: silos share model weights plus per-feature sums and class
counts (for standardisation and class balancing)."""

import numpy as np
import torch
import torch.nn.functional as F

from .models import LogReg


def _stats(parts):
    n = sum(len(p[0]) for p in parts)
    s1 = sum(p[0].sum(0) for p in parts)
    s2 = sum((p[0] ** 2).sum(0) for p in parts)
    mean = s1 / n
    std = torch.sqrt((s2 / n - mean ** 2).clamp(min=0))
    std[std < 1e-8] = 1.0
    pos = sum(int(p[1].sum()) for p in parts)
    w = torch.tensor([n / (2.0 * max(n - pos, 1)), n / (2.0 * max(pos, 1))])
    return mean, std, w, n


def logreg(parts, mode='fedavg', C=0.5, rounds=30, local_epochs=5, lr=0.05,
           prox_mu=0.0, seed=0):
    """parts: list of (X_train, y_train, {name: X_eval}) tensors per silo.
    mode: 'fedavg' | 'local' | 'central'. Returns {name: [probs per silo]}
    and communicated bytes."""
    torch.manual_seed(seed)
    d = parts[0][0].shape[1]
    comm = 0
    if mode == 'local':
        groups = [[p] for p in parts]
    elif mode == 'central':
        groups = [[(torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]), {})]]
    else:
        groups = [parts]

    trained = []
    for g in groups:
        mean, std, cw, n = _stats(g)
        if mode == 'fedavg':
            comm += len(g) * (3 * d + 2) * 4
        glob = LogReg(d)
        n_rounds = rounds if (mode == 'fedavg' and len(g) > 1) else 1
        n_epochs = local_epochs if n_rounds > 1 else 300
        for _ in range(n_rounds):
            states, sizes = [], []
            for Xp, yp, _ in g:
                m = LogReg(d)
                m.load_state_dict(glob.state_dict())
                g0 = [p.detach().clone() for p in glob.parameters()]
                opt = torch.optim.Adam(m.parameters(), lr=lr)
                Xn = (Xp - mean) / std
                for _ in range(n_epochs):
                    opt.zero_grad()
                    loss = F.cross_entropy(m(Xn), yp, weight=cw)
                    loss = loss + m.lin.weight.pow(2).sum() / (2 * C * n)
                    if prox_mu > 0:
                        loss = loss + prox_mu / 2 * sum(
                            (p - q).pow(2).sum() for p, q in zip(m.parameters(), g0))
                    loss.backward()
                    opt.step()
                states.append(m.state_dict())
                sizes.append(len(Xp))
            tot = sum(sizes)
            glob.load_state_dict({k: sum(s[k] * (z / tot) for s, z in zip(states, sizes))
                                  for k in states[0]})
            if mode == 'fedavg':
                comm += 2 * len(g) * (d + 1) * 2 * 4
        trained.append((glob, mean, std))

    out = {}
    for i, p in enumerate(parts):
        glob, mean, std = trained[i] if mode == 'local' else trained[0]
        for name, Xe in p[2].items():
            with torch.no_grad():
                pr = F.softmax(glob((Xe - mean) / std), 1)[:, 1].numpy()
            out.setdefault(name, []).append(pr)
    return out, comm


def federated_lr_head(head_data, federated=True, seed=0):
    """Calibrated head on [X || H] for FedTGNN-SS.
    head_data: list of (silo, H_train, {name: (logits, H_eval)})."""
    parts = []
    for s, h_tr, ev in head_data:
        Xtr = torch.cat([s.X, h_tr], 1)[s.lab]
        ytr = s.y[s.lab]
        evX = {n: torch.cat([s.eval_X[n], ev[n][1]], 1) for n in ev}
        evX['members'] = Xtr           # for the membership-inference audit
        parts.append((Xtr, ytr, evX))
    out, comm = logreg(parts, mode='fedavg' if federated else 'local', seed=seed)
    out['_comm'] = comm
    return out
