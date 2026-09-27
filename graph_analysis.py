"""Exploratory mechanistic analysis of the patient-similarity graphs
(PROTOCOL.md, Amendment 4). DIAGNOSTIC ONLY: the withheld outcomes of
unlabelled training patients are used solely to describe graph quality,
never for training.

For each dataset, repetition and fold (same partitions as the main study):
A. Graph quality, per silo, for the initial feature-space graph and for the
   refined (AGR) embedding graph produced during FedTGNN-SS training:
     * edge homophily      share of edges joining patients with the same outcome
     * random baseline     p^2 + (1-p)^2 for the silo's prevalence p
     * node purity         mean over patients of the share of same-outcome neighbours
     * purity by class     node purity among positive / negative patients
B. Access to labels at scarcity rho: share of unlabelled training patients
   with at least one labelled neighbour, and outcome agreement with the
   labelled neighbours only.
C. Whether local neighbourhood structure carries predictive information
   beyond a tabular model: distance-weighted k-NN (k = 10) as a one-shot
   federated ensemble, on the same inductive evaluation, compared with the
   main-study LR, GCN, GraphSAGE and FedTGNN-SS results.

Outputs: results/graph_quality.csv, results/graph_knn.csv
Usage:   python graph_analysis.py [--repeats 3] [--workers 3]
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DATASETS = {'gdm_early': 3, 'pima': 2, 'early': 2}


def graph_stats(ei, y_true, lab_mask=None):
    src, dst = ei[0].numpy(), ei[1].numpy()
    same = (y_true[src] == y_true[dst]).astype(float)
    n = len(y_true)
    agree = np.bincount(dst, weights=same, minlength=n)
    deg = np.bincount(dst, minlength=n)
    has = deg > 0
    purity = agree[has] / deg[has]
    yh = y_true[has]
    p = y_true.mean()
    out = dict(edge_homophily=same.mean(), random_homophily=p ** 2 + (1 - p) ** 2,
               node_purity=purity.mean(),
               purity_pos=purity[yh == 1].mean() if (yh == 1).any() else np.nan,
               purity_neg=purity[yh == 0].mean() if (yh == 0).any() else np.nan,
               prevalence=p)
    if lab_mask is not None:
        lm = lab_mask
        lab_src = lm[src]
        # for each unlabelled target: labelled in-neighbours and their agreement
        cnt = np.bincount(dst, weights=lab_src.astype(float), minlength=n)
        agr = np.bincount(dst, weights=(lab_src & (y_true[src] == y_true[dst])).astype(float), minlength=n)
        unl = ~lm
        out['unl_with_labelled_nb'] = (cnt[unl] > 0).mean()
        ok = unl & (cnt > 0)
        out['labelled_nb_purity'] = (agr[ok] / cnt[ok]).mean() if ok.any() else np.nan
    return out


def run_one(task):
    import torch
    torch.set_num_threads(1)
    from sklearn.metrics import roc_auc_score
    from sklearn.neighbors import KNeighborsClassifier
    from fedtgnn import data, splits
    from fedtgnn.context import build_context
    from fedtgnn.fedtgnn import FedTGNNConfig, train_fedtgnn

    ds, S, rho, r, f = task
    X, y, _ = data.load(ds)
    sp = next(s for s in splits.make_splits(y, rho, S, 0.5, n_repeats=r + 1)
              if s.repeat == r and s.fold == f)
    ctx = build_context(X, y, sp)
    qual, knn = [], []

    def hook(silo, ei, stage):
        st = graph_stats(ei, silo.y_hidden.numpy(), silo.lab.numpy())
        st.update(dataset=ds, rho=rho, repeat=r, fold=f, silo=silo.sid, graph=stage)
        qual.append(st)

    train_fedtgnn(ctx, FedTGNNConfig(), seed=sp.seed, graph_hook=hook)

    # C. one-shot federated k-NN ensemble, same inductive evaluation
    models, w = [], []
    for s in ctx.silos:
        yl = s.y[s.lab].numpy()
        if len(np.unique(yl)) < 2:
            continue
        m = KNeighborsClassifier(n_neighbors=min(10, len(yl)), weights='distance')
        models.append(m.fit(s.X[s.lab].numpy(), yl))
        w.append(len(yl))
    w = np.array(w) / sum(w)
    p, yt = [], []
    for s in ctx.silos:
        Xe = s.eval_X['test'].numpy()
        if len(Xe):
            p.append(sum(wi * m.predict_proba(Xe)[:, 1] for wi, m in zip(w, models)))
            yt.append(y[s.eval_idx['test']])
    knn.append(dict(dataset=ds, rho=rho, repeat=r, fold=f,
                    auroc=roc_auc_score(np.concatenate(yt), np.concatenate(p))))
    return qual, knn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--workers', type=int, default=3)
    args = ap.parse_args()
    tasks = [(ds, S, rho, r, f) for ds, S in DATASETS.items() for rho in (0.1, 0.8)
             for r in range(args.repeats) for f in range(5)]
    Q, K = [], []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_one, t) for t in tasks]
        for i, fu in enumerate(as_completed(futs)):
            q, k = fu.result()
            Q += q
            K += k
            print(f'[{i + 1}/{len(tasks)}]', flush=True)
    q = pd.DataFrame(Q)
    k = pd.DataFrame(K)
    q.to_csv(os.path.join(HERE, 'results', 'graph_quality.csv'), index=False)
    k.to_csv(os.path.join(HERE, 'results', 'graph_knn.csv'), index=False)
    print(q.groupby(['dataset', 'rho', 'graph'])[['edge_homophily', 'random_homophily', 'node_purity',
                                                  'purity_pos', 'purity_neg', 'unl_with_labelled_nb',
                                                  'labelled_nb_purity']].mean().round(3).to_string())
    print(k.groupby(['dataset', 'rho']).auroc.agg(['mean', 'std']).round(3).to_string())


if __name__ == '__main__':
    main()
