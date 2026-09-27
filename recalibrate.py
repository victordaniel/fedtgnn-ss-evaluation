"""Exploratory post-hoc recalibration (PROTOCOL.md, Amendment 3).

For every run of `run.py recal` (FedTGNN-SS and FedAvg-LR, rho = 0.8,
10 x 5 folds), two recalibration maps are fitted on the VALIDATION
predictions only and applied unchanged to the test fold:
  * Platt scaling:        logit p' = a + b * logit p
  * temperature scaling:  logit p' = logit p / T
Only predicted scores and outcomes of validation patients are used (two
numbers per patient), which a federation can fit by averaging.

Outputs: results/recal_summary.csv and results/recal_runs.csv
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import brier_score_loss, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fedtgnn.metrics import calibration, ece   # noqa: E402

RES = os.path.join(HERE, 'results')
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def nll(eta, y):
    return float(np.sum(np.logaddexp(0, eta) - y * eta))


def fit_platt(y, p):
    z = np.clip(logit(p), -15, 15)
    r = minimize(lambda t: nll(t[0] + t[1] * z, y), x0=[0.0, 1.0], method='L-BFGS-B',
                 bounds=[(-10, 10), (1e-3, 20)])
    a, b = r.x
    return lambda q: 1 / (1 + np.exp(-(a + b * np.clip(logit(q), -15, 15))))


def fit_temperature(y, p):
    z = np.clip(logit(p), -15, 15)
    r = minimize_scalar(lambda t: nll(z / t, y), bounds=(0.05, 50), method='bounded')
    T = r.x
    return lambda q: 1 / (1 + np.exp(-np.clip(logit(q), -15, 15) / T))


def metrics(y, p):
    ci, cs = calibration(y, p)
    return dict(auroc=roc_auc_score(y, p), brier=brier_score_loss(y, p),
                cal_intercept=ci, cal_slope=cs, ece=ece(y, p))


def main():
    pr = pd.read_csv(os.path.join(RES, 'recal_preds.csv.gz'))
    rows = []
    for (ds, m, r, f), g in pr.groupby(['dataset', 'method', 'repeat', 'fold']):
        v, t = g[g.part == 'val'], g[g.part == 'test']
        yv, pv, yt, pt = v.y.values, v.p.values, t.y.values, t.p.values
        for name, pnew in [('None', pt),
                           ('Platt scaling', fit_platt(yv, pv)(pt)),
                           ('Temperature scaling', fit_temperature(yv, pv)(pt))]:
            rows.append(dict(dataset=ds, method=m, repeat=r, fold=f, recalibration=name,
                             **metrics(yt, pnew)))
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(RES, 'recal_runs.csv'), index=False)
    g = runs.groupby(['dataset', 'method', 'recalibration'])
    summ = g[['auroc', 'brier', 'cal_intercept', 'cal_slope', 'ece']].agg(['mean', 'std'])
    summ.columns = [f'{a}_{b}' for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(os.path.join(RES, 'recal_summary.csv'), index=False)
    print(summ.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
