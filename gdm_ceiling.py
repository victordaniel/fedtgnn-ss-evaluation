"""Exploratory analyses of the GDM cohort (added after the primary analysis;
see PROTOCOL.md, Amendment 2).

1. Baseline characteristics by outcome (for the manuscript table).
2. Single-predictor discrimination: for every predictor, a logistic
   regression on that predictor alone, fitted on the training part and
   scored on the test fold (same 10 x 5 outer folds as the main study).
3. Leave-one-predictor-out: logistic regression on all early predictors
   except one, to see whether any single predictor carries the ceiling.

Outputs: results/gdm_baseline.csv, results/gdm_single_feature.csv,
         results/gdm_drop_one.csv
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fedtgnn.data import GDM_TARGET, DEFAULT_GDM_PATH   # noqa: E402

RES = os.path.join(HERE, 'results')


def baseline(df):
    rows = []
    y = df[GDM_TARGET]
    for c in df.columns:
        if c in (GDM_TARGET, 'Case Number'):
            continue
        x = df[c]
        miss = 100 * x.isna().mean()
        binary = x.dropna().nunique() <= 2
        g1, g0 = x[y == 1].dropna(), x[y == 0].dropna()
        if binary:
            f = lambda v: f"{int(v.sum())} ({100 * v.mean():.1f}%)"
            tab = pd.crosstab(x, y)
            p = stats.chi2_contingency(tab)[1] if tab.shape == (2, 2) else np.nan
        else:
            f = lambda v: f"{v.mean():.1f} ± {v.std():.1f}"
            p = stats.mannwhitneyu(g1, g0).pvalue
        rows.append(dict(variable=c, type='binary' if binary else 'continuous',
                         gdm=f(g1), no_gdm=f(g0), p_value=p, missing_pct=miss))
    return pd.DataFrame(rows)


def cv_auc(X, y, n_repeats=10, n_folds=5, base_seed=2026):
    aucs = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_folds, shuffle=True, random_state=base_seed + r)
        for tr, te in skf.split(X, y):
            m = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(),
                              LogisticRegression(max_iter=2000))
            m.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return np.array(aucs)


def main():
    df = pd.read_excel(DEFAULT_GDM_PATH)
    df = df.drop(columns=[c for c in ['Case Number'] if c in df.columns])
    y = df[GDM_TARGET].astype(int).values
    b = baseline(df)
    b.to_csv(os.path.join(RES, 'gdm_baseline.csv'), index=False)

    feats = [c for c in df.columns if c != GDM_TARGET]
    rows = []
    for c in feats:
        a = cv_auc(df[[c]].values.astype(float), y)
        rows.append(dict(predictor=c, auroc_mean=a.mean(), auroc_sd=a.std()))
    sf = pd.DataFrame(rows).sort_values('auroc_mean', ascending=False)
    sf.to_csv(os.path.join(RES, 'gdm_single_feature.csv'), index=False)
    print(sf.round(3).to_string(index=False))

    early = [c for c in feats if c != 'OGTT']
    full = cv_auc(df[early].values.astype(float), y)
    rows = [dict(dropped='(none)', auroc_mean=full.mean(), auroc_sd=full.std(), delta=0.0)]
    for c in early:
        keep = [f for f in early if f != c]
        a = cv_auc(df[keep].values.astype(float), y)
        rows.append(dict(dropped=c, auroc_mean=a.mean(), auroc_sd=a.std(),
                         delta=(a - full).mean()))
    do = pd.DataFrame(rows).sort_values('delta')
    do.to_csv(os.path.join(RES, 'gdm_drop_one.csv'), index=False)
    print(do.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
