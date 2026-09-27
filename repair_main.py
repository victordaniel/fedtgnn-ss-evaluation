"""One-off repair of results/main.csv (documented in REVISION_NOTES.md).

1. Rows of 2-silo datasets were appended under a 3-silo header, which
   shifted the four overall count columns into silo2_*. Only count columns
   were affected (metrics precede them in every row).
2. Calibration intercept/slope were computed with an unbounded Newton
   iteration that diverged for some runs; they are recomputed from the saved
   test predictions with the bounded estimator now in fedtgnn/metrics.py.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fedtgnn.metrics import calibration   # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
path = os.path.join(RES, 'main.csv')
backup = os.path.join(RES, 'main_before_repair.csv')
if not os.path.exists(backup):
    pd.read_csv(path).to_csv(backup, index=False)
df = pd.read_csv(backup)       # always repair from the untouched original

bad = (df.n_silos == 2) & df.n_train.isna()
shift = ['silo2_n', 'silo2_pos', 'silo2_labeled', 'silo2_labeled_pos']
tgt = ['n_train', 'n_val', 'n_test', 'n_labeled']
df.loc[bad, tgt] = df.loc[bad, shift].values
df.loc[bad, shift] = np.nan
print('realigned rows:', int(bad.sum()))

preds = pd.read_csv(os.path.join(RES, 'main_preds.csv.gz'))
key = ['dataset', 'method', 'rho', 'repeat', 'fold']
rows = [dict(zip(key, k), cal_intercept=c[0], cal_slope=c[1])
        for k, g in preds.groupby(key) for c in [calibration(g.y.values, g.p.values)]]
cal = pd.DataFrame(rows)
for c in ('dataset', 'method'):
    cal[c] = cal[c].astype(str)
    df[c] = df[c].astype(str)
df = df.drop(columns=['cal_intercept', 'cal_slope']).merge(cal, on=key, how='left')
print('calibration recomputed for', int(df.cal_slope.notna().sum()), 'of', len(df), 'rows')
assert df.cal_slope.notna().all()
assert df.n_train.notna().all()
df.to_csv(path, index=False)
