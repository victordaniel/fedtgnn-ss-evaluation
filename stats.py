"""Prespecified statistical analysis (see PROTOCOL.md).

For each dataset x scarcity level, FedTGNN-SS is compared with every
baseline on paired (repeat, fold) results:
  * mean difference with a 95% CI from the Nadeau-Bengio corrected
    resampled t-test (accounts for overlapping training sets in CV),
  * two-sided exact/normal-approximation Wilcoxon signed-rank p-value,
  * Holm adjustment within each hypothesis family
    (family = dataset x metric: all baselines x all scarcity levels),
  * effect size: matched-pairs rank-biserial correlation.

Usage: python stats.py results/main.csv --out results/stats_main.csv
"""

import argparse

import numpy as np
import pandas as pd
from scipy import stats

PROPOSED = 'FedTGNN-SS'
PRIMARY = dict(dataset='gdm_early', rho=0.8, metric='auroc', baseline='FedAvg-LR')
METRICS = {'auroc': +1, 'auprc': +1, 'macro_f1': +1, 'brier': -1}


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.maximum.accumulate((m - np.arange(m)) * p[ok][idx])
    tmp = np.empty(m)
    tmp[idx] = np.minimum(adj, 1.0)
    out[ok] = tmp
    return out


def nb_corrected_t(diff, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test."""
    k = len(diff)
    mean = diff.mean()
    var = diff.var(ddof=1)
    if var == 0:
        return mean, (mean, mean), 1.0 if mean == 0 else 0.0
    se = np.sqrt((1 / k + n_test / n_train) * var)
    t = mean / se
    p = 2 * stats.t.sf(abs(t), k - 1)
    h = stats.t.ppf(0.975, k - 1) * se
    return mean, (mean - h, mean + h), p


def rank_biserial(diff):
    d = diff[diff != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()


def compare(df):
    rows = []
    keys = ['repeat', 'fold']
    for (ds, rho), g in df.groupby(['dataset', 'rho']):
        prop = g[g.method == PROPOSED].set_index(keys)
        n_train = g['n_train'].iloc[0]
        n_test = g['n_test'].iloc[0]
        for bl in sorted(set(g.method) - {PROPOSED}):
            b = g[g.method == bl].set_index(keys)
            common = prop.index.intersection(b.index)
            if len(common) < 5:
                continue
            for met, sign in METRICS.items():
                a, c = prop.loc[common, met].values, b.loc[common, met].values
                diff = a - c
                mean, ci, p_t = nb_corrected_t(diff, n_train, n_test)
                try:
                    p_w = stats.wilcoxon(a, c, alternative='two-sided').pvalue
                except ValueError:
                    p_w = 1.0
                rows.append(dict(dataset=ds, rho=rho, metric=met, baseline=bl, n_pairs=len(common),
                                 proposed_mean=a.mean(), baseline_mean=c.mean(), diff=mean,
                                 ci_low=ci[0], ci_high=ci[1], p_corrected_t=p_t,
                                 p_wilcoxon=p_w, rank_biserial=rank_biserial(diff),
                                 better_direction='higher' if sign > 0 else 'lower'))
    out = pd.DataFrame(rows)
    for col in ('p_corrected_t', 'p_wilcoxon'):
        out[col + '_holm'] = np.nan
        for _, idx in out.groupby(['dataset', 'metric']).groups.items():
            out.loc[idx, col + '_holm'] = holm(out.loc[idx, col].values)
    sign = out.metric.map(METRICS)
    out['sig_better'] = (out.p_corrected_t_holm < 0.05) & (out['diff'] * sign > 0)
    out['sig_worse'] = (out.p_corrected_t_holm < 0.05) & (out['diff'] * sign < 0)
    return out


def summarise(df):
    num = ['auroc', 'auprc', 'sensitivity', 'specificity', 'ppv', 'npv', 'f1_pos', 'macro_f1',
           'brier', 'cal_intercept', 'cal_slope', 'ece', 'train_time_s', 'comm_bytes']
    g = df.groupby(['dataset', 'rho', 'method'])[num]
    m, s, n = g.mean(), g.std(), g.count()['auroc']
    out = m.add_suffix('_mean').join(s.add_suffix('_sd'))
    out['n_runs'] = n
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.results)
    comp = compare(df)
    comp.to_csv(args.out, index=False)
    summarise(df).to_csv(args.out.replace('.csv', '_summary.csv'), index=False)
    pr = comp[(comp.dataset == PRIMARY['dataset']) & (comp.rho == PRIMARY['rho']) &
              (comp.metric == PRIMARY['metric']) & (comp.baseline == PRIMARY['baseline'])]
    print('PRIMARY ANALYSIS:')
    print(pr.to_string(index=False))
    print('\nHolm-significant differences (corrected t-test), by dataset/metric:')
    print(comp.groupby(['dataset', 'metric'])[['sig_better', 'sig_worse']].sum())


if __name__ == '__main__':
    main()
