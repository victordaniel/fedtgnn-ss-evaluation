"""Generate every manuscript table (LaTeX), figure (PDF) and number macro
directly from saved results, so no number is typed by hand.

Outputs go to ../mdpi/generated/ :
  tables/*.tex, figures/*.pdf, macros.tex
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

from fedtgnn.metrics import net_benefit   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
OUT = os.path.join(HERE, '..', 'mdpi', 'generated')
TAB, FIG = os.path.join(OUT, 'tables'), os.path.join(OUT, 'figures')
for d in (TAB, FIG):
    os.makedirs(d, exist_ok=True)

DS_NAME = {'gdm_early': 'GDM (early, no OGTT)', 'gdm_diag': 'GDM (diagnostic, with OGTT)',
           'pima': 'Pima Indians', 'early': 'Early-Stage Diabetes'}
GROUPS = [
    ('Proposed', ['FedTGNN-SS']),
    ('Federated, supervised', ['FedAvg-LR', 'FedProx-LR', 'FedAvg-MLP',
                               'FedEns-RF', 'FedEns-XGB', 'FedEns-SVM']),
    ('Federated, semi-supervised (no graph)', ['FedST-XGB', 'FedMatch-tab', 'RSCFed-tab']),
    ('Federated GNN (supervised)', ['FedAvg-GCN', 'FedAvg-SAGE']),
    ('Local only (no federation)', ['Local-LR', 'Local-GCN', 'Local-TGNN']),
    ('Centralised, non-private reference', ['Cent-LR', 'Cent-XGB', 'Cent-GRAND']),
]
RHOS = [0.1, 0.3, 0.5, 0.7, 0.8]

# print palette (reference categorical slots 1-3, light mode) + text inks
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.6,
                     'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})

macros = []


def macro(name, value):
    macros.append(f'\\newcommand{{\\{name}}}{{{value}}}')


def _label(m):
    """Display name of an ablation variant (the prediction head is not
    called 'calibrated', since it is not)."""
    return m.replace('calibrated LR head', 'logistic-regression head').replace('&', r'\&')


def fmt_p(p):
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def load(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def ms(x):
    return f'{x.mean():.3f} $\\pm$ {x.std():.3f}'


# ----------------------------------------------------------------- tables
def table_main(df, stats, ds, metric='auroc', label=None):
    d = df[(df.dataset == ds)]
    st = stats[(stats.dataset == ds) & (stats.metric == metric)] if stats is not None else None
    lines = [r'\begin{tabularx}{\fulllength}{L' + 'C' * len(RHOS) + '}', r'\toprule',
             r'\textbf{Method} & ' + ' & '.join(rf'\textbf{{$\rho={r:g}$}}' for r in RHOS) + r'\\',
             r'\midrule']
    best = {r: d[d.rho == r].groupby('method')[metric].mean().max() for r in RHOS}
    for gname, methods in GROUPS:
        lines.append(rf'\multicolumn{{{len(RHOS) + 1}}}{{l}}{{\textit{{{gname}}}}}\\')
        for m in methods:
            cells = []
            for r in RHOS:
                x = d[(d.method == m) & (d.rho == r)][metric]
                if len(x) == 0:
                    cells.append('--')
                    continue
                c = ms(x)
                if abs(x.mean() - best[r]) < 5e-4:
                    c = r'\underline{' + c + '}'
                if st is not None and m != 'FedTGNN-SS':
                    row = st[(st.baseline == m) & (st.rho == r)]
                    if len(row) and row.p_corrected_t_holm.iloc[0] < 0.05:
                        c += r'$^{\dagger}$' if row['diff'].iloc[0] > 0 else r'$^{\ddagger}$'
                cells.append(c)
            name = r'\textbf{FedTGNN-SS}' if m == 'FedTGNN-SS' else m
            lines.append(name + ' & ' + ' & '.join(cells) + r'\\')
    lines += [r'\bottomrule', r'\end{tabularx}']
    with open(os.path.join(TAB, label or f'main_{ds}_{metric}.tex'), 'w') as f:
        f.write('\n'.join(lines))


def table_clinical(df, ds, rho=0.8):
    d = df[(df.dataset == ds) & (df.rho == rho)]
    cols = [('auprc', 'AUPRC'), ('sensitivity', 'Sens.'), ('specificity', 'Spec.'),
            ('ppv', 'PPV'), ('npv', 'NPV'), ('macro_f1', 'Macro-F1'), ('brier', 'Brier'),
            ('cal_intercept', 'Cal. int.'), ('cal_slope', 'Cal. slope')]
    lines = [r'\begin{tabularx}{\fulllength}{L' + 'C' * len(cols) + '}', r'\toprule',
             r'\textbf{Method} & ' + ' & '.join(rf'\textbf{{{c}}}' for _, c in cols) + r'\\',
             r'\midrule']
    for _, methods in GROUPS:
        for m in methods:
            x = d[d.method == m]
            if len(x) == 0:
                continue
            name = r'\textbf{FedTGNN-SS}' if m == 'FedTGNN-SS' else m
            lines.append(name + ' & ' + ' & '.join(f'{x[c].mean():.3f}' for c, _ in cols) + r'\\')
    lines += [r'\bottomrule', r'\end{tabularx}']
    with open(os.path.join(TAB, f'clinical_{ds}.tex'), 'w') as f:
        f.write('\n'.join(lines))


def table_counts(df):
    rows = []
    for ds in ['gdm_early', 'pima', 'early']:
        d = df[(df.dataset == ds) & (df.method == 'FedTGNN-SS')]
        if len(d) == 0:
            continue
        S = int(d.n_silos.iloc[0])
        for r in RHOS:
            x = d[d.rho == r]
            if len(x) == 0:
                continue
            silo = '; '.join(f"{x[f'silo{s}_n'].mean():.0f} ({100 * (x[f'silo{s}_pos'] / x[f'silo{s}_n']).mean():.0f}\\%)"
                             for s in range(S))
            rows.append(f"{DS_NAME[ds] if r == RHOS[0] else ''} & {r:g} & {x.n_train.mean():.0f} & "
                        f"{x.n_labeled.mean():.0f} ({100 * (x.n_labeled / x.n_train).mean():.1f}\\%) & "
                        f"{x.n_val.mean():.0f} & {x.n_test.mean():.0f} & {silo}\\\\")
    lines = [r'\begin{tabularx}{\fulllength}{lcccccL}', r'\toprule',
             r'\textbf{Dataset} & $\boldsymbol{\rho}$ & \textbf{Train} & \textbf{Labelled (\%)} & '
             r'\textbf{Val.} & \textbf{Test} & \textbf{Silo sizes (positive \%)}\\', r'\midrule'] + rows + \
            [r'\bottomrule', r'\end{tabularx}']
    with open(os.path.join(TAB, 'counts.tex'), 'w') as f:
        f.write('\n'.join(lines))


def table_ablation(ab):
    if ab is None:
        return
    for ds in ab.dataset.unique():
        d = ab[ab.dataset == ds]
        rhos = sorted(d.rho.unique())
        full = {r: d[(d.method == 'Full') & (d.rho == r)].set_index(['repeat', 'fold']) for r in rhos}
        lines = [r'\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{5.2cm}' + 'CC' * len(rhos) + '}',
                 r'\toprule',
                 r'\textbf{Configuration} & ' + ' & '.join(
                     rf'\multicolumn{{2}}{{c}}{{$\rho={r:g}$}}' for r in rhos) + r'\\',
                 ' & ' + ' & '.join(r'AUROC & $\Delta$ (95\% CI)' for _ in rhos) + r'\\', r'\midrule']
        for m in d.method.unique():
            cells = []
            for r in rhos:
                x = d[(d.method == m) & (d.rho == r)].set_index(['repeat', 'fold'])
                common = x.index.intersection(full[r].index)
                diff = x.loc[common, 'auroc'] - full[r].loc[common, 'auroc']
                if m == 'Full':
                    cells += [f"{x.auroc.mean():.3f}", '--']
                else:
                    h = 1.96 * diff.std() / np.sqrt(len(diff))
                    cells += [f"{x.auroc.mean():.3f}", f"{diff.mean():+.3f} ({diff.mean() - h:+.3f}, {diff.mean() + h:+.3f})"]
            lines.append(_label(m) + ' & ' + ' & '.join(cells) + r'\\')
        lines += [r'\bottomrule', r'\end{tabularx}']
        with open(os.path.join(TAB, f'ablation_{ds}.tex'), 'w') as f:
            f.write('\n'.join(lines))


def table_setting(df, name):
    """Heterogeneity / clients: AUROC of FedTGNN-SS vs FedAvg-LR and the best other method."""
    if df is None:
        return
    lines = [r'\begin{tabularx}{\textwidth}{llCCC}', r'\toprule',
             r'\textbf{Dataset} & \textbf{Setting} & \textbf{FedTGNN-SS} & \textbf{FedAvg-LR} & '
             r'\textbf{Best other federated method}\\', r'\midrule']
    fed = [m for _, ms_ in GROUPS[:4] for m in ms_ if m != 'FedTGNN-SS']
    for ds in df.dataset.unique():
        for s in df[df.dataset == ds].setting.unique():
            d = df[(df.dataset == ds) & (df.setting == s)]
            g = d.groupby('method').auroc
            best = g.mean()[fed].idxmax()
            lines.append(f"{DS_NAME[ds]} & {s} & {ms(d[d.method == 'FedTGNN-SS'].auroc)} & "
                         f"{ms(d[d.method == 'FedAvg-LR'].auroc)} & {best}: {ms(d[d.method == best].auroc)}\\\\")
    lines += [r'\bottomrule', r'\end{tabularx}']
    with open(os.path.join(TAB, f'{name}.tex'), 'w') as f:
        f.write('\n'.join(lines))


def table_sweep(sw):
    if sw is None:
        return
    lines = [r'\begin{tabularx}{\textwidth}{llCC}', r'\toprule',
             r'\textbf{Dataset} & \textbf{Setting} & \textbf{AUROC} & \textbf{Brier}\\', r'\midrule']
    for ds in sw.dataset.unique():
        for m in sw[sw.dataset == ds].method.unique():
            x = sw[(sw.dataset == ds) & (sw.method == m)]
            lines.append(f"{DS_NAME[ds]} & \\texttt{{{m.replace('_', '\\_')}}} & {ms(x.auroc)} & {ms(x.brier)}\\\\")
    lines += [r'\bottomrule', r'\end{tabularx}']
    with open(os.path.join(TAB, 'sweep.tex'), 'w') as f:
        f.write('\n'.join(lines))


def table_cost(df):
    d = df[(df.dataset == 'gdm_early') & (df.rho == 0.8)]
    lines = [r'\begin{tabularx}{\textwidth}{LCCC}', r'\toprule',
             r'\textbf{Method} & \textbf{Training time (s)} & \textbf{Communication (MB)} & \textbf{MIA AUROC}\\',
             r'\midrule']
    for _, methods in GROUPS[:5]:
        for m in methods:
            x = d[d.method == m]
            if len(x) == 0:
                continue
            comm = '--' if x.comm_bytes.isna().all() else f"{x.comm_bytes.mean() / 1e6:.3f}"
            mia = ms(x.mia_auc) if 'mia_auc' in x and x.mia_auc.notna().any() else '--'
            lines.append(f"{m} & {x.train_time_s.mean():.1f} & {comm} & {mia}\\\\")
    lines += [r'\bottomrule', r'\end{tabularx}']
    with open(os.path.join(TAB, 'cost.tex'), 'w') as f:
        f.write('\n'.join(lines))


SUM_DS = [('gdm_early', 'GDM-early'), ('gdm_diag', 'GDM-diag.'), ('pima', 'Pima'), ('early', 'Early-Stage')]


def table_summary(df, stats, rho=0.8, metric='auroc', name='summary_auroc'):
    """Main-text table: every method x every dataset at one scarcity level."""
    dss = [(d, n) for d, n in SUM_DS if d in df.dataset.unique()]
    d = df[df.rho == rho]
    best = {ds: d[d.dataset == ds].groupby('method')[metric].mean().max() for ds, _ in dss}
    lines = [r'\begin{tabularx}{\textwidth}{L' + 'C' * len(dss) + '}', r'\toprule',
             r'\textbf{Method} & ' + ' & '.join(rf'\textbf{{{n}}}' for _, n in dss) + r'\\', r'\midrule']
    for gname, methods in GROUPS:
        lines.append(rf'\multicolumn{{{len(dss) + 1}}}{{l}}{{\textit{{{gname}}}}}\\')
        for m in methods:
            cells = []
            for ds, _ in dss:
                x = d[(d.dataset == ds) & (d.method == m)][metric]
                c = ms(x)
                if abs(x.mean() - best[ds]) < 5e-4:
                    c = r'\underline{' + c + '}'
                if stats is not None and m != 'FedTGNN-SS':
                    r_ = stats[(stats.dataset == ds) & (stats.metric == metric) & (stats.rho == rho)
                               & (stats.baseline == m)]
                    if len(r_) and r_.p_corrected_t_holm.iloc[0] < 0.05:
                        c += r'$^{\dagger}$' if r_['diff'].iloc[0] > 0 else r'$^{\ddagger}$'
                cells.append(c)
            nm = r'\textbf{FedTGNN-SS}' if m == 'FedTGNN-SS' else m
            lines.append(nm + ' & ' + ' & '.join(cells) + r'\\')
    lines += [r'\bottomrule', r'\end{tabularx}']
    with open(os.path.join(TAB, f'{name}.tex'), 'w') as f:
        f.write('\n'.join(lines))


def table_primary_contrasts(stats, rho=0.8):
    """Paired AUROC differences FedTGNN-SS minus selected comparators."""
    sel = ['FedAvg-LR', 'FedAvg-MLP', 'FedEns-RF', 'FedAvg-GCN', 'Local-TGNN']
    lines = [r'\begin{tabularx}{\textwidth}{llCCC}', r'\toprule',
             r'\textbf{Dataset} & \textbf{Comparator} & \textbf{Difference (95\% CI)} & \textbf{$p$} & '
             r'\textbf{$p$ (Holm)}\\', r'\midrule']
    for ds, n in SUM_DS:
        s = stats[(stats.dataset == ds) & (stats.metric == 'auroc') & (stats.rho == rho)]
        for i, b in enumerate(sel):
            r_ = s[s.baseline == b]
            if not len(r_):
                continue
            r_ = r_.iloc[0]
            lines.append(f"{n if i == 0 else ''} & {b} & {r_['diff']:+.3f} ({r_.ci_low:+.3f}, {r_.ci_high:+.3f}) & "
                         f"{fmt_p(r_.p_corrected_t)} & {fmt_p(r_.p_corrected_t_holm)}\\\\")
        lines.append(r'\midrule' if ds != SUM_DS[-1][0] else '')
    lines += [r'\bottomrule', r'\end{tabularx}']
    with open(os.path.join(TAB, 'contrasts.tex'), 'w') as f:
        f.write('\n'.join(l for l in lines if l))


def table_ablation_summary(ab, rho=0.8):
    if ab is None:
        return
    dss = [(d, n) for d, n in SUM_DS if d in ab.dataset.unique()]
    lines = [r'\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{4.6cm}' + 'C' * len(dss) + '}',
             r'\toprule',
             r'\textbf{Configuration} & ' + ' & '.join(rf'\textbf{{{n}}}' for _, n in dss) + r'\\', r'\midrule']
    methods = list(dict.fromkeys(ab.method))
    for m in methods:
        cells = []
        for ds, _ in dss:
            d = ab[(ab.dataset == ds) & (ab.rho == rho)]
            full = d[d.method == 'Full'].set_index(['repeat', 'fold']).auroc
            x = d[d.method == m].set_index(['repeat', 'fold']).auroc
            if m == 'Full':
                cells.append(f'{x.mean():.3f}')
                continue
            diff = (x - full).dropna()
            h = 1.96 * diff.std() / np.sqrt(len(diff))
            cells.append(f'{diff.mean():+.3f} ({diff.mean() - h:+.3f}, {diff.mean() + h:+.3f})')
        lines.append(_label(m) + ' & ' + ' & '.join(cells) + r'\\')
        if m == 'Full':
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabularx}']
    with open(os.path.join(TAB, 'ablation_summary.tex'), 'w') as f:
        f.write('\n'.join(lines))


def _tex(s):
    return str(s).replace('%', r'\%').replace('±', r'$\pm$').replace('_', r'\_')


def table_gdm_baseline():
    b = load('gdm_baseline.csv')
    if b is None:
        return
    lines = [r'\begin{tabularx}{\textwidth}{LCCCC}', r'\toprule',
             r'\textbf{Predictor} & \textbf{GDM ($n=1372$)} & \textbf{No GDM ($n=2153$)} & '
             r'\textbf{$p$} & \textbf{Missing (\%)}\\', r'\midrule']
    for _, r in b.iterrows():
        p = '<0.001' if r.p_value < 0.001 else f'{r.p_value:.3f}'
        lines.append(f"{_tex(r.variable)} & {_tex(r.gdm)} & {_tex(r.no_gdm)} & {p} & {r.missing_pct:.0f}\\\\")
    lines += [r'\bottomrule', r'\end{tabularx}']
    with open(os.path.join(TAB, 'gdm_baseline.tex'), 'w', encoding='utf8') as f:
        f.write('\n'.join(lines))


def table_gdm_predictors():
    sf, do = load('gdm_single_feature.csv'), load('gdm_drop_one.csv')
    if sf is None or do is None:
        return
    full = do[do.dropped == '(none)'].auroc_mean.iloc[0]
    do = do.set_index('dropped')
    lines = [r'\begin{tabularx}{\textwidth}{LCC}', r'\toprule',
             r'\textbf{Predictor} & \textbf{AUROC of predictor alone} & '
             rf'\textbf{{Change on removal (all early predictors: {full:.3f})}}\\', r'\midrule']
    for _, r in sf.iterrows():
        rem = '--' if r.predictor == 'OGTT' else f"{do.loc[r.predictor, 'delta']:+.4f}"
        lines.append(f"{_tex(r.predictor)} & {r.auroc_mean:.3f} $\\pm$ {r.auroc_sd:.3f} & {rem}\\\\")
    lines += [r'\bottomrule', r'\end{tabularx}']
    with open(os.path.join(TAB, 'gdm_predictors.tex'), 'w', encoding='utf8') as f:
        f.write('\n'.join(lines))


def table_leakage():
    lk = load('leakage.csv')
    if lk is None:
        return
    from stats import nb_corrected_t
    dss = [(d, n) for d, n in SUM_DS if d in lk.dataset.unique()]
    rhos = sorted(lk.rho.unique())
    cols = [(d, n, r) for d, n in dss for r in rhos]
    lines = [r'\begin{tabularx}{\fulllength}{>{\raggedright\arraybackslash}p{4.2cm}' + 'C' * len(cols) + '}',
             r'\toprule',
             r'\textbf{Evaluation} & ' + ' & '.join(rf'\textbf{{{n}, $\rho={r:g}$}}' for _, n, r in cols) + r'\\',
             r'\midrule']
    ref_of = lambda m: 'FedAvg-LR, correct protocol' if m.startswith('FedAvg-LR') else 'Correct protocol (this study)'
    for m in list(dict.fromkeys(lk.method)):
        cells = []
        for d, _, r in cols:
            x = lk[(lk.dataset == d) & (lk.rho == r)]
            a = x[x.method == m].set_index(['repeat', 'fold'])
            b = x[x.method == ref_of(m)].set_index(['repeat', 'fold'])
            if m == ref_of(m):
                cells.append(f'{a.auroc.mean():.3f}')
                continue
            common = a.index.intersection(b.index)
            diff = (a.loc[common, 'auroc'] - b.loc[common, 'auroc']).values
            mean, ci, p = nb_corrected_t(diff, a.n_train.iloc[0], a.n_test.iloc[0])
            cells.append(f'{mean:+.3f} ({ci[0]:+.3f}, {ci[1]:+.3f})')
        lines.append(m + ' & ' + ' & '.join(cells) + r'\\')
        if m in ('All four shortcuts',):
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabularx}']
    with open(os.path.join(TAB, 'leakage.tex'), 'w', encoding='utf8') as f:
        f.write('\n'.join(lines))


def table_recal():
    s = load('recal_summary.csv')
    if s is None:
        return
    lines = [r'\begin{tabularx}{\textwidth}{llLCCCC}', r'\toprule',
             r'\textbf{Dataset} & \textbf{Model} & \textbf{Recalibration} & \textbf{Brier} & '
             r'\textbf{Cal. intercept} & \textbf{Cal. slope} & \textbf{ECE}\\', r'\midrule']
    order = ['None', 'Platt scaling', 'Temperature scaling']
    for ds, n in SUM_DS:
        d = s[s.dataset == ds]
        if d.empty:
            continue
        first = True
        for m in ['FedTGNN-SS', 'FedAvg-LR']:
            for i, rc in enumerate(order):
                x = d[(d.method == m) & (d.recalibration == rc)]
                if x.empty:
                    continue
                x = x.iloc[0]
                lines.append(f"{n if first else ''} & {m if i == 0 else ''} & {rc} & "
                             f"{x.brier_mean:.3f} & {x.cal_intercept_mean:+.2f} & {x.cal_slope_mean:.2f} & "
                             f"{x.ece_mean:.3f}\\\\")
                first = False
        lines.append(r'\midrule')
    if lines[-1] == r'\midrule':
        lines.pop()
    lines += [r'\bottomrule', r'\end{tabularx}']
    with open(os.path.join(TAB, 'recal.tex'), 'w') as f:
        f.write('\n'.join(lines))


def table_graph():
    q, k, m = load('graph_quality.csv'), load('graph_knn.csv'), load('main.csv')
    if q is None or k is None:
        return
    lines = [r'\begin{tabularx}{\textwidth}{llCCCCCC}', r'\toprule',
             r'\textbf{Dataset} & \textbf{Graph} & \textbf{Edge homophily} & \textbf{Chance} & '
             r'\textbf{Purity, positives} & \textbf{Purity, negatives} & \textbf{Labelled-neighbour agreement} & '
             r'\textbf{Unlabelled with labelled neighbour}\\', r'\midrule']
    for ds, n in SUM_DS:
        d = q[(q.dataset == ds) & (q.rho == 0.8)]
        if d.empty:
            continue
        for i, gname in enumerate(['initial', 'refined']):
            x = d[d.graph == gname].mean(numeric_only=True)
            lines.append(f"{n if i == 0 else ''} & {'feature space' if gname == 'initial' else 'refined (embeddings)'} & "
                         f"{x.edge_homophily:.3f} & {x.random_homophily:.3f} & {x.purity_pos:.3f} & "
                         f"{x.purity_neg:.3f} & {x.labelled_nb_purity:.3f} & {x.unl_with_labelled_nb:.3f}\\\\")
    lines += [r'\bottomrule', r'\end{tabularx}']
    with open(os.path.join(TAB, 'graph_quality.tex'), 'w') as f:
        f.write('\n'.join(lines))
    # local-neighbourhood models vs tabular models, same runs (repeats 0-2)
    sel = ['FedAvg-LR', 'FedAvg-MLP', 'FedAvg-GCN', 'FedAvg-SAGE', 'FedTGNN-SS']
    lines = [r'\begin{tabularx}{\textwidth}{lC' + 'C' * (len(sel) + 1) + '}', r'\toprule',
             r'\textbf{Dataset} & $\boldsymbol{\rho}$ & \textbf{k-NN} & ' + ' & '.join(rf'\textbf{{{s}}}' for s in sel) + r'\\',
             r'\midrule']
    for ds, n in SUM_DS:
        for r in (0.1, 0.8):
            kk = k[(k.dataset == ds) & (k.rho == r)]
            if kk.empty:
                continue
            mm = m[(m.dataset == ds) & (m.rho == r) & (m.repeat < 3)]
            cells = [f'{kk.auroc.mean():.3f}'] + [f'{mm[mm.method == s].auroc.mean():.3f}' for s in sel]
            lines.append(f"{n} & {r:g} & " + ' & '.join(cells) + r'\\')
    lines += [r'\bottomrule', r'\end{tabularx}']
    with open(os.path.join(TAB, 'graph_knn.tex'), 'w') as f:
        f.write('\n'.join(lines))


def table_mar():
    a = load('mar.csv')
    if a is None:
        return
    meths = ['FedTGNN-SS', 'FedAvg-LR', 'FedAvg-MLP', 'FedEns-RF', 'FedAvg-GCN', 'FedMatch-tab', 'Local-TGNN']
    lines = [r'\begin{tabularx}{\fulllength}{ll' + 'C' * len(meths) + '}', r'\toprule',
             r'\textbf{Dataset} & $\boldsymbol{\beta}$ & ' + ' & '.join(rf'\textbf{{{x}}}' for x in meths) + r'\\',
             r'\midrule']
    for ds, n in SUM_DS:
        for i, st in enumerate(['risk_beta=1', 'risk_beta=2.5']):
            d = a[(a.dataset == ds) & (a.setting == st)]
            if d.empty:
                continue
            best = d.groupby('method').auroc.mean().max()
            cells = []
            for x in meths:
                v = d[d.method == x].auroc.mean()
                c = f'{v:.3f}'
                cells.append(r'\underline{' + c + '}' if abs(v - best) < 5e-4 else c)
            lines.append(f"{n if i == 0 else ''} & {st.split('=')[1]} & " + ' & '.join(cells) + r'\\')
    lines += [r'\bottomrule', r'\end{tabularx}']
    with open(os.path.join(TAB, 'mar.tex'), 'w') as f:
        f.write('\n'.join(lines))


def table_mar_selection():
    """Outcome prevalence among labelled vs unlabelled training patients
    under random (main study) and risk-dependent label availability."""
    a, m = load('mar.csv'), load('main.csv')
    if a is None:
        return

    def prev(d):
        S = int(d.n_silos.iloc[0])
        lp = sum(d[f'silo{s}_labeled_pos'].sum() for s in range(S))
        ln = sum(d[f'silo{s}_labeled'].sum() for s in range(S))
        tp = sum(d[f'silo{s}_pos'].sum() for s in range(S))
        tn = sum(d[f'silo{s}_n'].sum() for s in range(S))
        return lp / ln, (tp - lp) / (tn - ln), ln / tn

    lines = [r'\begin{tabularx}{\textwidth}{llCCC}', r'\toprule',
             r'\textbf{Dataset} & \textbf{Label availability} & \textbf{Prevalence, labelled} & '
             r'\textbf{Prevalence, unlabelled} & \textbf{Share labelled}\\', r'\midrule']
    for ds, n in SUM_DS:
        base = m[(m.dataset == ds) & (m.rho == 0.8) & (m.method == 'FedTGNN-SS') & (m.repeat < 5)]
        if base.empty or a[a.dataset == ds].empty:
            continue
        rows = [('random within class (main study)', base)]
        for st, lab in [('risk_beta=1', r'risk-dependent, $\beta=1.0$'), ('risk_beta=2.5', r'risk-dependent, $\beta=2.5$')]:
            rows.append((lab, a[(a.dataset == ds) & (a.setting == st) & (a.method == 'FedTGNN-SS')]))
        for i, (lab, d) in enumerate(rows):
            pl, pu, sh = prev(d)
            lines.append(f"{n if i == 0 else ''} & {lab} & {pl:.3f} & {pu:.3f} & {sh:.3f}\\\\")
    lines += [r'\bottomrule', r'\end{tabularx}']
    with open(os.path.join(TAB, 'mar_selection.tex'), 'w') as f:
        f.write('\n'.join(lines))


def table_tuned():
    t, m = load('tuned.csv'), load('main.csv')
    if t is None:
        return
    base = ['FedAvg-LR', 'FedAvg-MLP', 'FedEns-RF', 'FedEns-XGB', 'FedEns-SVM']
    lines = [r'\begin{tabularx}{\textwidth}{llCCC}', r'\toprule',
             r'\textbf{Dataset, $\rho$} & \textbf{Method} & \textbf{Default} & \textbf{Tuned on validation} & '
             r'\textbf{FedTGNN-SS minus tuned}\\', r'\midrule']
    from stats import nb_corrected_t
    for ds, n in SUM_DS:
        for r in (0.1, 0.8):
            mm = m[(m.dataset == ds) & (m.rho == r) & (m.repeat < 5)]
            tt = t[(t.dataset == ds) & (t.rho == r)]
            if tt.empty:
                continue
            prop = mm[mm.method == 'FedTGNN-SS'].set_index(['repeat', 'fold'])
            for i, b in enumerate(base):
                x = tt[tt.method == f'{b} (tuned)'].set_index(['repeat', 'fold'])
                c = prop.index.intersection(x.index)
                diff = (prop.loc[c, 'auroc'] - x.loc[c, 'auroc']).values
                mean, ci, _ = nb_corrected_t(diff, prop.n_train.iloc[0], prop.n_test.iloc[0])
                lines.append(f"{(n + ', ' + format(r, 'g')) if i == 0 else ''} & {b} & "
                             f"{mm[mm.method == b].auroc.mean():.3f} & {x.auroc.mean():.3f} & "
                             f"{mean:+.3f} ({ci[0]:+.3f}, {ci[1]:+.3f})\\\\")
            lines.append(r'\midrule')
    if lines[-1] == r'\midrule':
        lines.pop()
    lines += [r'\bottomrule', r'\end{tabularx}']
    with open(os.path.join(TAB, 'tuned.tex'), 'w') as f:
        f.write('\n'.join(lines))


# ---------------------------------------------------------------- figures
def _style(ax):
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def fig_auroc_vs_rho(df):
    dss = [d for d in ['gdm_early', 'pima', 'early'] if d in df.dataset.unique()]
    fig, axes = plt.subplots(1, len(dss), figsize=(7.2, 2.5))
    axes = np.atleast_1d(axes)
    fed = [m for _, ms_ in GROUPS[1:4] for m in ms_]
    for ax, ds in zip(axes, dss):
        d = df[df.dataset == ds]
        if d.empty:
            continue
        g = d.groupby(['method', 'rho']).auroc.mean().unstack()
        others = [m for m in fed if m != 'FedAvg-LR']
        best = g.loc[others].mean(1).idxmax()
        series = [('FedTGNN-SS', C1, 'o', 'FedTGNN-SS'), ('FedAvg-LR', C2, 's', 'FedAvg-LR'),
                  (best, C3, '^', 'Best other federated method')]
        for m, c, mk, lab in series:
            y = g.loc[m]
            se = d[d.method == m].groupby('rho').auroc.sem()
            ax.fill_between(y.index, y - 1.96 * se, y + 1.96 * se, color=c, alpha=0.12, linewidth=0)
            ax.plot(y.index, y.values, color=c, marker=mk, markersize=5, linewidth=1.5, label=lab)
        ax.annotate(best, (RHOS[0], g.loc[best].iloc[0]), xytext=(0, 6), textcoords='offset points',
                    fontsize=7, color=INK2)
        ax.set_title(DS_NAME[ds], fontsize=8.5, color=INK, pad=12)
        ax.set_xlabel(r'Fraction of training labels removed ($\rho$)')
        ax.set_xticks(RHOS)
        _style(ax)
    axes[0].set_ylabel('Test AUROC (mean, 95% CI)')
    handles, labels = [], []
    for ax in axes:
        h, l_ = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l_):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(FIG, 'auroc_vs_rho.pdf'), bbox_inches='tight')
    plt.close(fig)


def fig_calibration_dca(preds, ds='gdm_early', rho=0.8):
    if preds is None:
        return
    p = preds[(preds.rho == rho)]
    if p.empty:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for m, c, mk in [('FedTGNN-SS', C1, 'o'), ('FedAvg-LR', C2, 's')]:
        x = p[p.method == m]
        bins = np.linspace(0, 1, 11)
        b = np.clip(np.digitize(x.p, bins[1:-1]), 0, 9)
        obs = x.groupby(b).y.mean()
        pr = x.groupby(b).p.mean()
        a1.plot(pr.values, obs.values, color=c, marker=mk, markersize=5, linewidth=1.5, label=m)
        th = np.linspace(0.05, 0.8, 40)
        nb = np.mean([net_benefit(g.y.values, g.p.values, th) for _, g in x.groupby(['repeat', 'fold'])], 0)
        a2.plot(th, nb, color=c, linewidth=1.5, label=m)
    a1.plot([0, 1], [0, 1], color=INK2, linestyle='--', linewidth=0.8, label='Perfect calibration')
    a1.set_xlabel('Predicted risk (decile mean)')
    a1.set_ylabel('Observed GDM proportion')
    a1.set_title('(a) Calibration (pooled test folds)', fontsize=8.5, color=INK)
    x = p[p.method == 'FedTGNN-SS']
    th = np.linspace(0.05, 0.8, 40)
    prev = np.mean([g.y.mean() for _, g in x.groupby(['repeat', 'fold'])])
    a2.plot(th, prev - (1 - prev) * th / (1 - th), color=INK2, linestyle=':', linewidth=1, label='Treat all')
    a2.axhline(0, color=INK2, linewidth=0.8, label='Treat none')
    a2.set_ylim(-0.05, max(0.05, prev + 0.05))
    a2.set_xlabel('Threshold probability')
    a2.set_ylabel('Net benefit')
    a2.set_title('(b) Decision curve', fontsize=8.5, color=INK)
    for a in (a1, a2):
        _style(a)
        a.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'calibration_dca.pdf'), bbox_inches='tight')
    plt.close(fig)


def fig_pseudolabels(pl):
    if pl is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    cols = [(0.5, C1, 'o'), (0.8, C2, 's')]
    dss = [d for d in ['gdm_early', 'pima', 'early'] if d in pl.dataset.unique()]
    for ds, ls in zip(dss, ['-', '--', ':']):
        for rho, c, mk in cols:
            x = pl[(pl.dataset == ds) & (pl.rho == rho)]
            if len(x) == 0:
                continue
            g = x.groupby('round')[['n_pseudo', 'n_correct', 'n_unlabeled']].sum()
            axes[0].plot(g.index, g.n_correct / g.n_pseudo.clip(lower=1), color=c, marker=mk,
                         markersize=4, linestyle=ls, linewidth=1.2, label=DS_NAME[ds] + r', $\rho$ = ' + f'{rho:g}')
            axes[1].plot(g.index, g.n_pseudo / g.n_unlabeled, color=c, marker=mk, markersize=4,
                         linestyle=ls, linewidth=1.2)
    axes[0].set_ylabel('Pseudo-label precision')
    axes[1].set_ylabel('Coverage of unlabelled patients')
    for a in axes:
        a.set_xlabel('Federation round')
        _style(a)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3, frameon=False,
               fontsize=6.5, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(os.path.join(FIG, 'pseudolabels.pdf'), bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    df = load('main.csv')
    stats = load('stats_main.csv')
    preds = None
    pp = os.path.join(RES, 'main_preds.csv.gz')
    if os.path.exists(pp):
        preds = pd.read_csv(pp).drop_duplicates(['dataset', 'method', 'rho', 'repeat', 'fold', 'idx'])
    if df is None:
        print('no results yet')
        return
    df = df.drop_duplicates(['dataset', 'rho', 'repeat', 'fold', 'method'])
    for ds in df.dataset.unique():
        table_main(df, stats, ds, 'auroc')
        table_main(df, stats, ds, 'macro_f1')
        table_main(df, stats, ds, 'brier')
        table_clinical(df, ds)
    table_counts(df)
    table_cost(df)
    fig_auroc_vs_rho(df)
    if preds is not None:
        fig_calibration_dca(preds[(preds.dataset == 'gdm_early')
                                  & preds.method.isin(['FedTGNN-SS', 'FedAvg-LR'])])
    fig_pseudolabels(load('main_pseudolabels.csv'))
    table_ablation(load('ablation.csv'))
    table_ablation_summary(load('ablation.csv'))
    table_gdm_baseline()
    table_gdm_predictors()
    table_leakage()
    table_recal()
    table_graph()
    table_mar()
    table_mar_selection()
    table_tuned()
    if stats is not None:
        table_summary(df, stats, 0.8, 'auroc', 'summary_auroc')
        table_summary(df, stats, 0.8, 'brier', 'summary_brier')
        table_primary_contrasts(stats)
    table_setting(load('hetero.csv'), 'hetero')
    table_setting(load('clients.csv'), 'clients')
    table_sweep(load('sweep.csv'))

    # ---- number macros for the text (primary analysis etc.)
    n_runs = df.groupby(['dataset', 'rho', 'method']).size()
    macro('NRunsPrimary', int(n_runs.get(('gdm_early', 0.8, 'FedTGNN-SS'), 0)))
    for ds, tag in [('gdm_early', 'Gdm'), ('pima', 'Pima'), ('early', 'Early'), ('gdm_diag', 'Diag')]:
        for m, mt in [('FedTGNN-SS', 'Prop'), ('FedAvg-LR', 'Lr')]:
            x = df[(df.dataset == ds) & (df.rho == 0.8) & (df.method == m)]
            if len(x):
                macro(f'Auc{tag}{mt}', f'{x.auroc.mean():.3f}')
                macro(f'CalSlope{tag}{mt}', f'{x.cal_slope.mean():.2f}')
                if 'mia_auc' in x:
                    macro(f'Mia{tag}{mt}', f'{x.mia_auc.mean():.3f}')
    if stats is not None:
        pr = stats[(stats.dataset == 'gdm_early') & (stats.rho == 0.8) & (stats.metric == 'auroc')
                   & (stats.baseline == 'FedAvg-LR')]
        if len(pr):
            r = pr.iloc[0]
            macro('PrimDiff', f"{r['diff']:+.3f}")
            macro('PrimCI', f"{r.ci_low:+.3f} to {r.ci_high:+.3f}")
            macro('PrimP', fmt_p(r.p_corrected_t))
            macro('PrimPholm', fmt_p(r.p_corrected_t_holm))
        for ds, tag in [('gdm_early', 'Gdm'), ('pima', 'Pima'), ('early', 'Early')]:
            s = stats[(stats.dataset == ds) & (stats.metric == 'auroc')]
            macro(f'SigBetter{tag}', int(s.sig_better.sum()))
            macro(f'SigWorse{tag}', int(s.sig_worse.sum()))
            macro(f'NComp{tag}', len(s))
    with open(os.path.join(OUT, 'macros.tex'), 'w') as f:
        f.write('\n'.join(macros) + '\n')
    print('tables/figures written to', OUT)


if __name__ == '__main__':
    main()
