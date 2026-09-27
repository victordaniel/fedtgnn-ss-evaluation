"""Experiment runner.

Examples
--------
Main comparison (prespecified primary analysis, see PROTOCOL.md):
    python run.py main --datasets gdm_early pima early gdm_diag

Heterogeneity / number of clients (rho = 0.8):
    python run.py hetero --datasets gdm_early pima early
    python run.py clients --datasets gdm_early

Ablation and sensitivity:
    python run.py ablation --datasets gdm_early pima early
    python run.py sweep --datasets gdm_early pima

Results are appended to results/<experiment>.csv (one row per
dataset x setting x repeat x fold x method); finished rows are skipped, so
an interrupted run can be resumed with the same command.
"""

import argparse
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEFAULT_SILOS = {'gdm_early': 3, 'gdm_diag': 3, 'pima': 2, 'early': 2}
SCARCITY = [0.1, 0.3, 0.5, 0.7, 0.8]
SILO_COLS = [f'silo{s}_{k}' for s in range(8) for k in ('n', 'pos', 'labeled', 'labeled_pos')]
RESULT_COLUMNS = (['experiment', 'dataset', 'n_silos', 'rho', 'setting', 'alpha', 'repeat', 'fold',
                   'seed', 'method', 'auroc', 'auprc', 'sensitivity', 'specificity', 'ppv', 'npv',
                   'f1_pos', 'macro_f1', 'brier', 'cal_intercept', 'cal_slope', 'ece', 'threshold',
                   'tp', 'tn', 'fp', 'fn', 'train_time_s', 'comm_bytes', 'n_pseudo', 'mia_auc']
                  + SILO_COLS + ['n_train', 'n_val', 'n_test', 'n_labeled'])
KEY = ['experiment', 'dataset', 'n_silos', 'alpha', 'rho', 'setting', 'repeat', 'fold', 'method']


def _alpha_str(a):
    return 'iid' if a is None else f'{a:g}'


def run_task(task):
    """Runs all requested methods for one (dataset, setting, rho, repeat, fold)."""
    import torch
    torch.set_num_threads(1)
    from fedtgnn import data, splits
    from fedtgnn.baselines import METHODS
    from fedtgnn.context import build_context
    from fedtgnn.fedtgnn import train_fedtgnn
    from fedtgnn.metrics import evaluate, mia_auc, youden_threshold

    X, y, _ = data.load(task['dataset'])
    sp = None
    for s in splits.make_splits(y, task['rho'], task['n_silos'], task['alpha'],
                                n_repeats=task['repeat'] + 1, n_folds=task['n_folds']):
        if s.repeat == task['repeat'] and s.fold == task['fold']:
            sp = s
            break
    splits.check_no_leakage(sp)
    rows, preds, diags = [], [], []
    ctx_cache = {}
    for job in task['jobs']:
        name, cfg = job[0], job[1]
        ctx_kw = job[2] if len(job) > 2 else {}
        method = job[3] if len(job) > 3 else None
        k = cfg.k if cfg is not None else 10
        ckey = (k, tuple(sorted(ctx_kw.items())))
        if ckey not in ctx_cache:
            ctx_cache[ckey] = build_context(X, y, sp, k=k, **ctx_kw)
        ctx = ctx_cache[ckey]
        seed = sp.seed
        try:
            diag = [] if task['diag'] and name == 'FedTGNN-SS' else None
            if method is not None:
                out = METHODS[method](ctx, seed=seed)
            elif cfg is None:
                fn = METHODS[name]
                out = fn(ctx, seed=seed, diag=diag) if name == 'FedTGNN-SS' else fn(ctx, seed=seed)
            else:
                out = train_fedtgnn(ctx, cfg, seed=seed, diag=diag)
        except Exception:
            traceback.print_exc()
            continue
        vi, vp = out['val']
        ti, tp = out['test']
        thr = youden_threshold(y[vi], vp)
        m = evaluate(y[ti], tp, thr)
        m.update(out['_meta'])
        if 'members' in out:
            m['mia_auc'] = mia_auc(y[out['members'][0]], out['members'][1], y[ti], tp)
        m.update(sp.counts)
        row = {k_: task[k_] for k_ in ('experiment', 'dataset', 'n_silos', 'rho', 'setting')}
        row.update(alpha=_alpha_str(task['alpha']), repeat=sp.repeat, fold=sp.fold,
                   seed=seed, method=name)
        row.update(m)
        rows.append(row)
        if task['save_preds']:
            preds.append(pd.DataFrame(dict(dataset=task['dataset'], method=name, repeat=sp.repeat, fold=sp.fold,
                                           rho=task['rho'], idx=ti, y=y[ti], p=tp)))
        if diag:
            for d in diag:
                d.update(dataset=task['dataset'], rho=task['rho'], repeat=sp.repeat, fold=sp.fold)
            diags.extend(diag)
    return rows, preds, diags


def build_tasks(args):
    from fedtgnn.variants import ABLATIONS, LEAKAGE, sweep_configs
    from fedtgnn.baselines import METHODS

    tasks = []
    exp = args.experiment
    for ds in args.datasets:
        settings = []   # (setting, n_silos, alpha, rhos, jobs, n_repeats)
        methods = args.methods or list(METHODS)
        if exp == 'main':
            settings.append(('main', DEFAULT_SILOS[ds], 0.5, args.rho or SCARCITY,
                             [(m, None) for m in methods], args.repeats))
        elif exp == 'hetero':
            for a in [0.1, 0.5, 1.0, None]:
                settings.append((f'alpha={_alpha_str(a)}', DEFAULT_SILOS[ds], a,
                                 args.rho or [0.8], [(m, None) for m in methods], args.repeats))
        elif exp == 'clients':
            for S in [2, 3, 4, 5]:
                settings.append((f'S={S}', S, 0.5, args.rho or [0.8],
                                 [(m, None) for m in methods], args.repeats))
        elif exp == 'ablation':
            settings.append(('ablation', DEFAULT_SILOS[ds], 0.5, args.rho or [0.5, 0.8],
                             list(ABLATIONS.items()), args.repeats))
        elif exp == 'leakage':
            settings.append(('leakage', DEFAULT_SILOS[ds], 0.5, args.rho or [0.1, 0.8],
                             LEAKAGE, args.repeats))
        elif exp == 'sweep':
            settings.append(('sweep', DEFAULT_SILOS[ds], 0.5, args.rho or [0.8],
                             list(sweep_configs()), args.repeats))
        for setting, S, a, rhos, jobs, R in settings:
            for rho in rhos:
                for r in range(R):
                    for f in range(args.folds):
                        tasks.append(dict(experiment=exp, dataset=ds, n_silos=S, alpha=a,
                                          rho=rho, setting=setting, repeat=r, fold=f,
                                          n_folds=args.folds, jobs=jobs,
                                          save_preds=args.save_preds,
                                          diag=(exp == 'main')))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('experiment', choices=['main', 'hetero', 'clients', 'ablation', 'sweep', 'leakage'])
    ap.add_argument('--datasets', nargs='+', default=['gdm_early', 'pima', 'early', 'gdm_diag'])
    ap.add_argument('--methods', nargs='+')
    ap.add_argument('--rho', nargs='+', type=float)
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--save-preds', action='store_true')
    ap.add_argument('--out', default=os.path.join(HERE, 'results'))
    args = ap.parse_args()
    if args.experiment == 'main':
        args.save_preds = True

    os.makedirs(args.out, exist_ok=True)
    res_path = os.path.join(args.out, f'{args.experiment}.csv')
    pred_path = os.path.join(args.out, f'{args.experiment}_preds.csv.gz')
    diag_path = os.path.join(args.out, f'{args.experiment}_pseudolabels.csv')
    done = set()
    if os.path.exists(res_path):
        prev = pd.read_csv(res_path)
        prev['alpha'] = prev['alpha'].astype(str)
        done = set(map(tuple, prev[KEY].astype(str).values))

    tasks = []
    for t in build_tasks(args):
        remaining = []
        for job in t['jobs']:
            name = job[0]
            key = tuple(map(str, [t['experiment'], t['dataset'], t['n_silos'], _alpha_str(t['alpha']),
                                  t['rho'], t['setting'], t['repeat'], t['fold'], name]))
            if key not in done:
                remaining.append(job)
        if remaining:
            t['jobs'] = remaining
            tasks.append(t)
    print(f'{len(tasks)} tasks to run -> {res_path}', flush=True)

    def write(rows, preds, diags):
        if rows:
            df = pd.DataFrame(rows)
            if os.path.exists(res_path):          # align to the existing header
                cols = list(pd.read_csv(res_path, nrows=0).columns)
                extra = [c for c in df.columns if c not in cols]
                if extra:
                    raise RuntimeError(f'new result columns {extra}; use a new results file')
                df = df.reindex(columns=cols)
            else:
                df = df.reindex(columns=RESULT_COLUMNS)
            df.to_csv(res_path, mode='a', index=False, header=not os.path.exists(res_path))
        if preds:
            pd.concat(preds).to_csv(pred_path, mode='a', index=False,
                                    header=not os.path.exists(pred_path), compression='gzip')
        if diags:
            pd.DataFrame(diags).to_csv(diag_path, mode='a', index=False,
                                       header=not os.path.exists(diag_path))

    if args.workers <= 1:
        for i, t in enumerate(tasks):
            write(*run_task(t))
            print(f'[{i + 1}/{len(tasks)}] {t["dataset"]} {t["setting"]} rho={t["rho"]} '
                  f'r={t["repeat"]} f={t["fold"]}', flush=True)
    else:
        with ProcessPoolExecutor(args.workers) as ex:
            futs = {ex.submit(run_task, t): t for t in tasks}
            for i, fu in enumerate(as_completed(futs)):
                t = futs[fu]
                write(*fu.result())
                print(f'[{i + 1}/{len(tasks)}] {t["dataset"]} {t["setting"]} rho={t["rho"]} '
                      f'r={t["repeat"]} f={t["fold"]}', flush=True)


if __name__ == '__main__':
    main()
