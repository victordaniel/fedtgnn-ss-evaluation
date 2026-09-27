"""Data splitting: outer CV -> validation hold-out -> Dirichlet silos ->
per-silo stratified label masking -> assignment of eval patients to silos.

Order matters: labels are removed only *after* the train/validation/test
split, and no test information is used by any step here.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass
class FoldSplit:
    repeat: int
    fold: int
    seed: int
    train: np.ndarray                 # outer-train indices (excl. validation)
    val: np.ndarray
    test: np.ndarray
    silos: list                       # list of arrays of training indices
    labeled: list                     # per-silo labeled indices
    unlabeled: list                   # per-silo unlabeled indices
    val_silo: np.ndarray              # silo id for each validation patient
    test_silo: np.ndarray             # silo id for each test patient
    counts: dict = field(default_factory=dict)


def dirichlet_partition(idx, y, n_silos, alpha, rng, min_per_class=5,
                        max_tries=1000):
    """Label-skew partition: class-c patients are split across silos with
    proportions ~ Dir(alpha * 1). alpha=None gives a stratified IID split."""
    classes = np.unique(y[idx])
    for _ in range(max_tries):
        parts = [[] for _ in range(n_silos)]
        for c in classes:
            ic = idx[y[idx] == c].copy()
            rng.shuffle(ic)
            if alpha is None:
                p = np.full(n_silos, 1.0 / n_silos)
            else:
                p = rng.dirichlet(alpha * np.ones(n_silos))
            cuts = (np.cumsum(p)[:-1] * len(ic)).astype(int)
            for s, chunk in enumerate(np.split(ic, cuts)):
                parts[s].extend(chunk.tolist())
        parts = [np.array(sorted(p), dtype=int) for p in parts]
        ok = all(min(np.sum(y[p] == c) for c in classes) >= min_per_class
                 for p in parts)
        if ok:
            return parts
    raise RuntimeError('Could not draw a Dirichlet partition with at least '
                       f'{min_per_class} patients per class in every silo; '
                       'increase alpha or reduce the number of silos.')


def mask_labels(silo, y, rho, rng):
    """Stratified MCAR label removal within one silo's training patients.
    Keeps round((1-rho) * n_c) labels per class, at least 1 per class."""
    lab, unl = [], []
    for c in np.unique(y[silo]):
        ic = silo[y[silo] == c].copy()
        rng.shuffle(ic)
        n_keep = max(1, int(round((1.0 - rho) * len(ic))))
        lab.extend(ic[:n_keep].tolist())
        unl.extend(ic[n_keep:].tolist())
    return np.array(sorted(lab), dtype=int), np.array(sorted(unl), dtype=int)


def assign_to_silos(idx, silos, rng):
    """Eval patients arrive at a hospital independently of their outcome:
    assign each one to a silo with probability proportional to silo size."""
    sizes = np.array([len(s) for s in silos], dtype=float)
    return rng.choice(len(silos), size=len(idx), p=sizes / sizes.sum())


def make_splits(y, rho, n_silos, alpha, n_repeats=10, n_folds=5,
                val_frac=0.15, base_seed=2026):
    """Yield FoldSplit objects for repeated stratified K-fold CV.

    The outer partition and the silo partition depend only on
    (repeat, fold), not on rho, so results are paired across scarcity
    levels and methods.
    """
    n = len(y)
    all_idx = np.arange(n)
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=base_seed + r)
        for f, (tr, te) in enumerate(skf.split(all_idx, y)):
            seed = base_seed + 1000 * r + f
            tr, va = train_test_split(tr, test_size=val_frac, stratify=y[tr],
                                      random_state=seed)
            rng = np.random.default_rng(seed)
            silos = dirichlet_partition(np.sort(tr), y, n_silos, alpha, rng)
            rng_mask = np.random.default_rng(seed + int(round(rho * 1000)))
            lab, unl = zip(*[mask_labels(s, y, rho, rng_mask) for s in silos])
            val_silo = assign_to_silos(va, silos, rng)
            test_silo = assign_to_silos(te, silos, rng)
            counts = {}
            for s in range(n_silos):
                counts[f'silo{s}_n'] = len(silos[s])
                counts[f'silo{s}_pos'] = int(y[silos[s]].sum())
                counts[f'silo{s}_labeled'] = len(lab[s])
                counts[f'silo{s}_labeled_pos'] = int(y[lab[s]].sum())
            counts.update(n_train=len(tr), n_val=len(va), n_test=len(te),
                          n_labeled=int(sum(len(l) for l in lab)))
            yield FoldSplit(r, f, seed, np.sort(tr), np.sort(va), np.sort(te),
                            list(silos), list(lab), list(unl), val_silo,
                            test_silo, counts)


def check_no_leakage(split):
    """Hard assertions used by the runner and the unit tests."""
    tr = set(split.train.tolist())
    va = set(split.val.tolist())
    te = set(split.test.tolist())
    assert not (tr & va) and not (tr & te) and not (va & te)
    silo_union = set(np.concatenate(split.silos).tolist())
    assert silo_union == tr, 'silos must partition the training set exactly'
    for s, l, u in zip(split.silos, split.labeled, split.unlabeled):
        assert set(l.tolist()) | set(u.tolist()) == set(s.tolist())
        assert not (set(l.tolist()) & set(u.tolist()))
