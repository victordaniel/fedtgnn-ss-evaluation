"""Evaluation metrics. The decision threshold is chosen on the validation set
(maximum Youden's J) and then applied unchanged to the test set."""

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, roc_auc_score, roc_curve)

EPS = 1e-6


def youden_threshold(y, p):
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    t = thr[int(np.argmax(j))]
    return float(min(max(t, EPS), 1 - EPS))


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _newton_logistic(X, y, offset=None, iters=50):
    """Unpenalised logistic regression by Newton-Raphson."""
    beta = np.zeros(X.shape[1])
    off = np.zeros(len(y)) if offset is None else offset
    for _ in range(iters):
        eta = X @ beta + off
        mu = 1 / (1 + np.exp(-eta))
        W = mu * (1 - mu) + 1e-9
        g = X.T @ (y - mu)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H + 1e-9 * np.eye(len(beta)), g)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def _nll(eta, y):
    return float(np.sum(np.logaddexp(0, eta) - y * eta))


def calibration(y, p):
    """Calibration intercept (calibration-in-the-large, slope fixed at 1)
    and calibration slope (logistic recalibration of logit(p)).
    Bounded maximum likelihood, so separable or extreme predictions give a
    finite value at the bound instead of a diverging Newton iterate."""
    from scipy.optimize import minimize, minimize_scalar
    y = np.asarray(y, float)
    lp = np.clip(_logit(p), -15, 15)
    a = minimize_scalar(lambda a: _nll(a + lp, y), bounds=(-10, 10), method='bounded').x
    res = minimize(lambda b: _nll(b[0] + b[1] * lp, y), x0=[0.0, 1.0], method='L-BFGS-B',
                   bounds=[(-10, 10), (-5, 20)])
    return float(a), float(res.x[1])


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(y[b == i].mean() - p[b == i].mean()) * (b == i).mean()
                     for i in range(bins) if (b == i).any()))


def evaluate(y_test, p_test, threshold):
    yhat = (p_test >= threshold).astype(int)
    tp = int(((yhat == 1) & (y_test == 1)).sum())
    tn = int(((yhat == 0) & (y_test == 0)).sum())
    fp = int(((yhat == 1) & (y_test == 0)).sum())
    fn = int(((yhat == 0) & (y_test == 1)).sum())
    ci, cs = calibration(y_test, p_test)
    return dict(
        auroc=roc_auc_score(y_test, p_test),
        auprc=average_precision_score(y_test, p_test),
        sensitivity=tp / max(tp + fn, 1),
        specificity=tn / max(tn + fp, 1),
        ppv=tp / max(tp + fp, 1) if (tp + fp) else np.nan,
        npv=tn / max(tn + fn, 1) if (tn + fn) else np.nan,
        f1_pos=f1_score(y_test, yhat, zero_division=0),
        macro_f1=f1_score(y_test, yhat, average='macro', zero_division=0),
        brier=brier_score_loss(y_test, p_test),
        cal_intercept=ci, cal_slope=cs, ece=ece(y_test, p_test),
        threshold=threshold, tp=tp, tn=tn, fp=fp, fn=fn)


def net_benefit(y, p, thresholds):
    n = len(y)
    out = []
    for t in thresholds:
        yhat = p >= t
        tp = np.sum(yhat & (y == 1))
        fp = np.sum(yhat & (y == 0))
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)


def mia_auc(y_mem, p_mem, y_non, p_non):
    """Loss-threshold membership-inference attack (Yeom et al. 2018 style):
    AUROC of separating training members from test non-members by the
    per-patient log-likelihood of the true label. 0.5 = no leakage."""
    def ll(y_, p_):
        p_ = np.clip(p_, EPS, 1 - EPS)
        return np.where(y_ == 1, np.log(p_), np.log(1 - p_))
    score = np.concatenate([ll(y_mem, p_mem), ll(y_non, p_non)])
    lab = np.concatenate([np.ones(len(y_mem)), np.zeros(len(y_non))])
    return roc_auc_score(lab, score)
