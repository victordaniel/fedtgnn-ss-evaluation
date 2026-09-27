"""Federated preprocessing fitted on training patients only.

Each silo shares only per-feature sufficient statistics (count, sum, sum of
squares of observed values) with the server. The server derives global
means/standard deviations, which are used for mean imputation and
standardisation of training, validation and test patients.
"""

import numpy as np


class FederatedStandardizer:
    def fit(self, X, silos):
        d = X.shape[1]
        cnt = np.zeros(d)
        s1 = np.zeros(d)
        s2 = np.zeros(d)
        for silo in silos:           # executed locally at each silo
            Xs = X[silo]
            obs = ~np.isnan(Xs)
            cnt += obs.sum(0)
            s1 += np.nansum(Xs, 0)
            s2 += np.nansum(Xs ** 2, 0)
        self.mean_ = s1 / np.maximum(cnt, 1)
        var = s2 / np.maximum(cnt, 1) - self.mean_ ** 2
        self.std_ = np.sqrt(np.maximum(var, 0))
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, X):
        Xi = np.where(np.isnan(X), self.mean_, X)
        return (Xi - self.mean_) / self.std_


def continuous_mask(X, silos):
    """A feature is 'continuous' if it takes more than two distinct values in
    the training data (binary indicators are left unperturbed by CAA)."""
    Xt = X[np.concatenate(silos)]
    return np.array([len(np.unique(Xt[~np.isnan(Xt[:, j]), j])) > 2
                     for j in range(X.shape[1])])
