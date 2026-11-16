"""Ways to divide samples into a calibration set and a validation set.

Every function returns a boolean array: True = validation sample.

* random          - simple random draw (fixed seed, reproducible)
* kennard_stone   - picks calibration samples that span the spectral space evenly;
                    the remaining (interior) samples are used for validation
* y_stratified    - sort by the reference value and send every k-th sample to
                    validation, so both sets cover the full concentration range
* by_group        - whole groups (batches, days, lots) go to validation together;
                    the honest choice when samples of one group are correlated
* by_order        - the last part of the data (by date or file order) is the
                    validation set: "train on the past, test on the future"
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METHODS = {
    "random": "Random",
    "kennard_stone": "Kennard-Stone (spectral coverage)",
    "y_stratified": "Stratified by reference value",
    "by_group": "By group (batch / day / lot)",
    "by_order": "By order / date (last part = validation)",
}


def random_split(n, frac=0.25, seed=0):
    rng = np.random.default_rng(seed)
    val = np.zeros(n, bool)
    val[rng.choice(n, max(1, int(round(frac * n))), replace=False)] = True
    return val


def kennard_stone(X, frac=0.25):
    """Classic Kennard-Stone: start from the two most distant samples, then repeatedly add
    the sample farthest from everything already chosen. Chosen samples = calibration."""
    X = np.asarray(X, float)
    n = len(X)
    n_cal = n - max(1, int(round(frac * n)))
    # work in a PCA space for speed and to ignore noise directions
    Xc = X - X.mean(axis=0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    k = min(10, len(S))
    Z = U[:, :k] * S[:k]
    D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    i, j = np.unravel_index(np.argmax(D), D.shape)
    chosen = [i, j]
    min_d = np.minimum(D[i], D[j])
    while len(chosen) < n_cal:
        min_d[chosen] = -1
        nxt = int(np.argmax(min_d))
        chosen.append(nxt)
        min_d = np.minimum(min_d, D[nxt])
    val = np.ones(n, bool)
    val[chosen] = False
    return val


def y_stratified(y, frac=0.25, seed=0):
    y = np.asarray(y, float)
    order = np.argsort(y)
    step = max(2, int(round(1 / frac)))
    offset = int(np.random.default_rng(seed).integers(step))
    val = np.zeros(len(y), bool)
    val[order[offset::step]] = True
    val[order[0]] = val[order[-1]] = False        # keep the extremes in calibration
    return val


def by_group(groups, frac=0.25, seed=0):
    groups = np.asarray(groups).astype(str)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(frac * len(uniq))))
    val_groups = rng.choice(uniq, n_val, replace=False)
    return np.isin(groups, val_groups)


def by_order(order_key, frac=0.25):
    """Last `frac` of the samples by `order_key` (dates or plain row order)."""
    key = pd.Series(order_key)
    if not np.issubdtype(key.dtype, np.number):
        key = pd.to_datetime(key, errors="coerce", dayfirst=True)
        if key.isna().all():
            key = pd.Series(np.arange(len(order_key)))
    ranks = key.rank(method="first").to_numpy()
    return ranks > len(ranks) * (1 - frac)


def make_split(method, X=None, y=None, groups=None, order_key=None, frac=0.25, seed=0):
    n = len(X) if X is not None else len(y)
    if method == "random":
        return random_split(n, frac, seed)
    if method == "kennard_stone":
        return kennard_stone(X, frac)
    if method == "y_stratified":
        return y_stratified(y, frac, seed)
    if method == "by_group":
        if groups is None:
            raise ValueError("This split needs a 'group' (batch / lot) column in the data.")
        return by_group(groups, frac, seed)
    if method == "by_order":
        return by_order(order_key if order_key is not None else np.arange(n), frac)
    raise ValueError(method)
