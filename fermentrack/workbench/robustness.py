"""Robustness tests: will the model survive real life?

1. Perturbation test - disturb the validation spectra the way a real instrument
   or process would, and watch how much RMSEP grows:
     * noise            (detector noise, short scans)
     * baseline offset  (temperature, window deposits)
     * baseline slope   (scattering, alignment)
     * gain             (lamp / source intensity, path length)
     * wavenumber shift (instrument drift, a different instrument)
   A robust model degrades slowly. The level at which RMSEP doubles is a useful
   "tolerance" number to compare models and preprocessing choices.

2. Resampling test - repeat the calibration/validation split many times at
   random. One lucky split can make a model look good; the spread of RMSEP
   over many splits shows how stable the result really is.

3. y-randomisation (permutation) test - shuffle the reference values, rebuild
   the model, and repeat. A real model must be far better than these "nonsense"
   models; if it is not, its apparent performance is chance correlation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import fit_model, regression_stats
from .preprocess import Pipeline
from .split import random_split

PERTURBATIONS = {
    "noise": ("Random noise", "% of spectral SD", [0, 0.5, 1, 2, 5]),
    "offset": ("Baseline offset", "% of mean absorbance", [0, 1, 2, 5, 10]),
    "slope": ("Baseline slope", "% of mean absorbance across range", [0, 1, 2, 5, 10]),
    "gain": ("Intensity gain", "% change", [0, 0.5, 1, 2, 5]),
    "shift": ("Wavenumber shift", "x point spacing", [0, 0.1, 0.25, 0.5, 1.0]),
}


def perturb(X, wn, kind, level, seed=0):
    """Return a disturbed copy of the spectra X (rows = samples)."""
    X = np.asarray(X, float).copy()
    if level == 0:
        return X
    scale = np.abs(X).mean()
    if kind == "noise":
        return X + np.random.default_rng(seed).normal(0, level / 100 * X.std(), X.shape)
    if kind == "offset":
        return X + level / 100 * scale
    if kind == "slope":
        z = np.linspace(-0.5, 0.5, X.shape[1])
        return X + level / 100 * scale * z
    if kind == "gain":
        return X * (1 + level / 100)
    if kind == "shift":
        step = float(np.median(np.abs(np.diff(wn))))
        order = np.argsort(wn)
        out = np.empty_like(X)
        for i, x in enumerate(X):
            out[i, order] = np.interp(wn[order] - level * step, wn[order], x[order])
        return out
    raise ValueError(kind)


def perturbation_test(model, X_val, y_val, kinds=None) -> pd.DataFrame:
    rows = []
    for kind in kinds or PERTURBATIONS:
        name, unit, levels = PERTURBATIONS[kind]
        for lv in levels:
            pred = model.predict(perturb(X_val, model.wn, kind, lv))
            st = regression_stats(y_val, pred)
            rows.append({"test": name, "kind": kind, "unit": unit, "level": lv,
                         "RMSEP": st["RMSE"], "bias": st["bias"]})
    df = pd.DataFrame(rows)
    base = df[df.level == 0].set_index("kind").RMSEP
    df["RMSEP_ratio"] = df.RMSEP / df.kind.map(base)
    return df


def tolerance(df: pd.DataFrame, factor=2.0) -> pd.DataFrame:
    """Smallest tested level at which RMSEP reaches `factor` x its undisturbed value."""
    rows = []
    for kind, g in df.groupby("kind", sort=False):
        hit = g[g.RMSEP_ratio >= factor]
        rows.append({"test": g.test.iat[0], "unit": g.unit.iat[0],
                     "RMSEP doubles at": hit.level.min() if len(hit) else f"> {g.level.max()}"})
    return pd.DataFrame(rows)


def resampling_test(spec, pipeline: Pipeline, X, wn, y, n_repeats=20, frac=0.25,
                    progress=None) -> pd.DataFrame:
    """RMSEC / RMSECV / RMSEP over many random calibration/validation splits."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    ok = ~np.isnan(y)
    X, y = X[ok], y[ok]
    rows = []
    for r in range(n_repeats):
        val = random_split(len(y), frac, seed=1000 + r)
        m = fit_model(spec, pipeline, X[~val], wn, y[~val])
        rows.append({"repeat": r + 1, "LVs": m.k, "RMSEC": m.cal_stats()["RMSE"],
                     "RMSECV": m.cv_stats()["RMSE"],
                     "RMSEP": regression_stats(y[val], m.predict(X[val]))["RMSE"]})
        if progress:
            progress((r + 1) / n_repeats)
    return pd.DataFrame(rows)


def y_randomization(spec, pipeline: Pipeline, X, wn, y, n_perm=10, progress=None) -> dict:
    """Compare the real model's RMSECV with models built on shuffled reference values."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    ok = ~np.isnan(y)
    X, y = X[ok], y[ok]
    real = fit_model(spec, pipeline, X, wn, y)
    real_rmse = real.cv_stats()["RMSE"]
    perm = []
    rng = np.random.default_rng(0)
    fixed = frozen_spec(spec, real.k)
    for i in range(n_perm):
        m = fit_model(fixed, pipeline, X, wn, rng.permutation(y))
        perm.append(m.cv_stats()["RMSE"])
        if progress:
            progress((i + 1) / n_perm)
    perm = np.array(perm)
    return {"real_RMSECV": real_rmse, "permuted_RMSECV": perm,
            "ratio": float(perm.min() / real_rmse),
            "passed": bool(real_rmse < 0.5 * perm.min())}


def frozen_spec(spec, k):
    """Same recipe but with the model size frozen (so shuffled data can't pick a lucky size)."""
    from .models import ModelSpec
    s = ModelSpec.from_settings(spec.settings())
    s.n_components = k
    return s
