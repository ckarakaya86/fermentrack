"""Spectral preprocessing and PLS calibration (spectra -> concentrations).

Workflow for each analyte:
1. Try a few preprocessing recipes (baseline removal, SNV, Savitzky-Golay derivative).
2. For each recipe, fit PLS models with 1..max components and score them with
   *batch-wise* cross-validation (all samples of a batch are left out together;
   samples from the same batch are correlated, so leaving out single samples
   would give over-optimistic errors).
3. Choose the recipe with the lowest RMSECV, and the smallest number of
   components whose RMSECV is within 5 % of the minimum (a guard against overfitting).
4. Refit on all calibration samples and report RMSEP on held-out test batches.

The PLS algorithm is a NumPy NIPALS PLS1 (one model per analyte), written following
Wold, Sjostrom & Eriksson (2001), Chemometrics and Intelligent Laboratory Systems 58, 109-130.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from . import config as C


# ============================================================================
# Preprocessing
# ============================================================================
@dataclass(frozen=True)
class Preprocessor:
    """A stateless spectral preprocessing recipe.

    baseline: "none" | "linear"  - subtract a straight line fitted to each spectrum
    snv:      Standard Normal Variate - centre and scale each spectrum individually
    deriv:    0 | 1 | 2          - Savitzky-Golay derivative (removes offsets / slopes)
    """
    baseline: str = "none"
    snv: bool = False
    deriv: int = 0
    window: int = 11
    poly: int = 2

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        if self.baseline == "linear":
            z = np.linspace(-1, 1, X.shape[1])
            V = np.vstack([np.ones_like(z), z]).T
            coef, *_ = np.linalg.lstsq(V, X.T, rcond=None)
            X = X - (V @ coef).T
        if self.snv:
            X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
        if self.deriv:
            X = savgol_filter(X, self.window, self.poly, deriv=self.deriv, axis=1)
        return X

    @property
    def name(self) -> str:
        parts = []
        if self.baseline == "linear":
            parts.append("linear baseline")
        if self.snv:
            parts.append("SNV")
        if self.deriv:
            parts.append(f"SG {self.deriv}{'st' if self.deriv == 1 else 'nd'} derivative "
                         f"({self.window} pt)")
        return " + ".join(parts) or "none (raw absorbance)"


# The recipes we let cross-validation choose from
CANDIDATE_PREPROCESSING = [
    Preprocessor(),
    Preprocessor(baseline="linear"),
    Preprocessor(snv=True),
    Preprocessor(deriv=1),
    Preprocessor(baseline="linear", deriv=1),
    Preprocessor(deriv=2, window=15),
]


# ============================================================================
# PLS1 (NIPALS)
# ============================================================================
def nipals_pls1(X: np.ndarray, y: np.ndarray, n_components: int):
    """NIPALS PLS1 on mean-centred X (n x p) and y (n).

    Returns B (p x K): column k-1 holds the regression vector of the model with
    k components, so one fit gives every model size at once (fast CV).
    Also returns R (p x K), the weights that turn centred X into scores T = X R.
    """
    X = X.copy()
    y = y.copy()
    p = X.shape[1]
    W, P, q = np.zeros((p, n_components)), np.zeros((p, n_components)), np.zeros(n_components)
    used = 0
    for a in range(n_components):
        w = X.T @ y                       # direction in X most covariant with y
        norm = np.linalg.norm(w)
        if norm < 1e-12:
            break
        w /= norm
        t = X @ w                         # scores
        tt = t @ t
        p_a = X.T @ t / tt                # X loadings
        q_a = (y @ t) / tt                # y loading
        X -= np.outer(t, p_a)             # deflate: remove what this component explains
        y = y - q_a * t
        W[:, a], P[:, a], q[a] = w, p_a, q_a
        used = a + 1
    W, P, q = W[:, :used], P[:, :used], q[:used]
    R = W @ np.linalg.inv(P.T @ W)
    B = np.cumsum(R * q, axis=1)
    return B, R


@dataclass
class PLSModel:
    x_mean: np.ndarray
    y_mean: float
    B: np.ndarray            # p x K regression vectors (one per model size)
    R: np.ndarray

    @classmethod
    def fit(cls, X, y, n_components):
        xm, ym = X.mean(axis=0), float(y.mean())
        k = max(1, min(n_components, X.shape[0] - 1, X.shape[1]))
        B, R = nipals_pls1(X - xm, y - ym, k)
        return cls(xm, ym, B, R)

    @property
    def max_components(self):
        return self.B.shape[1]

    def predict_all(self, X):
        """Predictions for every model size: (n x K)."""
        return (X - self.x_mean) @ self.B + self.y_mean

    def predict(self, X, k):
        return (X - self.x_mean) @ self.B[:, min(k, self.max_components) - 1] + self.y_mean


# ============================================================================
# Metrics and cross-validation
# ============================================================================
def rmse(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def r2(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def group_folds(groups: np.ndarray, n_folds: int, seed: int = 0) -> list[np.ndarray]:
    """Split sample indices into folds so that each batch sits in exactly one fold."""
    uniq = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(uniq)
    return [np.where(np.isin(groups, chunk))[0] for chunk in np.array_split(perm, n_folds)]


def cv_predictions(X, y, groups, max_components, n_folds=8):
    """Batch-wise cross-validated predictions for every model size: (n x K)."""
    out = np.full((len(y), max_components), np.nan)
    for test_idx in group_folds(groups, n_folds):
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        m = PLSModel.fit(X[train_idx], y[train_idx], max_components)
        out[test_idx, :m.max_components] = m.predict_all(X[test_idx])
    return out


def choose_components(rmsecv_curve: np.ndarray, tolerance: float = 0.05) -> int:
    """Smallest number of components whose RMSECV is within `tolerance` of the best."""
    best = np.nanmin(rmsecv_curve)
    return int(np.argmax(rmsecv_curve <= best * (1 + tolerance))) + 1


# ============================================================================
# Calibration of one analyte, and of all analytes
# ============================================================================
@dataclass
class AnalyteCalibration:
    analyte: str
    preprocessor: Preprocessor
    n_components: int
    model: PLSModel
    rmsecv: float
    r2cv: float
    rmsecv_curve: np.ndarray
    cv_pred: np.ndarray
    y_cal: np.ndarray
    candidates: pd.DataFrame          # RMSECV of every preprocessing recipe tried

    def predict(self, X_raw):
        return self.model.predict(self.preprocessor.transform(X_raw), self.n_components)


def calibrate_analyte(X_raw, y, groups, analyte, max_components=12,
                      candidates=CANDIDATE_PREPROCESSING) -> AnalyteCalibration:
    trials = []
    for pre in candidates:
        Xp = pre.transform(X_raw)
        cvp = cv_predictions(Xp, y, groups, max_components)
        curve = np.array([rmse(y, cvp[:, k]) for k in range(cvp.shape[1])])
        k = choose_components(curve)
        trials.append((curve[k - 1], pre, k, curve, cvp[:, k - 1]))
    best_err, pre, k, curve, cvp = min(trials, key=lambda t: t[0])
    model = PLSModel.fit(pre.transform(X_raw), y, max_components)
    table = pd.DataFrame([{"preprocessing": t[1].name, "components": t[2], "RMSECV": t[0]}
                          for t in trials]).sort_values("RMSECV")
    return AnalyteCalibration(analyte, pre, k, model, best_err, r2(y, cvp), curve, cvp, y, table)


@dataclass
class SpectralCalibration:
    """One PLS model per analyte, plus a method to predict them all from spectra."""
    models: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, X_raw, Y: pd.DataFrame, groups, analytes=C.ANALYTES, **kw):
        return cls({a: calibrate_analyte(X_raw, Y[a].to_numpy(), groups, a, **kw)
                    for a in analytes})

    def predict(self, X_raw) -> pd.DataFrame:
        """Concentrations for every spectrum. Negative values are clipped to 0."""
        return pd.DataFrame({a: np.clip(m.predict(X_raw), 0, None) for a, m in self.models.items()})

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "analyte": a, "label": C.LABELS.get(a, a), "preprocessing": m.preprocessor.name,
            "components": m.n_components, "RMSECV": m.rmsecv, "R2_CV": m.r2cv,
            "range_min": float(m.y_cal.min()), "range_max": float(m.y_cal.max()),
        } for a, m in self.models.items()])
