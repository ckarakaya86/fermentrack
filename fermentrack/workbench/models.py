"""PLS and ANN (neural network) regression models with one common interface.

Both model types share the same workflow, so they can be compared fairly:

    spec  = ModelSpec(kind="PLS" or "ANN", ...)          # the recipe
    model = fit_model(spec, pipeline, X, wn, y, groups)  # preprocess + CV + final fit
    yhat  = model.predict(X_new)

PLS  - NIPALS PLS1 (one property at a time). Number of latent variables (LVs)
       chosen from the cross-validation curve: the smallest number within 5 % of
       the best RMSECV (guards against overfitting), unless fixed by the user.

ANN  - a small multilayer perceptron (scikit-learn MLPRegressor). Spectra have
       hundreds of correlated points but we only have tens/hundreds of samples, so
       the network does NOT see raw spectra: it sees the first k PLS scores
       (a compact, noise-reduced summary). This "PLS-ANN" is a standard way to let
       a neural network capture curvature (non-linearity) without overfitting.
       Several networks with different random starts are trained (an ensemble);
       their spread is a simple uncertainty estimate.

Cross-validation: if samples have groups (batches), whole groups are left out
together; otherwise leave-one-out for small sets (< 40) or 10 random folds.
The PLS score extraction is refitted inside every CV fold, so no information
leaks from the left-out samples.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy import stats as sstats
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

from ..chemometrics import nipals_pls1
from .preprocess import Pipeline


# ============================================================================
# Recipe
# ============================================================================
@dataclass
class ModelSpec:
    kind: str = "PLS"                 # "PLS" or "ANN"
    n_components: int | None = None   # PLS LVs (None = choose by CV); ANN: number of PLS scores fed in
    max_components: int = 15
    scaling: str = "mean"             # "mean" (mean-centring) or "auto" (autoscaling)
    hidden: tuple = (8,)              # ANN hidden layer sizes
    alpha: float = 1e-2               # ANN L2 penalty (higher = smoother, less overfitting)
    n_ensemble: int = 5               # ANN: number of networks averaged
    cv: str = "auto"                  # "auto", "loo", "5", "10", "group"
    seed: int = 0

    def settings(self) -> dict:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        return d

    @classmethod
    def from_settings(cls, d):
        d = dict(d)
        d["hidden"] = tuple(d.get("hidden", (8,)))
        return cls(**d)

    def describe(self) -> str:
        if self.kind == "PLS":
            k = self.n_components or "auto"
            return f"PLS, {k} LV, {self.scaling}-scaled"
        return (f"ANN {list(self.hidden)} on {self.n_components or 'auto'} PLS scores, "
                f"alpha {self.alpha:g}, ensemble of {self.n_ensemble}")


# ============================================================================
# Statistics
# ============================================================================
def regression_stats(y, yhat) -> dict:
    """Standard figures of merit for a set of (reference, predicted) pairs."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = ~(np.isnan(y) | np.isnan(yhat))
    y, yhat = y[ok], yhat[ok]
    if len(y) < 2:
        return {"n": int(len(y)), "RMSE": np.nan, "R2": np.nan, "bias": np.nan,
                "SEP": np.nan, "slope": np.nan, "RPD": np.nan}
    e = yhat - y
    sep = float(np.std(e, ddof=1))
    slope = float(np.polyfit(y, yhat, 1)[0]) if np.ptp(y) > 0 else np.nan
    return {"n": int(len(y)), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "R2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)),
            "bias": float(e.mean()), "SEP": sep, "slope": slope,
            "RPD": float(np.std(y, ddof=1) / sep) if sep > 0 else np.nan}


def cv_folds(n, cv="auto", groups=None, seed=0) -> list[np.ndarray]:
    """Index arrays of the left-out samples for each CV fold."""
    if cv == "group" or (cv == "auto" and groups is not None and len(np.unique(groups)) >= 3):
        g = np.asarray(groups).astype(str)
        uniq = np.random.default_rng(seed).permutation(np.unique(g))
        chunks = np.array_split(uniq, min(len(uniq), 10))
        return [np.where(np.isin(g, c))[0] for c in chunks]
    if cv == "loo" or (cv == "auto" and n < 40):
        return [np.array([i]) for i in range(n)]
    k = 10 if cv == "auto" else int(cv)
    perm = np.random.default_rng(seed).permutation(n)
    return np.array_split(perm, min(k, n))


# ============================================================================
# Estimators (work on already-preprocessed spectra)
# ============================================================================
@dataclass
class PLSCore:
    """Scaling + NIPALS PLS1. B[:, k-1] = regression vector with k LVs."""
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    B: np.ndarray
    R: np.ndarray

    @classmethod
    def fit(cls, X, y, k_max, scaling="mean"):
        xm = X.mean(axis=0)
        xs = X.std(axis=0, ddof=1) if scaling == "auto" else np.ones(X.shape[1])
        xs = np.where(xs < 1e-12, 1.0, xs)
        k = max(1, min(k_max, X.shape[0] - 1, X.shape[1]))
        B, R = nipals_pls1((X - xm) / xs, y - y.mean(), k)
        return cls(xm, xs, float(y.mean()), B, R)

    @property
    def k_max(self):
        return self.B.shape[1]

    def z(self, X):
        return (X - self.x_mean) / self.x_scale

    def predict_all(self, X):
        return self.z(X) @ self.B + self.y_mean

    def predict(self, X, k):
        return self.z(X) @ self.B[:, min(k, self.k_max) - 1] + self.y_mean

    def scores(self, X, k):
        return self.z(X) @ self.R[:, :min(k, self.R.shape[1])]


@dataclass
class ANNCore:
    """PLS scores -> standardised -> ensemble of small MLPs."""
    pls: PLSCore
    k: int
    t_mean: np.ndarray
    t_scale: np.ndarray
    y_mean: float
    y_scale: float
    nets: list

    @classmethod
    def fit(cls, X, y, spec: ModelSpec, k):
        pls = PLSCore.fit(X, y, k, spec.scaling)
        k = min(k, pls.k_max)
        T = pls.scores(X, k)
        tm, ts = T.mean(axis=0), T.std(axis=0) + 1e-12
        ym, ysd = float(y.mean()), float(y.std() + 1e-12)
        nets = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for s in range(spec.n_ensemble):
                net = MLPRegressor(hidden_layer_sizes=tuple(spec.hidden), activation="tanh",
                                   solver="lbfgs", alpha=spec.alpha, max_iter=3000,
                                   random_state=spec.seed + s)
                net.fit((T - tm) / ts, (y - ym) / ysd)
                nets.append(net)
        return cls(pls, k, tm, ts, ym, ysd, nets)

    def predict_members(self, X):
        Tz = (self.pls.scores(X, self.k) - self.t_mean) / self.t_scale
        return np.array([n.predict(Tz) for n in self.nets]) * self.y_scale + self.y_mean

    def predict(self, X):
        return self.predict_members(X).mean(axis=0)


# ============================================================================
# Fitted model = pipeline + estimator + everything needed to judge it
# ============================================================================
@dataclass
class FittedModel:
    property: str
    spec: ModelSpec
    pipeline: Pipeline
    wn: np.ndarray                    # wavenumber axis the model expects (full, before regions)
    core: object                      # PLSCore or ANNCore
    k: int                            # LVs used (PLS) or PLS scores fed to the ANN
    y_cal: np.ndarray
    fitted_cal: np.ndarray            # predictions for calibration samples (RMSEC)
    cv_pred: np.ndarray               # cross-validated predictions (RMSECV)
    cv_curve: np.ndarray              # RMSECV for 1..k_max components
    cal_ids: np.ndarray = field(default_factory=lambda: np.array([]))
    diag: dict = field(default_factory=dict)   # outlier statistics reference (see diagnostics.py)

    # --- prediction -------------------------------------------------------------
    def _prep(self, X_raw, wn=None):
        X_raw = np.asarray(X_raw, float)
        if wn is not None and (len(wn) != len(self.wn) or not np.allclose(wn, self.wn)):
            order = np.argsort(wn)
            X_raw = np.array([np.interp(self.wn, wn[order], x[order]) for x in X_raw])
        return self.pipeline.transform(X_raw, self.wn)

    def predict(self, X_raw, wn=None):
        Xp = self._prep(X_raw, wn)
        if self.spec.kind == "PLS":
            return self.core.predict(Xp, self.k)
        return self.core.predict(Xp)

    def predict_spread(self, X_raw, wn=None):
        """ANN only: standard deviation across the ensemble (a rough uncertainty)."""
        if self.spec.kind != "ANN":
            return None
        return self.core.predict_members(self._prep(X_raw, wn)).std(axis=0)

    def pls_part(self) -> tuple[PLSCore, int]:
        """The PLS layer (for scores / outlier statistics) of either model type."""
        return (self.core, self.k) if self.spec.kind == "PLS" else (self.core.pls, self.core.k)

    # --- figures of merit ---------------------------------------------------------
    def cal_stats(self):
        return regression_stats(self.y_cal, self.fitted_cal)

    def cv_stats(self):
        return regression_stats(self.y_cal, self.cv_pred)


def _fit_core(spec, X, y, k):
    if spec.kind == "PLS":
        return PLSCore.fit(X, y, k, spec.scaling)
    return ANNCore.fit(X, y, spec, k)


def _predict_core(spec, core, X, k):
    return core.predict(X, k) if spec.kind == "PLS" else core.predict(X)


def choose_components(curve, tol=0.05) -> int:
    best = np.nanmin(curve)
    return int(np.argmax(curve <= best * (1 + tol))) + 1


def fit_model(spec: ModelSpec, pipeline: Pipeline, X_raw, wn, y, prop="y",
              groups=None, ids=None, progress=None) -> FittedModel:
    """Preprocess, cross-validate, pick the model size and fit the final model."""
    X_raw, y = np.asarray(X_raw, float), np.asarray(y, float)
    ok = ~np.isnan(y)
    X_raw, y = X_raw[ok], y[ok]
    groups = None if groups is None else np.asarray(groups)[ok]
    ids = np.arange(len(y)).astype(str) if ids is None else np.asarray(ids)[ok]
    if len(y) < 5:
        raise ValueError("At least 5 calibration samples with a reference value are needed.")

    folds = cv_folds(len(y), spec.cv, groups, spec.seed)
    k_max = max(1, min(spec.max_components, len(y) - 2))

    # 1) PLS component curve (always computed: it also guides the ANN input size)
    curve_pred = np.full((len(y), k_max), np.nan)
    for te in folds:
        tr = np.setdiff1d(np.arange(len(y)), te)
        pipe = Pipeline.from_settings(pipeline.settings()).fit(X_raw[tr], wn)
        core = PLSCore.fit(pipe.transform(X_raw[tr], wn), y[tr], k_max, spec.scaling)
        curve_pred[te, :core.k_max] = core.predict_all(pipe.transform(X_raw[te], wn))
    curve = np.sqrt(np.nanmean((curve_pred - y[:, None]) ** 2, axis=0))
    k = spec.n_components or choose_components(curve)
    k = int(min(k, k_max))

    # 2) cross-validated predictions of the actual model
    if spec.kind == "PLS":
        cv_pred = curve_pred[:, k - 1]
    else:
        cv_pred = np.full(len(y), np.nan)
        for i, te in enumerate(folds):
            tr = np.setdiff1d(np.arange(len(y)), te)
            pipe = Pipeline.from_settings(pipeline.settings()).fit(X_raw[tr], wn)
            core = ANNCore.fit(pipe.transform(X_raw[tr], wn), y[tr], spec, k)
            cv_pred[te] = core.predict(pipe.transform(X_raw[te], wn))
            if progress:
                progress((i + 1) / len(folds))

    # 3) final model on all calibration samples
    pipe = Pipeline.from_settings(pipeline.settings()).fit(X_raw, wn)
    Xp = pipe.transform(X_raw, wn)
    core = _fit_core(spec, Xp, y, k)
    model = FittedModel(prop, spec, pipe, np.asarray(wn, float), core, k, y,
                        _predict_core(spec, core, Xp, k), cv_pred, curve, ids)
    from .diagnostics import reference_statistics      # local import avoids a cycle
    model.diag = reference_statistics(model, X_raw)
    return model


def f_limit(k, n, alpha=0.95):
    """Hotelling T² limit for k components and n calibration samples."""
    return k * (n - 1) * (n + 1) / (n * (n - k)) * sstats.f.ppf(alpha, k, max(n - k, 1))
