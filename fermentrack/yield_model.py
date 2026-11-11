"""Early prediction of final ethanol (and yield) from the first hours of a batch.

Hybrid (physics + machine learning) approach:
* Chemistry gives the ceiling: theoretical ethanol = 0.511 x starting sugar.
* Machine learning predicts only the *yield efficiency* (fraction of that ceiling
  the batch will reach) from early signs of health: how fast ethanol is rising,
  how much glycerol is being made per ethanol (a classic yield-loss driver),
  whether acids are building up.
* Final ethanol = predicted efficiency x theoretical ethanol.
The model therefore does not have to re-learn "more sugar gives more ethanol"
from a few dozen batches; it only learns the part that chemistry cannot tell.

Why a robust (Huber) regression? Some training batches were later hit by faults
that no early signal could foresee (e.g. contamination entering at 15 h). An
ordinary least-squares fit would bend itself to explain them; Huber loss gives
such outliers less weight, so the model learns the typical behaviour.

Honesty rules:
* Features come ONLY from data available at the cut-off time (no peeking ahead).
* Training batches are the plant's past, test batches its future (chronological split).
* Uncertainty comes from leave-one-batch-out cross-validation on the training
  batches. We use the empirical 5th/95th percentiles of those errors, so the band
  is asymmetric: a batch can turn out much worse than expected, rarely much better.
* We always compare against a naive baseline (typical yield x starting sugar).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C

FEATURES = {
    "progress": "Ethanol at cut-off / theoretical ethanol",
    "ethanol_rate": "Ethanol rate, last 4 h (%/h)",
    "glycerol_ratio": "Glycerol / ethanol at cut-off",
    "lactic_now": "Lactic acid at cut-off",
    "acetic_now": "Acetic acid at cut-off",
}


def _window_mean(t, v, t_end, width):
    m = (t <= t_end) & (t > t_end - width)
    return float(np.mean(v[m])) if m.any() else float(v[t <= t_end][-1])


def trajectory_features(traj: pd.DataFrame, cutoff_h: float) -> dict:
    """Features from one batch's predicted trajectory, using data up to cutoff_h only.

    `traj` needs columns time_h + analytes. Short windows are averaged to damp noise.
    """
    traj = traj[traj["time_h"] <= cutoff_h + 1e-9]
    t = traj["time_h"].to_numpy()
    g = {a: traj[a].to_numpy() for a in C.ANALYTES}
    sugar = sum(g[a] for a in C.SUGARS)
    now = lambda v: _window_mean(t, v, cutoff_h, 1.0)          # mean of the last hour
    before = lambda v: _window_mean(t, v, cutoff_h - 4, 1.0)   # mean of the hour 4 h earlier
    initial_sugar = _window_mean(t, sugar, 1.0, 1.01)
    theoretical = C.ETHANOL_YIELD * initial_sugar
    return {
        "initial_sugar": initial_sugar,
        "theoretical_ethanol": theoretical,
        "progress": now(g["ethanol"]) / theoretical,
        "ethanol_rate": (now(g["ethanol"]) - before(g["ethanol"])) / 4,
        "glucose_now": now(g["glucose"]),
        "glycerol_ratio": now(g["glycerol"]) / max(now(g["ethanol"]), 0.1),
        "lactic_now": now(g["lactic"]),
        "lactic_rate": (now(g["lactic"]) - before(g["lactic"])) / 4,
        "acetic_now": now(g["acetic"]),
    }


def feature_table(predictions: pd.DataFrame, cutoff_h: float) -> pd.DataFrame:
    """One feature row per batch. `predictions` holds batch_id, time_h and analytes."""
    rows = [{"batch_id": b, **trajectory_features(df, cutoff_h)}
            for b, df in predictions.groupby("batch_id")]
    return pd.DataFrame(rows).set_index("batch_id")


@dataclass
class CutoffModel:
    cutoff_h: float
    model: object                 # sklearn pipeline (scaler + ridge)
    cv_rmse: float                # leave-one-batch-out RMSE (final ethanol, % w/v), training batches
    err_low: float                # 5th percentile of (actual - predicted) in that CV
    err_high: float               # 95th percentile
    baseline_yield: float         # typical (median) fraction of theoretical, for the naive baseline

    def predict(self, X: pd.DataFrame):
        """Final ethanol (% w/v) with a 90 % prediction interval (low, high)."""
        efficiency = self.model.predict(X[list(FEATURES)])
        mean = efficiency * X["theoretical_ethanol"].to_numpy()
        return mean, mean + self.err_low, mean + self.err_high

    def baseline(self, X: pd.DataFrame):
        return self.baseline_yield * C.ETHANOL_YIELD * X["initial_sugar"].to_numpy()

    def coefficients(self) -> pd.Series:
        """Standardised coefficients: effect of a 1-SD change of each feature."""
        reg = self.model[-1]
        return pd.Series(reg.coef_, index=list(FEATURES)).sort_values(key=np.abs, ascending=False)


class EarlyYieldPredictor:
    """One robust linear model per cut-off time (8, 12, 16, 20, 24 h)."""

    def __init__(self, cutoffs=C.YIELD_CUTOFFS_H):
        self.cutoffs = list(cutoffs)
        self.models: dict[float, CutoffModel] = {}

    def fit(self, predictions: pd.DataFrame, final_ethanol: pd.Series):
        """final_ethanol: lab-measured final ethanol, indexed by batch_id (training batches)."""
        for c in self.cutoffs:
            X = feature_table(predictions, c).loc[final_ethanol.index]
            theo = X["theoretical_ethanol"].to_numpy()
            y = final_ethanol.to_numpy()
            efficiency = y / theo                       # what the ML part has to learn
            pipe = make_pipeline(StandardScaler(), HuberRegressor(alpha=1.0, epsilon=1.35))
            # leave-one-batch-out: each batch is predicted by a model that never saw it
            cv_eff = cross_val_predict(pipe, X[list(FEATURES)], efficiency, cv=LeaveOneOut())
            pipe.fit(X[list(FEATURES)], efficiency)
            err = y - cv_eff * theo
            self.models[c] = CutoffModel(
                c, pipe, float(np.sqrt(np.mean(err ** 2))),
                float(np.percentile(err, 5)), float(np.percentile(err, 95)),
                float(np.median(efficiency)))
        return self

    def model_for(self, t_now: float) -> CutoffModel | None:
        """The latest cut-off model that can be used at time t_now (None if too early)."""
        usable = [c for c in self.cutoffs if c <= t_now + 1e-9]
        return self.models[max(usable)] if usable else None

    def predict_batch(self, traj: pd.DataFrame, t_now: float):
        """Live prediction for a single batch at time t_now. Returns dict or None."""
        m = self.model_for(t_now)
        if m is None:
            return None
        X = pd.DataFrame([trajectory_features(traj, m.cutoff_h)])
        mean, lo, hi = (float(v[0]) for v in m.predict(X))
        theo = float(X["theoretical_ethanol"].iat[0])
        return {"cutoff_h": m.cutoff_h, "final_ethanol": mean, "low": lo, "high": hi,
                "yield_pct": 100 * mean / theo, "theoretical_ethanol": theo}

    def evaluate(self, predictions: pd.DataFrame, final_ethanol: pd.Series,
                 normal_ids=None) -> pd.DataFrame:
        """Test-set scores for every cut-off, against the naive baseline.

        normal_ids: optional batch ids without faults, to also report the error on
        batches where nothing unforeseeable happened after the cut-off.
        """
        rmse = lambda e: float(np.sqrt(np.mean(np.asarray(e) ** 2)))
        rows = []
        for c, m in self.models.items():
            X = feature_table(predictions, c).loc[final_ethanol.index]
            y = final_ethanol.to_numpy()
            mean, lo, hi = m.predict(X)
            base = m.baseline(X)
            row = {
                "cutoff_h": c,
                "RMSE_model": rmse(mean - y),
                "RMSE_baseline": rmse(base - y),
                "RMSE_LOBO_CV_train": m.cv_rmse,
                "interval_coverage": float(np.mean((y >= lo) & (y <= hi))),
            }
            if normal_ids is not None:
                n = np.isin(final_ethanol.index, list(normal_ids))
                row["RMSE_model_normal"] = rmse(mean[n] - y[n])
                row["RMSE_baseline_normal"] = rmse(base[n] - y[n])
                row["RMSE_model_faulty"] = rmse(mean[~n] - y[~n])
                row["RMSE_baseline_faulty"] = rmse(base[~n] - y[~n])
            rows.append(row)
        return pd.DataFrame(rows)

    def test_predictions(self, predictions, final_ethanol, cutoff_h) -> pd.DataFrame:
        m = self.models[cutoff_h]
        X = feature_table(predictions, cutoff_h).loc[final_ethanol.index]
        mean, lo, hi = m.predict(X)
        return pd.DataFrame({"batch_id": final_ethanol.index, "actual": final_ethanol.to_numpy(),
                             "predicted": mean, "low": lo, "high": hi,
                             "baseline": m.baseline(X)})
