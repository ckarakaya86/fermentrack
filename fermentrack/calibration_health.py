"""Calibration health: is the PLS model still agreeing with the lab?

Every time a lab (HPLC) result comes back we compare it with what the in-line
model predicted for the same spectrum (residual = in-line - lab). While the
calibration is healthy these residuals scatter randomly around zero.

Samples from the same batch share the same matrix, so their residuals are
correlated. We therefore summarise each batch by ONE number: the median
residual of its lab samples. The natural batch-to-batch scatter of that number
is learned from the calibration's cross-validation.

We watch the batch medians with an EWMA control chart (exponentially weighted
moving average). EWMA is good at catching *small, slow, persistent* shifts,
such as an ageing light source or a slowly misaligned interferometer, that a
simple "one point outside 3 sigma" rule would miss for weeks.

When the chart signals, we recommend a rolling bias update (cheap: before each
batch, subtract the average in-line - lab difference of the last 3 batches)
and show how much it would have helped, plus a full recalibration later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

WATCHED = ["ethanol", "glucose", "dp4plus", "glycerol", "lactic"]


def residual_table(lab: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    """Lab samples in chronological order with in-line prediction - lab for each analyte.

    `predicted` holds model predictions for the same rows as `lab` (same order).
    """
    out = lab[["batch_id", "time_h"]].reset_index(drop=True).copy()
    out["sequence"] = np.arange(len(out))
    for a in C.ANALYTES:
        out[f"{a}_lab"] = lab[a].to_numpy()
        out[f"{a}_pred"] = predicted[a].to_numpy()
        out[f"{a}_resid"] = out[f"{a}_pred"] - out[f"{a}_lab"]
    return out


def batch_sigma(cv_resid: np.ndarray, groups: np.ndarray) -> float:
    """Expected scatter of a batch's median residual, from cross-validation residuals."""
    med = pd.Series(cv_resid).groupby(groups).median()
    return float(med.std(ddof=1))


def ewma(x: np.ndarray, lam: float) -> np.ndarray:
    z = np.zeros_like(x, dtype=float)
    prev = 0.0
    for i, v in enumerate(x):
        prev = lam * v + (1 - lam) * prev
        z[i] = prev
    return z


def ewma_chart(x: np.ndarray, sigma: float, lam: float = 0.2, L: float = 3.0):
    """EWMA statistic and its (time-varying) control limit."""
    i = np.arange(1, len(x) + 1)
    limit = L * sigma * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * i)))
    return ewma(x, lam), limit


def check_analyte(res: pd.DataFrame, analyte: str, sigma: float, start_batch: int,
                  lam: float = 0.2, L: float = 3.0, n_bias: int = 3) -> dict:
    """EWMA chart of batch-median residuals from `start_batch` onwards."""
    sub = res[res.batch_id >= start_batch]
    med = sub.groupby("batch_id")[f"{analyte}_resid"].median()
    x = med.to_numpy()
    z, lim = ewma_chart(x, sigma, lam, L)
    out = np.abs(z) > lim
    idx = int(np.argmax(out)) if out.any() else None
    result = {"analyte": analyte, "sigma": sigma, "batch_ids": med.index.to_numpy(),
              "batch_median": x, "ewma": z, "limit": lim, "alarm_index": idx,
              "alarm_batch": None, "bias_estimate": None}
    if idx is not None:
        alarm_batch = int(med.index[idx])
        recent = sub[sub.batch_id.isin(med.index[max(0, idx + 1 - n_bias): idx + 1])]
        bias = float(recent[f"{analyte}_resid"].mean())
        result.update(alarm_batch=alarm_batch, bias_estimate=bias)
        # After the alarm: before each new batch, subtract the mean residual of the
        # previous n_bias batches (a rolling bias update, as plants do with each lab result).
        later = [b for b in med.index if b > alarm_batch]
        raw, fixed = [], []
        for b in later:
            prev = med.index[(med.index < b)][-n_bias:]
            corr = sub.loc[sub.batch_id.isin(prev), f"{analyte}_resid"].mean()
            r_b = sub.loc[sub.batch_id == b, f"{analyte}_resid"].to_numpy()
            raw.append(r_b)
            fixed.append(r_b - corr)
        if later:
            raw, fixed = np.concatenate(raw), np.concatenate(fixed)
            result["rmse_after_no_fix"] = float(np.sqrt(np.mean(raw ** 2)))
            result["rmse_after_bias_fix"] = float(np.sqrt(np.mean(fixed ** 2)))
    return result


def health_alerts(checks: list[dict]) -> list[dict]:
    """One 'Calibration' alert per analyte whose EWMA chart went out of control."""
    alerts = []
    for c in checks:
        if c["alarm_index"] is None:
            continue
        tail = ""
        if "rmse_after_bias_fix" in c:
            tail = (f" A rolling bias update (latest 3 batches of lab results) from this point "
                    f"would have cut the error on later lab samples from "
                    f"{c['rmse_after_no_fix']:.3f} to {c['rmse_after_bias_fix']:.3f} {C.UNIT}.")
        name = C.LABELS[c["analyte"]]
        alerts.append({
            "batch_id": c["alarm_batch"], "time_h": C.BATCH_DURATION_H,
            "category": "Calibration", "severity": "medium",
            "title": f"Calibration drift: {name}",
            "message": (f"The EWMA of the batch-median (in-line - lab) difference for {name.lower()} "
                        f"crossed its control limit. Recent bias {c['bias_estimate']:+.3f} {C.UNIT}, "
                        f"while the normal batch-to-batch scatter is {c['sigma']:.3f}. The instrument "
                        f"or the matrix has changed since calibration: apply a bias update now and "
                        f"plan a recalibration with fresh lab samples.{tail}"),
        })
    return alerts
