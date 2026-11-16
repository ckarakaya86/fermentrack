"""Outlier diagnostics: which samples (or new spectra) should not be trusted?

Three independent questions, each with its own statistic:

* Hotelling T²  - is the sample *extreme* within the model's space?
                  (unusually high/low concentrations, an unusual mixture).
                  Leverage h is the same idea scaled 0..1: how strongly one
                  sample pulls the model towards itself.
* Q residual    - does the spectrum contain something the model cannot
                  describe at all? (contamination, fouling, wrong sample, air bubble)
* Studentised residual - is the *reference value* suspicious? The prediction
                  (from cross-validation, i.e. without that sample) disagrees with
                  the lab value by more than ~3 typical errors: a typing mistake
                  or a bad lab measurement.

Limits are 95 % (warning) and 99 % (action). A sample above the action limit is
a candidate for removal, but the decision stays with the user: remove only with
a reason, and keep a record (the workbench logs every exclusion).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sstats

from .models import f_limit


def _q_limit(q, alpha):
    """Box's approximation: Q ~ g * chi²(h), matched to the calibration Q values."""
    m, v = float(np.mean(q)), float(np.var(q))
    if v <= 0 or m <= 0:
        return float(np.max(q)) * 1.5 + 1e-12
    g, h = v / (2 * m), 2 * m ** 2 / v
    return float(g * sstats.chi2.ppf(alpha, h))


def _scores_and_q(model, X_raw, P=None):
    pls, k = model.pls_part()
    Z = pls.z(model.pipeline.transform(np.asarray(X_raw, float), model.wn))
    T = Z @ pls.R[:, :k]
    if P is None:
        P = Z.T @ T @ np.linalg.pinv(T.T @ T)       # X loadings: regress spectra on scores
    E = Z - T @ P.T
    return T, np.sum(E ** 2, axis=1), P


def reference_statistics(model, X_cal_raw) -> dict:
    """Statistics of the calibration set that later samples are compared with."""
    T, q, P = _scores_and_q(model, X_cal_raw)
    n, k = T.shape
    t_var = T.var(axis=0, ddof=1) + 1e-12
    resid = model.cv_pred - model.y_cal
    return {
        "P": P, "t_var": t_var, "TtT_inv": np.linalg.pinv(T.T @ T), "n": n, "k": k,
        "t2_lim95": float(f_limit(k, n, 0.95)), "t2_lim99": float(f_limit(k, n, 0.99)),
        "q_lim95": _q_limit(q, 0.95), "q_lim99": _q_limit(q, 0.99),
        "resid_sd": float(np.sqrt(np.mean(resid ** 2))),
        "y_range": (float(np.min(model.y_cal)), float(np.max(model.y_cal))),
    }


def sample_diagnostics(model, X_raw, y=None, ids=None, use_cv=False) -> pd.DataFrame:
    """One row per sample: T², Q, leverage, residual, flags and a plain-English reason.

    use_cv=True: for the calibration samples, judge the residual with the
    cross-validated prediction (the honest one); otherwise use model.predict.
    """
    d = model.diag
    T, q, _ = _scores_and_q(model, X_raw, d["P"])
    t2 = np.sum(T ** 2 / d["t_var"], axis=1)
    lev = 1 / d["n"] + np.einsum("ij,jk,ik->i", T, d["TtT_inv"], T)
    pred = model.cv_pred if use_cv else model.predict(X_raw)
    out = pd.DataFrame({"sample_id": ids if ids is not None else np.arange(len(t2)),
                        "predicted": pred, "T2": t2, "Q": q, "leverage": lev})
    if y is not None:
        y = np.asarray(y, float)
        out["reference"] = y
        out["residual"] = pred - y
        out["stud_residual"] = out["residual"] / d["resid_sd"]

    reasons, levels = [], []
    lo, hi = d["y_range"]
    for i in range(len(out)):
        r, level = [], 0
        if t2[i] > d["t2_lim99"]:
            r.append(f"extreme in model space (T² {t2[i]:.1f} > 99% limit {d['t2_lim99']:.1f})")
            level = 2
        elif t2[i] > d["t2_lim95"]:
            r.append("somewhat extreme (T² above 95% limit)")
            level = max(level, 1)
        if q[i] > d["q_lim99"]:
            r.append(f"spectrum not explained by the model (Q {q[i] / d['q_lim99']:.1f}x the 99% limit)")
            level = 2
        elif q[i] > d["q_lim95"]:
            r.append("unusual spectral residual (Q above 95% limit)")
            level = max(level, 1)
        if y is not None and not np.isnan(y[i]):
            s = out["stud_residual"].iat[i]
            if abs(s) > 3:
                r.append(f"reference value disagrees with spectrum ({s:+.1f} typical errors) - "
                         f"check lab value / sample label")
                level = 2
            elif abs(s) > 2.5:
                r.append("large residual (> 2.5 typical errors)")
                level = max(level, 1)
        if pred[i] < lo - 0.1 * (hi - lo) or pred[i] > hi + 0.1 * (hi - lo):
            r.append("prediction outside the calibration range (extrapolation)")
            level = max(level, 1)
        reasons.append("; ".join(r))
        levels.append(["ok", "warning", "outlier"][level])
    out["status"] = levels
    out["reason"] = reasons
    return out
