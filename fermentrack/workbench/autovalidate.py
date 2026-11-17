"""Automatic self-check of a saved model against new lab (reference) results.

Run it whenever new lab values arrive, by hand or on a schedule:

    python -m fermentrack.workbench.autovalidate --model Ethanol --data new_lab.xlsx
    python -m fermentrack.workbench.autovalidate --model Ethanol --data new_lab.xlsx --auto-update

What it checks (all relative to what the model promised at calibration time):
* accuracy  - RMSEP of the new samples vs the model's own RMSECV
* bias      - is the average error significantly different from zero? (t-test)
* trend     - EWMA of the average error over past checks of THIS version (slow drift)
* spectra   - how many new spectra fall outside the model (Q / T² outliers)

Verdict:
    OK       nothing to do
    WARNING  keep an eye on it (one criterion mildly off)
    ACTION   the model needs attention; the report says whether a bias update,
             an update with the new samples, or a lab check is the likely fix

With --auto-update, an ACTION verdict triggers registry.update_candidates();
the recommended candidate is saved as a new version and only made active if it
beat the current version on held-out new samples. Every check is appended to
checks.csv so the history can be plotted and audited.
"""
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats as sstats

from . import io, registry
from .diagnostics import sample_diagnostics
from .models import regression_stats

EWMA_LAMBDA = 0.3


def check(pkg: registry.ModelPackage, X, wn, y, ids=None, history: pd.DataFrame | None = None) -> dict:
    """Judge the model on new samples that have reference values."""
    y = np.asarray(y, float)
    ok = ~np.isnan(y)
    X, y = np.asarray(X, float)[ok], y[ok]
    ids = np.arange(len(y)).astype(str) if ids is None else np.asarray(ids)[ok]
    if pkg.model.wn.shape != np.asarray(wn).shape or not np.allclose(pkg.model.wn, wn):
        order = np.argsort(wn)
        X = np.array([np.interp(pkg.model.wn, np.asarray(wn)[order], x[order]) for x in X])
    pred = pkg.predict(X)
    st = regression_stats(y, pred)
    expected = pkg.model.cv_stats()["RMSE"]
    e = pred - y
    p_bias = float(sstats.ttest_1samp(e, 0).pvalue) if len(e) >= 3 else 1.0
    diag = sample_diagnostics(pkg.model, X, y + pkg.bias_correction, ids)   # residual in model units
    frac_spec_out = float(diag.reason.str.contains("spectrum not explained|extreme in model").mean())

    # EWMA of the mean error across checks (history from earlier runs)
    past = [] if history is None or not len(history) else list(history["bias"])
    z = 0.0
    for b in past + [st["bias"]]:
        z = EWMA_LAMBDA * b + (1 - EWMA_LAMBDA) * z
    n_checks = len(past) + 1
    ewma_lim = 3 * expected / np.sqrt(max(len(y), 1)) * np.sqrt(
        EWMA_LAMBDA / (2 - EWMA_LAMBDA) * (1 - (1 - EWMA_LAMBDA) ** (2 * n_checks)))

    ratio = st["RMSE"] / expected
    findings, level = [], 0
    if ratio > 2:
        findings.append(f"error is {ratio:.1f}x what the model promised (RMSECV {expected:.3g})")
        level = 2
    elif ratio > 1.5:
        findings.append(f"error is {ratio:.1f}x the model's RMSECV")
        level = max(level, 1)
    if p_bias < 0.01 and abs(st["bias"]) > 0.5 * expected:
        findings.append(f"systematic bias {st['bias']:+.3g} (p = {p_bias:.3g})")
        level = 2
    elif p_bias < 0.05:
        findings.append(f"possible bias {st['bias']:+.3g} (p = {p_bias:.3g})")
        level = max(level, 1)
    if abs(z) > ewma_lim and n_checks >= 2:
        findings.append(f"slow drift: EWMA of bias {z:+.3g} beyond limit ±{ewma_lim:.3g}")
        level = 2
    if frac_spec_out > 0.2:
        findings.append(f"{100 * frac_spec_out:.0f}% of new spectra are outside the model "
                        f"(new variation the calibration has not seen)")
        level = 2
    elif frac_spec_out > 0.05:        # at a 99 % limit ~1 % is expected by pure chance
        findings.append(f"{100 * frac_spec_out:.0f}% of new spectra flagged as unusual")
        level = max(level, 1)

    # what is the most likely fix?
    if level == 2:
        if frac_spec_out > 0.2 or (st["SEP"] > 1.5 * expected):
            action = ("Update the model with the new samples (augment) - the spread of errors grew "
                      "or the spectra contain new variation.")
        elif abs(st["bias"]) > 0.5 * expected:
            action = "Bias update - errors are shifted by a constant amount but their spread is normal."
        else:
            action = "Check the lab values and sample handling for the worst samples."
    else:
        action = "No action needed." if level == 0 else "Monitor: re-check with the next lab samples."

    return {"time": datetime.now().isoformat(timespec="seconds"), "model": pkg.name,
            "version": pkg.version, "n": st["n"], "RMSEP": st["RMSE"], "bias": st["bias"],
            "SEP": st["SEP"], "R2": st["R2"], "expected_RMSE": expected, "ratio": ratio,
            "p_bias": p_bias, "ewma_bias": z, "ewma_limit": ewma_lim,
            "frac_spectral_outliers": frac_spec_out,
            "verdict": ["OK", "WARNING", "ACTION"][level], "findings": "; ".join(findings) or "-",
            "recommended_action": action, "samples": diag}


def run(name, X, wn, y, ids=None, groups=None, auto_update=False, root=registry.DEFAULT_ROOT) -> dict:
    """Check the active version, log it, and (optionally) update the model."""
    pkg = registry.load(name, root=root)
    hp = registry.history_path(name, root)
    history = pd.read_csv(hp) if hp.exists() else None
    if history is not None and "version" in history:
        # the drift test (EWMA) starts fresh for every version: an old model's errors
        # say nothing about the new one
        history = history[history.version == pkg.version]
    res = check(pkg, X, wn, y, ids, history)
    res["update"] = None
    if auto_update and res["verdict"] == "ACTION":
        cands = registry.update_candidates(pkg, X, y, ids if ids is not None else np.arange(len(y)), groups)
        rec = cands["recommended"]
        res["update"] = {"table": cands["table"], "recommended": rec, "saved": None}
        if rec != "current":
            new = cands["bias" if rec == "bias update" else "augment"]
            new.notes += " (automatic update)"
            try:
                res["update"]["saved"] = registry.save(new, root, activate=True)
            except registry.DuplicateVersion:
                res["update"]["recommended"] += " (identical version already saved)"
    log = {k: v for k, v in res.items() if k not in ("samples", "update")}
    log["auto_update"] = None if res["update"] is None else (res["update"]["saved"] or "kept current")
    pd.DataFrame([log]).to_csv(hp, mode="a", header=not hp.exists(), index=False)
    return res


def _print(res):
    print(f"\nModel {res['model']} {res['version']}: {res['verdict']}")
    print(f"  {res['n']} new samples, RMSEP {res['RMSEP']:.4g} (model promised {res['expected_RMSE']:.4g}), "
          f"bias {res['bias']:+.4g}")
    print(f"  Findings: {res['findings']}")
    print(f"  Recommended: {res['recommended_action']}")
    if res["update"]:
        print("\n" + res["update"]["table"].round(4).to_string(index=False))
        print(f"  -> {res['update']['recommended']}"
              + (f", saved and activated as {res['update']['saved']}" if res["update"]["saved"] else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check a saved FermenTrack model against new lab data.")
    ap.add_argument("--model", required=True, help="model name in the registry (e.g. Ethanol)")
    ap.add_argument("--data", required=True, help="spectra file (Excel/CSV), may include reference values")
    ap.add_argument("--reference", help="separate reference file (sample_id + property columns)")
    ap.add_argument("--property", help="column with the reference value (default: the model's property)")
    ap.add_argument("--auto-update", action="store_true", help="save an improved version if needed")
    ap.add_argument("--registry", default=str(registry.DEFAULT_ROOT))
    a = ap.parse_args(argv)
    ds = io.read_spectra(a.data, name=a.data)
    if a.reference:
        ds = io.attach_reference(ds, *io.read_reference(a.reference, name=a.reference))
    pkg = registry.load(a.model, root=a.registry)
    prop = a.property or pkg.property
    if prop not in ds.Y.columns:
        raise SystemExit(f"Column '{prop}' not found. Available: {list(ds.Y.columns)}")
    groups = ds.meta["group"].to_numpy() if "group" in ds.meta else None
    res = run(a.model, ds.X, ds.wn, ds.Y[prop].to_numpy(), ds.ids, groups, a.auto_update, a.registry)
    _print(res)
    return 0 if res["verdict"] != "ACTION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
