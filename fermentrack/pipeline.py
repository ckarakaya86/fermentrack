"""Run the whole FermenTrack workflow end to end and save the results.

    python -m fermentrack.pipeline

Steps
1. Simulate the plant history (80 batches, spectra, lab samples).
2. Calibrate PLS models on lab samples from the training batches.
3. Predict concentrations for every spectrum (the "in-line" trajectories).
4. Build the anomaly detector and scan every batch.
5. Check calibration health on the test period (EWMA of in-line minus lab).
6. Train and test the early final-ethanol predictor.
7. Save everything for the dashboard (data/ and outputs/).
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from . import dataset
from .anomaly import batch_signals, build_detector, evaluate as evaluate_alerts
from .calibration_health import WATCHED, batch_sigma, check_analyte, health_alerts, residual_table
from .chemometrics import SpectralCalibration, rmse, r2
from . import model_source
from .workbench import registry as wb_registry
from .yield_model import EarlyYieldPredictor


@dataclass
class Results:
    calibration: SpectralCalibration
    detector: object
    yield_predictor: EarlyYieldPredictor
    health_checks: list
    metrics: dict
    saved_models: dict = None        # analyte -> active ModelPackage from the workbench (if used)


def log(msg, t0):
    print(f"[{time.time() - t0:5.1f}s] {msg}", flush=True)


def run(seed: int = C.SEED, save: bool = True, verbose: bool = True,
        use_saved_models: bool = True, registry_root=None):
    """Run everything. With use_saved_models, an active Model-Workbench version replaces the
    built-in PLS model for its analyte (see model_source.py)."""
    t0 = time.time()
    say = (lambda m: log(m, t0)) if verbose else (lambda m: None)

    # 1) data -------------------------------------------------------------------
    data = dataset.generate(seed)
    if save:
        dataset.save(data)
    b = data.batches.set_index("batch_id")
    say(f"Simulated {len(b)} batches, {len(data.process)} spectra, {len(data.lab)} lab samples")

    lab = data.lab.merge(data.batches[["batch_id", "split", "scenario", "instrument_drift"]])
    # Calibration set: lab samples of training batches. Batches with a probe-maintenance
    # event in the log (fouling) are left out, as a plant chemometrician would do.
    cal = lab[(lab.split == "train") & (lab.scenario != "probe_fouling")]

    # 2) PLS calibration ---------------------------------------------------------
    calib = SpectralCalibration.fit(data.spectra[cal.spectrum_row], cal, cal.batch_id.to_numpy())
    say("PLS calibration done")

    # 3) in-line predictions for every spectrum ---------------------------------
    pred = calib.predict(data.spectra)
    root = wb_registry.DEFAULT_ROOT if registry_root is None else registry_root
    saved, skipped = (model_source.active_saved_models(data.wavenumbers, root)
                      if use_saved_models else ({}, []))
    for a, pkg in saved.items():                      # your maintained model replaces the built-in one
        pred[a] = np.clip(pkg.predict(data.spectra, data.wavenumbers), 0, None)
        say(f"{C.LABELS[a]}: using saved model {model_source.describe(pkg)}")
    for name, why in skipped:
        say(f"Saved model {name} not used: {why}")
    pred.insert(0, "time_h", data.process["time_h"].to_numpy())
    pred.insert(0, "batch_id", data.process["batch_id"].to_numpy())
    lab_pred = pred.iloc[lab.spectrum_row.to_numpy()].reset_index(drop=True)

    perf = calib.summary()
    perf["model_source"] = "built-in PLS"
    for a, pkg in saved.items():                      # report the model that is actually used
        i = perf.index[perf.analyte == a][0]
        cv = pkg.model.cv_stats()
        perf.loc[i, ["model_source", "preprocessing", "components", "RMSECV", "R2_CV"]] = [
            "saved: " + model_source.describe(pkg), pkg.model.pipeline.describe(), pkg.model.k,
            cv["RMSE"], cv["R2"]]
        perf.loc[i, ["range_min", "range_max"]] = [float(pkg.model.y_cal.min()), float(pkg.model.y_cal.max())]
    groups = {
        "RMSEP_test_clean": (lab.split == "test") & ~lab.instrument_drift & (lab.scenario != "probe_fouling"),
        "RMSEP_test_drift": lab.instrument_drift,
        "RMSEP_test_fouling": (lab.split == "test") & (lab.scenario == "probe_fouling"),
    }
    for name, m in groups.items():
        m = m.to_numpy()
        perf[name] = [rmse(lab.loc[m, a], lab_pred.loc[m, a]) for a in perf.analyte]
    m = groups["RMSEP_test_clean"].to_numpy()
    perf["R2_test_clean"] = [r2(lab.loc[m, a], lab_pred.loc[m, a]) for a in perf.analyte]

    # 4) anomaly detection ----------------------------------------------------------
    cal_batches = cal.batch_id.unique()
    cal_rows = np.isin(data.process.batch_id, cal_batches)
    normal_train = b[(b.split == "train") & (b.scenario == "normal")].index
    normal_signals = [batch_signals(pred[pred.batch_id == i]) for i in normal_train]
    detector = build_detector(normal_signals, data.spectra[cal_rows], data.wavenumbers)

    monitoring, alerts = [], []
    for i in b.index:
        rows = data.batch_rows(i)
        s = detector.monitor_batch(pred.iloc[rows], data.spectra[rows])
        s.insert(0, "batch_id", i)
        monitoring.append(s)
        alerts += detector.alerts_for_batch(i, s, data.spectra[rows])
    monitoring = pd.concat(monitoring, ignore_index=True)
    say(f"Anomaly scan done (PCA: {detector.monitor.n_components} components)")

    # 5) calibration health (test period only, skip samples taken while the probe failed)
    res = residual_table(lab, lab_pred)
    probe_bad = monitoring["flag_balance"].to_numpy()[lab.spectrum_row.to_numpy()]
    res["probe_ok"] = ~probe_bad
    groups_cal = cal.batch_id.to_numpy()
    def sigma(a):        # expected batch-to-batch scatter of the model that is actually used
        if a in saved:
            m = saved[a].model
            g = saved[a].train_groups if saved[a].train_groups is not None else m.cal_ids
            return batch_sigma(m.cv_pred - m.y_cal, np.asarray(g))
        return batch_sigma(calib.models[a].cv_pred - calib.models[a].y_cal, groups_cal)
    checks = [check_analyte(res[res.probe_ok], a, sigma(a), C.N_TRAIN_BATCHES) for a in WATCHED]
    alerts += health_alerts(checks)
    alerts = pd.DataFrame(alerts).sort_values(["batch_id", "time_h"]).reset_index(drop=True)
    verdicts = evaluate_alerts(alerts, data.batches)

    # 6) early yield prediction -------------------------------------------------------
    final_lab = lab[lab.time_h == C.BATCH_DURATION_H].set_index("batch_id")["ethanol"]
    train_ids = b[b.split == "train"].index
    test_ids = b[b.split == "test"].index
    yp = EarlyYieldPredictor().fit(pred, final_lab.loc[train_ids])
    yield_eval = yp.evaluate(pred, final_lab.loc[test_ids], b[b.scenario == "normal"].index)
    yield_test = pd.concat([yp.test_predictions(pred, final_lab.loc[test_ids], c).assign(cutoff_h=c)
                            for c in yp.cutoffs], ignore_index=True)
    yield_test = yield_test.merge(data.batches[["batch_id", "scenario"]])
    say("Early yield models done")

    # 7) metrics + save ----------------------------------------------------------------
    test_v = verdicts[verdicts.split == "test"]
    faulty = test_v[test_v.scenario != "normal"]
    metrics = {
        "n_batches": int(len(b)), "n_train": int(len(train_ids)), "n_test": int(len(test_ids)),
        "pca_components": detector.monitor.n_components,
        "pca_explained": detector.monitor.explained,
        "test_fault_detection_rate": float(faulty.detected.mean()),
        "test_false_alarm_rate_normal": float(test_v[test_v.scenario == "normal"].false_alarm.mean()),
        "test_median_delay_h": float(faulty.delay_h.median()),
        "calibration_alarms": {c["analyte"]: c["alarm_batch"] for c in checks},
        "yield_20h": yield_eval.set_index("cutoff_h").loc[20].round(3).to_dict(),
        "model_sources": {a: model_source.describe(pkg) for a, pkg in saved.items()},
        "saved_models_skipped": [f"{n}: {why}" for n, why in skipped],
        # saved models that already saw test-period batches give optimistic test errors
        "saved_models_seen_test_batches": {
            a: sorted(b_ for b_ in model_source.trained_on_batches(pkg) if b_ >= C.N_TRAIN_BATCHES)
            for a, pkg in saved.items()
            if any(b_ >= C.N_TRAIN_BATCHES for b_ in model_source.trained_on_batches(pkg))},
        "registry_signature": wb_registry.signature(root) if use_saved_models else "",
    }
    results = Results(calib, detector, yp, checks, metrics, saved)
    if save:
        out = C.OUTPUT_DIR
        out.mkdir(parents=True, exist_ok=True)
        pred.to_csv(out / "inline_predictions.csv", index=False)
        monitoring.to_csv(out / "monitoring.csv", index=False)
        alerts.to_csv(out / "alerts.csv", index=False)
        verdicts.to_csv(out / "alert_evaluation.csv", index=False)
        res.to_csv(out / "lab_vs_inline.csv", index=False)
        perf.to_csv(out / "pls_performance.csv", index=False)
        yield_eval.to_csv(out / "yield_performance.csv", index=False)
        yield_test.to_csv(out / "yield_test_predictions.csv", index=False)
        with open(out / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(out / "models.pkl", "wb") as f:
            pickle.dump(results, f)
        say(f"Saved results to {out}")
    if verbose:
        _report(perf, yield_eval, verdicts, checks, metrics)
    return results


def _report(perf, yield_eval, verdicts, checks, metrics):
    pd.set_option("display.width", 160)
    print("\n=== PLS calibration (% w/v) ===")
    print(perf[["analyte", "preprocessing", "components", "RMSECV", "RMSEP_test_clean",
                "RMSEP_test_drift", "R2_test_clean"]].round(3).to_string(index=False))
    print("\n=== Early final-ethanol prediction, test batches (% w/v) ===")
    print(yield_eval.round(3).to_string(index=False))
    print("\n=== Anomaly detection, test batches ===")
    print(verdicts[verdicts.split == "test"].to_string(index=False))
    print("\n=== Calibration health (first EWMA alarm, batch) ===")
    for c in checks:
        print(f"  {c['analyte']:9s} alarm at batch {c['alarm_batch']}  bias {c['bias_estimate']}")
    print("\n", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    # import through the package so pickled classes are stored as fermentrack.pipeline.*
    # (not __main__.*), otherwise the dashboard could not load them
    from fermentrack.pipeline import run as _run
    _run()
