"""FermenTrack dashboard (Streamlit).

Run with:  streamlit run app.py      (or double-click FermenTrack.bat)

Pages
* Batch monitor     - replay one batch "live": trajectories, final-ethanol forecast, alerts
* Plant overview    - every batch at a glance, alerts table, calibration health chart
* Model performance - how good each model is, measured honestly on held-out batches
* About             - what is simulated and how the pieces fit together
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fermentrack import config as C
from fermentrack import pipeline
from fermentrack.yield_model import FEATURES

st.set_page_config(page_title="FermenTrack", page_icon="🧪", layout="wide")

# One fixed colour per analyte, everywhere (colour follows the entity, never its rank)
COLOR = {
    "ethanol": "#2a78d6", "glucose": "#eb6834", "dp2": "#1baf7a", "dp3": "#eda100",
    "dp4plus": "#e87ba4", "glycerol": "#008300", "lactic": "#6250d6", "acetic": "#e34948",
}
MUTED = "#898781"
SEVERITY_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}


# ============================================================================
# Data loading (runs the pipeline once if results are missing)
# ============================================================================
@st.cache_resource(show_spinner=False)
def load_all():
    out, data = C.OUTPUT_DIR, C.DATA_DIR
    if not (out / "models.pkl").exists() or not (data / "spectra.npz").exists():
        with st.spinner("First start: simulating the plant and training the models (~20 s)..."):
            pipeline.run(verbose=False)
    with open(out / "models.pkl", "rb") as f:
        res = pickle.load(f)
    z = np.load(data / "spectra.npz")
    return {
        "res": res,
        "batches": pd.read_csv(data / "batches.csv"),
        "truth": pd.read_csv(data / "process_true.csv"),
        "lab": pd.read_csv(data / "lab_hplc.csv"),
        "spectra": z["spectra"].astype(float), "wn": z["wavenumbers"],
        "pred": pd.read_csv(out / "inline_predictions.csv"),
        "mon": pd.read_csv(out / "monitoring.csv"),
        "alerts": pd.read_csv(out / "alerts.csv"),
        "verdicts": pd.read_csv(out / "alert_evaluation.csv"),
        "resid": pd.read_csv(out / "lab_vs_inline.csv"),
        "pls": pd.read_csv(out / "pls_performance.csv"),
        "yeval": pd.read_csv(out / "yield_performance.csv"),
        "ytest": pd.read_csv(out / "yield_test_predictions.csv"),
        "metrics": json.loads((out / "metrics.json").read_text()),
    }


def base_layout(fig, title, ytitle, height=300, xtitle="Time (h)"):
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=15)), height=height,
        margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="left", x=0),
        xaxis_title=xtitle, yaxis_title=ytitle)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridwidth=1, zeroline=False)
    return fig


def show(fig):
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def analyte_chart(title, analytes, pred_b, lab_b, t_now, truth_b=None, envelope=None,
                  hlines=(), height=300):
    fig = go.Figure()
    if envelope is not None:
        t, lo, hi = envelope
        fig.add_trace(go.Scatter(x=np.r_[t, t[::-1]], y=np.r_[hi, lo[::-1]], fill="toself",
                                 fillcolor="rgba(137,135,129,0.18)", line=dict(width=0),
                                 name="Good-batch range (±2 SD)", hoverinfo="skip"))
    for a in analytes:
        p = pred_b[pred_b.time_h <= t_now]
        fig.add_trace(go.Scatter(x=p.time_h, y=p[a], mode="lines", name=C.LABELS[a],
                                 line=dict(color=COLOR[a], width=2),
                                 hovertemplate="%{y:.2f} " + C.UNIT))
        l = lab_b[lab_b.time_h <= t_now]
        fig.add_trace(go.Scatter(x=l.time_h, y=l[a], mode="markers", name=f"{C.LABELS[a]} (lab)",
                                 marker=dict(color=COLOR[a], size=9, symbol="diamond",
                                             line=dict(color="white", width=1.5)),
                                 showlegend=False, hovertemplate="lab %{y:.2f}"))
        if truth_b is not None:
            fig.add_trace(go.Scatter(x=truth_b.time_h, y=truth_b[a], mode="lines",
                                     line=dict(color=COLOR[a], width=1, dash="dot"),
                                     name=f"{C.LABELS[a]} (true)", showlegend=False,
                                     hovertemplate="true %{y:.2f}"))
    for y, label in hlines:
        fig.add_hline(y=y, line=dict(color=MUTED, width=1, dash="dash"),
                      annotation_text=label, annotation_position="top left")
    fig.update_xaxes(range=[0, C.BATCH_DURATION_H])
    return base_layout(fig, title, C.UNIT, height)


# ============================================================================
# Page: batch monitor
# ============================================================================
def page_batch(D):
    res, batches = D["res"], D["batches"]
    st.sidebar.markdown("### Batch")
    labels = {r.batch_id: f"Batch {r.batch_id:02d} · {'test' if r.split == 'test' else 'history'}"
              for r in batches.itertuples()}
    qp = st.query_params                       # e.g. ?batch=68&t=30 opens that view directly
    default = int(qp.get("batch", batches[batches.split == "test"].batch_id.iloc[0]))
    bid = st.sidebar.selectbox("Select batch", list(labels), index=default,
                               format_func=labels.get)
    t_default = round(float(qp.get("t", C.BATCH_DURATION_H)) * 3) / 3
    t_now = st.sidebar.slider("Simulated clock (h since fill)", 0.0, C.BATCH_DURATION_H,
                              t_default, 1 / 3, format="%.1f h")
    show_truth = st.sidebar.checkbox("Overlay true concentrations (simulation only)", False)

    pred_b = D["pred"][D["pred"].batch_id == bid]
    lab_b = D["lab"][D["lab"].batch_id == bid]
    mon_b = D["mon"][D["mon"].batch_id == bid]
    truth_b = D["truth"][D["truth"].batch_id == bid] if show_truth else None
    al_b = D["alerts"][(D["alerts"].batch_id == bid) & (D["alerts"].time_h <= t_now + 1e-6)]
    now = pred_b[pred_b.time_h <= t_now + 1e-6].iloc[-1]

    st.title(f"Batch {bid:02d} — live monitor")
    st.caption(f"Replaying the in-line mid-IR predictions up to **{t_now:.1f} h**. "
               "Lines = in-line PLS predictions every 20 min, diamonds = lab HPLC samples.")

    # --- headline numbers ------------------------------------------------------
    # Do not forecast from data the probe itself has flagged as unreliable:
    # once a probe failure is raised, keep the last forecast made before it.
    probe_fail = al_b[(al_b.category == "Measurement") & (al_b.severity == "high")]
    t_fc = min(t_now, float(probe_fail.time_h.min()) - 1e-3) if len(probe_fail) else t_now
    fc = res.yield_predictor.predict_batch(pred_b, t_fc)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ethanol now", f"{now.ethanol:.2f} {C.UNIT}")
    c2.metric("Glucose now", f"{now.glucose:.2f} {C.UNIT}")
    if fc:
        c3.metric("Forecast final ethanol", f"{fc['final_ethanol']:.2f} {C.UNIT}",
                  help="90 % interval from leave-one-batch-out cross-validation")
        c3.caption(f"90 % range {fc['low']:.2f} – {fc['high']:.2f} · model from {fc['cutoff_h']:.0f} h data"
                   + (" · frozen: probe flagged" if t_fc < t_now else ""))
        c4.metric("Forecast yield", f"{fc['yield_pct']:.1f} %",
                  help=f"Theoretical ethanol from starting sugar: {fc['theoretical_ethanol']:.2f} {C.UNIT}")
        c4.caption("of theoretical (0.511 × starting sugar)")
    else:
        c3.metric("Forecast final ethanol", "—", help="Available from 8 h")
        c3.caption("Needs at least 8 h of data")
        c4.metric("Forecast yield", "—")

    # --- alerts ------------------------------------------------------------------
    st.subheader(f"Alerts ({len(al_b)})")
    if al_b.empty:
        st.success("No alerts so far: spectra inside the calibration space, mass balance closed, "
                   "trajectory within the good-batch range.")
    for a in al_b.itertuples():
        box = st.error if a.severity == "high" else st.warning
        box(f"{SEVERITY_ICON.get(a.severity, '')} **{a.title}** — {a.category}, at {a.time_h:.1f} h\n\n"
            f"{a.message}")

    # --- trajectories ------------------------------------------------------------------
    env = res.detector.envelope
    t_env = C.TIMES_H[: len(env.mean)]
    left, right = st.columns(2)
    with left:
        show(analyte_chart("Ethanol", ["ethanol"], pred_b, lab_b, t_now, truth_b,
                           (t_env, env.mean.ethanol - 2 * env.sd.ethanol,
                            env.mean.ethanol + 2 * env.sd.ethanol)))
        show(analyte_chart("Organic acids (contamination markers)", ["lactic", "acetic"], pred_b,
                           lab_b, t_now, truth_b,
                           hlines=[(C.LACTIC_ALERT, "lactic action level"),
                                   (C.ACETIC_ALERT, "acetic action level")]))
    with right:
        show(analyte_chart("Sugars", ["glucose", "dp2", "dp3", "dp4plus"], pred_b, lab_b, t_now,
                           truth_b))
        show(analyte_chart("Glycerol", ["glycerol"], pred_b, lab_b, t_now, truth_b))

    # --- forecast evolution --------------------------------------------------------
    rows = [res.yield_predictor.predict_batch(pred_b, c) for c in res.yield_predictor.cutoffs
            if c <= t_fc + 1e-6]
    if rows:
        f = pd.DataFrame(rows)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=f.cutoff_h, y=f.final_ethanol, mode="lines+markers",
                                 name="Forecast", line=dict(color=COLOR["ethanol"], width=2),
                                 marker=dict(size=9),
                                 error_y=dict(type="data", symmetric=False,
                                              array=f.high - f.final_ethanol,
                                              arrayminus=f.final_ethanol - f.low,
                                              color=COLOR["ethanol"], thickness=1.5),
                                 hovertemplate="%{y:.2f} " + C.UNIT))
        final_lab = lab_b[lab_b.time_h == C.BATCH_DURATION_H]
        if t_now >= C.BATCH_DURATION_H - 1e-6 and len(final_lab):
            fig.add_hline(y=float(final_lab.ethanol.iat[0]), line=dict(color=MUTED, dash="dash"),
                          annotation_text="actual final (lab)", annotation_position="bottom right")
        show(base_layout(fig, "How the final-ethanol forecast evolved", C.UNIT, 280,
                         "Forecast made at (h)"))

    # --- measurement trust --------------------------------------------------------------
    with st.expander("Measurement trust: spectral Q / T² and mass balance", expanded=bool(
            (al_b.category == "Measurement").any())):
        m = mon_b[mon_b.time_h <= t_now + 1e-6]
        mon = res.detector.monitor
        a, b = st.columns(2)
        with a:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=m.time_h, y=m.Q / mon.q_limit, name="Q / limit",
                                     line=dict(color=COLOR["ethanol"], width=2)))
            fig.add_trace(go.Scatter(x=m.time_h, y=m.T2 / mon.t2_limit, name="T² / limit",
                                     line=dict(color=COLOR["glucose"], width=2)))
            fig.add_hline(y=1, line=dict(color=MUTED, dash="dash"), annotation_text="99.9 % limit")
            fig.update_yaxes(type="log")
            show(base_layout(fig, "Spectral PCA monitor (ratio to limit, log scale)", "ratio"))
            st.caption(f"PCA on calibration spectra, {mon.n_components} components. "
                       "Q = what the model cannot explain (new substance, fouling, drift); "
                       "T² = unusual but known variation.")
        with b:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=m.time_h, y=100 * (m.balance_ratio - 1), name="This batch",
                                     line=dict(color=COLOR["ethanol"], width=2)))
            fig.add_trace(go.Scatter(x=t_env, y=100 * (env.mean.balance_ratio - 1), name="Good batches",
                                     line=dict(color=MUTED, width=1, dash="dot")))
            show(base_layout(fig, "Mass balance: change in total glucose-equivalent", "%"))
            st.caption("Sugars + ethanol, glycerol and acids (as glucose equivalents) should stay "
                       "constant. Real process faults keep the balance; a dirty probe breaks it.")

    with st.expander("Latest spectrum"):
        rows_b = np.where(D["mon"].batch_id.to_numpy() == bid)[0]
        i = rows_b[min(int(round(t_now / C.SAMPLE_INTERVAL_H)), len(rows_b) - 1)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=D["wn"], y=D["spectra"][i], name=f"t = {t_now:.1f} h",
                                 line=dict(color=COLOR["ethanol"], width=2)))
        fig.update_xaxes(autorange="reversed")
        show(base_layout(fig, "Mid-IR absorbance", "Absorbance (AU)", 280, "Wavenumber (cm⁻¹)"))

    with st.expander("Ground truth (only the simulator knows this)"):
        r = batches.set_index("batch_id").loc[bid]
        st.write(f"**Scenario:** {C.SCENARIO_LABELS[r.scenario]}"
                 + (f" (fault starts at {r.fault_onset_h:.1f} h)" if pd.notna(r.fault_onset_h) else "")
                 + (f" · instrument drift {r.wn_shift_cm1:.2f} cm⁻¹" if r.instrument_drift else ""))
        st.write(f"**True final ethanol:** {r.final_ethanol_true:.2f} {C.UNIT} "
                 f"({r.yield_pct_true:.1f} % of theoretical)")


# ============================================================================
# Page: plant overview
# ============================================================================
def page_overview(D):
    st.title("Plant overview")
    b = D["batches"].merge(D["verdicts"][["batch_id", "alerts"]])
    final = D["lab"][D["lab"].time_h == C.BATCH_DURATION_H][["batch_id", "ethanol"]]
    b = b.merge(final.rename(columns={"ethanol": "final_ethanol_lab"}))
    b["yield_pct_lab"] = 100 * b.final_ethanol_lab / (C.ETHANOL_YIELD * b.initial_sugar)

    m = D["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Batches", f"{m['n_batches']}", help=f"{m['n_train']} history (training) + {m['n_test']} test")
    c2.metric("Test faults detected", f"{100 * m['test_fault_detection_rate']:.0f} %")
    c3.metric("False alarms on normal test batches", f"{100 * m['test_false_alarm_rate_normal']:.0f} %")
    c4.metric("Ethanol calibration drift flagged at", f"batch {m['calibration_alarms']['ethanol']}",
              help=f"Instrument drift was simulated from batch {C.DRIFT_START_BATCH}")

    fig = go.Figure()
    for flagged, color, name in [(False, COLOR["ethanol"], "No process/measurement alert"),
                                 (True, COLOR["acetic"], "Alert raised")]:
        s = b[(b.alerts != "-") == flagged]
        fig.add_trace(go.Scatter(x=s.batch_id, y=s.yield_pct_lab, mode="markers", name=name,
                                 marker=dict(color=color, size=10, line=dict(color="white", width=1.5)),
                                 customdata=s[["alerts"]], hovertemplate=
                                 "Batch %{x}<br>%{y:.1f} % of theoretical<br>%{customdata[0]}<extra></extra>"))
    fig.add_vline(x=C.N_TRAIN_BATCHES - 0.5, line=dict(color=MUTED, dash="dash"),
                  annotation_text="test period →")
    fig.add_vline(x=C.DRIFT_START_BATCH - 0.5, line=dict(color=MUTED, dash="dot"),
                  annotation_text="instrument drift starts", annotation_position="bottom right")
    show(base_layout(fig, "Final yield per batch (lab HPLC)", "% of theoretical", 360, "Batch"))

    st.subheader("All alerts")
    cats = st.multiselect("Category", sorted(D["alerts"].category.unique()),
                          default=sorted(D["alerts"].category.unique()))
    a = D["alerts"][D["alerts"].category.isin(cats)]
    st.dataframe(a[["batch_id", "time_h", "category", "severity", "title", "message"]],
                 use_container_width=True, hide_index=True,
                 column_config={"time_h": st.column_config.NumberColumn("time (h)", format="%.1f"),
                                "message": st.column_config.TextColumn(width="large")})

    st.subheader("Batch table")
    show_truth = st.checkbox("Show simulation ground truth", True)
    cols = ["batch_id", "split", "final_ethanol_lab", "yield_pct_lab", "alerts"]
    if show_truth:
        cols += ["scenario", "instrument_drift", "fault_onset_h"]
    st.dataframe(b[cols].round(2), use_container_width=True, hide_index=True)


# ============================================================================
# Page: model performance
# ============================================================================
def page_performance(D):
    st.title("Model performance")
    st.caption("Everything is scored on batches the models never saw. Training = batches 0–55 "
               "(the plant's past), test = batches 56–79 (its future).")
    sections = {"pls": ("Spectral calibration (PLS)", perf_pls),
                "yield": ("Early yield forecast", perf_yield),
                "anomaly": ("Anomaly detection", perf_anomaly),
                "health": ("Calibration health", page_health)}
    keys = list(sections)
    wanted = st.query_params.get("section")            # e.g. ?page=performance&section=yield
    start = keys.index(wanted) if wanted in keys else 0
    key = st.radio("Section", keys, index=start, horizontal=True, label_visibility="collapsed",
                   format_func=lambda k: sections[k][0])
    sections[key][1](D)


def perf_pls(D):
    res = D["res"]
    p = D["pls"].copy()
    show_cols = {"label": "Analyte", "preprocessing": "Preprocessing (chosen by CV)",
                 "components": "PLS comps", "RMSECV": "RMSECV", "RMSEP_test_clean": "RMSEP test",
                 "RMSEP_test_drift": "RMSEP drift period", "R2_CV": "R² CV",
                 "range_min": "cal. min", "range_max": "cal. max"}
    st.dataframe(p[list(show_cols)].rename(columns=show_cols).round(3), hide_index=True,
                 use_container_width=True)
    st.caption(f"Errors in {C.UNIT}. RMSECV = batch-wise 8-fold cross-validation on training "
               "lab samples. RMSEP test = clean test batches. The drift period shows how an "
               "ageing instrument degrades a fixed calibration.")
    a = st.selectbox("Analyte", C.ANALYTES, format_func=C.LABELS.get)
    cal = res.calibration.models[a]
    r = D["resid"].merge(D["batches"][["batch_id", "split", "instrument_drift"]])
    test = r[(r.split == "test") & r.probe_ok]
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cal.y_cal, y=cal.cv_pred, mode="markers", name="CV (training)",
                                 marker=dict(color=MUTED, size=8, opacity=0.6)))
        for drift, color, name in [(False, COLOR["ethanol"], "Test"),
                                   (True, COLOR["glucose"], "Test, drift period")]:
            s = test[test.instrument_drift == drift]
            fig.add_trace(go.Scatter(x=s[f"{a}_lab"], y=s[f"{a}_pred"], mode="markers", name=name,
                                     marker=dict(color=color, size=8,
                                                 line=dict(color="white", width=1))))
        lim = [0, float(max(cal.y_cal.max(), test[f"{a}_lab"].max())) * 1.05]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="1:1",
                                 line=dict(color=MUTED, dash="dash", width=1)))
        show(base_layout(fig, f"{C.LABELS[a]}: predicted vs lab", f"In-line ({C.UNIT})", 380,
                         f"Lab HPLC ({C.UNIT})"))
    with c2:
        fig = go.Figure()
        k = np.arange(1, len(cal.rmsecv_curve) + 1)
        fig.add_trace(go.Scatter(x=k, y=cal.rmsecv_curve, mode="lines+markers", name="RMSECV",
                                 line=dict(color=COLOR["ethanol"], width=2)))
        fig.add_vline(x=cal.n_components, line=dict(color=MUTED, dash="dash"),
                      annotation_text=f"chosen: {cal.n_components}")
        show(base_layout(fig, "Choosing the number of PLS components", f"RMSECV ({C.UNIT})",
                         380, "Components"))
    st.markdown("**Preprocessing recipes tried** (best first)")
    st.dataframe(cal.candidates.round(4), hide_index=True, use_container_width=True)


def perf_yield(D):
    res = D["res"]
    y = D["yeval"]
    fig = go.Figure()
    for col, name, color, dash in [
            ("RMSE_model", "Model, all test batches", COLOR["ethanol"], "solid"),
            ("RMSE_baseline", "Baseline, all test batches", COLOR["ethanol"], "dot"),
            ("RMSE_model_faulty", "Model, batches with faults", COLOR["glucose"], "solid"),
            ("RMSE_baseline_faulty", "Baseline, batches with faults", COLOR["glucose"], "dot"),
            ("RMSE_model_normal", "Model, normal batches", COLOR["dp2"], "solid"),
            ("RMSE_baseline_normal", "Baseline, normal batches", COLOR["dp2"], "dot")]:
        fig.add_trace(go.Scatter(x=y.cutoff_h, y=y[col], mode="lines+markers", name=name,
                                 line=dict(color=color, width=2, dash=dash), marker=dict(size=8)))
    show(base_layout(fig, "Final-ethanol forecast error vs. how early it is made", f"RMSE ({C.UNIT})",
                     380, "Forecast made at (h)"))
    st.info("**How to read this.** Baseline = starting sugar × typical yield of past batches. "
            "On normal batches both are already close to the lab's own error, because yield "
            "varies little. The model earns its keep on problem batches: by 20–24 h it picks up "
            "slow ethanol production and acid build-up and roughly halves the error.")
    c = st.select_slider("Forecast made at", res.yield_predictor.cutoffs, 20)
    t = D["ytest"][D["ytest"].cutoff_h == c]
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        for normal, color, name in [(True, COLOR["ethanol"], "Normal"),
                                    (False, COLOR["glucose"], "Fault scenario")]:
            s = t[(t.scenario == "normal") == normal]
            fig.add_trace(go.Scatter(
                x=s.actual, y=s.predicted, mode="markers", name=name,
                marker=dict(color=color, size=9, line=dict(color="white", width=1)),
                error_y=dict(type="data", symmetric=False, array=s.high - s.predicted,
                             arrayminus=s.predicted - s.low, color=color, thickness=1),
                customdata=s[["batch_id", "scenario"]],
                hovertemplate="Batch %{customdata[0]} (%{customdata[1]})<br>actual %{x:.2f}"
                              "<br>forecast %{y:.2f}<extra></extra>"))
        lim = [t[["actual", "low"]].min().min() - 0.2, t[["actual", "high"]].max().max() + 0.2]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="1:1",
                                 line=dict(color=MUTED, dash="dash", width=1)))
        show(base_layout(fig, f"Test batches: forecast at {c} h vs actual", f"Forecast ({C.UNIT})",
                         400, f"Actual final ethanol, lab ({C.UNIT})"))
    with c2:
        coef = res.yield_predictor.models[c].coefficients()
        fig = go.Figure(go.Bar(x=coef.values * 100, y=[FEATURES[k] for k in coef.index],
                               orientation="h", marker=dict(color=COLOR["ethanol"]),
                               hovertemplate="%{x:+.2f} points of yield<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        show(base_layout(fig, "What drives the forecast", "", 400,
                         "Effect of +1 SD (yield, % points)"))
    st.dataframe(y.round(3), hide_index=True, use_container_width=True)


def perf_anomaly(D):
    v = D["verdicts"]
    test = v[v.split == "test"]
    st.dataframe(test, hide_index=True, use_container_width=True)
    summ = (v[v.scenario != "normal"].groupby(["split", "scenario"])
            .agg(batches=("batch_id", "size"), detected=("detected", "mean"),
                 median_delay_h=("delay_h", "median")).reset_index())
    st.markdown("**Detection by fault type** (delay = hours from the moment the fault starts "
                "in the simulation, e.g. bacteria entering, to the alarm)")
    st.dataframe(summ.round(2), hide_index=True, use_container_width=True)
    fa = v[v.scenario == "normal"].groupby("split").false_alarm.mean()
    st.write("False-alarm rate on normal batches: " +
             ", ".join(f"{k}: {100 * x:.0f} %" for k, x in fa.items()))


def page_health(D):
    checks = {c["analyte"]: c for c in D["res"].health_checks}
    a = st.selectbox("Analyte to monitor", list(checks), format_func=C.LABELS.get, key="health_a")
    c = checks[a]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c["batch_ids"], y=c["batch_median"], mode="markers",
                             name="Batch median (in-line − lab)",
                             marker=dict(color=MUTED, size=8)))
    fig.add_trace(go.Scatter(x=c["batch_ids"], y=c["ewma"], mode="lines+markers", name="EWMA",
                             line=dict(color=COLOR[a], width=2), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=c["batch_ids"], y=c["limit"], mode="lines", name="Control limit",
                             line=dict(color=MUTED, dash="dash", width=1)))
    fig.add_trace(go.Scatter(x=c["batch_ids"], y=-c["limit"], mode="lines", showlegend=False,
                             line=dict(color=MUTED, dash="dash", width=1)))
    if c["alarm_batch"] is not None:
        fig.add_vline(x=c["alarm_batch"], line=dict(color=COLOR["acetic"], width=1),
                      annotation_text=f"alarm: batch {c['alarm_batch']}")
    fig.add_vline(x=C.DRIFT_START_BATCH - 0.5, line=dict(color=MUTED, dash="dot"),
                  annotation_text="drift starts (simulated)", annotation_position="bottom right")
    show(base_layout(fig, f"Calibration health: {C.LABELS[a]}", f"Bias ({C.UNIT})", 380, "Batch"))
    if c["alarm_batch"] is not None:
        txt = (f"Alarm at batch **{c['alarm_batch']}**, estimated bias **{c['bias_estimate']:+.3f} "
               f"{C.UNIT}**.")
        if "rmse_after_bias_fix" in c:
            txt += (f" A rolling bias update from that point (latest 3 batches of lab results) would "
                    f"have reduced the error on the following lab samples from "
                    f"{c['rmse_after_no_fix']:.3f} to {c['rmse_after_bias_fix']:.3f} {C.UNIT}.")
        st.warning(txt)
    else:
        st.success("No drift detected for this analyte.")


# ============================================================================
# Page: about
# ============================================================================
def page_about(D):
    st.title("About FermenTrack")
    st.markdown(f"""
FermenTrack is a **portfolio prototype** for process analytics in fuel-ethanol fermentation,
built to show an end-to-end workflow from raw in-line mid-IR spectra to decisions.
All data are **simulated** (fixed random seed {C.SEED}); no plant data are used.

**1. Simulated plant** — {C.N_BATCHES} batches of {C.BATCH_DURATION_H:.0f} h corn-mash SSF:
enzyme hydrolysis (DP4+ → DP3 → DP2 → glucose), yeast fermentation with ethanol and acid
inhibition, plus faults: lactic or acetic bacteria, stuck fermentation (yeast stress),
probe fouling and slow instrument drift (from batch {C.DRIFT_START_BATCH}).
A spectrum every 20 min from Gaussian pure-component bands, with noise, baseline wander and a
batch-specific matrix; ten HPLC lab samples per batch.

**2. Chemometrics** — per analyte: preprocessing chosen by batch-wise cross-validation
(baseline, SNV, Savitzky–Golay derivatives), NumPy NIPALS PLS1, number of components from the
RMSECV curve.

**3. Process intelligence**
* *Early yield forecast*: chemistry gives the theoretical ceiling; a robust regression predicts
  the efficiency from early signs.
* *Anomaly detection*: first “can I trust the measurement?” (PCA Q/T², mass balance), then
  “is the chemistry going wrong?” (acid action levels, golden-batch envelope), with a reason for
  every alarm.
* *Calibration health*: EWMA chart of in-line minus lab, with a bias-update recommendation.
""")


# ============================================================================
def main():
    D = load_all()
    st.sidebar.title("🧪 FermenTrack")
    st.sidebar.caption("Fermentation process intelligence · simulated data")
    pages = ["Batch monitor", "Plant overview", "Model performance", "About"]
    keys = {"monitor": 0, "overview": 1, "performance": 2, "about": 3}
    page = st.sidebar.radio("Page", pages, index=keys.get(st.query_params.get("page", ""), 0))
    {"Batch monitor": page_batch, "Plant overview": page_overview,
     "Model performance": page_performance, "About": page_about}[page](D)


main()
