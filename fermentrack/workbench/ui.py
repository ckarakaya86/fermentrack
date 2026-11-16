"""Streamlit interface of the FermenTrack Model Workbench.

`render()` draws the whole workbench; it is used both as a page inside the
FermenTrack dashboard and by the stand-alone launcher (workbench_app.py).

Steps (sidebar):
  1 Data          load spectra + reference values, inspect, exclude samples by hand
  2 Spectra       overlay plots, wavenumber regions, preprocessing
  3 Split         calibration / validation division
  4 Model         PLS or ANN, cross-validation, figures of merit, comparison table
  5 Outliers      T² / Q / residual diagnostics, exclude with a reason (logged)
  6 Robustness    perturbation, resampling and y-randomisation tests
  7 Save & update model registry: versions, model cards, update with new samples
  8 Auto-check    validate a saved model against new lab data (also schedulable)
All state lives in st.session_state under keys starting with "wb_".
"""
from __future__ import annotations

import io as _io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

from . import autovalidate, diagnostics, io, registry, robustness, split
from .models import ModelSpec, fit_model, regression_stats
from .preprocess import BASELINES, SCATTER, Pipeline, region_mask

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]     # one-hue sequential ramp
STEPS = ["1 Data", "2 Spectra", "3 Split", "4 Model", "5 Outliers", "6 Robustness",
         "7 Save & update", "8 Auto-check"]
STEP_KEYS = {"data": 0, "spectra": 1, "split": 2, "model": 3, "outliers": 4, "robustness": 5,
             "registry": 6, "autocheck": 7}


# ============================================================================
# helpers
# ============================================================================
def S():
    return st.session_state


def layout(fig, title, ytitle, xtitle, height=380):
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=15)), height=height,
                      margin=dict(l=10, r=10, t=40, b=10), xaxis_title=xtitle, yaxis_title=ytitle,
                      legend=dict(orientation="h", yanchor="top", y=-0.25, x=0))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(zeroline=False)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", theme="streamlit")


def need(*keys, msg="Load data in step 1 first."):
    if any(S().get(k) is None for k in keys):
        st.info(msg)
        st.stop()


def ds():
    return S()["wb_ds"]


def prop():
    return S().get("wb_prop")


def y_all():
    return ds().Y[prop()].to_numpy(float)


def groups_all():
    d = ds()
    return d.meta["group"].to_numpy() if "group" in d.meta.columns else None


def included_mask():
    """Samples that are not excluded and have a reference value for the chosen property."""
    exc = S().get("wb_excluded", {})
    return ~np.isin(ds().ids, list(exc)) & ~np.isnan(y_all())


def current_pipeline() -> Pipeline:
    return Pipeline.from_settings(S().get("wb_pipe", Pipeline().settings()))


def val_mask():
    v = S().get("wb_val")
    return np.zeros(ds().n, bool) if v is None or len(v) != ds().n else v


def colors_for(values):
    v = np.asarray(values, float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    t = np.zeros_like(v) if hi <= lo else (v - lo) / (hi - lo)
    scale = [[i / (len(RAMP) - 1), c] for i, c in enumerate(RAMP)]
    return sample_colorscale(scale, np.nan_to_num(t, nan=0.0).tolist())


def reset_models():
    S()["wb_model"] = None


def set_dataset(d: io.SpectralDataset):
    S()["wb_ds"] = d
    S()["wb_excluded"] = {}
    S()["wb_val"] = None
    S()["wb_models"] = {}
    S()["wb_model"] = None
    S()["wb_prop"] = d.properties[0] if d.properties else None
    S()["wb_pipe"] = Pipeline().settings()


def flash(msg):
    """Show a success message after the page reloads (st.rerun would erase it otherwise)."""
    S()["wb_flash"] = msg


def show_flash():
    msg = S().pop("wb_flash", None)
    if msg:
        st.success(msg)


def sidebar_status():
    st.sidebar.markdown("---")
    if S().get("wb_ds") is None:
        st.sidebar.caption("No data loaded.")
        return
    d = ds()
    st.sidebar.caption(f"**Data:** {d.n} samples × {len(d.wn)} points  \n"
                       f"**Property:** {prop()}  \n"
                       f"**Excluded:** {len(S().get('wb_excluded', {}))}  \n"
                       f"**Preprocessing:** {current_pipeline().describe()}")
    m = S().get("wb_model")
    if m is not None:
        st.sidebar.caption(f"**Model:** {m.spec.describe().split(',')[0]}, k = {m.k}, "
                           f"RMSECV {m.cv_stats()['RMSE']:.4g}")


# ============================================================================
# 1 Data
# ============================================================================
def step_data():
    st.header("1 · Data")
    st.caption("Spectra file: rows = samples with wavenumbers as column headers, or columns = samples "
               "with wavenumbers in the first column. Reference values "
               "can be in the same file or in a separate file matched by sample id.")
    src = st.radio("Source", ["Upload my files", "Example: simulated FTIR (fermentation)"],
                   horizontal=True, key="wb_src")
    if src.startswith("Example"):
        if st.button("Load example calibration data", type="primary"):
            set_dataset(read_example("ftir_calibration.xlsx"))
            S()["wb_prop"] = "Ethanol"
            st.rerun()
    else:
        c1, c2 = st.columns(2)
        spec_file = c1.file_uploader("Spectra (Excel / CSV)", type=["xlsx", "xls", "csv", "txt"])
        ref_file = c2.file_uploader("Reference values (optional)", type=["xlsx", "xls", "csv"])
        sheet = 0
        if spec_file is not None and spec_file.name.lower().endswith(("xlsx", "xls")):
            sheets = io.excel_sheets(spec_file)
            if len(sheets) > 1:
                sheet = c1.selectbox("Sheet", sheets)
        if spec_file is not None and st.button("Read files", type="primary"):
            try:
                d = io.read_spectra(spec_file, sheet=sheet, name=spec_file.name)
                if ref_file is not None:
                    d = io.attach_reference(d, *io.read_reference(ref_file, name=ref_file.name))
                set_dataset(d)
                st.rerun()
            except Exception as e:                       # show the reason in plain words
                st.error(f"Could not read the file: {e}")

    if S().get("wb_ds") is None:
        return
    d = ds()
    if not d.properties:
        st.warning("No reference values found. Add a reference file to build models.")
        return
    s = d.summary()
    c = st.columns(4)
    c[0].metric("Samples", s["samples"])
    c[1].metric("Spectral points", s["points"])
    c[2].metric("Range (cm⁻¹)", s["range_cm1"])
    c[3].metric("Excluded", len(S()["wb_excluded"]))
    new_prop = st.selectbox("Property to model", d.properties, index=d.properties.index(prop()))
    if new_prop != prop():
        S()["wb_prop"] = new_prop
        reset_models()

    st.subheader("Samples")
    st.caption("Tick **Exclude** to leave a sample out of everything that follows, and write why "
               "(the reason is saved with the model).")
    exc = S()["wb_excluded"]
    table = pd.DataFrame({"sample_id": d.ids})
    for col in d.meta.columns:
        table[col] = d.meta[col].astype(str).to_numpy()
    for col in d.properties:
        table[col] = d.Y[col].to_numpy()
    table.insert(0, "Exclude", [i in exc for i in d.ids])
    table.insert(1, "Reason", [exc.get(i, "") for i in d.ids])
    edited = st.data_editor(table, hide_index=True, width="stretch", height=320,
                            disabled=[c for c in table.columns if c not in ("Exclude", "Reason")],
                            key="wb_data_editor")
    new_exc = {r.sample_id: (r.Reason or "excluded by hand") for r in edited.itertuples() if r.Exclude}
    if new_exc != exc:
        S()["wb_excluded"] = new_exc
        reset_models()

    st.subheader("Reference values")
    y = y_all()
    fig = go.Figure(go.Histogram(x=y[included_mask()], marker_color=BLUE, nbinsx=30,
                                 name="included"))
    show(layout(fig, f"Distribution of {prop()}", "samples", prop(), 280))

    buf = _io.BytesIO()
    io.to_excel(d.subset(~np.isin(d.ids, list(exc))), buf)
    st.download_button("Download cleaned data (Excel)", buf.getvalue(), "cleaned_data.xlsx")


# ============================================================================
# 2 Spectra & preprocessing
# ============================================================================
def step_spectra():
    st.header("2 · Spectra & preprocessing")
    need("wb_ds")
    d = ds()
    p = current_pipeline()
    wmin, wmax = float(d.wn.min()), float(d.wn.max())

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        reg_text = c1.text_input("Wavenumber regions (cm⁻¹)",
                                 ", ".join(f"{min(r):g}-{max(r):g}" for r in p.regions),
                                 placeholder=f"all ({wmin:g}-{wmax:g}); e.g. 980-1200, 1250-1480")
        baseline = c2.selectbox("Baseline", list(BASELINES), list(BASELINES).index(p.baseline),
                                format_func=BASELINES.get)
        scatter = c3.selectbox("Scatter correction", list(SCATTER), list(SCATTER).index(p.scatter),
                               format_func=SCATTER.get)
        c4, c5, c6 = st.columns(3)
        smooth = c4.select_slider("SG smoothing window", [0, 5, 7, 9, 11, 15, 21], p.smooth or 0)
        deriv = c5.radio("Derivative", [0, 1, 2], p.deriv, horizontal=True,
                         format_func=lambda k: ["none", "1st", "2nd"][k])
        dwin = c6.select_slider("Derivative window", [5, 7, 9, 11, 15, 21, 31], p.deriv_window)
    try:
        regions = []
        for part in [x for x in reg_text.replace(";", ",").split(",") if x.strip()]:
            lo, hi = [float(v) for v in part.replace("–", "-").split("-") if v.strip()]
            regions.append((lo, hi))
    except ValueError:
        st.error("Regions must look like 980-1200, 1250-1480")
        regions = p.regions
    new = Pipeline(regions, baseline, scatter, smooth, deriv, dwin)
    if new.settings() != p.settings():
        S()["wb_pipe"] = new.settings()
        reset_models()
        p = new

    inc = included_mask()
    color_by = st.radio("Colour by", [prop(), "calibration / validation", "excluded"],
                        horizontal=True)
    max_lines = st.slider("Maximum spectra drawn", 20, 500, 150, 10,
                          help="Drawing many lines is slow; a random selection is shown.")
    idx = np.arange(d.n)
    if len(idx) > max_lines:
        idx = np.sort(np.random.default_rng(0).choice(idx, max_lines, replace=False))
    y, val = y_all(), val_mask()
    if color_by == prop():
        cols = colors_for(np.where(inc, y, np.nanmean(y)))
    elif color_by.startswith("calibration"):
        cols = [ORANGE if v else BLUE for v in val]
    else:
        cols = [MUTED if not i else BLUE for i in inc]

    Xp = p.fit(d.X[inc], d.wn).transform(d.X, d.wn) if inc.any() else p.transform(d.X, d.wn)
    wn_p = p.wavenumbers(d.wn)
    for title, X, w in [("Raw spectra", d.X, d.wn), (f"After preprocessing: {p.describe()}", Xp, wn_p)]:
        fig = go.Figure()
        for i in idx:
            fig.add_trace(go.Scatter(x=w, y=X[i], mode="lines", showlegend=False,
                                       line=dict(color=cols[i], width=1,
                                                 dash="dot" if not inc[i] else "solid"),
                                       opacity=0.8 if inc[i] else 0.5,
                                       hovertemplate=f"{d.ids[i]}<br>{prop()} = {y[i]:.4g}<extra></extra>"))
        if title == "Raw spectra" and p.regions:
            for lo, hi in p.regions:
                fig.add_vrect(x0=min(lo, hi), x1=max(lo, hi), fillcolor=BLUE, opacity=0.07, line_width=0)
        fig.update_xaxes(autorange="reversed")
        show(layout(fig, title, "Absorbance" if title == "Raw spectra" else "processed", "Wavenumber (cm⁻¹)"))
    if color_by == prop():
        st.caption(f"Light → dark blue = low → high {prop()}. Dotted = excluded samples.")

    with st.expander("Mean spectrum ± 2 SD and correlation with the reference value"):
        m, sd = Xp[inc].mean(axis=0), Xp[inc].std(axis=0)
        fig = go.Figure([
            go.Scatter(x=np.r_[wn_p, wn_p[::-1]], y=np.r_[m + 2 * sd, (m - 2 * sd)[::-1]], fill="toself",
                       fillcolor="rgba(42,120,214,0.15)", line=dict(width=0), name="± 2 SD"),
            go.Scatter(x=wn_p, y=m, line=dict(color=BLUE, width=2), name="mean")])
        fig.update_xaxes(autorange="reversed")
        show(layout(fig, "Mean spectrum", "processed", "Wavenumber (cm⁻¹)", 300))
        yc = y[inc] - y[inc].mean()
        Xc = Xp[inc] - m
        corr = (Xc.T @ yc) / (np.sqrt((Xc ** 2).sum(0)) * np.sqrt((yc ** 2).sum()) + 1e-12)
        fig = go.Figure(go.Scatter(x=wn_p, y=corr, line=dict(color=ORANGE, width=2), name="r"))
        fig.update_xaxes(autorange="reversed")
        show(layout(fig, f"Correlation of each wavenumber with {prop()}", "r", "Wavenumber (cm⁻¹)", 300))
        st.caption("Regions with strong positive or negative correlation carry information about the "
                   "property; flat regions near zero are candidates to leave out.")


# ============================================================================
# 3 Split
# ============================================================================
def step_split():
    st.header("3 · Calibration / validation split")
    need("wb_ds")
    d = ds()
    inc = included_mask()
    c1, c2, c3 = st.columns(3)
    method = c1.selectbox("Method", list(split.METHODS), format_func=split.METHODS.get,
                          index=list(split.METHODS).index("by_group") if groups_all() is not None else 0)
    frac = c2.slider("Validation share", 0.1, 0.5, 0.25, 0.05)
    seed = c3.number_input("Random seed", 0, 9999, 0)
    st.caption({
        "random": "Simple random draw.",
        "kennard_stone": "Calibration samples are chosen to cover the spectral space evenly; validation "
                         "samples are the interior ones. Good coverage, but optimistic validation.",
        "y_stratified": "Every k-th sample by reference value goes to validation: both sets cover the "
                        "whole range.",
        "by_group": "Whole batches / days / lots go to validation together - the honest choice when "
                    "samples of one group are correlated.",
        "by_order": "The newest samples (by date, else file order) are the validation set - "
                    "'train on the past, test on the future'.",
    }[method])
    if st.button("Apply split", type="primary"):
        try:
            Xp = current_pipeline().fit_transform(d.X[inc], d.wn)
            g = groups_all()
            order = d.meta["date"].to_numpy() if "date" in d.meta.columns else None
            v = split.make_split(method, Xp, y_all()[inc], None if g is None else g[inc],
                                 None if order is None else order[inc], frac, int(seed))
            full = np.zeros(d.n, bool)
            full[np.where(inc)[0]] = v
            S()["wb_val"] = full
            reset_models()
            st.rerun()
        except Exception as e:
            st.error(str(e))
    if S().get("wb_val") is None:
        st.info("No split yet: all included samples are used for calibration (cross-validation only).")
        return
    val = val_mask() & inc
    cal = ~val_mask() & inc
    c = st.columns(3)
    c[0].metric("Calibration", int(cal.sum()))
    c[1].metric("Validation", int(val.sum()))
    y = y_all()
    c[2].metric("Validation inside calibration range",
                f"{100 * np.mean((y[val] >= y[cal].min()) & (y[val] <= y[cal].max())):.0f} %"
                if val.any() and cal.any() else "-")
    l, r = st.columns(2)
    with l:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=y[cal], name="calibration", marker_color=BLUE, opacity=0.75))
        fig.add_trace(go.Histogram(x=y[val], name="validation", marker_color=ORANGE, opacity=0.75))
        fig.update_layout(barmode="overlay")
        show(layout(fig, f"{prop()} in each set", "samples", prop(), 340))
    with r:
        Xp = current_pipeline().fit(d.X[cal], d.wn).transform(d.X, d.wn)
        Xc = Xp - Xp[cal].mean(axis=0)
        _, sv, Vt = np.linalg.svd(Xc[cal], full_matrices=False)
        T = Xc @ Vt[:2].T
        ev = sv ** 2 / np.sum(sv ** 2)
        fig = go.Figure()
        for m, colr, name in [(cal, BLUE, "calibration"), (val, ORANGE, "validation")]:
            fig.add_trace(go.Scatter(x=T[m, 0], y=T[m, 1], mode="markers", name=name,
                                       marker=dict(color=colr, size=8, line=dict(color="white", width=1)),
                                       text=d.ids[m], hovertemplate="%{text}<extra></extra>"))
        show(layout(fig, "PCA scores of the spectra", f"PC2 ({100 * ev[1]:.1f} %)",
                    f"PC1 ({100 * ev[0]:.1f} %)", 340))


# ============================================================================
# 4 Model
# ============================================================================
def model_label(m):
    return f"{m.spec.kind} k={m.k} | {m.pipeline.describe()}"


def step_model():
    st.header("4 · Model")
    need("wb_ds")
    d = ds()
    inc = included_mask()
    val, cal = val_mask() & inc, ~val_mask() & inc
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        kind = c1.radio("Model type", ["PLS", "ANN"], horizontal=True,
                        help="ANN = small neural network fed with PLS scores (captures curvature).")
        auto_k = c2.checkbox("Choose size by cross-validation", True)
        k = None if auto_k else c2.number_input("PLS components (LVs)", 1, 30, 5)
        scaling = c3.selectbox("Scaling", ["mean", "auto"],
                               format_func={"mean": "Mean-centring", "auto": "Autoscaling"}.get)
        cv = c4.selectbox("Cross-validation", ["auto", "group", "loo", "5", "10"],
                          format_func={"auto": "Auto", "group": "By group", "loo": "Leave-one-out",
                                       "5": "5-fold", "10": "10-fold"}.get,
                          help="Auto: by group if groups exist, else leave-one-out (< 40 samples) or 10-fold.")
        hidden, alpha, n_ens = (8,), 1e-2, 5
        if kind == "ANN":
            a1, a2, a3 = st.columns(3)
            htxt = a1.text_input("Hidden layers (neurons)", "8", help="e.g. 8  or  16,8")
            hidden = tuple(int(x) for x in htxt.replace(" ", "").split(",") if x)
            alpha = a2.select_slider("Regularisation (alpha)", [1e-4, 1e-3, 1e-2, 1e-1, 1.0], 1e-2)
            n_ens = a3.slider("Networks in ensemble", 1, 10, 5)
        max_k = st.slider("Maximum components tried", 3, 30, 15)
    spec = ModelSpec(kind, k, max_k, scaling, hidden, alpha, n_ens, cv)
    if st.button("Fit model", type="primary"):
        bar = st.progress(0.0, "Fitting and cross-validating...")
        try:
            g = groups_all()
            m = fit_model(spec, current_pipeline(), d.X[cal], d.wn, y_all()[cal], prop(),
                          None if g is None else g[cal], d.ids[cal],
                          progress=lambda f: bar.progress(f, "Cross-validating the ANN..."))
            S()["wb_model"] = m
            S().setdefault("wb_models", {})[model_label(m)] = m
        except Exception as e:
            st.error(str(e))
        bar.empty()
    m = S().get("wb_model")
    if m is None:
        st.info("Set the options and press **Fit model**.")
        return

    y = y_all()
    rows = [{"set": "Calibration (fit)", **m.cal_stats()}, {"set": "Cross-validation", **m.cv_stats()}]
    if val.any():
        rows.append({"set": "Validation", **regression_stats(y[val], m.predict(d.X[val]))})
    stats = pd.DataFrame(rows)
    st.dataframe(stats.round(4), hide_index=True, width="stretch")
    st.caption("RMSE = typical error in the property's units. RMSEC (fit) is always optimistic; judge the "
               "model by RMSECV and the validation RMSEP. Bias = average error. SEP = spread after removing "
               "bias. RPD = SD of the reference ÷ SEP (> 3 good for screening, > 5 for quality control).")

    l, r = st.columns(2)
    with l:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=m.y_cal, y=m.cv_pred, mode="markers", name="calibration (CV)",
                                   marker=dict(color=BLUE, size=8, line=dict(color="white", width=1)),
                                   text=m.cal_ids, hovertemplate="%{text}<br>ref %{x:.4g}<br>pred %{y:.4g}<extra></extra>"))
        if val.any():
            fig.add_trace(go.Scatter(x=y[val], y=m.predict(d.X[val]), mode="markers", name="validation",
                                       marker=dict(color=ORANGE, size=8, line=dict(color="white", width=1)),
                                       text=d.ids[val], hovertemplate="%{text}<br>ref %{x:.4g}<br>pred %{y:.4g}<extra></extra>"))
        lim = [float(np.nanmin(y[inc])), float(np.nanmax(y[inc]))]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="1:1", line=dict(color=MUTED, dash="dash")))
        show(layout(fig, "Predicted vs reference", "predicted", "reference"))
    with r:
        kk = np.arange(1, len(m.cv_curve) + 1)
        fig = go.Figure(go.Scatter(x=kk, y=m.cv_curve, mode="lines+markers", name="RMSECV (PLS)",
                                   line=dict(color=BLUE, width=2)))
        fig.add_vline(x=m.k, line=dict(color=MUTED, dash="dash"), annotation_text=f"used: {m.k}")
        show(layout(fig, "Choosing the number of components", "RMSECV", "components"))
        st.caption("Pick the point where the curve stops falling clearly; more components fit noise. "
                   + ("For the ANN, this is the number of PLS scores fed to the network." if m.spec.kind == "ANN" else ""))

    if m.spec.kind == "PLS":
        b = m.core.B[:, m.k - 1] / m.core.x_scale
        fig = go.Figure(go.Scatter(x=m.pipeline.wavenumbers(m.wn), y=b, line=dict(color=BLUE, width=2),
                                   name="coefficient"))
        fig.update_xaxes(autorange="reversed")
        show(layout(fig, "Regression coefficients", "coefficient", "Wavenumber (cm⁻¹)", 300))

    comp = S().get("wb_models", {})
    if len(comp) > 1:
        st.subheader("Compare models fitted in this session")
        rows = []
        for lab, mm in comp.items():
            r_ = {"model": lab, "RMSEC": mm.cal_stats()["RMSE"], "RMSECV": mm.cv_stats()["RMSE"],
                  "R2 CV": mm.cv_stats()["R2"]}
            if val.any() and mm.property == prop():
                r_["RMSEP"] = regression_stats(y[val], mm.predict(d.X[val]))["RMSE"]
            rows.append(r_)
        st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
        st.caption("Prefer the simplest model whose RMSECV and RMSEP are close to the best. A model "
                   "with a much lower RMSEC than RMSECV is overfitting.")
        if st.button("Clear comparison"):
            S()["wb_models"] = {model_label(m): m}
            st.rerun()


# ============================================================================
# 5 Outliers
# ============================================================================
def step_outliers():
    st.header("5 · Outliers")
    need("wb_ds")
    need("wb_model", msg="Fit a model in step 4 first.")
    d, m = ds(), S()["wb_model"]
    inc = included_mask()
    val, cal = val_mask() & inc, ~val_mask() & inc
    if len(m.y_cal) != cal.sum():
        st.warning("The data changed since the model was fitted. Re-fit in step 4.")
        st.stop()
    dc = diagnostics.sample_diagnostics(m, d.X[cal], y_all()[cal], d.ids[cal], use_cv=True)
    dc["set"] = "calibration"
    parts = [dc]
    if val.any():
        dv = diagnostics.sample_diagnostics(m, d.X[val], y_all()[val], d.ids[val])
        dv["set"] = "validation"
        parts.append(dv)
    D = pd.concat(parts, ignore_index=True)
    lim = m.diag

    c = st.columns(3)
    c[0].metric("Flagged as outlier", int((D.status == "outlier").sum()))
    c[1].metric("Warnings", int((D.status == "warning").sum()))
    c[2].metric("Typical CV error", f"{lim['resid_sd']:.4g}")

    l, r = st.columns(2)
    sym = {"calibration": "circle", "validation": "diamond"}
    colr = {"ok": BLUE, "warning": "#eda100", "outlier": "#e34948"}
    with l:
        fig = go.Figure()
        for (st_, se), g in D.groupby(["status", "set"]):
            fig.add_trace(go.Scatter(x=g.T2, y=g.Q, mode="markers", name=f"{st_} ({se})",
                                       marker=dict(color=colr[st_], symbol=sym[se], size=9,
                                                   line=dict(color="white", width=1)),
                                       text=g.sample_id, hovertemplate="%{text}<br>T² %{x:.3g}<br>Q %{y:.3g}<extra></extra>"))
        fig.add_vline(x=lim["t2_lim99"], line=dict(color=MUTED, dash="dash"), annotation_text="T² 99 %")
        fig.add_hline(y=lim["q_lim99"], line=dict(color=MUTED, dash="dash"), annotation_text="Q 99 %")
        show(layout(fig, "Spectral outliers: T² vs Q", "Q residual", "Hotelling T²"))
        st.caption("Right = extreme but understood by the model. Top = the spectrum contains something "
                   "the model cannot describe (fouling, wrong sample, contamination).")
    with r:
        fig = go.Figure()
        for (st_, se), g in D.groupby(["status", "set"]):
            fig.add_trace(go.Scatter(x=g.leverage, y=g.stud_residual, mode="markers", name=f"{st_} ({se})",
                                       marker=dict(color=colr[st_], symbol=sym[se], size=9,
                                                   line=dict(color="white", width=1)),
                                       text=g.sample_id, hovertemplate="%{text}<br>lev %{x:.3g}<br>resid %{y:.2f}<extra></extra>"))
        for yv in (-3, 3):
            fig.add_hline(y=yv, line=dict(color=MUTED, dash="dash"))
        show(layout(fig, "Reference outliers: residual vs leverage", "studentised residual", "leverage"))
        st.caption("Far above/below ±3: the lab value does not match the spectrum - check for typing "
                   "errors or swapped labels before excluding.")

    st.subheader("Samples needing attention")
    flagged = D[D.status != "ok"].sort_values("status")          # "outlier" before "warning"
    if flagged.empty:
        st.success("No sample is above the 95 % limits.")
        return
    tab = flagged[["sample_id", "set", "status", "reference", "predicted", "T2", "Q",
                   "stud_residual", "reason"]].copy()
    # pre-tick only strong evidence; borderline cases are listed for the user's judgement
    strong = ((tab.stud_residual.abs() > 3) | (tab.Q > 2 * lim["q_lim99"]) | (tab.T2 > 2 * lim["t2_lim99"]))
    tab.insert(0, "Exclude", strong.to_numpy())
    st.caption("Pre-ticked: strong evidence only (residual > 3 typical errors, or Q / T² more than twice "
               "the 99 % limit). Borderline samples are listed for your judgement - a whole batch just "
               "above the limit is usually real variation the model should learn, not an outlier.")
    ed = st.data_editor(tab.round(4), hide_index=True, width="stretch",
                        disabled=[c for c in tab.columns if c != "Exclude"], key="wb_out_editor")
    if st.button("Exclude ticked samples and re-fit", type="primary"):
        exc = dict(S()["wb_excluded"])
        for r_ in ed.itertuples():
            if r_.Exclude:
                exc[r_.sample_id] = f"outlier ({r_.set}): {r_.reason}"
        S()["wb_excluded"] = exc
        g = groups_all()
        inc2 = included_mask()
        cal2 = ~val_mask() & inc2
        spec = ModelSpec.from_settings(m.spec.settings())
        m2 = fit_model(spec, current_pipeline(), d.X[cal2], d.wn, y_all()[cal2], prop(),
                       None if g is None else g[cal2], d.ids[cal2])
        S()["wb_model"] = m2
        S().setdefault("wb_models", {})[model_label(m2) + " (cleaned)"] = m2
        flash(f"Re-fitted without the excluded samples: RMSECV {m.cv_stats()['RMSE']:.4g} → "
              f"{m2.cv_stats()['RMSE']:.4g}")
        st.rerun()

    if S()["wb_excluded"]:
        with st.expander(f"Exclusion log ({len(S()['wb_excluded'])}) - untick to restore"):
            log = pd.DataFrame({"Keep excluded": True, "sample_id": list(S()["wb_excluded"]),
                                "reason": list(S()["wb_excluded"].values())})
            e2 = st.data_editor(log, hide_index=True, width="stretch",
                                disabled=["sample_id", "reason"], key="wb_log_editor")
            if st.button("Apply restore"):
                S()["wb_excluded"] = {r_.sample_id: r_.reason for r_ in e2.itertuples() if r_[1]}
                reset_models()
                st.rerun()


# ============================================================================
# 6 Robustness
# ============================================================================
def step_robustness():
    st.header("6 · Robustness tests")
    need("wb_ds")
    need("wb_model", msg="Fit a model in step 4 first.")
    d, m = ds(), S()["wb_model"]
    inc = included_mask()
    val, cal = val_mask() & inc, ~val_mask() & inc
    test_mask = val if val.any() else cal
    st.caption("Uses the validation set (or calibration set if there is no split). The model itself "
               "is not changed by these tests.")

    st.subheader("A · Disturbed spectra")
    if st.button("Run perturbation test", type="primary"):
        S()["wb_pert"] = robustness.perturbation_test(m, d.X[test_mask], y_all()[test_mask])
    pt = S().get("wb_pert")
    if pt is not None:
        kinds = list(dict.fromkeys(pt.kind))
        fig = make_subplots(rows=1, cols=len(kinds), subplot_titles=[pt[pt.kind == k].test.iat[0] for k in kinds],
                            shared_yaxes=True)
        for i, k in enumerate(kinds):
            g = pt[pt.kind == k]
            fig.add_trace(go.Scatter(x=g.level, y=g.RMSEP_ratio, mode="lines+markers", showlegend=False,
                                     line=dict(color=BLUE, width=2), marker=dict(size=8),
                                     hovertemplate="level %{x}<br>RMSEP × %{y:.2f}<extra></extra>"),
                          row=1, col=i + 1)
            fig.update_xaxes(title_text=g.unit.iat[0], row=1, col=i + 1, showgrid=False)
        fig.add_hline(y=2, line=dict(color=MUTED, dash="dash"))
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10),
                          yaxis_title="RMSEP ÷ undisturbed RMSEP")
        show(fig)
        st.dataframe(robustness.tolerance(pt), hide_index=True, width="stretch")
        st.caption("Lower lines = more robust. The dashed line marks a doubled error. Compare "
                   "preprocessing choices here: derivatives usually tolerate baseline changes, "
                   "SNV/MSC tolerate gain changes.")

    st.subheader("B · Many random splits")
    n_rep = st.slider("Repeats", 5, 50, 20)
    if st.button("Run resampling test"):
        bar = st.progress(0.0)
        S()["wb_resamp"] = robustness.resampling_test(m.spec, current_pipeline(), d.X[inc], d.wn,
                                                      y_all()[inc], n_rep, progress=bar.progress)
        bar.empty()
    rs = S().get("wb_resamp")
    if rs is not None:
        fig = go.Figure()
        for col, colr in [("RMSEC", MUTED), ("RMSECV", BLUE), ("RMSEP", ORANGE)]:
            fig.add_trace(go.Box(y=rs[col], name=col, marker_color=colr, boxpoints="all", jitter=0.4))
        show(layout(fig, f"Error over {len(rs)} random splits", "RMSE", "", 330))
        st.caption(f"RMSEP {rs.RMSEP.mean():.4g} ± {rs.RMSEP.std():.4g}. A wide spread means one split "
                   f"alone cannot be trusted; LVs chosen ranged {rs.LVs.min()}-{rs.LVs.max()}.")

    st.subheader("C · y-randomisation (chance-correlation check)")
    n_perm = st.slider("Shuffles", 5, 30, 10)
    if st.button("Run y-randomisation"):
        bar = st.progress(0.0)
        S()["wb_yrand"] = robustness.y_randomization(m.spec, current_pipeline(), d.X[cal], d.wn,
                                                     y_all()[cal], n_perm, progress=bar.progress)
        bar.empty()
    yr = S().get("wb_yrand")
    if yr is not None:
        fig = go.Figure(go.Bar(x=["real model"] + [f"shuffle {i + 1}" for i in range(len(yr["permuted_RMSECV"]))],
                               y=[yr["real_RMSECV"], *yr["permuted_RMSECV"]],
                               marker_color=[BLUE] + [MUTED] * len(yr["permuted_RMSECV"])))
        show(layout(fig, "RMSECV: real vs shuffled reference values", "RMSECV", "", 300))
        (st.success if yr["passed"] else st.error)(
            f"The best shuffled model is {yr['ratio']:.1f}× worse than the real one. "
            + ("The model learns a real relationship." if yr["passed"]
               else "Too close: the model may be fitting chance correlations - simplify it."))


# ============================================================================
# 7 Save & update
# ============================================================================
def step_registry():
    st.header("7 · Save & update models")
    root = registry.DEFAULT_ROOT
    st.caption(f"Models are stored as numbered versions in `{root}`.")

    st.subheader("Save the current model")
    m = S().get("wb_model")
    if m is None:
        st.info("Fit a model in step 4 to save it.")
    else:
        d = ds()
        c1, c2 = st.columns([1, 2])
        name = c1.text_input("Model name", prop())
        notes = c2.text_input("Notes", f"{m.spec.describe()}; {m.pipeline.describe()}")
        if st.button("Save as new version", type="primary"):
            inc = included_mask()
            val = val_mask() & inc
            g = groups_all()
            exc = [{"sample_id": k, "reason": v} for k, v in S()["wb_excluded"].items()]
            # rebuild from the same recipe with the frozen model size, so the package is self-contained
            spec = ModelSpec.from_settings(m.spec.settings())
            spec.n_components = m.k
            pkg = registry.build_package(name, spec, current_pipeline(), d.X[inc], d.wn, y_all()[inc],
                                         d.ids[inc], None if g is None else g[inc], val[inc], exc,
                                         notes, d.fingerprint())
            ver = registry.save(pkg, root)
            st.success(f"Saved {name} {ver}.")

    models = registry.list_models(root)
    if models.empty:
        return
    st.subheader("Saved models")
    st.dataframe(models.round(4), hide_index=True, width="stretch")
    name = st.selectbox("Model", models.model.tolist(), key="wb_reg_model")
    vers = registry.list_versions(name, root)
    st.dataframe(vers.round(4), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    v_sel = c1.selectbox("Version", vers.version.tolist()[::-1])
    if c2.button("Make this version active"):
        registry.set_active(name, v_sel, root)
        st.rerun()
    with st.expander("Model card (everything about this version)"):
        st.json(json.loads((root / name / f"{v_sel}.json").read_text()))

    st.subheader("Update with new lab-checked samples")
    st.caption("Upload new spectra with reference values. Two candidates are built - a bias update and a "
               "refit with old + new samples - and compared on the newest 30 % of the new samples, "
               "which the candidates never see.")
    f = st.file_uploader("New samples (spectra + reference values)", type=["xlsx", "csv"], key="wb_upd_file")
    use_example = st.checkbox("Use example: ftir_new_lab_drift.xlsx")
    if st.button("Build update candidates") and (f is not None or use_example):
        nd = read_example("ftir_new_lab_drift.xlsx") if use_example else io.read_spectra(f, name=f.name)
        pkg = registry.load(name, root=root)
        if pkg.property not in nd.Y.columns:
            st.error(f"New file has no column '{pkg.property}'.")
        else:
            nX = np.array([np.interp(pkg.model.wn, np.sort(nd.wn), x[np.argsort(nd.wn)]) for x in nd.X])
            g = nd.meta["group"].to_numpy() if "group" in nd.meta.columns else None
            S()["wb_update"] = (name, registry.update_candidates(pkg, nX, nd.Y[pkg.property].to_numpy(),
                                                                 nd.ids, g))
    upd = S().get("wb_update")
    if upd and upd[0] == name:
        cands = upd[1]
        st.dataframe(cands["table"].round(4), hide_index=True, width="stretch")
        rec = cands["recommended"]
        st.info(f"Recommended: **{rec}**" + ("" if rec != "current" else " (no candidate was clearly better)"))
        if rec != "current" and st.button(f"Save '{rec}' as new active version", type="primary"):
            ver = registry.save(cands["bias" if rec == "bias update" else "augment"], root, activate=True)
            S()["wb_update"] = None
            st.success(f"Saved and activated {name} {ver}.")


# ============================================================================
# 8 Auto-check
# ============================================================================
def step_autocheck():
    st.header("8 · Automatic validation")
    root = registry.DEFAULT_ROOT
    models = registry.list_models(root)
    if models.empty:
        st.info("Save a model in step 7 first.")
        return
    st.caption("Checks the active version of a model against new lab results: accuracy vs what the "
               "model promised, bias, slow drift (EWMA over all checks) and spectra outside the model. "
               "The same check can run unattended from the command line (see below).")
    name = st.selectbox("Model", models.model.tolist(), key="wb_chk_model")
    c1, c2 = st.columns(2)
    f = c1.file_uploader("New lab samples (spectra + reference)", type=["xlsx", "csv"], key="wb_chk_file")
    ex = c2.radio("…or an example", ["none", "ftir_new_lab_ok.xlsx", "ftir_new_lab_drift.xlsx"])
    auto = st.checkbox("Allow automatic update (saves a better version if action is needed)")
    if st.button("Run check", type="primary") and (f is not None or ex != "none"):
        nd = read_example(ex) if ex != "none" else io.read_spectra(f, name=f.name)
        pkg = registry.load(name, root=root)
        if pkg.property not in nd.Y.columns:
            st.error(f"New file has no column '{pkg.property}'.")
        else:
            g = nd.meta["group"].to_numpy() if "group" in nd.meta.columns else None
            S()["wb_check"] = autovalidate.run(name, nd.X, nd.wn, nd.Y[pkg.property].to_numpy(),
                                               nd.ids, g, auto, root)
    res = S().get("wb_check")
    if res and res["model"] == name:
        box = {"OK": st.success, "WARNING": st.warning, "ACTION": st.error}[res["verdict"]]
        box(f"**{res['verdict']}** - {name} {res['version']} on {res['n']} new samples: RMSEP "
            f"{res['RMSEP']:.4g} (model promised {res['expected_RMSE']:.4g}), bias {res['bias']:+.4g}.\n\n"
            f"Findings: {res['findings']}\n\n**Recommended:** {res['recommended_action']}")
        if res["update"]:
            st.dataframe(res["update"]["table"].round(4), hide_index=True, width="stretch")
            st.write(f"Automatic update: {res['update']['recommended']}"
                     + (f" → saved and activated as {res['update']['saved']}" if res["update"]["saved"] else ""))
        with st.expander("Per-sample results"):
            st.dataframe(res["samples"].round(4), hide_index=True, width="stretch")

    hp = registry.history_path(name, root)
    if hp.exists():
        h = pd.read_csv(hp)
        st.subheader("Check history")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, len(h) + 1)), y=h.RMSEP, mode="lines+markers",
                                 name="RMSEP", line=dict(color=BLUE, width=2)))
        fig.add_trace(go.Scatter(x=list(range(1, len(h) + 1)), y=h.bias, mode="lines+markers",
                                 name="bias", line=dict(color=ORANGE, width=2)))
        fig.add_trace(go.Scatter(x=list(range(1, len(h) + 1)), y=h.expected_RMSE, mode="lines",
                                 name="promised (RMSECV)", line=dict(color=MUTED, dash="dash")))
        show(layout(fig, f"{name}: error at each check", "error", "check number", 300))
        st.dataframe(h.drop(columns=["model"], errors="ignore").round(4), hide_index=True,
                     width="stretch")

    with st.expander("Run it automatically (Windows Task Scheduler)"):
        root_dir = Path(__file__).resolve().parents[2]
        cmd = (f'py -3.12 -m fermentrack.workbench.autovalidate --model "{name}" '
               f'--data "C:\\path\\to\\new_lab.xlsx" --auto-update')
        st.markdown("1. Put each new lab export (spectra + reference values) in a fixed place.\n"
                    "2. Task Scheduler → *Create Basic Task* → daily → *Start a program*:\n"
                    f"   - Program: `py`  \n   - Arguments: `{cmd[3:]}`  \n   - Start in: `{root_dir}`\n"
                    "3. Each run is appended to the check history above. Exit code 2 = action needed "
                    "(can trigger an e-mail in Task Scheduler).")
        st.code(cmd, language="bash")


# ============================================================================
def render(standalone: bool = False):
    if standalone:
        st.sidebar.title("🔬 FermenTrack Workbench")
    st.sidebar.markdown("### Model workbench")
    start = STEP_KEYS.get(st.query_params.get("step", ""), 0)
    step = st.sidebar.radio("Step", STEPS, index=start, key="wb_step", label_visibility="collapsed")
    if st.query_params.get("demo") == "1" and S().get("wb_ds") is None:
        _demo_state()
    sidebar_status()
    show_flash()
    {0: step_data, 1: step_spectra, 2: step_split, 3: step_model, 4: step_outliers,
     5: step_robustness, 6: step_registry, 7: step_autocheck}[STEPS.index(step)]()


@st.cache_data(show_spinner=False)
def read_example(name: str) -> io.SpectralDataset:
    """Example files are read once and kept in memory (Excel reading is slow)."""
    return io.read_spectra(EXAMPLES / name)


def _demo_state():
    """?demo=1 : example data, SG 1st derivative, split by batch, PLS fitted (for screenshots/demos)."""
    set_dataset(read_example("ftir_calibration.xlsx"))
    S()["wb_prop"] = "Ethanol"
    S()["wb_pipe"] = Pipeline(deriv=1, deriv_window=11).settings()
    d = ds()
    S()["wb_val"] = split.by_group(groups_all(), 0.25, 0)
    cal = ~val_mask() & included_mask()
    m = fit_model(ModelSpec(), current_pipeline(), d.X[cal], d.wn, y_all()[cal], "Ethanol",
                  groups_all()[cal], d.ids[cal])
    S()["wb_model"] = m
    S()["wb_models"] = {model_label(m): m}
