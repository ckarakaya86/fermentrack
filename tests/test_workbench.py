"""Model Workbench: data reading, preprocessing, splits, PLS/ANN, outliers, registry, auto-check."""
import numpy as np
import pandas as pd
import pytest

from fermentrack import config as C
from fermentrack.workbench import autovalidate, diagnostics, io, registry, robustness, split
from fermentrack.workbench.models import ModelSpec, fit_model, regression_stats
from fermentrack.workbench.preprocess import Pipeline

EX = C.ROOT / "examples"


def _toy(n=60, seed=0, quadratic=False):
    rng = np.random.default_rng(seed)
    wn = np.linspace(1500, 950, 120)
    K = np.vstack([np.exp(-((wn - c) / 25) ** 2) for c in (1050, 1150, 1300)])
    Cn = rng.uniform(0, 5, (n, 3))
    X = Cn @ K + rng.normal(0, 1e-3, (n, len(wn)))
    y = Cn[:, 0] ** 2 if quadratic else Cn[:, 0]
    return X, wn, y


# --- data reading -------------------------------------------------------------
def test_rows_layout_round_trip(tmp_path):
    X, wn, y = _toy(10)
    ds = io.SpectralDataset(np.array([f"s{i}" for i in range(10)]), wn, X,
                            pd.DataFrame({"Ethanol": y}), pd.DataFrame({"group": ["a", "b"] * 5}))
    io.to_excel(ds, tmp_path / "d.xlsx")
    back = io.read_spectra(tmp_path / "d.xlsx")
    assert back.n == 10 and np.allclose(back.X, X) and np.allclose(back.wn, wn)
    assert np.allclose(back.Y["Ethanol"], y) and list(back.meta["group"]) == ["a", "b"] * 5


def test_columns_layout_one_column_per_sample(tmp_path):
    X, wn, y = _toy(6)
    top = pd.DataFrame([["No", *range(1, 7)], ["API", *np.round(y, 3)]])
    body = pd.DataFrame(np.column_stack([wn, X.T]))
    pd.concat([top, body], ignore_index=True).to_excel(tmp_path / "c.xlsx", header=False, index=False)
    d = io.read_spectra(tmp_path / "c.xlsx")
    assert d.n == 6 and list(d.ids) == [str(i) for i in range(1, 7)]
    assert np.allclose(d.X, X) and np.allclose(d.Y["API"], np.round(y, 3))


def test_reference_attached_by_sample_id():
    X, wn, _ = _toy(4)
    ds = io.SpectralDataset(np.array(["a", "b", "c", "d"]), wn, X, pd.DataFrame(index=range(4)))
    out = io.attach_reference(ds, np.array(["c", "a"]), pd.DataFrame({"Glucose": [3.0, 1.0]}),
                              pd.DataFrame())
    assert out.Y["Glucose"].tolist()[:3] == [1.0, None, 3.0] or np.isnan(out.Y["Glucose"][1])
    assert out.Y["Glucose"][0] == 1.0 and out.Y["Glucose"][2] == 3.0


# --- preprocessing and splits -----------------------------------------------------
def test_msc_removes_multiplicative_and_additive_effects():
    X, wn, _ = _toy(5)
    Xd = X * np.array([[1.0], [1.3], [0.8], [1.1], [0.9]]) + np.array([[0], [0.1], [-0.05], [0.02], [0]])
    p = Pipeline(scatter="msc").fit(X, wn)
    assert np.allclose(p.transform(Xd, wn), p.transform(X, wn), atol=1e-8)


def test_pipeline_settings_round_trip_and_regions():
    p = Pipeline(regions=[(1000, 1200)], baseline="linear", deriv=1)
    q = Pipeline.from_settings(p.settings())
    X, wn, _ = _toy(3)
    assert np.allclose(p.fit_transform(X, wn), q.fit_transform(X, wn))
    assert q.transform(X, wn).shape[1] == np.sum((wn >= 1000) & (wn <= 1200))


def test_splits():
    X, wn, y = _toy(40)
    ks = split.kennard_stone(X, 0.25)
    assert ks.sum() == 10
    g = np.repeat(np.arange(8), 5)
    v = split.by_group(g, 0.25)
    assert not set(g[v]) & set(g[~v])
    assert split.by_order(np.arange(40), 0.25).tolist() == [False] * 30 + [True] * 10


# --- models ---------------------------------------------------------------------------
def test_pls_is_accurate_on_linear_data():
    X, wn, y = _toy(60)
    m = fit_model(ModelSpec(kind="PLS"), Pipeline(), X, wn, y)
    assert m.cv_stats()["RMSE"] < 0.01 and m.k <= 5


def test_ann_beats_pls_on_curved_relationship():
    X, wn, y = _toy(80, quadratic=True)
    pls = fit_model(ModelSpec(kind="PLS", cv="10"), Pipeline(), X, wn, y)
    ann = fit_model(ModelSpec(kind="ANN", n_components=3, cv="10", n_ensemble=3), Pipeline(), X, wn, y)
    assert ann.cv_stats()["RMSE"] < 0.5 * pls.cv_stats()["RMSE"]
    assert ann.predict_spread(X[:3]).shape == (3,)


def test_regression_stats():
    s = regression_stats([1, 2, 3, 4], [1.1, 2.1, 3.1, 4.1])
    assert abs(s["bias"] - 0.1) < 1e-12 and abs(s["SEP"]) < 1e-12 and s["R2"] > 0.99


# --- outliers and robustness (on the example file with planted problems) --------------
@pytest.fixture(scope="module")
def example():
    if not (EX / "ftir_calibration.xlsx").exists():
        from fermentrack.workbench.examples import make
        make()
    return io.read_spectra(EX / "ftir_calibration.xlsx")


def test_planted_reference_errors_are_flagged(example):
    y = example.Y["Ethanol"].to_numpy()
    m = fit_model(ModelSpec(), Pipeline(deriv=1, deriv_window=11), example.X, example.wn, y,
                  groups=example.meta["group"].to_numpy(), ids=example.ids)
    d = diagnostics.sample_diagnostics(m, example.X, y, example.ids, use_cv=True)
    flagged = set(d[d.status == "outlier"].sample_id)
    assert {"B03-32.3h", "B03-08.0h", "B09-60.0h"} <= flagged          # typo + swapped labels
    q_bad = d.set_index("sample_id").loc["B02-40.0h"]                   # fouled spectrum
    assert q_bad.Q > m.diag["q_lim99"]


def test_robustness_tools(example):
    clean = ~np.isin(example.ids, ["B03-32.3h", "B03-08.0h", "B09-60.0h", "B02-40.0h"])
    X, y = example.X[clean], example.Y["Ethanol"].to_numpy()[clean]
    m = fit_model(ModelSpec(), Pipeline(deriv=1, deriv_window=11), X, example.wn, y)
    assert np.allclose(robustness.perturb(X, example.wn, "shift", 0), X)
    pt = robustness.perturbation_test(m, X[:40], y[:40], kinds=["offset", "noise"])
    assert pt[pt.kind == "offset"].RMSEP_ratio.max() < 1.2             # derivative ignores offsets
    yr = robustness.y_randomization(ModelSpec(), Pipeline(deriv=1, deriv_window=11), X, example.wn, y, 3)
    assert yr["passed"]


# --- registry, update and auto-check ---------------------------------------------------
def test_registry_update_and_autocheck(example, tmp_path):
    clean = ~np.isin(example.ids, ["B03-32.3h", "B03-08.0h", "B09-60.0h", "B02-40.0h"])
    d = example.subset(clean)
    g = d.meta["group"].to_numpy()
    pkg = registry.build_package("Ethanol", ModelSpec(), Pipeline(deriv=1, deriv_window=11), d.X, d.wn,
                                 d.Y["Ethanol"].to_numpy(), d.ids, g, split.by_group(g, 0.25))
    assert registry.save(pkg, tmp_path) == "v001"
    assert registry.load("Ethanol", root=tmp_path).version == "v001"

    ok = io.read_spectra(EX / "ftir_new_lab_ok.xlsx")
    r1 = autovalidate.run("Ethanol", ok.X, ok.wn, ok.Y["Ethanol"].to_numpy(), ok.ids, root=tmp_path)
    assert r1["verdict"] == "OK"

    drift = io.read_spectra(EX / "ftir_new_lab_drift.xlsx")
    r2 = autovalidate.run("Ethanol", drift.X, drift.wn, drift.Y["Ethanol"].to_numpy(), drift.ids,
                          drift.meta["group"].to_numpy(), auto_update=True, root=tmp_path)
    assert r2["verdict"] == "ACTION"
    assert r2["update"]["recommended"] == "augment (refit)" and r2["update"]["saved"] == "v002"
    assert registry.load("Ethanol", root=tmp_path).version == "v002"
    assert len(pd.read_csv(registry.history_path("Ethanol", tmp_path))) == 2


def test_cli_exit_codes(example, tmp_path):
    d = example.subset(~np.isin(example.ids, ["B03-32.3h", "B03-08.0h", "B09-60.0h", "B02-40.0h"]))
    pkg = registry.build_package("Eth", ModelSpec(), Pipeline(deriv=1, deriv_window=11), d.X, d.wn,
                                 d.Y["Ethanol"].to_numpy(), d.ids)
    registry.save(pkg, tmp_path)
    args = ["--model", "Eth", "--property", "Ethanol", "--registry", str(tmp_path)]
    assert autovalidate.main(args + ["--data", str(EX / "ftir_new_lab_ok.xlsx")]) == 0
    assert autovalidate.main(args + ["--data", str(EX / "ftir_new_lab_drift.xlsx")]) == 2


# --- data preparation: one file per spectrum, templates ---------------------------------
def test_single_spectrum_files_with_mixed_formats(tmp_path):
    wn = np.arange(950, 1502, 2.0)
    a = np.exp(-((wn - 1045) / 10) ** 2)
    (tmp_path / "s1.csv").write_text("Title: FTIR\nWavenumber,Absorbance\n" +
                                     "\n".join(f"{w:g},{v:.6f}" for w, v in zip(wn[::-1], a[::-1])))
    (tmp_path / "s2.txt").write_text("Wellenzahl;Absorbanz\n" +
                                     "\n".join(f"{w:g};{2 * v:.6f}".replace(".", ",") for w, v in zip(wn, a)))
    wn3 = np.arange(940, 1512, 1.0)                     # a different grid and a wider range
    (tmp_path / "s3.csv").write_text("\n".join(f"{w:g}\t{3 * np.exp(-((w - 1045) / 10) ** 2):.6f}"
                                               for w in wn3))
    d = io.read_spectrum_files([(f.name, f) for f in sorted(tmp_path.iterdir())])
    assert list(d.ids) == ["s1", "s2", "s3"] and np.allclose(d.wn, wn)
    assert np.allclose(d.X[1], 2 * d.X[0], atol=1e-5) and np.allclose(d.X[2], 3 * d.X[0], atol=1e-4)


def test_decimal_comma_lab_values(tmp_path):
    pd.DataFrame({"sample_id": ["a", "b"], "Ethanol": ["12,5", "3,25"]}).to_csv(tmp_path / "r.csv", index=False, sep=";")
    ids, Y, _ = io.read_reference(tmp_path / "r.csv")
    assert Y["Ethanol"].tolist() == [12.5, 3.25]


def test_all_templates_give_the_same_data():
    from fermentrack.workbench.examples import make_templates
    T = EX / "templates"
    if not (T / "single_spectra").exists():
        make_templates()
    a = io.read_spectra(T / "1_spectra_and_lab_values_in_one_file.xlsx")
    b = io.attach_reference(io.read_spectra(T / "2_spectra_only.xlsx"), *io.read_reference(T / "3_lab_values.xlsx"))
    c = io.read_spectra(T / "4_columns_layout.xlsx")
    files = sorted((T / "single_spectra").glob("*.csv"))
    d = io.read_spectrum_files([(f.name, f) for f in files])
    for other in (b, c, d):
        assert np.allclose(other.X, a.X, atol=1e-5)
    assert np.allclose(b.Y["Ethanol"], a.Y["Ethanol"]) and np.allclose(c.Y["Ethanol"], a.Y["Ethanol"])
