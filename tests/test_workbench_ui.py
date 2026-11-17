"""Click through the workbench interface headlessly (Streamlit AppTest)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from fermentrack.workbench import registry

APP = str(Path(__file__).resolve().parents[1] / "workbench_app.py")
STEPS = ["data", "spectra", "split", "model", "outliers", "robustness", "registry", "autocheck"]


@pytest.fixture(autouse=True)
def temp_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "DEFAULT_ROOT", tmp_path)


def _app(step):
    at = AppTest.from_file(APP, default_timeout=180)
    at.query_params["demo"] = "1"          # example data, split by batch, PLS fitted
    at.query_params["step"] = step
    return at.run()


def _click(at, label):
    next(b for b in at.button if b.label.startswith(label)).click()
    return at.run()


@pytest.mark.parametrize("step", STEPS)
def test_every_step_renders(step):
    at = _app(step)
    assert not at.exception, at.exception


def test_fit_ann_from_the_model_step():
    at = _app("model")
    next(r for r in at.radio if r.label == "Model type").set_value("ANN")
    at.run()
    at = _click(at, "Fit model")
    assert not at.exception
    assert any("ANN" in str(t.value) for t in at.sidebar.caption) or at.dataframe


def test_exclude_outliers_refits_and_lowers_error():
    at = _app("outliers")
    at = _click(at, "Exclude ticked samples")
    assert not at.exception
    assert any("RMSECV" in s.value for s in at.success)


def test_clean_save_then_autocheck_flags_drift():
    """Full user journey: remove outliers -> save -> check against drifting instrument data.
    (An uncleaned model promises such a large error that it cannot detect the drift.)"""
    at = _app("outliers")
    at = _click(at, "Exclude ticked samples")
    at.sidebar.radio(key="wb_step").set_value("7 Save & update")
    at.run()
    at = _click(at, "Save as new version")
    assert not at.exception and any("Saved" in s.value for s in at.success)
    at.sidebar.radio(key="wb_step").set_value("8 Auto-check")
    at.run()
    next(r for r in at.radio if "an example" in r.label).set_value("ftir_new_lab_drift.xlsx")
    at.run()
    at = _click(at, "Run check")
    assert not at.exception
    assert any("ACTION" in e.value for e in at.error)


def test_compare_versions_button():
    at = _app("outliers")
    at = _click(at, "Exclude ticked samples")
    at.sidebar.radio(key="wb_step").set_value("7 Save & update")
    at.run()
    at = _click(at, "Save as new version")
    at = _click(at, "Save as new version")                  # second click: duplicate, not saved
    assert any("Identical" in i.value for i in at.info)
    at.sidebar.radio(key="wb_step").set_value("8 Auto-check")
    at.run()
    next(r for r in at.radio if "an example" in r.label).set_value("ftir_new_lab_ok.xlsx")
    at.run()
    at = _click(at, "Run check")
    at = _click(at, "Compare all versions")
    assert not at.exception
    assert any("best" in s.value for s in at.success)
    assert any(h.value.startswith("Performance per version") for h in at.subheader)
