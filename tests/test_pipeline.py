"""End-to-end: the full workflow runs and gives sensible, reproducible results (~15 s)."""
import pytest

from fermentrack import config as C
from fermentrack import dataset
from fermentrack.pipeline import run


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    # an empty model registry, so the built-in models are tested (not models saved on this PC)
    return run(save=False, verbose=False, registry_root=tmp_path_factory.mktemp("empty_registry"))


def test_data_is_reproducible():
    a, b = dataset.generate(), dataset.generate()
    assert (a.spectra == b.spectra).all()
    assert a.batches.equals(b.batches)


def test_pls_calibration_is_accurate(results):
    s = results.calibration.summary().set_index("analyte")
    assert s.loc["ethanol", "RMSECV"] < 0.3          # % w/v
    assert s.loc["glucose", "RMSECV"] < 0.3
    assert (s.R2_CV > 0.9).all()


def test_faults_are_detected_without_false_alarms(results):
    m = results.metrics
    assert m["test_fault_detection_rate"] >= 0.8
    assert m["test_false_alarm_rate_normal"] <= 0.1


def test_calibration_drift_is_caught_after_it_starts(results):
    alarm = results.metrics["calibration_alarms"]["ethanol"]
    assert alarm is not None and C.DRIFT_START_BATCH <= alarm <= C.DRIFT_START_BATCH + 6


def test_early_forecast_helps_on_faulty_batches(results):
    y = results.metrics["yield_20h"]
    assert y["RMSE_model_faulty"] < y["RMSE_baseline_faulty"]
    assert y["RMSE_model"] < y["RMSE_baseline"]


def test_active_saved_model_is_used_by_the_dashboard(tmp_path):
    """A model saved and activated in the workbench replaces the built-in one for its analyte."""
    import numpy as np
    from fermentrack.model_source import match_analyte
    from fermentrack.workbench import io, registry
    from fermentrack.workbench.models import ModelSpec
    from fermentrack.workbench.preprocess import Pipeline
    ex = C.ROOT / "examples" / "ftir_calibration.xlsx"
    if not ex.exists():
        from fermentrack.workbench.examples import make
        make()
    d = io.read_spectra(ex)
    d = d.subset(~np.isin(d.ids, ["B03-32.3h", "B03-08.0h", "B09-60.0h", "B02-40.0h"]))
    pkg = registry.build_package("Ethanol (% w/v)", ModelSpec(), Pipeline(deriv=1, deriv_window=11),
                                 d.X, d.wn, d.Y["Ethanol"].to_numpy(), d.ids)
    registry.save(pkg, tmp_path)
    nir = registry.build_package("API", ModelSpec(), Pipeline(), d.X, d.wn + 3000,   # wrong range
                                 d.Y["Ethanol"].to_numpy(), d.ids)
    registry.save(nir, tmp_path)

    r = run(save=False, verbose=False, registry_root=tmp_path)
    assert list(r.saved_models) == ["ethanol"]
    assert "Ethanol" in r.metrics["model_sources"]["ethanol"]
    assert any(s.startswith("API") for s in r.metrics["saved_models_skipped"])
    assert r.metrics["test_fault_detection_rate"] >= 0.8
    assert match_analyte("DP4+ (dextrins)") == "dp4plus" and match_analyte("API") is None
