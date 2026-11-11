"""End-to-end: the full workflow runs and gives sensible, reproducible results (~15 s)."""
import pytest

from fermentrack import config as C
from fermentrack import dataset
from fermentrack.pipeline import run


@pytest.fixture(scope="module")
def results():
    return run(save=False, verbose=False)


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
