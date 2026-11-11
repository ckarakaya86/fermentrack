"""Anomaly rules, calibration-health chart and the early yield features."""
import numpy as np
import pandas as pd

from fermentrack import config as C
from fermentrack.anomaly import SpectralMonitor, first_persistent, persistent_mask
from fermentrack.calibration_health import check_analyte, ewma_chart
from fermentrack.yield_model import trajectory_features


def test_persistence_ignores_single_spikes():
    flags = np.array([0, 1, 0, 1, 1, 0, 1, 1, 1, 1], bool)
    assert first_persistent(flags, 3) == 8
    assert persistent_mask(flags, 3).tolist() == [False] * 8 + [True, True]


def test_q_statistic_flags_unknown_band():
    rng = np.random.default_rng(0)
    wn = np.linspace(950, 1500, 120)
    K = np.vstack([np.exp(-((wn - c) / 15) ** 2) for c in (1040, 1100, 1280)])
    X = rng.uniform(0, 1, (300, 3)) @ K + rng.normal(0, 1e-3, (300, 120))
    mon = SpectralMonitor.fit(X)
    normal = rng.uniform(0, 1, (1, 3)) @ K + rng.normal(0, 1e-3, (1, 120))
    foreign = normal + 0.05 * np.exp(-((wn - 1400) / 20) ** 2)   # something new in the spectrum
    _, q_ok, _ = mon.score(normal)
    _, q_bad, _ = mon.score(foreign)
    assert q_ok[0] < mon.q_limit < q_bad[0]


def test_ewma_quiet_when_unbiased_and_alarms_on_drift():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.1, 40)
    z, lim = ewma_chart(noise, 0.1)
    assert not (np.abs(z) > lim).any()
    drift = noise + np.r_[np.zeros(20), np.linspace(0.05, 0.4, 20)]
    res = pd.DataFrame({"batch_id": np.arange(40), "ethanol_resid": drift})
    out = check_analyte(res, "ethanol", 0.1, start_batch=0)
    assert out["alarm_batch"] is not None and out["alarm_batch"] >= 20
    assert out["bias_estimate"] > 0


def test_features_do_not_peek_into_the_future():
    """Changing data after the cut-off must not change the features."""
    t = C.TIMES_H
    traj = pd.DataFrame({"time_h": t, **{a: np.linspace(0.1, 1, len(t)) for a in C.ANALYTES}})
    traj["ethanol"] = 0.3 * t
    f1 = trajectory_features(traj, 16)
    traj.loc[traj.time_h > 16, C.ANALYTES] = 99.0
    f2 = trajectory_features(traj, 16)
    assert f1 == f2
    assert abs(f1["ethanol_rate"] - 0.3) < 1e-9
