"""The simulator must obey basic chemistry and physics."""
import numpy as np

from fermentrack import config as C
from fermentrack.anomaly import hexose_equivalent
from fermentrack.kinetics import sample_params, simulate_batch
from fermentrack.spectroscopy import (SensorParams, fouling_level, pure_spectra,
                                      simulate_spectra)


def _batch(scenario, seed=0):
    p = sample_params(np.random.default_rng(seed), 0, scenario)
    return p, simulate_batch(p)


def test_time_grid_and_no_negative_concentrations():
    _, df = _batch("normal")
    assert np.allclose(df.time_h, C.TIMES_H)
    assert (df[C.ANALYTES] >= -1e-9).all().all()


def test_normal_batch_converts_sugar_to_ethanol():
    _, df = _batch("normal")
    assert df.ethanol.iat[-1] > 10            # realistic final ethanol (% w/v)
    assert df.dp4plus.iat[-1] < 2             # dextrins mostly hydrolysed
    assert df.glucose.iat[-1] < 1             # glucose used up
    assert df.glucose.max() > 4               # typical SSF glucose peak


def test_mass_balance_closes():
    """Glucose-equivalents only drop by the ~3 % that goes into yeast cells."""
    _, df = _batch("normal")
    hx = hexose_equivalent(df)
    assert 0.94 < hx[-1] / hx[0] <= 1.0


def test_contamination_and_stuck_lower_the_yield():
    """Same random draw, different scenario: faults must cost ethanol."""
    base_p, base = _batch("normal", seed=3)
    for sc in ["contamination_lactic", "stuck"]:
        p, df = _batch(sc, seed=3)
        assert df.ethanol.iat[-1] < base.ethanol.iat[-1]
    _, lac = _batch("contamination_lactic", seed=3)
    assert lac.lactic.iat[-1] > C.LACTIC_ALERT
    _, stuck = _batch("stuck", seed=3)
    assert stuck.glucose.iat[-1] > base.glucose.iat[-1]


def test_beer_lambert_linearity():
    """Without noise or artefacts, doubling the concentrations doubles the absorbance."""
    s = SensorParams(noise_sd=0, baseline_walk_sd=0, matrix_level=0)
    rng = np.random.default_rng(0)
    c = np.full((1, len(C.ANALYTES)), 1.0)
    a1 = simulate_spectra(c, np.array([0.0]), s, rng)
    a2 = simulate_spectra(2 * c, np.array([0.0]), s, rng)
    assert np.allclose(a2, 2 * a1)
    assert np.allclose(a1[0], pure_spectra().sum(axis=0))


def test_fouling_only_inside_its_window():
    s = SensorParams(fouling_onset_h=10, fouling_rate=0.1, fouling_clean_h=20)
    f = fouling_level(np.array([5.0, 10.0, 15.0, 25.0]), s)
    assert f[0] == 0 and f[1] == 0 and 0 < f[2] <= 1 and f[3] == 0
