"""Preprocessing and PLS behave as the textbook says."""
import numpy as np
import pandas as pd

from fermentrack.chemometrics import (PLSModel, Preprocessor, SpectralCalibration,
                                      choose_components, group_folds, rmse)


def test_snv_gives_zero_mean_unit_sd():
    X = np.random.default_rng(0).normal(5, 2, (4, 50))
    Z = Preprocessor(snv=True).transform(X)
    assert np.allclose(Z.mean(axis=1), 0) and np.allclose(Z.std(axis=1), 1)


def test_derivative_and_baseline_remove_offsets():
    x = np.sin(np.linspace(0, 6, 100))
    X = np.vstack([x, x + 3.0])                          # same spectrum, shifted up
    for pre in [Preprocessor(deriv=1), Preprocessor(baseline="linear")]:
        Z = pre.transform(X)
        assert np.allclose(Z[0], Z[1])


def test_pls_recovers_linear_mixture():
    """Noise-free two-component mixture: PLS with 2 components must be exact."""
    rng = np.random.default_rng(1)
    wn = np.linspace(0, 1, 80)
    k1, k2 = np.exp(-((wn - 0.3) / 0.05) ** 2), np.exp(-((wn - 0.6) / 0.08) ** 2)
    C = rng.uniform(0, 5, (30, 2))
    X = C[:, :1] * k1 + C[:, 1:] * k2
    m = PLSModel.fit(X, C[:, 0], 5)
    assert rmse(C[:, 0], m.predict(X, 2)) < 1e-8


def test_group_folds_keep_batches_together():
    groups = np.repeat(np.arange(12), 5)
    folds = group_folds(groups, 4)
    assert sorted(np.concatenate(folds)) == list(range(60))
    for f in folds:
        others = np.setdiff1d(np.arange(60), f)
        assert not set(groups[f]) & set(groups[others])


def test_choose_components_prefers_simpler_model():
    curve = np.array([1.0, 0.5, 0.29, 0.285, 0.28])
    assert choose_components(curve) == 3            # 0.29 is within 5 % of the best (0.28)


def test_spectral_calibration_end_to_end():
    rng = np.random.default_rng(2)
    wn = np.linspace(0, 1, 60)
    K = np.vstack([np.exp(-((wn - c) / 0.06) ** 2) for c in (0.3, 0.5, 0.7)])
    Y = pd.DataFrame(rng.uniform(0, 3, (40, 3)), columns=["a", "b", "c"])
    X = Y.to_numpy() @ K + rng.normal(0, 1e-3, (40, 60))
    cal = SpectralCalibration.fit(X, Y, np.repeat(np.arange(10), 4), analytes=["a", "b"],
                                  max_components=6)
    pred = cal.predict(X)
    assert list(pred.columns) == ["a", "b"]
    assert rmse(Y.a, pred.a) < 0.02
    assert set(cal.summary().columns) >= {"preprocessing", "components", "RMSECV"}
