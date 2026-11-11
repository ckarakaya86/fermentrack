"""Synthetic in-line mid-IR (ATR) spectra of fermenting mash.

Beer-Lambert idea: the absorbance spectrum is (approximately) the sum of each
component's pure spectrum multiplied by its concentration:

    A(wavenumber) = sum_i  c_i * K_i(wavenumber)  + baseline + noise + artefacts

Each pure spectrum K_i is built from a few Gaussian bands at plausible
mid-IR positions (C-O / C-C stretching region 950-1500 cm-1). The band
positions are literature-inspired approximations, good enough to make the
calibration problem realistic: the sugars overlap strongly, ethanol and
glycerol overlap, and the acids sit on the shoulders.

On top of the chemistry we add what a real probe in a real plant sees:
* random noise and a slowly wandering baseline,
* an unknown "matrix" background (corn solids, proteins) that differs per batch,
* probe fouling: a growing film that dims the signal and adds broad bands,
* instrument drift: a slow shift of the wavenumber axis over many batches.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C

# (centre cm-1, width sigma cm-1, height in absorbance per 1 % w/v)
BANDS = {
    "glucose":  [(1035, 12, 0.0110), (1080, 11, 0.0065), (1108, 10, 0.0050), (1150, 12, 0.0032)],
    "dp2":      [(1025, 13, 0.0090), (1078, 12, 0.0055), (1120, 10, 0.0030), (1150, 11, 0.0048)],
    "dp3":      [(1022, 14, 0.0085), (1074, 12, 0.0050), (1105, 11, 0.0032), (1153, 11, 0.0052)],
    "dp4plus":  [(1018, 16, 0.0080), (1080, 14, 0.0045), (1155, 13, 0.0058), (1000, 12, 0.0025)],
    "ethanol":  [(1045, 8, 0.0120), (1087, 8, 0.0068), (1275, 9, 0.0012), (1380, 10, 0.0010),
                 (1455, 12, 0.0008)],
    "glycerol": [(1040, 9, 0.0095), (1110, 9, 0.0082), (995, 8, 0.0040), (1415, 12, 0.0018)],
    "lactic":   [(1125, 9, 0.0085), (1045, 10, 0.0040), (1215, 12, 0.0045), (1375, 10, 0.0035),
                 (1460, 12, 0.0020)],
    "acetic":   [(1280, 10, 0.0100), (1018, 9, 0.0040), (1400, 12, 0.0060), (1350, 10, 0.0025)],
}

# Broad bands that are NOT any of our analytes (unknown matrix / fouling film)
MATRIX_BANDS = [(1240, 40, 1.0), (1400, 55, 0.7), (1080, 90, 0.5)]
FOULING_BANDS = [(1400, 60, 1.0), (1080, 120, 0.8), (1240, 35, 0.4)]


def gaussian(wn, centre, sigma, height):
    return height * np.exp(-0.5 * ((wn - centre) / sigma) ** 2)


def band_sum(wn, bands):
    return sum(gaussian(wn, c, s, h) for c, s, h in bands)


def pure_spectra(wn: np.ndarray = C.WAVENUMBERS) -> np.ndarray:
    """Matrix K (n_analytes x n_wavenumbers): absorbance of 1 % w/v of each analyte."""
    return np.array([band_sum(wn, BANDS[a]) for a in C.ANALYTES])


@dataclass
class SensorParams:
    """Everything about the probe/instrument for one batch."""
    noise_sd: float = 2.5e-4          # absorbance units
    baseline_walk_sd: float = 4e-5    # random-walk step of the baseline offset
    matrix_level: float = 0.02        # amount of unknown matrix background
    fouling_onset_h: float | None = None
    fouling_rate: float = 0.0         # film growth per hour (0..1 scale)
    fouling_clean_h: float | None = None   # time when the probe is cleaned again
    wn_shift_cm1: float = 0.0         # instrument wavenumber shift for this batch
    gain: float = 1.0                 # instrument gain (1 = perfect)


def sample_sensor_params(rng: np.random.Generator, batch_id: int, scenario: str) -> SensorParams:
    s = SensorParams(matrix_level=rng.normal(0.02, 0.004))
    if scenario == "probe_fouling":
        s.fouling_onset_h = rng.uniform(15, 35)
        s.fouling_rate = rng.uniform(0.05, 0.09)
        s.fouling_clean_h = min(s.fouling_onset_h + rng.uniform(12, 20), C.BATCH_DURATION_H)
    if batch_id >= C.DRIFT_START_BATCH:
        # slow ageing of the instrument: grows linearly over the remaining batches
        frac = (batch_id - C.DRIFT_START_BATCH + 1) / (C.N_BATCHES - C.DRIFT_START_BATCH)
        s.wn_shift_cm1 = 0.35 * frac
        s.gain = 1.0 + 0.035 * frac
    return s


def fouling_level(t: np.ndarray, s: SensorParams) -> np.ndarray:
    """Film thickness (0 = clean, 1 = heavy) at each time."""
    f = np.zeros_like(t, dtype=float)
    if s.fouling_onset_h is None:
        return f
    on = (t >= s.fouling_onset_h) & (t < s.fouling_clean_h)
    f[on] = np.minimum(s.fouling_rate * (t[on] - s.fouling_onset_h), 1.0)
    return f


def simulate_spectra(conc: np.ndarray, times: np.ndarray, s: SensorParams,
                     rng: np.random.Generator, wn: np.ndarray = C.WAVENUMBERS) -> np.ndarray:
    """Turn a (n_times x n_analytes) concentration table into (n_times x n_wn) spectra."""
    K = pure_spectra(wn)
    A = conc @ K                                                   # the chemistry
    A = A + s.matrix_level * band_sum(wn, MATRIX_BANDS) / 1.0      # unknown background

    # slowly wandering baseline: offset random walk + small tilt
    n = len(times)
    offset = np.cumsum(rng.normal(0, s.baseline_walk_sd, n))
    tilt = np.cumsum(rng.normal(0, s.baseline_walk_sd / 2, n))
    z = (wn - wn.mean()) / (wn.max() - wn.min())
    A = A + offset[:, None] + tilt[:, None] * z[None, :]

    # probe fouling: the film dims the evanescent wave and adds its own bands
    f = fouling_level(times, s)
    if f.any():
        A = A * (1 - 0.25 * f[:, None]) + 0.03 * f[:, None] * band_sum(wn, FOULING_BANDS)[None, :]

    # instrument drift: the true spectrum appears at slightly shifted positions
    if s.wn_shift_cm1:
        A = np.array([np.interp(wn - s.wn_shift_cm1, wn, a) for a in A])
    A = A * s.gain

    return A + rng.normal(0, s.noise_sd, A.shape)
