"""Spectral preprocessing pipeline for the workbench.

A `Pipeline` is an ordered recipe that is *fitted on calibration samples only*
(important for MSC, whose reference spectrum must not see validation data) and
then applied unchanged to validation, new and unknown samples.

Order of steps (the usual chemometric order):
    1. wavenumber regions  - keep only informative parts of the spectrum
    2. baseline            - offset / linear / 2nd-order polynomial removal per spectrum
    3. scatter correction  - SNV or MSC
    4. smoothing           - Savitzky-Golay
    5. derivative          - Savitzky-Golay 1st or 2nd derivative
Mean-centring / autoscaling is done inside the model (see models.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.signal import savgol_filter

BASELINES = {"none": "None", "offset": "Offset (min = 0)", "linear": "Linear", "poly2": "2nd-order polynomial"}
SCATTER = {"none": "None", "snv": "SNV", "msc": "MSC"}


def _odd(n: int) -> int:
    n = int(n)
    return n if n % 2 else n + 1


def region_mask(wn: np.ndarray, regions) -> np.ndarray:
    """True for wavenumbers inside any (low, high) region; all True when no regions are set."""
    if not regions:
        return np.ones(len(wn), bool)
    m = np.zeros(len(wn), bool)
    for lo, hi in regions:
        lo, hi = min(lo, hi), max(lo, hi)
        m |= (wn >= lo) & (wn <= hi)
    return m


@dataclass
class Pipeline:
    regions: list = field(default_factory=list)   # [(low, high), ...] in cm-1; empty = all
    baseline: str = "none"
    scatter: str = "none"
    smooth: int = 0                               # SG smoothing window, 0 = off
    deriv: int = 0                                # 0, 1, 2
    deriv_window: int = 15
    poly: int = 2
    msc_ref: np.ndarray | None = None             # learned in fit()

    # --- recipe as plain data (for saving / showing) ---------------------------
    def settings(self) -> dict:
        d = asdict(self)
        d.pop("msc_ref")
        d["regions"] = [list(map(float, r)) for r in self.regions]
        return d

    @classmethod
    def from_settings(cls, d: dict) -> "Pipeline":
        d = dict(d)
        d["regions"] = [tuple(r) for r in d.get("regions", [])]
        return cls(**d)

    def describe(self) -> str:
        parts = []
        if self.regions:
            parts.append("regions " + ", ".join(f"{min(r):.0f}-{max(r):.0f}" for r in self.regions))
        if self.baseline != "none":
            parts.append(f"baseline {BASELINES[self.baseline].lower()}")
        if self.scatter != "none":
            parts.append(SCATTER[self.scatter])
        if self.smooth >= 3:
            parts.append(f"SG smoothing {_odd(self.smooth)}")
        if self.deriv:
            parts.append(f"SG {self.deriv}. derivative ({_odd(self.deriv_window)} pt)")
        return ", ".join(parts) or "none"

    # --- fitting and applying ------------------------------------------------------
    def _baseline(self, X):
        if self.baseline == "none":
            return X
        if self.baseline == "offset":
            return X - X.min(axis=1, keepdims=True)
        deg = {"linear": 1, "poly2": 2}[self.baseline]
        z = np.linspace(-1, 1, X.shape[1])
        V = np.vander(z, deg + 1)
        coef, *_ = np.linalg.lstsq(V, X.T, rcond=None)
        return X - (V @ coef).T

    def _before_scatter(self, X, wn):
        X = np.asarray(X, float)[:, region_mask(wn, self.regions)]
        return self._baseline(X)

    def fit(self, X, wn) -> "Pipeline":
        """Learn what needs learning from calibration spectra (MSC reference)."""
        self.msc_ref = self._before_scatter(X, wn).mean(axis=0) if self.scatter == "msc" else None
        return self

    def transform(self, X, wn) -> np.ndarray:
        X = self._before_scatter(X, wn)
        if self.scatter == "snv":
            X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
        elif self.scatter == "msc":
            ref = self.msc_ref if self.msc_ref is not None else X.mean(axis=0)
            A = np.vstack([np.ones_like(ref), ref]).T          # x = a + b * ref
            coef, *_ = np.linalg.lstsq(A, X.T, rcond=None)
            X = (X - coef[0][:, None]) / coef[1][:, None]
        if self.smooth and self.smooth >= 3:
            w = _odd(self.smooth)
            if w < X.shape[1]:
                X = savgol_filter(X, w, min(self.poly, w - 1), axis=1)
        if self.deriv:
            w = _odd(max(self.deriv_window, self.poly + 2, self.deriv + 2))
            if w < X.shape[1]:
                X = savgol_filter(X, w, max(self.poly, self.deriv), deriv=self.deriv, axis=1)
        return X

    def fit_transform(self, X, wn):
        return self.fit(X, wn).transform(X, wn)

    def wavenumbers(self, wn) -> np.ndarray:
        return wn[region_mask(wn, self.regions)]
