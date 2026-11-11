"""Anomaly detection with a reason for every alarm.

Three layers, in the order an engineer would think about them:

1. "Can I trust the measurement?"
   * Spectral PCA monitor (Hotelling T² and Q / SPE). The PCA model is built on
     the calibration spectra, so Q tells us "this spectrum contains something the
     calibration has never seen" (fouling, bubbles, instrument drift).
   * Mass (carbon) balance. Sugar that disappears must reappear as ethanol,
     glycerol or acids. Real process faults keep the balance; measurement faults
     (a dirty probe dims everything) break it.

2. "Is the chemistry going wrong?" (only trusted when the mass balance holds)
   * Contamination: lactic or acetic acid above action level and still rising.
   * Stuck / sluggish fermentation: compared with a "golden batch" envelope
     (mean ± SD of good historical batches at the same hour), glucose is too high
     while ethanol production is too slow.

3. Persistence: a condition must hold for several readings in a row before an
   alarm is raised, which removes most one-off false alarms.

All checks are causal: an alarm at time t only uses data up to time t,
exactly as it would run live in the plant.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from .chemometrics import Preprocessor

STUCK_GLUCOSE = 1.5      # % w/v of glucose still available...
STUCK_RATE = 0.15        # ...but ethanol rising slower than this (% w/v per h)


# ============================================================================
# 1a) Spectral PCA monitor
# ============================================================================
@dataclass
class SpectralMonitor:
    pre: Preprocessor
    mean: np.ndarray
    loadings: np.ndarray        # p x k
    eigvals: np.ndarray         # variance of each score
    t2_limit: float
    q_limit: float
    n_components: int
    explained: float

    @classmethod
    def fit(cls, X_raw, noise_factor=5.0, max_components=20, alpha=0.999,
            pre=Preprocessor(baseline="linear")):
        """Keep every principal component that carries clearly more variance than noise.

        The noise level is estimated from the small trailing eigenvalues (pure noise);
        a component is kept while its variance is > noise_factor x that level.
        """
        X = pre.transform(X_raw)
        mean = X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
        var = S ** 2 / (len(X) - 1)
        cum = np.cumsum(var) / var.sum()
        noise = np.median(var[len(var) // 2:])
        k = int(min(max(np.sum(var > noise_factor * noise), 1), max_components))
        P, lam = Vt[:k].T, var[:k]
        n = len(X)
        # T² limit from the F distribution
        t2_lim = k * (n - 1) * (n + 1) / (n * (n - k)) * stats.f.ppf(alpha, k, n - k)
        # Q limit: Box's approximation, Q ~ g * chi²(h), matched to the training Q values
        E = (X - mean) - (X - mean) @ P @ P.T
        q = np.sum(E ** 2, axis=1)
        g, h = q.var() / (2 * q.mean()), 2 * q.mean() ** 2 / q.var()
        q_lim = g * stats.chi2.ppf(alpha, h)
        return cls(pre, mean, P, lam, float(t2_lim), float(q_lim), k, float(cum[k - 1]))

    def score(self, X_raw):
        """Returns T², Q and the residual matrix (for contribution plots)."""
        Xc = self.pre.transform(X_raw) - self.mean
        T = Xc @ self.loadings
        t2 = np.sum(T ** 2 / self.eigvals, axis=1)
        E = Xc - T @ self.loadings.T
        return t2, np.sum(E ** 2, axis=1), E


def top_residual_region(residual_row, wn, width=40):
    """Wavenumber window (cm-1) that carries the largest unexplained signal."""
    e2 = residual_row ** 2
    step = wn[1] - wn[0]
    n = max(int(width / step), 1)
    s = np.convolve(e2, np.ones(n), mode="valid")
    i = int(np.argmax(s))
    return float(wn[i]), float(wn[i + n - 1])


# ============================================================================
# 1b) Mass balance and 2) golden-batch envelope
# ============================================================================
def hexose_equivalent(df: pd.DataFrame) -> np.ndarray:
    """Total glucose-equivalent in sugars + products (should stay ~constant)."""
    total = df[C.SUGARS].sum(axis=1).to_numpy()
    for a, cost in C.HEXOSE_COST.items():
        total = total + cost * df[a].to_numpy()
    return total


def causal_smooth(v: np.ndarray, window=5) -> np.ndarray:
    """Trailing rolling median (uses only the current and past readings)."""
    return pd.Series(v).rolling(window, min_periods=1).median().to_numpy()


def causal_rate(v: np.ndarray, hours=3.0) -> np.ndarray:
    """Change per hour over the last `hours`, using past readings only."""
    lag = int(round(hours / C.SAMPLE_INTERVAL_H))
    out = np.zeros_like(v)
    out[lag:] = (v[lag:] - v[:-lag]) / hours
    return out


def batch_signals(pred: pd.DataFrame) -> pd.DataFrame:
    """Smoothed trajectories + derived signals for one batch (row order = time)."""
    s = pd.DataFrame({"time_h": pred["time_h"].to_numpy()})
    for a in C.ANALYTES:
        s[a] = causal_smooth(pred[a].to_numpy())
    hx = hexose_equivalent(s)
    s["balance_ratio"] = hx / np.mean(hx[:3])
    s["ethanol_rate"] = causal_rate(s["ethanol"].to_numpy())
    s["lactic_rate"] = causal_rate(s["lactic"].to_numpy())
    return s


@dataclass
class GoldenEnvelope:
    """Mean and SD of good historical batches at each reading time."""
    mean: pd.DataFrame
    sd: pd.DataFrame
    SIGNALS = ("glucose", "ethanol", "ethanol_rate", "balance_ratio")
    SD_FLOOR = {"glucose": 0.3, "ethanol": 0.3, "ethanol_rate": 0.03, "balance_ratio": 0.006}

    @classmethod
    def fit(cls, signals: list[pd.DataFrame]):
        stack = {k: np.vstack([s[k].to_numpy() for s in signals]) for k in cls.SIGNALS}
        mean = pd.DataFrame({k: v.mean(axis=0) for k, v in stack.items()})
        sd = pd.DataFrame({k: np.maximum(v.std(axis=0, ddof=1), cls.SD_FLOOR[k])
                           for k, v in stack.items()})
        return cls(mean, sd)

    def z(self, signals: pd.DataFrame, key: str) -> np.ndarray:
        n = len(signals)
        return (signals[key].to_numpy() - self.mean[key].to_numpy()[:n]) / self.sd[key].to_numpy()[:n]


# ============================================================================
# 3) Rules -> alerts
# ============================================================================
def first_persistent(flags: np.ndarray, n=C.PERSISTENCE) -> int | None:
    """Index of the reading at which `flags` has been true n times in a row."""
    run = 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        if run >= n:
            return i
    return None


def persistent_mask(flags: np.ndarray, n=C.PERSISTENCE) -> np.ndarray:
    """True wherever the flag has held for at least n readings in a row."""
    out, run = np.zeros(len(flags), bool), 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        out[i] = run >= n
    return out


@dataclass
class AnomalyDetector:
    monitor: SpectralMonitor
    envelope: GoldenEnvelope
    wavenumbers: np.ndarray
    balance_z: float = 5.0

    def monitor_batch(self, pred: pd.DataFrame, X_raw: np.ndarray) -> pd.DataFrame:
        """Per-reading monitoring table for one batch (all signals + flags)."""
        s = batch_signals(pred)
        t2, q, _ = self.monitor.score(X_raw)
        s["T2"], s["Q"] = t2, q
        s["z_glucose"] = self.envelope.z(s, "glucose")
        s["z_ethanol"] = self.envelope.z(s, "ethanol")
        s["z_ethanol_rate"] = self.envelope.z(s, "ethanol_rate")
        s["z_balance"] = self.envelope.z(s, "balance_ratio")
        t = s["time_h"].to_numpy()

        s["flag_Q"] = persistent_mask(q > self.monitor.q_limit)
        s["flag_balance"] = persistent_mask(np.abs(s["z_balance"].to_numpy()) > self.balance_z)
        trusted = ~s["flag_balance"].to_numpy()
        s["flag_lactic"] = persistent_mask((s["lactic"] > C.LACTIC_ALERT).to_numpy()
                                           & (s["lactic_rate"] > 0.005).to_numpy() & trusted)
        s["flag_acetic"] = persistent_mask((s["acetic"] > C.ACETIC_ALERT).to_numpy() & trusted)
        # stuck: plenty of glucose is available and ethanol is behind good batches,
        # yet the yeast has almost stopped making ethanol
        s["flag_stuck"] = persistent_mask((t >= 12) & (s["glucose"] > STUCK_GLUCOSE).to_numpy()
                                          & (s["ethanol_rate"] < STUCK_RATE).to_numpy()
                                          & (s["z_ethanol"] < -2.0).to_numpy() & trusted, 6)
        return s

    def alerts_for_batch(self, batch_id: int, s: pd.DataFrame, X_raw: np.ndarray) -> list[dict]:
        """Turn the per-reading flags into at most one alert per category, with reasons."""
        out = []
        t = s["time_h"].to_numpy()

        def add(i, category, severity, title, message):
            out.append({"batch_id": batch_id, "time_h": float(t[i]), "category": category,
                        "severity": severity, "title": title, "message": message})

        # --- measurement layer ---------------------------------------------------
        iq = first_persistent(s["flag_Q"].to_numpy(), 1)
        ib = first_persistent(s["flag_balance"].to_numpy(), 1)
        if ib is not None:
            _, _, E = self.monitor.score(X_raw[ib:ib + 1])
            lo, hi = top_residual_region(E[0], self.wavenumbers)
            add(ib, "Measurement", "high", "Probe fouling / measurement failure suspected",
                f"Mass balance is off by {100 * (s['balance_ratio'].iat[ib] - 1):+.1f} % "
                f"(z = {s['z_balance'].iat[ib]:+.1f} vs. good batches): sugar and products no longer "
                f"add up, which a real process change cannot do. Spectral Q = "
                f"{s['Q'].iat[ib] / self.monitor.q_limit:.1f}x its limit, largest unexplained signal "
                f"near {lo:.0f}-{hi:.0f} cm-1. Clean/inspect the probe; predictions are not reliable "
                f"until this clears.")
        elif iq is not None:
            _, _, E = self.monitor.score(X_raw[iq:iq + 1])
            lo, hi = top_residual_region(E[0], self.wavenumbers)
            add(iq, "Measurement", "medium", "Unusual spectrum (outside calibration space)",
                f"Spectral Q residual has stayed above its 99.9 % limit for {C.PERSISTENCE} readings "
                f"({s['Q'].iat[iq] / self.monitor.q_limit:.1f}x limit), largest unexplained signal near "
                f"{lo:.0f}-{hi:.0f} cm-1. The mass balance still holds, so the chemistry is probably "
                f"fine but the spectrum looks different from the calibration: possible instrument "
                f"drift or a new matrix. Check against the next lab sample.")

        # --- chemistry layer -----------------------------------------------------
        il = first_persistent(s["flag_lactic"].to_numpy(), 1)
        if il is not None:
            add(il, "Contamination", "high", "Bacterial contamination suspected (lactic acid)",
                f"Lactic acid is {s['lactic'].iat[il]:.2f} {C.UNIT} and rising by "
                f"{s['lactic_rate'].iat[il]:.3f} %/h (action level {C.LACTIC_ALERT}). "
                f"Lactic acid bacteria compete with yeast for glucose and every 1 % lactic acid "
                f"costs ~0.5 % ethanol. Consider antibiotic dosing and check the propagator / heat "
                f"exchangers.")
        ia = first_persistent(s["flag_acetic"].to_numpy(), 1)
        if ia is not None:
            add(ia, "Contamination", "high", "Bacterial contamination suspected (acetic acid)",
                f"Acetic acid is {s['acetic'].iat[ia]:.2f} {C.UNIT} (action level {C.ACETIC_ALERT}). "
                f"Acetic acid is strongly toxic to yeast even at low levels; expect the fermentation "
                f"to slow down. Check sanitation and air ingress.")
        ist = first_persistent(s["flag_stuck"].to_numpy(), 1)
        if ist is not None:
            cause = (" Contamination was flagged earlier, so the slowdown is most likely its "
                     "consequence." if (il is not None and il < ist) or (ia is not None and ia < ist)
                     else " No acid build-up is seen, so suspect yeast stress (temperature, "
                          "nutrients, pH).")
            add(ist, "Process", "medium", "Sluggish / stuck fermentation",
                f"At {t[ist]:.1f} h there is still {s['glucose'].iat[ist]:.2f} {C.UNIT} glucose, but "
                f"ethanol is rising only {s['ethanol_rate'].iat[ist]:.3f} %/h (alarm below "
                f"{STUCK_RATE}) and is {s['ethanol'].iat[ist]:.2f} {C.UNIT}, behind good batches "
                f"at the same hour (z = {s['z_ethanol'].iat[ist]:+.1f}).{cause}")
        return out


def build_detector(signals_normal: list[pd.DataFrame], X_calibration: np.ndarray, wavenumbers):
    return AnomalyDetector(SpectralMonitor.fit(X_calibration), GoldenEnvelope.fit(signals_normal),
                           wavenumbers)


# ============================================================================
# Evaluation against the simulator's ground truth
# ============================================================================
EXPECTED = {
    "normal": set(),
    "contamination_lactic": {"Contamination"},
    "contamination_acetic": {"Contamination"},
    "stuck": {"Process"},
    "probe_fouling": {"Measurement"},
}
ALLOWED_SECONDARY = {
    "contamination_lactic": {"Process"},      # contamination can slow the yeast down
    "contamination_acetic": {"Process"},
}


def evaluate(alerts: pd.DataFrame, batches: pd.DataFrame) -> pd.DataFrame:
    """Per-batch verdict: was the true fault caught, how fast, and any false alarms?"""
    rows = []
    for _, b in batches.iterrows():
        # calibration alerts are about the instrument over many batches, not about one batch
        a = alerts[(alerts.batch_id == b.batch_id) & (alerts.category != "Calibration")]
        cats = set(a.category)
        expected = set(EXPECTED[b.scenario])
        allowed = expected | ALLOWED_SECONDARY.get(b.scenario, set())
        if b.instrument_drift:
            allowed = allowed | {"Measurement"}   # drift batches may legitimately flag spectra
        detected = expected <= cats if expected else None
        delay = None
        if expected and detected and pd.notna(b.fault_onset_h):
            delay = float(a[a.category.isin(expected)].time_h.min() - b.fault_onset_h)
        rows.append({"batch_id": b.batch_id, "scenario": b.scenario, "split": b.split,
                     "instrument_drift": b.instrument_drift,
                     "alerts": ", ".join(sorted(cats)) or "-",
                     "detected": detected, "delay_h": delay,
                     "false_alarm": bool(cats - allowed)})
    return pd.DataFrame(rows)
