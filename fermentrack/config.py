"""Central settings for FermenTrack.

Everything that defines the simulated "plant" lives here, so the whole
project is reproducible from one place (fixed random seed, fixed batch plan).
"""
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Analytes (all concentrations in % w/v, i.e. g per 100 mL, as HPLC reports them)
# ---------------------------------------------------------------------------
ANALYTES = ["glucose", "dp2", "dp3", "dp4plus", "ethanol", "glycerol", "lactic", "acetic"]
LABELS = {
    "glucose": "Glucose",
    "dp2": "DP2 (maltose)",
    "dp3": "DP3 (maltotriose)",
    "dp4plus": "DP4+ (dextrins)",
    "ethanol": "Ethanol",
    "glycerol": "Glycerol",
    "lactic": "Lactic acid",
    "acetic": "Acetic acid",
}
SUGARS = ["glucose", "dp2", "dp3", "dp4plus"]
UNIT = "% w/v"

# Glucose-equivalent (hexose) cost of each product, in g sugar per g product.
# Used for the mass-balance check: process faults keep the balance, sensor faults break it.
ETHANOL_YIELD = 0.511            # theoretical g ethanol per g glucose (Gay-Lussac)
HEXOSE_COST = {
    "ethanol": 1 / ETHANOL_YIELD,  # glucose -> 2 ethanol + 2 CO2
    "glycerol": 1.0,               # ~1 g glucose per g glycerol
    "lactic": 1.0,                 # glucose -> 2 lactic acid
    "acetic": 1.5,                 # glucose -> 2 acetic acid + 2 CO2 (simplified)
}

# ---------------------------------------------------------------------------
# Time base
# ---------------------------------------------------------------------------
BATCH_DURATION_H = 60.0
SAMPLE_INTERVAL_H = 1 / 3        # one in-line spectrum every 20 minutes
TIMES_H = np.round(np.arange(0, BATCH_DURATION_H + 1e-9, SAMPLE_INTERVAL_H), 4)

# Nominal times when an operator pulls a sample for the HPLC lab
LAB_TIMES_H = [0, 4, 8, 12, 16, 24, 32, 40, 48, 60]

# ---------------------------------------------------------------------------
# Spectrometer (mid-IR fingerprint region)
# ---------------------------------------------------------------------------
WAVENUMBERS = np.arange(950.0, 1502.0, 2.0)  # cm-1, 276 points

# ---------------------------------------------------------------------------
# Plant history: 80 batches in chronological order
# ---------------------------------------------------------------------------
N_BATCHES = 80
N_TRAIN_BATCHES = 56             # first 56 = "past" (training), last 24 = "future" (test)
DRIFT_START_BATCH = 66           # from this batch on, the instrument slowly drifts
SCENARIO_COUNTS = {
    "normal": 52,
    "contamination_lactic": 7,
    "contamination_acetic": 4,
    "stuck": 9,
    "probe_fouling": 8,
}
SCENARIO_LABELS = {
    "normal": "Normal",
    "contamination_lactic": "Lactic bacteria contamination",
    "contamination_acetic": "Acetic bacteria contamination",
    "stuck": "Stuck / sluggish fermentation",
    "probe_fouling": "Probe fouling",
}

# ---------------------------------------------------------------------------
# Process-intelligence settings
# ---------------------------------------------------------------------------
YIELD_CUTOFFS_H = [8, 12, 16, 20, 24]   # early-prediction horizons
LACTIC_ALERT = 0.45      # % w/v, typical plant action level is 0.3-0.6
ACETIC_ALERT = 0.20      # % w/v
PERSISTENCE = 3          # readings in a row needed before an alarm is raised

# ---------------------------------------------------------------------------
# File locations
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
