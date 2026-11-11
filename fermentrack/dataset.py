"""Build the whole synthetic plant history: 80 batches, spectra and lab samples.

Outputs (in-memory `PlantData`, and saved to data/ by `save`):
* batches   - one row per batch: scenario, train/test split, key parameters
* process   - TRUE concentrations every 20 min (only the simulator knows these)
* spectra   - one mid-IR spectrum per reading (what the probe sees)
* lab       - sparse HPLC lab results (what the plant actually knows), with lab error
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .kinetics import sample_params, simulate_batch
from .spectroscopy import sample_sensor_params, simulate_spectra

# HPLC error: relative part + absolute floor (% w/v), per analyte
LAB_REL_ERR = 0.008
LAB_ABS_ERR = {"glucose": 0.03, "dp2": 0.03, "dp3": 0.03, "dp4plus": 0.08, "ethanol": 0.03,
               "glycerol": 0.01, "lactic": 0.01, "acetic": 0.005}


@dataclass
class PlantData:
    batches: pd.DataFrame
    process: pd.DataFrame
    spectra: np.ndarray        # (n_readings, n_wavenumbers), row-aligned with `process`
    lab: pd.DataFrame
    wavenumbers: np.ndarray

    def batch_rows(self, batch_id: int) -> np.ndarray:
        return np.where(self.process["batch_id"].to_numpy() == batch_id)[0]


def scenario_plan(rng: np.random.Generator) -> list[str]:
    """Shuffle the scenario mix, making sure every fault type appears in the test period."""
    plan = [s for s, n in C.SCENARIO_COUNTS.items() for _ in range(n)]
    assert len(plan) == C.N_BATCHES
    while True:
        rng.shuffle(plan)
        test = plan[C.N_TRAIN_BATCHES:]
        need = {"contamination_lactic": 2, "contamination_acetic": 1, "stuck": 2,
                "probe_fouling": 2, "normal": 10}
        if all(test.count(s) >= n for s, n in need.items()):
            return list(plan)


def lab_samples(rng, batch_id, true_df) -> pd.DataFrame:
    """Pick the readings closest to the nominal lab times and add HPLC error."""
    rows = []
    for nominal in C.LAB_TIMES_H:
        t = nominal + (rng.uniform(-0.7, 0.7) if 0 < nominal < C.BATCH_DURATION_H else 0)
        i = int(np.argmin(np.abs(true_df["time_h"].to_numpy() - t)))
        row = {"batch_id": batch_id, "time_h": float(true_df["time_h"].iat[i]), "row_in_batch": i}
        for a in C.ANALYTES:
            true = float(true_df[a].iat[i])
            sd = LAB_REL_ERR * true + LAB_ABS_ERR[a]
            row[a] = max(true + rng.normal(0, sd), 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def generate(seed: int = C.SEED) -> PlantData:
    rng = np.random.default_rng(seed)
    plan = scenario_plan(rng)
    batch_rows, proc, spec, labs = [], [], [], []
    for b, scenario in enumerate(plan):
        pp = sample_params(rng, b, scenario)
        sp = sample_sensor_params(rng, b, scenario)
        true = simulate_batch(pp)
        X = simulate_spectra(true[C.ANALYTES].to_numpy(), true["time_h"].to_numpy(), sp, rng)

        true.insert(0, "batch_id", b)
        proc.append(true)
        spec.append(X)
        labs.append(lab_samples(rng, b, true))

        initial_sugar = float(true.loc[true.index[0], C.SUGARS].sum())
        final_eth = float(true["ethanol"].iat[-1])
        batch_rows.append({
            "batch_id": b,
            "scenario": scenario,
            "split": "train" if b < C.N_TRAIN_BATCHES else "test",
            "instrument_drift": b >= C.DRIFT_START_BATCH,
            "wn_shift_cm1": sp.wn_shift_cm1,
            "fault_onset_h": pp.contamination_onset_h or pp.stuck_onset_h or sp.fouling_onset_h,
            "initial_sugar": initial_sugar,
            "final_ethanol_true": final_eth,
            "yield_pct_true": 100 * final_eth / (C.ETHANOL_YIELD * initial_sugar),
        })

    process = pd.concat(proc, ignore_index=True)
    lab = pd.concat(labs, ignore_index=True)
    # map each lab sample to its row in the global spectra matrix
    starts = process.groupby("batch_id").cumcount() == 0
    first_row = dict(zip(process.loc[starts, "batch_id"], np.where(starts)[0]))
    lab.insert(2, "spectrum_row", [first_row[b] + i for b, i in zip(lab.batch_id, lab.row_in_batch)])
    lab = lab.drop(columns="row_in_batch")
    return PlantData(pd.DataFrame(batch_rows), process, np.vstack(spec), lab, C.WAVENUMBERS.copy())


def save(data: PlantData, folder=C.DATA_DIR):
    folder.mkdir(parents=True, exist_ok=True)
    data.batches.to_csv(folder / "batches.csv", index=False)
    data.process.to_csv(folder / "process_true.csv", index=False)
    data.lab.to_csv(folder / "lab_hplc.csv", index=False)
    np.savez_compressed(folder / "spectra.npz", spectra=data.spectra.astype(np.float32),
                        wavenumbers=data.wavenumbers)


def load(folder=C.DATA_DIR) -> PlantData:
    z = np.load(folder / "spectra.npz")
    return PlantData(pd.read_csv(folder / "batches.csv"), pd.read_csv(folder / "process_true.csv"),
                     z["spectra"].astype(float), pd.read_csv(folder / "lab_hplc.csv"),
                     z["wavenumbers"])
