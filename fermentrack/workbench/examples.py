"""Create example FTIR files for trying out the workbench.

    python -m fermentrack.workbench.examples

Writes to examples/ (Excel, rows = samples, wavenumbers as column headers):
* ftir_calibration.xlsx  - lab-checked samples from the simulated plant's past,
                           with three planted problems for the outlier step:
                           a fouled-probe spectrum, a typing error in a lab value
                           (decimal point slipped), and two swapped sample labels
* ftir_new_lab_ok.xlsx   - later samples, instrument healthy  -> auto-check says OK
* ftir_new_lab_drift.xlsx- even later samples, instrument drifting -> auto-check says ACTION
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from .. import dataset

OUT = C.ROOT / "examples"
PROPS = {"ethanol": "Ethanol", "glucose": "Glucose", "glycerol": "Glycerol", "lactic": "Lactic acid"}


def _frame(data, lab: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp("2026-01-05 06:00")
    when = [start + pd.Timedelta(days=2.5 * b) + pd.Timedelta(hours=t)
            for b, t in zip(lab.batch_id, lab.time_h)]
    df = pd.DataFrame({
        "sample_id": [f"B{b:02d}-{t:04.1f}h" for b, t in zip(lab.batch_id, lab.time_h)],
        "group": [f"batch {b:02d}" for b in lab.batch_id],
        "date": [w.strftime("%Y-%m-%d %H:%M") for w in when],
    })
    for k, name in PROPS.items():
        df[name] = lab[k].round(3).to_numpy()
    spec = pd.DataFrame(np.round(data.spectra[lab.spectrum_row.to_numpy()], 5),
                        columns=[f"{w:g}" for w in data.wavenumbers])
    return pd.concat([df.reset_index(drop=True), spec], axis=1)


def make(out: Path = OUT, seed: int = C.SEED) -> dict:
    data = dataset.generate(seed)
    b = data.batches.set_index("batch_id")
    lab = data.lab.merge(data.batches[["batch_id", "scenario", "split", "instrument_drift"]])
    lab = lab[lab.time_h > 0]                              # skip the trivial t = 0 samples

    cal = lab[(lab.split == "train") & (lab.scenario != "probe_fouling")]
    cal = cal[cal.batch_id < 40]                           # keep the file small: 40 batches
    cal_df = _frame(data, cal).reset_index(drop=True)

    # planted problem 1: a spectrum taken while the probe was fouled
    foul_batch = int(b[(b.scenario == "probe_fouling") & (b.split == "train")].index[0])
    onset = b.loc[foul_batch, "fault_onset_h"]
    fl = lab[(lab.batch_id == foul_batch) & (lab.time_h > onset + 6)].head(1)
    cal_df = pd.concat([cal_df, _frame(data, fl)], ignore_index=True)
    # planted problem 2: decimal point slipped when typing a lab value (12.4 -> 1.24)
    i_typo = int(cal_df.index[cal_df["Ethanol"] > 10][5])
    cal_df.loc[i_typo, "Ethanol"] = round(cal_df.loc[i_typo, "Ethanol"] / 10, 3)
    # planted problem 3: two sample labels swapped (reference values belong to the other spectrum)
    i, j = int(cal_df.index[cal_df["Ethanol"].between(2, 4)][3]), int(cal_df.index[cal_df["Ethanol"] > 11][20])
    for c in PROPS.values():
        cal_df.loc[i, c], cal_df.loc[j, c] = cal_df.loc[j, c], cal_df.loc[i, c]
    planted = {"fouled spectrum": cal_df.sample_id.iat[-1], "typing error": cal_df.sample_id.iat[i_typo],
               "swapped labels": [cal_df.sample_id.iat[i], cal_df.sample_id.iat[j]]}

    ok = lab[(lab.split == "test") & ~lab.instrument_drift & (lab.scenario != "probe_fouling")]
    drift = lab[lab.instrument_drift & (lab.batch_id >= 72) & (lab.scenario != "probe_fouling")]

    out.mkdir(parents=True, exist_ok=True)
    cal_df.to_excel(out / "ftir_calibration.xlsx", index=False)
    _frame(data, ok).to_excel(out / "ftir_new_lab_ok.xlsx", index=False)
    _frame(data, drift).to_excel(out / "ftir_new_lab_drift.xlsx", index=False)
    (out / "README.md").write_text(
        "# Example FTIR files (simulated)\n\n"
        "Rows = samples; `sample_id`, `group` (batch), `date`, reference values "
        "(Ethanol, Glucose, Glycerol, Lactic acid in % w/v), then one column per wavenumber (cm-1).\n\n"
        "* `ftir_calibration.xlsx` - build a model from scratch. Planted problems to find in the "
        f"Outliers step: fouled spectrum `{planted['fouled spectrum']}`, typing error "
        f"`{planted['typing error']}`, swapped labels `{planted['swapped labels'][0]}` / "
        f"`{planted['swapped labels'][1]}`.\n"
        "* `ftir_new_lab_ok.xlsx` - new lab results while the instrument is healthy (auto-check: OK).\n"
        "* `ftir_new_lab_drift.xlsx` - new lab results while the instrument drifts (auto-check: ACTION).\n\n"
        "Regenerate with `python -m fermentrack.workbench.examples`.\n", encoding="utf-8")
    return planted


if __name__ == "__main__":
    print(make())
