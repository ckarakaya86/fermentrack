# FermenTrack user guide

A step-by-step guide for using FermenTrack without writing any code.
FermenTrack has two parts, both in the same folder and the same repository:

| Part | What it is for | How to open |
|---|---|---|
| **Dashboard** | Watch simulated fermentation batches "live": concentrations, final-ethanol forecast, alerts, model performance | Double-click **`FermenTrack.bat`** → browser opens at `http://localhost:8501` |
| **Model Workbench** | Build, check, update and auto-validate your **own** PLS / ANN models from FTIR (or NIR) spectra + lab values | Double-click **`Workbench.bat`** → `http://localhost:8502`, **or** choose *Model workbench* in the dashboard's left menu |

Keep the black window that opens with the `.bat` file open while you work; closing it stops the app.

---

## 1. First start

1. Install Python 3.10 or newer (once).
2. Double-click `FermenTrack.bat` or `Workbench.bat`. The first time, missing packages are
   installed automatically, which takes a few minutes.
3. The dashboard simulates its 80 batches the first time it opens (~15 s); later starts are instant.

---

## 2. Dashboard

| Page (left menu) | What to look at |
|---|---|
| **Batch monitor** | Pick a batch, move the *Simulated clock* slider. Lines = in-line predictions, diamonds = lab samples. Alerts appear with a reason and a suggested action. Good examples: **batch 68** (bacterial contamination flagged at 30 h) and **batch 62** (probe fouling; the forecast is frozen because the measurement cannot be trusted). |
| **Plant overview** | All 80 batches: final yield, every alert, ground truth of the simulation. |
| **Model performance** | How good each model is, scored on batches it never saw: PLS calibration, early yield forecast, anomaly detection, calibration-drift chart. |
| **Model workbench** | The workbench (section 3). |
| **About** | Short summary of the methods. |

Shareable links: `?page=monitor&batch=68&t=30`, `?page=performance&section=yield`.

---

## 3. Model Workbench, step by step

### Example files (in the `examples/` folder)

| File | Used in | What it shows |
|---|---|---|
| `ftir_calibration.xlsx` | Step 1 | 316 lab-checked samples to build a model. Contains **3 planted problems** for step 5: a fouled-probe spectrum (`B02-40.0h`), a typing error (`B03-32.3h`), two swapped labels (`B03-08.0h`, `B09-60.0h`). |
| `ftir_new_lab_ok.xlsx` | Step 8 | New lab results, instrument healthy → verdict **OK** |
| `ftir_new_lab_drift.xlsx` | Steps 7 and 8 | New lab results, instrument drifting → verdict **ACTION**, model is updated |

You do not need to browse for these: steps 1, 7 and 8 each have an *example* option.

### Suggested first run (about 5 minutes)

1. **1 Data** → *Source: Example* → **Load example calibration data**. Property: *Ethanol*.
2. **2 Spectra** → look at the overlaid spectra (light → dark blue = low → high ethanol).
   Set *Derivative* to **1st**.
3. **3 Split** → method *By group* → **Apply split**.
4. **4 Model** → *PLS* → **Fit model**. Note RMSECV is about 1.2 and only 2 components are used:
   the planted problems are spoiling the model.
5. **5 Outliers** → the planted samples are pre-ticked with their reasons →
   **Exclude ticked samples and re-fit**. RMSECV drops to about **0.10**.
6. **6 Robustness** → run the three tests and see how the model copes with disturbed spectra.
7. **7 Save & update** → **Save as new version** (saved as `Ethanol v001`).
8. **8 Auto-check** → example `ftir_new_lab_ok.xlsx` → **Run check** → **OK**.
   Then `ftir_new_lab_drift.xlsx` with *Allow automatic update* ticked → **ACTION** →
   a better version `v002` is saved and activated.

### What each step does

| Step | You do | Good to know |
|---|---|---|
| **1 Data** | Upload spectra (+ optional separate reference file); choose the property; tick *Exclude* and type a reason for any sample you want to leave out | Two layouts are recognised: **rows = samples** (wavenumbers as column headers) or **columns = samples** (wavenumbers in column A). For Excel files with several sheets, pick the sheet. A reference file needs a `sample_id` column; optional `group` (batch/lot) and `date` columns enable better splits. *Download cleaned data* saves the edited set. |
| **2 Spectra** | Choose wavenumber regions (e.g. `980-1200, 1250-1480`), baseline, SNV/MSC, smoothing, derivative | Raw and preprocessed spectra are drawn on top of each other. The expander shows the mean spectrum and the correlation of every wavenumber with the property, which helps choose regions. |
| **3 Split** | Choose a split method and validation share → **Apply split** | *By group* is the honest default when samples of one batch are related; *By order/date* tests "train on the past, predict the future". The PCA plot shows whether validation samples sit inside the calibration space. |
| **4 Model** | PLS or ANN, size by cross-validation or fixed → **Fit model** | Judge models by **RMSECV** and validation **RMSEP**, not by RMSEC. RPD > 3 is fine for screening, > 5 for quality control. Each fitted model is added to a comparison table. ANN = small neural network fed with PLS scores; useful when the relationship is curved. |
| **5 Outliers** | Review flagged samples, tick, **Exclude … and re-fit** | **T²** = extreme but understood; **Q** = spectrum contains something unknown (fouling, wrong sample); **studentised residual** = lab value disagrees with the spectrum (typing error, swapped label). Only strong cases are pre-ticked. Every exclusion is logged and can be undone in the *Exclusion log*. |
| **6 Robustness** | Run perturbation, resampling and y-randomisation tests | Lower lines = more robust. "RMSEP doubles at" is a handy single number for comparing preprocessing choices. y-randomisation must fail clearly for shuffled data; otherwise the model is fitting chance. |
| **7 Save & update** | Save a version with notes; later upload new lab-checked samples → **Build update candidates** | Versions `v001`, `v002`… with a *model card* (settings, statistics, data fingerprint, excluded samples). Two update candidates are compared on new samples they never saw: a **bias update** (fixes a constant offset) or a **refit with old + new samples** (fixes new variation). |
| **8 Auto-check** | Pick a model, give new lab results → **Run check** | Verdict **OK / WARNING / ACTION** with findings and the likely fix. The history chart shows every check. With *Allow automatic update*, a new version is activated only if it is better on held-out new samples and not worse on the original validation. |

### Using your own data

* Spectra and lab values can be in one file (lab values as extra columns) or in two files
  (matched by `sample_id`).
* Non-numeric columns other than `group` and `date` are kept as sample information.
* Your saved models live in the `model_registry/` folder, which is **not** uploaded to GitHub.

### Running the check automatically every day

```bash
py -3.12 -m fermentrack.workbench.autovalidate --model Ethanol --data "C:\path\to\new_lab.xlsx" --auto-update
```

Windows Task Scheduler → *Create Basic Task* → daily → *Start a program*: program `py`, arguments
as above (without `py`), *Start in* = the FermenTrack folder. Exit code `2` means action is
needed and can trigger an e-mail. Step 8 shows the exact command for the selected model.

---

## 4. Troubleshooting

| Problem | Fix |
|---|---|
| Browser does not open | Go to `http://localhost:8501` (dashboard) or `http://localhost:8502` (workbench). |
| "Could not find wavenumbers" | Wavenumbers must be numbers between 200 and 15000, either as column headers or in the first column. |
| "At least 5 calibration samples…" | The chosen property has too few lab values after exclusions. |
| "The data changed since the model was fitted" | You changed exclusions, preprocessing or the split: press **Fit model** again in step 4. |
| A second app does not start | The dashboard uses port 8501 and the workbench 8502; close old black windows first. |
