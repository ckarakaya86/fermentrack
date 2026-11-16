# Changelog

## 2026-11-16: Data preparation

* Import of **one file per spectrum** (the usual FTIR export): header lines, comma / semicolon / tab separators and decimal commas understood; spectra on different wavenumber grids are aligned; the file name becomes the sample id.
* Decimal commas accepted in spectra tables and lab values.
* **Templates** in `examples/templates/` for every supported layout (one table, spectra + lab file, columns = samples, per-spectrum files), downloadable from step 1.
* Step 1 explains with small examples how files should look; the user guide has a new *Preparing your own data* section with good-practice tips.
* 3 new tests (48 in total).

## 2026-11-16: Model Workbench

A second part of FermenTrack for **building and maintaining your own spectral models**
(FTIR / NIR spectra + lab or physical measurements). It opens with `Workbench.bat` or as the
*Model workbench* page of the dashboard. See [docs/USER_GUIDE.md](docs/USER_GUIDE.md).

* **Data**: Excel/CSV import in both layouts (rows = samples, or columns = samples like the
  column-per-sample file); separate reference files matched by sample id; manual exclusion with reasons.
* **Spectra & preprocessing**: overlay plots coloured by value / set / exclusion; regions,
  baseline, SNV, MSC, Savitzky-Golay smoothing and derivatives (fitted on calibration only).
* **Calibration / validation split**: random, Kennard-Stone, stratified by value, by group, by date.
* **Models**: NIPALS PLS and PLS-ANN (neural-network ensemble on PLS scores), cross-validated;
  RMSEC, RMSECV, RMSEP, R², bias, SEP, slope, RPD; side-by-side comparison.
* **Outliers**: Hotelling T², Q residuals, leverage, studentised residuals, each with a plain reason;
  logged and reversible exclusions.
* **Robustness**: perturbation tests (noise, offset, slope, gain, wavenumber shift), repeated random
  splits, y-randomisation.
* **Model registry**: numbered versions with model cards; updates by bias correction or by refitting
  with old + new samples, compared on held-out new samples.
* **Auto-check**: OK / WARNING / ACTION against new lab data (accuracy, bias test, EWMA drift,
  spectral outliers), optional safe automatic update, command-line tool for daily scheduling.
* **Examples**: simulated FTIR files with planted problems (fouled spectrum, typing error, swapped
  labels) plus "healthy" and "drifting" new-lab files.
* **Tests**: 24 new (45 in total), including a headless click-through of every workbench step.

## 2026-11-11: First version

* Simulated fuel-ethanol plant: 80 batches with kinetics (enzymes, yeast, bacteria), mid-IR spectra,
  HPLC lab samples, faults (contamination, stuck fermentation, probe fouling, instrument drift).
* PLS calibration per analyte with preprocessing chosen by batch-wise cross-validation.
* Early final-ethanol forecast (theoretical yield × robust ML efficiency model) with uncertainty.
* Anomaly detection with reasons: PCA T²/Q, mass balance, contamination and stuck-fermentation rules.
* Calibration health: EWMA chart of in-line minus lab, rolling bias update.
* Streamlit dashboard (`FermenTrack.bat`), README with results, 21 tests.
