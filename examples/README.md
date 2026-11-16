# Example FTIR files (simulated)

Rows = samples; `sample_id`, `group` (batch), `date`, reference values (Ethanol, Glucose, Glycerol, Lactic acid in % w/v), then one column per wavenumber (cm-1).

* `ftir_calibration.xlsx` - build a model from scratch. Planted problems to find in the Outliers step: fouled spectrum `B02-40.0h`, typing error `B03-32.3h`, swapped labels `B03-08.0h` / `B09-60.0h`.
* `ftir_new_lab_ok.xlsx` - new lab results while the instrument is healthy (auto-check: OK).
* `ftir_new_lab_drift.xlsx` - new lab results while the instrument drifts (auto-check: ACTION).

Regenerate with `python -m fermentrack.workbench.examples`.
