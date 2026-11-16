"""Reading spectra and reference (lab / physical) measurements into one dataset.

Supported layouts (Excel .xlsx/.xls or .csv), detected automatically:

* Rows = samples (most instrument exports):
      sample_id | 4000 | 3998 | ... | 650
  Every column whose header is a number is a wavenumber; other columns are
  treated as sample information (id, date, batch...). Numeric non-wavenumber
  columns in the same sheet become reference values.

* Columns = samples (one spectrum per column):
      No       | 1     | 2     | ...
      Date     | ...   | ...
      Ethanol  | 12.1  | 11.8  | ...      <- property rows (optional)
      4000.0   | 0.12  | 0.13  | ...      <- wavenumber in column A, one spectrum per column

A separate reference file (rows = samples, a sample-id column + one column per
property, optional "group" / "date" columns) is matched to the spectra by sample id.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ID_NAMES = ("sample_id", "sample", "id", "no", "sample no", "name", "numune")
GROUP_NAMES = ("group", "batch", "batch_id", "lot", "parti")
DATE_NAMES = ("date", "time", "datetime", "tarih")


@dataclass
class SpectralDataset:
    """Spectra + reference values + sample information, all row-aligned."""
    ids: np.ndarray                  # sample ids (strings)
    wn: np.ndarray                   # wavenumbers (cm-1), ascending or descending as read
    X: np.ndarray                    # spectra, n_samples x n_wavenumbers
    Y: pd.DataFrame                  # reference values, one column per property (NaN = not measured)
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)   # group, date, ...
    source: str = ""

    @property
    def n(self):
        return len(self.ids)

    @property
    def properties(self):
        return list(self.Y.columns)

    def subset(self, mask) -> "SpectralDataset":
        mask = np.asarray(mask)
        return SpectralDataset(self.ids[mask], self.wn, self.X[mask],
                               self.Y[mask].reset_index(drop=True),
                               self.meta[mask].reset_index(drop=True) if len(self.meta) else self.meta,
                               self.source)

    def fingerprint(self) -> str:
        """Short hash of the data, stored with every model so we know what it was trained on."""
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(self.X, dtype=np.float32).tobytes())
        h.update(self.Y.to_csv(index=False).encode())
        return h.hexdigest()[:12]

    def summary(self) -> dict:
        return {"samples": self.n, "points": len(self.wn),
                "range_cm1": f"{self.wn.min():.0f}-{self.wn.max():.0f}",
                "properties": ", ".join(self.properties) or "(none)"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_number(x):
    try:
        return float(str(x).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def _num_or_nan(x) -> float:
    v = _as_number(x)
    return np.nan if v is None else v


def _clean_name(x) -> str:
    s = re.sub(r"\s+", " ", str(x)).strip()
    return s.replace("�", "°")          # repair a degree sign broken by old encodings


def _read_raw(path_or_buffer, sheet=0, name="") -> pd.DataFrame:
    name = str(name or getattr(path_or_buffer, "name", path_or_buffer)).lower()
    if name.endswith(".csv") or name.endswith(".txt"):
        raw = pd.read_csv(path_or_buffer, header=None, sep=None, engine="python")
    else:
        raw = pd.read_excel(path_or_buffer, sheet_name=sheet, header=None)
    return raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)


def excel_sheets(path_or_buffer) -> list[str]:
    return pd.ExcelFile(path_or_buffer).sheet_names


def _id_str(v) -> str:
    """Sample id as text; 12.0 (Excel number) becomes '12'."""
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v).strip()


def _looks_like_wavenumber(v) -> bool:
    x = _as_number(v)
    return x is not None and 200 <= x <= 15000


def _split_info(info: pd.DataFrame):
    """Separate id / group / date columns from numeric property columns."""
    cols = {c: _clean_name(c) for c in info.columns}
    lower = {c: n.lower() for c, n in cols.items()}
    id_col = next((c for c in info.columns if lower[c] in ID_NAMES), None)
    meta, Y = pd.DataFrame(index=info.index), pd.DataFrame(index=info.index)
    for c in info.columns:
        if c == id_col:
            continue
        if lower[c] in GROUP_NAMES:
            meta["group"] = info[c].astype(str)
        elif lower[c] in DATE_NAMES:
            meta["date"] = info[c]
        else:
            num = info[c].map(_num_or_nan).astype(float)           # "12,3" counts as a number too
            if num.notna().mean() >= 0.5:
                Y[cols[c]] = num
            else:
                meta[cols[c]] = info[c]
    ids = (np.array([_id_str(v) for v in info[id_col]]) if id_col is not None
           else np.array([f"S{i + 1}" for i in range(len(info))]))
    return ids, Y.reset_index(drop=True), meta.reset_index(drop=True)


# ---------------------------------------------------------------------------
# main readers
# ---------------------------------------------------------------------------
def read_spectra(path_or_buffer, sheet=0, name="") -> SpectralDataset:
    """Read a spectra file in either layout (see module docstring)."""
    raw = _read_raw(path_or_buffer, sheet, name)
    header = raw.iloc[0].tolist()
    n_wn_header = sum(_looks_like_wavenumber(v) for v in header)
    n_wn_col0 = sum(_looks_like_wavenumber(v) for v in raw.iloc[:, 0])

    if n_wn_header >= 10 and n_wn_header >= n_wn_col0:          # rows = samples
        wn_cols = [i for i, v in enumerate(header) if _looks_like_wavenumber(v)]
        info_cols = [i for i in range(raw.shape[1]) if i not in wn_cols]
        body = raw.iloc[1:].reset_index(drop=True)
        wn = np.array([_as_number(header[i]) for i in wn_cols])
        X = body.iloc[:, wn_cols].map(_num_or_nan).to_numpy(float)     # also reads "0,123"
        info = body.iloc[:, info_cols]
        info.columns = [_clean_name(header[i]) for i in info_cols]
        ids, Y, meta = _split_info(info)
    elif n_wn_col0 >= 10:                                      # columns = samples
        col0 = raw.iloc[:, 0]
        is_wn = col0.map(_looks_like_wavenumber).to_numpy()
        first = int(np.argmax(is_wn))
        wn = col0.iloc[first:][is_wn[first:]].map(_as_number).to_numpy(float)
        spec_rows = np.where(is_wn)[0]
        spec_rows = spec_rows[spec_rows >= first]
        sample_cols = [c for c in range(1, raw.shape[1]) if pd.notna(raw.iat[first, c])]
        X = raw.iloc[spec_rows, sample_cols].map(_num_or_nan).to_numpy(float).T
        label_rows = {_clean_name(raw.iat[r, 0]): r for r in range(first) if pd.notna(raw.iat[r, 0])}
        info = pd.DataFrame({lab: raw.iloc[r, sample_cols].to_numpy() for lab, r in label_rows.items()})
        ids, Y, meta = _split_info(info)
    else:
        raise ValueError("Could not find wavenumbers: expected them either as column headers "
                         "(rows = samples) or in the first column (columns = samples).")

    keep = ~np.isnan(X).all(axis=0)
    X, wn = X[:, keep], wn[keep]
    good = ~np.isnan(X).any(axis=1)
    if not good.all():                                          # drop incomplete spectra
        X, ids, Y = X[good], ids[good], Y[good].reset_index(drop=True)
        meta = meta[good].reset_index(drop=True) if len(meta.columns) else meta
    return SpectralDataset(ids, wn, X, Y, meta, str(name or path_or_buffer))


def read_reference(path_or_buffer, sheet=0, name="") -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Reference file: rows = samples, an id column + property columns (+ group/date)."""
    raw = _read_raw(path_or_buffer, sheet, name)
    info = raw.iloc[1:].reset_index(drop=True)
    info.columns = [_clean_name(c) for c in raw.iloc[0]]
    return _split_info(info)


def attach_reference(ds: SpectralDataset, ids, Y: pd.DataFrame, meta: pd.DataFrame) -> SpectralDataset:
    """Match reference values to spectra by sample id (reference values win on conflicts)."""
    ref = Y.copy()
    ref.index = pd.Index(np.asarray(ids).astype(str))
    ref = ref[~ref.index.duplicated(keep="last")]
    merged = ds.Y.copy()
    merged.index = pd.Index(ds.ids)
    for c in ref.columns:
        merged[c] = ref[c].reindex(ds.ids).to_numpy()
    m = ds.meta.copy() if len(ds.meta.columns) else pd.DataFrame(index=range(ds.n))
    if len(meta.columns):
        mm = meta.copy()
        mm.index = pd.Index(np.asarray(ids).astype(str))
        mm = mm[~mm.index.duplicated(keep="last")]
        for c in mm.columns:
            m[c] = mm[c].reindex(ds.ids).to_numpy()
    return SpectralDataset(ds.ids, ds.wn, ds.X, merged.reset_index(drop=True),
                           m.reset_index(drop=True), ds.source)


# ---------------------------------------------------------------------------
# one file per spectrum (typical FTIR software export)
# ---------------------------------------------------------------------------
_NUM = re.compile(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")


def _two_column_lines(text: str) -> np.ndarray:
    """(wavenumber, absorbance) pairs from a text export; header / info lines are skipped.

    Handles comma, semicolon, tab or space separators and decimal commas (1234,5;0,123).
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0].isalpha() or line.startswith(("#", "##", "%")):
            continue
        if ";" in line or "\t" in line:
            parts = [p.strip().replace(",", ".") for p in re.split(r"[;\t]", line) if p.strip()]
        else:
            parts = [p for p in re.split(r"[,\s]+", line) if p]
        nums = [_as_number(p) for p in parts[:2]]
        if len(nums) == 2 and None not in nums:
            rows.append(nums)
    return np.array(rows, float)


def read_single_spectrum(buffer, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Read one spectrum file with two columns: wavenumber, absorbance (or transmittance)."""
    if name.lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(buffer, header=None)
        num = raw.map(_as_number).dropna(how="any", subset=raw.columns[:2])
        arr = num.iloc[:, :2].to_numpy(float)
    else:
        data = buffer.read() if hasattr(buffer, "read") else Path(buffer).read_bytes()
        text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
        arr = _two_column_lines(text)
    if len(arr) < 10:
        raise ValueError(f"{name}: expected two columns (wavenumber, absorbance) with at least 10 rows.")
    if not (_looks_like_wavenumber(arr[0, 0]) and _looks_like_wavenumber(arr[-1, 0])):
        raise ValueError(f"{name}: the first column does not look like wavenumbers (200-15000 cm-1).")
    return arr[:, 0], arr[:, 1]


def read_spectrum_files(files) -> SpectralDataset:
    """Combine many single-spectrum files into one dataset.

    `files` is a list of (name, file_or_path). The sample id is the file name without
    its extension, so name the files after your samples (e.g. B12-24h.csv).
    If the files have slightly different wavenumber points, all spectra are
    interpolated onto the points of the first file, within the range they all share.
    """
    if not files:
        raise ValueError("No files selected.")
    spectra = [(Path(n).stem, *read_single_spectrum(f, n)) for n, f in files]
    _, wn0, _ = spectra[0]
    lo = max(min(w.min(), w.max()) for _, w, _ in spectra)
    hi = min(max(w.min(), w.max()) for _, w, _ in spectra)
    grid = np.sort(wn0[(wn0 >= lo) & (wn0 <= hi)])    # ascending, whatever order the files use
    X = []
    for _, w, a in spectra:
        o = np.argsort(w)
        X.append(np.interp(grid, w[o], a[o]))
    ids = np.array([s[0] for s in spectra])
    if len(set(ids)) != len(ids):
        raise ValueError("Two files have the same name; sample ids must be unique.")
    return SpectralDataset(ids, grid, np.array(X), pd.DataFrame(index=range(len(ids))),
                           pd.DataFrame(index=range(len(ids))), f"{len(ids)} spectrum files")


def to_excel(ds: SpectralDataset, path):
    """Write a dataset back out in the rows = samples layout (round-trips with read_spectra)."""
    df = pd.DataFrame(ds.X, columns=[f"{w:g}" for w in ds.wn])
    front = pd.DataFrame({"sample_id": ds.ids})
    if len(ds.meta.columns):
        front = pd.concat([front, ds.meta.reset_index(drop=True)], axis=1)
    pd.concat([front, ds.Y.reset_index(drop=True), df], axis=1).to_excel(path, index=False)


def load_any(path: str | Path) -> SpectralDataset:
    """Convenience for scripts: one file that holds spectra (+ optional reference values)."""
    return read_spectra(path, name=str(path))
