"""Model registry: every saved model is a numbered, documented version.

Folder layout (one folder per model, e.g. one per property):

    model_registry/
      Ethanol/
        registry.json       which version is active + list of versions
        v001.pkl            the full model package (model + its training data)
        v001.json           human-readable "model card": settings, statistics, data hash
        v002.pkl / .json    an updated version (parent = v001)
        checks.csv          history of automatic validation checks

Why keep the training data inside the package? Updating a model means refitting
it with the old samples PLUS the new ones, using exactly the same recipe; and an
auditor can always see what a model was built from.

Two ways to update a model:
* "bias"    - keep the model, subtract the average (in-line - lab) difference of the
              new samples. Cheap, fixes a constant offset (e.g. after cleaning a probe).
* "augment" - refit with old + new samples (same preprocessing and model type,
              number of LVs re-chosen by cross-validation). Fixes new variation.
A candidate version is only recommended if it is better on held-out new samples
and not worse on the original validation set.
"""
from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .models import FittedModel, ModelSpec, fit_model, regression_stats
from .preprocess import Pipeline
from .split import by_order

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "model_registry"


@dataclass
class ModelPackage:
    name: str
    model: FittedModel
    train_X: np.ndarray
    train_y: np.ndarray
    train_ids: np.ndarray
    train_groups: np.ndarray | None = None
    val_X: np.ndarray | None = None
    val_y: np.ndarray | None = None
    val_ids: np.ndarray | None = None
    bias_correction: float = 0.0
    version: str = ""
    parent: str = ""
    created: str = ""
    notes: str = ""
    excluded: list = field(default_factory=list)       # [{"sample_id":..., "reason":...}]
    data_fingerprint: str = ""

    @property
    def property(self):
        return self.model.property

    def predict(self, X_raw, wn=None):
        return self.model.predict(X_raw, wn) - self.bias_correction

    def val_stats(self):
        if self.val_X is None or not len(self.val_y):
            return None
        return regression_stats(self.val_y, self.predict(self.val_X))

    def card(self) -> dict:
        """Everything a person needs to judge this version, as plain data."""
        m = self.model
        return {
            "name": self.name, "version": self.version, "parent": self.parent,
            "created": self.created, "property": self.property, "notes": self.notes,
            "model": m.spec.describe(), "model_settings": m.spec.settings(),
            "preprocessing": m.pipeline.describe(), "preprocessing_settings": m.pipeline.settings(),
            "components_used": m.k, "bias_correction": self.bias_correction,
            "n_calibration": int(len(self.train_y)),
            "n_validation": int(0 if self.val_y is None else len(self.val_y)),
            "calibration": m.cal_stats(), "cross_validation": m.cv_stats(),
            "validation": self.val_stats(),
            "wavenumber_range": [float(m.wn.min()), float(m.wn.max())], "n_points": int(len(m.wn)),
            "excluded_samples": self.excluded, "data_fingerprint": self.data_fingerprint,
        }


# ---------------------------------------------------------------------------
# saving / loading
# ---------------------------------------------------------------------------
def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "model"


def _folder(name, root=DEFAULT_ROOT) -> Path:
    return Path(root) / _safe(name)


def _index(name, root=DEFAULT_ROOT) -> dict:
    p = _folder(name, root) / "registry.json"
    return json.loads(p.read_text()) if p.exists() else {"active": None, "versions": []}


def _write_index(name, idx, root=DEFAULT_ROOT):
    (_folder(name, root) / "registry.json").write_text(json.dumps(idx, indent=2))


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save(pkg: ModelPackage, root=DEFAULT_ROOT, activate=None) -> str:
    """Store a package as the next version. The first version becomes active automatically."""
    folder = _folder(pkg.name, root)
    folder.mkdir(parents=True, exist_ok=True)
    idx = _index(pkg.name, root)
    pkg.version = f"v{len(idx['versions']) + 1:03d}"
    pkg.created = datetime.now().isoformat(timespec="seconds")
    with open(folder / f"{pkg.version}.pkl", "wb") as f:
        pickle.dump(pkg, f)
    card = pkg.card()
    (folder / f"{pkg.version}.json").write_text(json.dumps(card, indent=2, default=_json_default))
    idx["versions"].append({"version": pkg.version, "created": pkg.created, "parent": pkg.parent,
                            "notes": pkg.notes, "RMSECV": card["cross_validation"]["RMSE"],
                            "RMSEP": (card["validation"] or {}).get("RMSE")})
    if activate or (activate is None and idx["active"] is None):
        idx["active"] = pkg.version
    _write_index(pkg.name, idx, root)
    return pkg.version


def load(name, version=None, root=DEFAULT_ROOT) -> ModelPackage:
    idx = _index(name, root)
    version = version or idx["active"]
    if version is None:
        raise FileNotFoundError(f"No saved versions for model '{name}'.")
    with open(_folder(name, root) / f"{version}.pkl", "rb") as f:
        return pickle.load(f)


def set_active(name, version, root=DEFAULT_ROOT):
    idx = _index(name, root)
    if version not in [v["version"] for v in idx["versions"]]:
        raise ValueError(f"{name} has no version {version}")
    idx["active"] = version
    _write_index(name, idx, root)


def list_models(root=DEFAULT_ROOT) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(root).glob("*/registry.json")):
        idx = json.loads(p.read_text())
        act = next((v for v in idx["versions"] if v["version"] == idx["active"]), {})
        rows.append({"model": p.parent.name, "active": idx["active"],
                     "versions": len(idx["versions"]), "RMSECV": act.get("RMSECV"),
                     "RMSEP": act.get("RMSEP"), "last_change": idx["versions"][-1]["created"]})
    return pd.DataFrame(rows)


def list_versions(name, root=DEFAULT_ROOT) -> pd.DataFrame:
    idx = _index(name, root)
    df = pd.DataFrame(idx["versions"])
    if len(df):
        df["active"] = df.version == idx["active"]
    return df


def history_path(name, root=DEFAULT_ROOT) -> Path:
    return _folder(name, root) / "checks.csv"


# ---------------------------------------------------------------------------
# building and updating
# ---------------------------------------------------------------------------
def build_package(name, spec: ModelSpec, pipeline: Pipeline, X, wn, y, ids, groups=None,
                  val_mask=None, excluded=None, notes="", fingerprint="") -> ModelPackage:
    """Fit a model from scratch on the calibration part and package it with its data."""
    X, y, ids = np.asarray(X, float), np.asarray(y, float), np.asarray(ids)
    val = np.zeros(len(y), bool) if val_mask is None else np.asarray(val_mask)
    has_y = ~np.isnan(y)
    cal, valm = ~val & has_y, val & has_y
    g = None if groups is None else np.asarray(groups)
    model = fit_model(spec, pipeline, X[cal], wn, y[cal], name,
                      None if g is None else g[cal], ids[cal])
    return ModelPackage(name, model, X[cal], y[cal], ids[cal], None if g is None else g[cal],
                        X[valm], y[valm], ids[valm], notes=notes, excluded=list(excluded or []),
                        data_fingerprint=fingerprint)


def update_candidates(pkg: ModelPackage, X_new, y_new, ids_new, groups_new=None,
                      holdout_frac=0.3) -> dict:
    """Build 'bias' and 'augment' candidates from new lab-checked samples and compare them.

    The newest `holdout_frac` of the new samples is kept aside to judge the candidates
    fairly (the candidates never see it).
    """
    X_new, y_new = np.asarray(X_new, float), np.asarray(y_new, float)
    ok = ~np.isnan(y_new)
    X_new, y_new, ids_new = X_new[ok], y_new[ok], np.asarray(ids_new)[ok]
    g_new = None if groups_new is None else np.asarray(groups_new)[ok]
    hold = by_order(np.arange(len(y_new)), holdout_frac) if len(y_new) >= 6 else np.zeros(len(y_new), bool)
    use = ~hold

    # 1) bias update: shift by the mean residual of the new (non-held-out) samples
    bias = float(np.mean(pkg.predict(X_new[use]) - y_new[use]))
    cand_bias = _clone(pkg, bias_correction=pkg.bias_correction + bias,
                       notes=f"bias update {bias:+.4g} from {use.sum()} new samples")

    # 2) augment: refit with old + new samples using the same recipe
    spec = ModelSpec.from_settings(pkg.model.spec.settings())
    spec.n_components = None                         # let CV re-choose the size
    groups = None
    if pkg.train_groups is not None and g_new is not None:
        groups = np.r_[pkg.train_groups, g_new[use]]
    pipe = Pipeline.from_settings(pkg.model.pipeline.settings())
    X_aug = np.vstack([pkg.train_X, X_new[use]])
    y_aug = np.r_[pkg.train_y, y_new[use]]
    ids_aug = np.r_[pkg.train_ids, ids_new[use]]
    model = fit_model(spec, pipe, X_aug, pkg.model.wn, y_aug, pkg.name, groups, ids_aug)
    cand_aug = _clone(pkg, model=model, train_X=X_aug, train_y=y_aug, train_ids=ids_aug,
                      train_groups=groups, bias_correction=0.0,
                      notes=f"augmented with {use.sum()} new samples")

    rows = []
    for label, p in [("current", pkg), ("bias update", cand_bias), ("augment (refit)", cand_aug)]:
        rows.append({"candidate": label,
                     "RMSE new (held-out)": regression_stats(y_new[hold], p.predict(X_new[hold]))["RMSE"]
                     if hold.any() else np.nan,
                     "RMSE new (all)": regression_stats(y_new, p.predict(X_new))["RMSE"],
                     "RMSEP original validation": (p.val_stats() or {}).get("RMSE", np.nan),
                     "RMSECV": p.model.cv_stats()["RMSE"]})
    table = pd.DataFrame(rows)
    return {"table": table, "bias": cand_bias, "augment": cand_aug,
            "recommended": _recommend(table)}


def _recommend(table: pd.DataFrame) -> str:
    """Pick the candidate that is best on new held-out data without hurting the old validation."""
    key = "RMSE new (held-out)" if table["RMSE new (held-out)"].notna().all() else "RMSE new (all)"
    cur = table.iloc[0]
    best, best_err = "current", cur[key]
    for _, r in table.iloc[1:].iterrows():
        old_ok = (np.isnan(r["RMSEP original validation"]) or np.isnan(cur["RMSEP original validation"])
                  or r["RMSEP original validation"] <= 1.10 * cur["RMSEP original validation"])
        if r[key] < 0.9 * best_err and old_ok:
            best, best_err = r["candidate"], r[key]
    return best


def _clone(pkg: ModelPackage, **changes) -> ModelPackage:
    d = dict(pkg.__dict__)
    d.update(changes)
    d.setdefault("excluded", list(pkg.excluded))
    new = ModelPackage(**d)
    new.parent, new.version, new.created = pkg.version, "", ""
    return new
