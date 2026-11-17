"""Which model predicts each analyte in the dashboard?

The dashboard starts with its own built-in PLS models. As soon as you save a model
in the Model Workbench and make a version *active*, the dashboard uses that version
instead for the matching analyte, so the in-line trajectories, alerts, yield forecast
and calibration health all run on the model you maintain.

A saved model is used when
* its name or property matches an analyte (e.g. "Ethanol", "ethanol (% w/v)", "DP4+"), and
* it covers the dashboard's wavenumber range (a NIR model cannot read mid-IR spectra).
Otherwise the built-in model is kept and the reason is reported.
"""
from __future__ import annotations

import re

import numpy as np

from . import config as C
from .workbench import registry


def _norm(text: str) -> str:
    text = re.sub(r"\(.*?\)", "", str(text))            # drop units / remarks in brackets
    return re.sub(r"[^a-z0-9+]", "", text.lower())


def match_analyte(*names) -> str | None:
    """Analyte key for a model name or property, e.g. 'Ethanol (% w/v)' -> 'ethanol'."""
    aliases = {}
    for key, label in C.LABELS.items():
        for alias in (key, label, label.split(" (")[0], re.sub(r".*\((.*)\)", r"\1", label)):
            aliases[_norm(alias)] = key
    aliases.update({_norm("lactic"): "lactic", _norm("acetic"): "acetic", _norm("dp4"): "dp4plus"})
    for n in names:
        if n and _norm(n) in aliases:
            return aliases[_norm(n)]
    return None


def covers(model_wn, wn) -> bool:
    """True if the dashboard spectra span every wavenumber the model needs."""
    step = float(np.median(np.abs(np.diff(wn)))) if len(wn) > 1 else 0.0
    return wn.min() <= model_wn.min() + step / 2 and wn.max() >= model_wn.max() - step / 2


def active_saved_models(wn, root=None) -> tuple[dict, list]:
    """{analyte: ModelPackage} for usable active versions, and a list of (model, reason) skipped."""
    root = registry.DEFAULT_ROOT if root is None else root
    used, skipped = {}, []
    models = registry.list_models(root)
    if models.empty:
        return used, skipped
    # most recently changed first, so the newest model wins if two map to the same analyte
    for name in models.sort_values("last_change", ascending=False).model:
        try:
            pkg = registry.load(name, root=root)
        except Exception as e:                            # unreadable file: keep built-in
            skipped.append((name, f"could not be loaded ({e})"))
            continue
        key = match_analyte(name, pkg.property)
        if key is None:
            skipped.append((name, "name does not match any dashboard analyte"))
        elif not covers(pkg.model.wn, wn):
            skipped.append((name, f"needs {pkg.model.wn.min():.0f}-{pkg.model.wn.max():.0f} cm-1, "
                                  f"dashboard spectra are {wn.min():.0f}-{wn.max():.0f} cm-1"))
        elif key in used:
            skipped.append((name, f"{C.LABELS[key]} already uses {used[key].name}"))
        else:
            used[key] = pkg
    return used, skipped


def describe(pkg) -> str:
    return f"{pkg.name} {pkg.version} ({pkg.model.spec.describe().split(',')[0]})"


def trained_on_batches(pkg) -> set[int]:
    """Plant batches a saved model has seen (from sample ids such as 'B72-24.0h')."""
    out = set()
    for sid in pkg.train_ids:
        m = re.match(r"B(\d+)-", str(sid))
        if m:
            out.add(int(m.group(1)))
    return out
