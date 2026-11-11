"""Simplified kinetic model of a corn-mash fuel-ethanol fermentation.

The model is a simultaneous saccharification and fermentation (SSF):

    DP4+ (dextrins) --enzyme--> DP3 --> DP2 --> glucose --yeast--> ethanol + glycerol

* Enzymes (glucoamylase) break long sugars into shorter ones (first-order steps).
* Yeast activity grows logistically and takes up glucose with Monod kinetics.
* Yeast is slowed down by ethanol and by organic acids (product inhibition).
* Optional faults: bacteria that steal glucose to make lactic/acetic acid,
  or a yeast-stress event that makes the fermentation stall ("stuck").

Everything is in glucose-equivalent % w/v (hydration water ignored), so the
sugar that disappears can be traced to the products (mass balance).
This is a teaching-grade model, not a validated plant model: it only needs
to produce realistic-looking trajectories and realistic fault signatures.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class ProcessParams:
    """All numbers that make one batch different from another."""
    batch_id: int
    scenario: str
    # starting composition (% w/v)
    dp4plus_0: float
    dp3_0: float
    dp2_0: float
    glucose_0: float
    ethanol_0: float
    glycerol_0: float
    lactic_0: float
    acetic_0: float
    # enzyme hydrolysis rates (1/h) and non-fermentable residue (% w/v)
    k4: float
    k3: float
    k2: float
    dp4_residual: float
    # yeast
    vmax: float          # max glucose uptake (% w/v per h) at full yeast activity
    mu: float            # yeast growth rate (1/h)
    yeast_0: float       # starting yeast activity (0..1)
    ethanol_max: float   # ethanol level (% w/v) that fully stops the yeast
    glycerol_fraction: float = 0.05   # share of yeast glucose diverted to glycerol (yield loss)
    # faults (None = not present)
    contamination_onset_h: float | None = None
    bacteria_mu: float = 0.0
    bacteria_rate: float = 0.0
    stuck_onset_h: float | None = None
    stuck_death_rate: float = 0.0

    def as_dict(self):
        return asdict(self)


def sample_params(rng: np.random.Generator, batch_id: int, scenario: str) -> ProcessParams:
    """Draw plausible random settings for one batch of the given scenario."""
    p = ProcessParams(
        batch_id=batch_id,
        scenario=scenario,
        dp4plus_0=rng.normal(22.5, 1.0),
        dp3_0=rng.normal(1.0, 0.15),
        dp2_0=rng.normal(2.5, 0.3),
        glucose_0=rng.normal(1.2, 0.2),
        ethanol_0=rng.normal(0.35, 0.05),
        glycerol_0=rng.normal(0.12, 0.02),
        lactic_0=rng.normal(0.10, 0.02),
        acetic_0=rng.normal(0.03, 0.005),
        k4=rng.normal(0.115, 0.012),
        k3=rng.normal(0.55, 0.05),
        k2=rng.normal(0.45, 0.04),
        dp4_residual=rng.normal(0.8, 0.12),
        vmax=rng.normal(2.1, 0.30),
        mu=rng.normal(0.45, 0.06),
        yeast_0=rng.uniform(0.05, 0.11),
        ethanol_max=rng.normal(18.0, 0.5),
        glycerol_fraction=rng.normal(0.05, 0.009),
    )
    if scenario in ("contamination_lactic", "contamination_acetic"):
        p.contamination_onset_h = rng.uniform(3, 12)
        p.bacteria_mu = rng.uniform(0.55, 0.75)
        # acetic bacteria make much less acid, but it is far more toxic to yeast
        p.bacteria_rate = rng.uniform(0.045, 0.075) if scenario == "contamination_lactic" \
            else rng.uniform(0.015, 0.025)
    if scenario == "stuck":
        p.stuck_onset_h = rng.uniform(10, 20)
        p.stuck_death_rate = rng.uniform(0.10, 0.22)
    return p


def simulate_batch(p: ProcessParams, dt: float = 0.05) -> pd.DataFrame:
    """Integrate the model with small Euler steps and return readings at C.TIMES_H.

    Returns a DataFrame with columns: time_h + every analyte in C.ANALYTES
    (these are the TRUE concentrations; instruments see them with errors).
    """
    dp4, dp3, dp2, glc = p.dp4plus_0, p.dp3_0, p.dp2_0, p.glucose_0
    eth, gly, lac, ace = p.ethanol_0, p.glycerol_0, p.lactic_0, p.acetic_0
    yeast, bact = p.yeast_0, 0.0

    record_steps = set(np.round(C.TIMES_H / dt).astype(int))
    n_steps = int(round(C.BATCH_DURATION_H / dt))
    rows = []
    for i in range(n_steps + 1):
        t = i * dt
        if i in record_steps:
            rows.append((t, glc, dp2, dp3, dp4, eth, gly, lac, ace))

        # 1) enzymes: long sugars -> short sugars -> glucose
        r4 = p.k4 * max(dp4 - p.dp4_residual, 0.0)
        r3 = p.k3 * dp3
        r2 = p.k2 * dp2

        # 2) yeast: Monod uptake, slowed by ethanol and acids
        monod = glc / (0.25 + glc)
        f_ethanol = max(0.0, 1.0 - eth / p.ethanol_max) ** 1.2
        f_acid = max(0.0, 1.0 - lac / 2.5) * max(0.0, 1.0 - ace / 0.6)
        uptake = p.vmax * yeast * monod * f_ethanol * f_acid

        death = 0.003
        if p.stuck_onset_h is not None and t >= p.stuck_onset_h:
            death += p.stuck_death_rate          # yeast stress (heat spike, nutrients...)
        d_yeast = p.mu * yeast * (1 - yeast) * monod * f_acid * f_ethanol - death * yeast

        # 3) bacteria (only after the contamination enters)
        lac_made = ace_made = 0.0
        if p.contamination_onset_h is not None and t >= p.contamination_onset_h:
            if bact == 0.0:
                bact = 2e-3
            bact_monod = glc / (0.4 + glc)
            d_bact = p.bacteria_mu * bact * (1 - bact) * max(bact_monod, 0.15)
            acid_rate = p.bacteria_rate * bact * bact_monod
            if p.scenario == "contamination_lactic":
                lac_made = acid_rate
            else:
                ace_made = acid_rate
            bact += d_bact * dt

        # baseline acids made by the yeast itself (small)
        lac_made += 0.0012 * yeast * monod
        ace_made += 0.0006 * yeast * monod

        # 4) update the state (glucose pays for everything that is produced)
        glc_used = uptake + lac_made * C.HEXOSE_COST["lactic"] + ace_made * C.HEXOSE_COST["acetic"]
        glc_used = min(glc_used, (glc + r2 * dt) / dt)      # cannot use more than exists
        uptake = max(glc_used - lac_made - 1.5 * ace_made, 0.0)

        dp4 -= r4 * dt
        dp3 += (r4 - r3) * dt
        dp2 += (r3 - r2) * dt
        glc += (r2 - glc_used) * dt
        # of the glucose taken by yeast: ~92% -> ethanol path, ~5% -> glycerol, 3% -> cells
        eth += (0.97 - p.glycerol_fraction) * uptake * C.ETHANOL_YIELD * dt
        gly += p.glycerol_fraction * uptake * dt
        lac += lac_made * dt
        ace += ace_made * dt
        yeast = min(max(yeast + d_yeast * dt, 0.0), 1.0)

    df = pd.DataFrame(rows, columns=["time_h"] + C.ANALYTES)
    df["time_h"] = C.TIMES_H[: len(df)]
    return df
