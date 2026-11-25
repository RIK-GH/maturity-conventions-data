# -*- coding: utf-8 -*-
"""
a02_mismatch_consistent.py — cross-convention cost + global/within-mix fits,
using EXACTLY the archived Stage 1 conventions so the whole manuscript rests
on one group definition and one fitting rule.

Reused verbatim from s1_holdout_archived.py (imported, not copied):
    binder_class()   binder normalisation to OPC/SF/FA/Slag/LowHeat
    fit()            exponential curve, same bounds and feasibility rules
    load_groups()    eligibility: isothermal/sealed/water/air-sealed rows,
                     wb stated, group = source x wb x binder_class,
                     >=2 temperatures, >=3 ages, >=8 rows

Conventions (maturity computed from stated temperature and age):
    NS0     TTF datum 0 C        M = (T-0)  * t_hr     [C·hr]
    TTF10   TTF datum -10 C      M = (T+10) * t_hr     [C·hr]
    EA335   equivalent age, Ea = 33.5 kJ/mol, Tref 20 C [hr]

Cross-cost, per group: fit the curve with convention A; re-evaluate the SAME
observations with convention-B maturity fed into A's curve.
    cost(B into A) = RMSE(B|A curve) - RMSE(A|A curve)
This is the penalty for reusing a published curve while mis-reading which
convention its maturity axis uses. In-sample by construction; no holdout.

Rows whose maturity would be non-positive under a convention are excluded by
fit()'s own feasibility rule (it raises), mirroring the archived behaviour.

Outputs (01_analysis/out/):
    fits_global.csv, fits_withinmix.csv, mismatch_costs.csv,
    mismatch_summary.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "s1", HERE / "s1_holdout_archived.py")
s1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s1)

R = 8.314
EA = 33500.0
TREF_K = 293.15


def m_conv(g, conv):
    T = g.curing_temp_C.to_numpy(float)
    t = g.age_hr.to_numpy(float)
    if conv == "NS0":
        return (T - 0.0) * t
    if conv == "TTF10":
        return (T + 10.0) * t
    if conv == "EA335":
        return np.exp(-EA / R * (1.0 / (T + 273.15) - 1.0 / TREF_K)) * t
    raise ValueError(conv)


CONVS = ["NS0", "TTF10", "EA335"]

groups = s1.load_groups()
print(f"eligible groups: {len(groups)}")

# ------------------------------------------------------------- global fits
big = pd.concat([g for _, g in groups])
glob_rows = []
for c in CONVS:
    try:
        p, r2 = s1.fit(m_conv(big, c),
                       big.compressive_strength_MPa.to_numpy(float))
        pred = s1.expo(m_conv(big, c), *p)
        rm = s1.rmse(big.compressive_strength_MPa.to_numpy(float), pred)
        glob_rows.append({"convention": c, "n": len(big),
                          "R2": round(float(r2), 3), "RMSE_MPa": round(rm, 2)})
    except (ValueError, RuntimeError):
        glob_rows.append({"convention": c, "n": len(big),
                          "R2": None, "RMSE_MPa": None})
pd.DataFrame(glob_rows).to_csv(OUT / "fits_global.csv", index=False)

# --------------------------------------------- within-mix fits + cross costs
within, costs = [], []
for (source, wb, binder), g in groups:
    S = g.compressive_strength_MPa.to_numpy(float)
    fits = {}
    for c in CONVS:
        try:
            p, r2 = s1.fit(m_conv(g, c), S)
            pred = s1.expo(m_conv(g, c), *p)
            fits[c] = (p, float(r2), s1.rmse(S, pred))
        except (ValueError, RuntimeError):
            fits[c] = None
    if fits["TTF10"]:
        within.append({"source_no": source, "wb": wb, "binder": binder,
                       "n": len(g), "R2": round(fits["TTF10"][1], 3),
                       "RMSE_MPa": round(fits["TTF10"][2], 3)})
    row = {"source_no": source, "wb": wb, "binder": binder, "n": len(g)}
    got = False
    for a in CONVS:
        if not fits[a]:
            continue
        pa, _, rmse_aa = fits[a]
        for b in CONVS:
            if b == a or not fits[b]:
                continue
            Mb = m_conv(g, b)
            if np.any(Mb <= 0):
                continue
            rmse_ba = s1.rmse(S, s1.expo(Mb, *pa))
            row[f"{b}_into_{a}"] = round(rmse_ba - rmse_aa, 3)
            got = True
    if got:
        costs.append(row)

within_df = pd.DataFrame(within)
cost_df = pd.DataFrame(costs)
within_df.to_csv(OUT / "fits_withinmix.csv", index=False)
cost_df.to_csv(OUT / "mismatch_costs.csv", index=False)

pair_cols = [c for c in cost_df.columns if "_into_" in c]
summary = {
    "eligible_groups": len(groups),
    "withinmix_groups_fit": int(len(within_df)),
    "withinmix_R2_median": float(within_df["R2"].median().round(3)),
    "withinmix_R2_q1": float(within_df["R2"].quantile(.25).round(3)),
    "withinmix_R2_q3": float(within_df["R2"].quantile(.75).round(3)),
    "global_fits": glob_rows,
    "mismatch_groups": int(len(cost_df)),
    "mismatch_median_MPa": {c: float(cost_df[c].median().round(2))
                            for c in pair_cols},
    "mismatch_q3_MPa": {c: float(cost_df[c].quantile(.75).round(2))
                        for c in pair_cols},
    "mismatch_max_MPa": {c: float(cost_df[c].max().round(2))
                         for c in pair_cols},
}
(OUT / "mismatch_summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
print(json.dumps(summary, indent=2))
