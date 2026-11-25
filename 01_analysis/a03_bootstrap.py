# -*- coding: utf-8 -*-
"""
a03_bootstrap.py — hierarchical bootstrap over the 251 paired held-temperature
tests produced by s1_holdout_archived.py.

Same design as the Technical Note's archived Node pipeline: 10,000 replicates,
sources resampled with replacement first, mixture groups resampled with
replacement within each sampled source, all rows of a sampled group kept
together. Estimands: row-weighted median, group-equal mean, source-equal mean.

The point estimates are deterministic and must equal the TN values
(row median 0.000, group-equal mean -0.447, source-equal mean -0.615 MPa).
The interval endpoints may differ in the second decimal from the TN because
the RNG differs (numpy PCG64 here, mulberry32 there); both are valid
percentile intervals of the same design. The TN reference intervals are
printed alongside for comparison.

Output: out/holdout_bootstrap.csv, out/holdout_summary.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SEED = 20260718
B = 10000
EPS = 1e-12

h = pd.read_csv(OUT / "nested_leave_temperature_out_paired.csv")
h["group_key"] = (h.source_no.astype(str) + "|" + h.wb.astype(str)
                  + "|" + h.binder)
delta = h["fixed_minus10_minus_nested_MPa"].to_numpy()

grp_deltas = {k: v["fixed_minus10_minus_nested_MPa"].to_numpy()
              for k, v in h.groupby("group_key")}
src_groups = {s: sorted(v["group_key"].unique())
              for s, v in h.groupby("source_no")}
sources = sorted(src_groups)

rng = np.random.default_rng(SEED)
boot = {"row_median": [], "group_equal_mean": [], "source_equal_mean": []}
for _ in range(B):
    rows_all, gmeans, smeans = [], [], []
    for s in rng.choice(sources, size=len(sources), replace=True):
        gs = src_groups[s]
        this = []
        for gk in rng.choice(gs, size=len(gs), replace=True):
            dd = grp_deltas[gk]
            rows_all.append(dd)
            m = dd.mean()
            gmeans.append(m)
            this.append(m)
        smeans.append(np.mean(this))
    boot["row_median"].append(np.median(np.concatenate(rows_all)))
    boot["group_equal_mean"].append(np.mean(gmeans))
    boot["source_equal_mean"].append(np.mean(smeans))

gm = h.groupby(["source_no", "group_key"])["fixed_minus10_minus_nested_MPa"].mean()
actual = {
    "row_median": float(np.median(delta)),
    "group_equal_mean": float(gm.mean()),
    "source_equal_mean": float(gm.groupby(level=0).mean().mean()),
}
TN_REF = {  # from the TN's generated_results_macros.tex
    "group_equal_mean": (-0.447, -1.213, 0.055),
    "source_equal_mean": (-0.615, -1.629, 0.106),
}

rows = []
for k in actual:
    lo, hi = np.quantile(boot[k], [.025, .975])
    rows.append({"estimand": k, "estimate_MPa": round(actual[k], 3),
                 "ci95_low": round(float(lo), 3),
                 "ci95_high": round(float(hi), 3),
                 "tn_reference": TN_REF.get(k)})
btab = pd.DataFrame(rows)
btab.to_csv(OUT / "holdout_bootstrap.csv", index=False)

direc = {"nested_better": int((delta > EPS).sum()),
         "tie": int((np.abs(delta) <= EPS).sum()),
         "fixed_better": int((delta < -EPS).sum())}
summary = {
    "tests": int(len(h)), "groups": int(h.group_key.nunique()),
    "sources": int(h.source_no.nunique()), "directions": direc,
    "delta_median_MPa": round(float(np.median(delta)), 3),
    "delta_q1_MPa": round(float(np.quantile(delta, .25)), 3),
    "delta_q3_MPa": round(float(np.quantile(delta, .75)), 3),
    "bootstrap": rows, "seed": SEED, "reps": B,
}
(OUT / "holdout_summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
print(json.dumps(summary, indent=2))
