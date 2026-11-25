# -*- coding: utf-8 -*-
"""
a01_conventions.py — maturity-convention reporting profile of the v3 corpus.

Recomputes, on the 2,471-row v3 dataset, what dataset_derived/ previously
reported on the 1,475-row v1: how the assembled literature reports (or fails
to report) its maturity convention.

Outputs (01_analysis/out/):
    conv_by_row.csv        rows per maturity_type x datum, corpus-wide
    conv_by_source.csv     per-source dominant convention + completeness
    conv_summary.json      headline numbers for the manuscript
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "00_dataset" / "maturity_corpus_v3.csv"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

d = pd.read_csv(DATA)
n = len(d)

# ---------------------------------------------------------------- normalise
# maturity_type is the extractor's classification; blank = the paper did not
# state a usable convention for that row.
d["conv"] = d["maturity_type"].fillna("unreported")
d.loc[d["conv"].str.strip() == "", "conv"] = "unreported"

# rows that carry a maturity NUMBER but no stated convention are the
# dangerous case: a reader must guess to reuse them
d["has_maturity_value"] = d["maturity_value"].notna()

by_row = (d.groupby("conv")
            .agg(rows=("row_id", "size"),
                 with_value=("has_maturity_value", "sum"),
                 sources=("source_no", "nunique"))
            .sort_values("rows", ascending=False))
by_row["pct_rows"] = (by_row["rows"] / n * 100).round(1)
by_row.to_csv(OUT / "conv_by_row.csv")

# ---------------------------------------------------------- per-source view
rows = []
for sno, g in d.groupby("source_no"):
    convs = g["conv"].value_counts()
    rows.append({
        "source_no": sno,
        "region": g["region"].iloc[0],
        "n_rows": len(g),
        "dominant_conv": convs.index[0],
        "n_conventions": int((convs.index != "unreported").sum()),
        "pct_unreported": round(float((g["conv"] == "unreported").mean() * 100), 1),
        "datum_stated_pct": round(float(g["maturity_datum_C"].notna().mean() * 100), 1),
        "wb_stated_pct": round(float(g["wb_ratio_pct"].notna().mean() * 100), 1),
        "curing_temp_stated_pct": round(float(g["curing_temp_C"].notna().mean() * 100), 1),
    })
by_src = pd.DataFrame(rows).sort_values("source_no")
by_src.to_csv(OUT / "conv_by_source.csv", index=False)

# ------------------------------------------------------------- headline json
mixed = by_src[by_src["n_conventions"] >= 2]
unrep_rows = int((d["conv"] == "unreported").sum())
unrep_with_value = int(((d["conv"] == "unreported") & d["has_maturity_value"]).sum())

summary = {
    "corpus_rows": n,
    "corpus_sources": int(d["source_no"].nunique()),
    "convention_rows": {k: int(v) for k, v in by_row["rows"].items()},
    "convention_pct": {k: float(v) for k, v in by_row["pct_rows"].items()},
    "unreported_rows": unrep_rows,
    "unreported_pct": round(unrep_rows / n * 100, 1),
    "unreported_rows_carrying_maturity_value": unrep_with_value,
    "sources_using_2plus_conventions": int(len(mixed)),
    "datum_unstated_rows": int(d["maturity_datum_C"].isna().sum()),
    "datum_unstated_pct": round(float(d["maturity_datum_C"].isna().mean() * 100), 1),
}
(OUT / "conv_summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
print(json.dumps(summary, indent=2))
