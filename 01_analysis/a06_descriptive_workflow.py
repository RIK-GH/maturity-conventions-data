# -*- coding: utf-8 -*-
"""
a06_descriptive_workflow.py — two presentation assets (2026-08-13), generated
from the data like everything else:

    02_manuscript/generated_descriptive_table.tex
        rows of the corpus descriptive-statistics table (Sect. 2.3);
        the manuscript supplies the table environment, caption and header.
    03_figures/fig0_workflow.{png,pdf}
        methodology flow diagram (corpus -> screen -> two experiments ->
        reporting set), with every count read from a01/a02/a03 outputs.

Node counts come from out/{conv,mismatch,holdout}_summary.json; the
descriptive rows are computed from maturity_corpus_v3.csv with the SAME
harmonisation and eligibility as the archived Stage 1 code (s1.load_groups).
Section numbers printed inside the diagram track the current manuscript
layout (3 profile / 4 cost / 5 transfer / 6 reporting set).
"""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DST = HERE.parent / "02_manuscript"
FIG = HERE.parent / "03_figures"

spec = importlib.util.spec_from_file_location("s1", HERE / "s1_holdout_archived.py")
s1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s1)

conv = json.loads((OUT / "conv_summary.json").read_text("utf-8"))
mm = json.loads((OUT / "mismatch_summary.json").read_text("utf-8"))
hold = json.loads((OUT / "holdout_summary.json").read_text("utf-8"))
val = json.loads((OUT / "validity_summary.json").read_text("utf-8"))  # a08 first

# ------------------------------------------------------------ descriptive rows
corpus = pd.read_csv(HERE.parent / "00_dataset" / "maturity_corpus_v3.csv",
                     encoding="utf-8-sig")
for c in ["curing_temp_C", "age_value", "compressive_strength_MPa", "wb_ratio_pct"]:
    corpus[c] = pd.to_numeric(corpus[c], errors="coerce")
corpus["age_day"] = np.where(corpus.age_unit.astype(str).str.strip().eq("hr"),
                             corpus.age_value / 24, corpus.age_value)

groups = s1.load_groups()
elig = pd.concat([g for _, g in groups])
elig["age_day"] = elig["age_hr"] / 24.0

VARS = [
    ("Curing temperature ($^{\\circ}$C)", "curing_temp_C", 1),
    ("Age (days)", "age_day", 2),
    ("Water--binder ratio (\\%)", "wb_ratio_pct", 1),
    ("Compressive strength (MPa)", "compressive_strength_MPa", 1),
]


def num(v, nd):
    s = f"{v:.{nd}f}"
    return "$-$" + s[1:] if s.startswith("-") else s


def stat_cells(s, nd):
    s = s.dropna()
    return (f"{len(s):,}",
            f"{num(s.min(), nd)}--{num(s.max(), nd)}",
            f"{num(s.mean(), nd)} $\\pm$ {num(s.std(), nd)}")


lines = []
for label, col, nd in VARS:
    full = stat_cells(corpus[col], nd)
    el = stat_cells(elig[col], nd)
    lines.append(f"{label} & {full[0]} & {full[1]} & {full[2]} & "
                 f"{el[0]} & {el[1]} & {el[2]} \\\\")
body = "\n".join(lines)
(DST / "generated_descriptive_table.tex").write_text(
    "\\newcommand{\\DescriptiveRows}{%\n" + body + "\n}\n", "utf-8")
print(f"wrote {len(lines)} table rows -> generated_descriptive_table.tex")

# ---------------------------------------------------------------- flow diagram
BLUE, ORANGE, TEAL = "#2563EB", "#EA580C", "#0D9488"
INK = "#1F2937"
TINT = {BLUE: "#EFF4FE", ORANGE: "#FDF2EB", TEAL: "#EBF7F6"}

plt.rcParams.update({"font.size": 9.5, "figure.dpi": 300, "savefig.dpi": 300,
                     "text.color": INK})

W = 160 / 25.4
fig, ax = plt.subplots(figsize=(W, 7.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, color, title, body):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.6,rounding_size=1.4",
                                linewidth=1.4, edgecolor=color,
                                facecolor=TINT[color], mutation_aspect=0.6))
    ax.text(x, y + h / 2 - 2.0, title, ha="center", va="top",
            fontsize=11, fontweight="bold", color=INK)
    ax.text(x, y + h / 2 - 6.8, body, ha="center", va="top",
            fontsize=10, color=INK, linespacing=1.4)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#6B7280",
                                linewidth=1.2, mutation_scale=14))


n_rows = conv["corpus_rows"]
n_src = conv["corpus_sources"]
n_grp = mm["eligible_groups"]
n_fit = next(g["n"] for g in mm["global_fits"] if g["convention"] == "TTF10")
n_tests, n_hgrp, n_hsrc = hold["tests"], hold["groups"], hold["sources"]
reps = hold["reps"]

# top row: corpus (data, blue) branching to the reporting profile (teal)
box(26, 92, 47, 12.5, BLUE, "Assembled corpus",
    f"{n_rows:,} strength observations\nfrom {n_src} source studies")
box(76, 92, 46, 12.5, TEAL, "Reporting profile (Sect. 3)",
    "convention, parameters, units\nas published, never converted")
arrow(50, 92, 52.5, 92)

# eligibility screen (data, blue)
box(27, 70.5, 54, 17, BLUE, "Eligibility screen",
    "isothermal-type curing, W/B stated,\n"
    "$\\geq$2 temperatures, $\\geq$3 ages, $\\geq$8 rows\n"
    f"$\\rightarrow$ {n_grp} mixture groups, {n_fit:,} rows")
arrow(26, 85.4, 26, 79)

# material validity (teal, Sect. 4) — the value at stake
box(51, 50.5, 90, 14, TEAL, "Material validity (Sect. 4)",
    f"within-mixture $R^2$ = {mm['withinmix_R2_median']:.2f},"
    f" Abrams $S_u$–W/B trend ($r_s$ = {val['spearman_Su_wb']:.2f}),\n"
    "slower kinetics for slag and fly-ash binders")
arrow(26, 61.7, 36, 58)

# two experiments (teal)
box(25.5, 28.5, 49, 19.5, TEAL, "Experiment 1 (Sect. 5)",
    "convention-mismatch cost:\ncurve fitted under A, evaluated\n"
    f"with maturity under B\n(6 ordered pairs, {n_grp} groups)")
box(76, 28.5, 47, 19.5, TEAL, "Experiment 2 (Sect. 6)",
    "held-temperature datum tuning:\n"
    f"{n_tests} paired tests,\n{n_hgrp} groups, {n_hsrc} sources,\n"
    f"bootstrap {reps:,} replicates")
arrow(33, 44, 27, 38.8)
arrow(69, 44, 75, 38.8)

# outcome (orange)
box(51, 6.5, 84, 12.5, ORANGE, "Five-item minimum reporting set (Sect. 7)",
    "convention and formula, parameter values, units,\n"
    "raw temperature–age–strength records, mixture identifier")
arrow(26, 18.3, 42, 13.2)
arrow(76, 18.3, 60, 13.2)

for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig0_workflow.{ext}", bbox_inches="tight")
plt.close(fig)
print("wrote fig0_workflow")
