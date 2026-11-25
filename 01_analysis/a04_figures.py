# -*- coding: utf-8 -*-
"""
a04_figures.py — the manuscript's three figures, generated from the outputs of
a01/a02/a03 (never from hard-coded numbers).

    fig1_reporting_profile   what the corpus states (convention + metadata)
    fig2_convention_cost     cost of reusing a curve under the wrong convention
    fig3_calibration         global vs within-mix fits + held-temperature test

Print-first: 300 dpi PNG + vector PDF, double-column width 190 mm, direct
labels, one hue per job. Palette validated with the dataviz six-checks script
(#2563EB, #EA580C, #0D9488 pass on the light surface; #6B7280 is neutral
context ink, not a categorical slot).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
FIG = HERE.parent / "03_figures"
FIG.mkdir(exist_ok=True)

BLUE, ORANGE, TEAL, GRAY = "#2563EB", "#EA580C", "#0D9488", "#6B7280"
INK, MUTED = "#1F2937", "#6B7280"

plt.rcParams.update({
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.edgecolor": "black", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#E5E7EB", "grid.linewidth": 0.6,
    "figure.dpi": 300, "savefig.dpi": 300,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})
# Panels are stacked VERTICALLY (one per row) so each gets the full figure
# width — larger marks and labels at print size.
W = 160 / 25.4  # figure width, inches


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def panel_label(ax, text):
    """Panel title centred below the panel, under its x-axis label."""
    fig = ax.figure
    fig.canvas.draw()
    below = (ax.bbox.y0 - ax.get_tightbbox(fig.canvas.get_renderer()).y0) * 72 / fig.dpi
    ax.annotate(text, xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -(below + 6)),
                textcoords="offset points", ha="center", va="top",
                fontsize=plt.rcParams["axes.titlesize"])


conv = json.loads((OUT / "conv_summary.json").read_text("utf-8"))
mm = json.loads((OUT / "mismatch_summary.json").read_text("utf-8"))
hold = json.loads((OUT / "holdout_summary.json").read_text("utf-8"))
corpus = pd.read_csv(HERE.parent / "00_dataset" / "maturity_corpus_v3.csv")
cost = pd.read_csv(OUT / "mismatch_costs.csv")
within = pd.read_csv(OUT / "fits_withinmix.csv")

# ============================================================ FIG 1 — profile
LBL = {"TTF_-10": "TTF, datum −10 °C",
       "TTF_NurseSaul": "TTF, datum 0 °C (Nurse–Saul)",
       "equivalent_age": "Equivalent age",
       "other": "Other / modified index",
       "unreported": "No maturity index reported"}
order = ["TTF_-10", "unreported", "other", "equivalent_age", "TTF_NurseSaul"]
rows = [conv["convention_rows"][k] for k in order]
pcts = [conv["convention_pct"][k] for k in order]
n = conv["corpus_rows"]

fields = [("Compressive strength", corpus.compressive_strength_MPa.notna()),
          ("Curing temperature", corpus.curing_temp_C.notna()),
          ("Age", corpus.age_value.notna()),
          ("Water–binder ratio", corpus.wb_ratio_pct.notna()),
          ("Maturity index value", corpus.maturity_value.notna()),
          ("Maturity datum / E$_a$", corpus.maturity_datum_C.notna())]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W, 5.6))

y = np.arange(len(order))[::-1]
colors = [ORANGE if k == "unreported" else BLUE for k in order]
ax1.barh(y, rows, color=colors, height=0.62)
for yi, r, p in zip(y, rows, pcts):
    ax1.text(r + 22, yi, f"{r:,}  ({p:.0f}%)", va="center", fontsize=8,
             color=INK)
ax1.set_yticks(y, [LBL[k] for k in order])
ax1.set_xlim(0, 1750)
ax1.set_xlabel(f"Rows in corpus (n = {n:,})")
panel_label(ax1, "(a) Maturity convention as published")
ax1.grid(axis="y", visible=False)

y2 = np.arange(len(fields))[::-1]
vals = [f[1].mean() * 100 for f in fields]
ax2.barh(y2, vals, color=[BLUE if v > 99 else (ORANGE if v < 80 else BLUE)
                          for v in vals], height=0.62)
for yi, v in zip(y2, vals):
    ax2.text(v - 2, yi, f"{v:.0f}%", va="center", ha="right", fontsize=8,
             color="white", fontweight="bold")
ax2.set_yticks(y2, [f[0] for f in fields])
ax2.set_xlim(0, 100)
ax2.set_xlabel("Rows stating the field (%)")
panel_label(ax2, "(b) Metadata completeness")
ax2.grid(axis="y", visible=False)

fig.tight_layout(h_pad=1.5)
save(fig, "fig1_reporting_profile")

# ======================================================= FIG 2 — cost matrix
PAIRS = [("TTF10_into_NS0", "TTF −10 → datum-0 curve"),
         ("NS0_into_TTF10", "Datum-0 → TTF −10 curve"),
         ("NS0_into_EA335", "Datum-0 → equiv.-age curve"),
         ("EA335_into_NS0", "Equiv. age → datum-0 curve"),
         ("TTF10_into_EA335", "TTF −10 → equiv.-age curve"),
         ("EA335_into_TTF10", "Equiv. age → TTF −10 curve")]
data = [cost[k].dropna().to_numpy() for k, _ in PAIRS]
within_family = {0, 1}

fig, ax = plt.subplots(figsize=(W, 4.0))
pos = np.arange(len(PAIRS))[::-1]
bp = ax.boxplot(data, positions=pos, vert=False, widths=0.55,
                showfliers=False, patch_artist=True,
                medianprops=dict(color=INK, linewidth=1.4),
                whiskerprops=dict(color=GRAY, linewidth=0.9),
                capprops=dict(color=GRAY, linewidth=0.9),
                boxprops=dict(linewidth=0.9))
for i, (patch, dd) in enumerate(zip(bp["boxes"], data)):
    c = BLUE if i in within_family else ORANGE
    patch.set_facecolor(c)
    patch.set_alpha(0.25)
    patch.set_edgecolor(c)
    rng = np.random.default_rng(7 + i)
    jitter = rng.uniform(-0.16, 0.16, len(dd))
    ax.plot(dd, pos[i] + jitter, "o", ms=2.4, mec="none",
            mfc=(BLUE if i in within_family else ORANGE), alpha=0.55)
    med = np.median(dd)
    ax.annotate(f"{med:.1f}", (med, pos[i] + 0.38), fontsize=8,
                ha="center", color=INK, fontweight="bold")
ax.axvline(0, color=GRAY, linewidth=0.8)
ax.set_ylim(-0.6, pos.max() + 0.8)  # room for the top median label inside the frame
ax.set_yticks(pos, [p[1] for p in PAIRS])
ax.set_xlabel("Added strength-prediction error, RMSE increase (MPa)\n"
              f"across {mm['mismatch_groups']} mixture groups")
ax.grid(axis="y", visible=False)
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor=BLUE, alpha=0.35, edgecolor=BLUE,
          label="Within the TTF family (datum swap)"),
    Patch(facecolor=ORANGE, alpha=0.35, edgecolor=ORANGE,
          label="Across families (TTF ↔ equivalent age)")],
    loc="upper right", frameon=False, fontsize=8)
fig.tight_layout()
save(fig, "fig2_convention_cost")

# ===================================================== FIG 3 — calibration
big_groups = None  # global scatter rebuilt from the archived loader
import importlib.util
spec = importlib.util.spec_from_file_location("s1", HERE / "s1_holdout_archived.py")
s1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s1)
groups = s1.load_groups()
big = pd.concat([g for _, g in groups])
M = (big.curing_temp_C.to_numpy(float) + 10.0) * big.age_hr.to_numpy(float)
S = big.compressive_strength_MPa.to_numpy(float)
glob = next(g for g in mm["global_fits"] if g["convention"] == "TTF10")
p_glob, _ = s1.fit(M, S)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(W, 9.6),
                                    gridspec_kw={"height_ratios": [1.35, 1, 1]})

ax1.scatter(M, S, s=4, color=GRAY, alpha=0.35, linewidths=0)
mg = np.geomspace(max(M.min(), 10), M.max(), 200)
ax1.plot(mg, s1.expo(mg, *p_glob), color=BLUE, linewidth=1.8)
ax1.set_xscale("log")
ax1.set_xlabel("Maturity, TTF datum −10 °C (°C·h)")
ax1.set_ylabel("Compressive strength (MPa)")
ax1.annotate(f"one pooled curve\nR² = {glob['R2']:.2f}, "
             f"RMSE = {glob['RMSE_MPa']:.0f} MPa\nn = {glob['n']:,}",
             xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=8,
             color=INK)
panel_label(ax1, "(a) All mixtures, one curve")

ax2.hist(within["R2"], bins=np.arange(0.5, 1.01, 0.025), color=BLUE,
         edgecolor="white", linewidth=0.5)
med = within["R2"].median()
ax2.axvline(med, color=ORANGE, linewidth=1.4)
ax2.annotate(f"median {med:.2f}", xy=(med - 0.01, ax2.get_ylim()[1] * 0.95),
             ha="right", va="top", fontsize=8, color=ORANGE,
             fontweight="bold")
ax2.set_xlabel(f"Within-group R² ({len(within)} mixture groups)")
ax2.set_ylabel("Groups")
ax2.set_xlim(0.5, 1.0)
panel_label(ax2, "(b) One curve per mixture")

est_lbl = {"row_median": "Row-weighted median",
           "group_equal_mean": "Group-equal mean",
           "source_equal_mean": "Source-equal mean"}
ypos = np.arange(len(hold["bootstrap"]))[::-1]
for yi, b in zip(ypos, hold["bootstrap"]):
    ax3.plot([b["ci95_low"], b["ci95_high"]], [yi, yi], color=BLUE,
             linewidth=1.6, solid_capstyle="butt")
    ax3.plot(b["estimate_MPa"], yi, "o", ms=6, color=BLUE)
    ax3.annotate(f"{b['estimate_MPa']:+.2f}", (b["estimate_MPa"], yi + 0.22),
                 ha="center", fontsize=8, color=INK, fontweight="bold")
ax3.axvline(0, color=GRAY, linewidth=0.9, linestyle="--")
ax3.set_yticks(ypos, [est_lbl[b["estimand"]] for b in hold["bootstrap"]])
ax3.set_xlim(-2.1, 0.7)
ax3.set_ylim(-0.7, len(ypos) - 0.3)
ax3.set_xlabel("Fixed(−10 °C) − nested RMSE (MPa)\n"
               "95% hierarchical-bootstrap interval")
d = hold["directions"]
ax3.annotate(f"{hold['tests']} tests: {d['nested_better']} nested better · "
             f"{d['tie']} tie · {d['fixed_better']} fixed better",
             xy=(0.5, -0.56), xycoords=("axes fraction", "data"),
             ha="center", fontsize=7.5, color=MUTED)
panel_label(ax3, "(c) Datum tuning at a withheld temperature")
ax3.grid(axis="y", visible=False)

fig.tight_layout(h_pad=1.5)
save(fig, "fig3_calibration")
