# -*- coding: utf-8 -*-
"""
a08_material_validity.py — material validity of the corpus under correctly
identified conventions (new Section 4 of the manuscript, 2026-08-13).

The deductive chain of the paper needs an explicit first step: when the
maturity axis IS identified, the corpus behaves like coherent concrete data,
so there is real calibration value for mis-identification to destroy.
Two material-science checks on the 73 within-mixture TTF10 fits:

    1. Abrams-type water-binder dependence: fitted ultimate strength S_u
       should fall as W/B rises (Spearman across groups);
    2. binder kinetics: the fitted time-scale parameter tau should be larger
       (slower) for slag and fly-ash groups than for ordinary Portland.

Outputs
    out/validity_summary.json, out/validity_groups.csv
    02_manuscript/generated_validity_macros.tex
    03_figures/fig6_validity.{png,pdf}
"""
import importlib.util
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DST = HERE.parent / "02_manuscript"
FIG = HERE.parent / "03_figures"

spec = importlib.util.spec_from_file_location("s1", HERE / "s1_holdout_archived.py")
s1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s1)

rows = []
for (source, wb, binder), g in s1.load_groups():
    try:
        S = g.compressive_strength_MPa.to_numpy(float)
        M = (g.curing_temp_C.to_numpy(float) + 10.0) * g.age_hr.to_numpy(float)
        (su, tau, beta), r2 = s1.fit(M, S)
        rows.append({"source_no": source, "wb": wb, "binder": binder,
                     "Su_MPa": su, "tau_Ch": tau, "beta": beta,
                     "R2": float(r2), "n": len(g)})
    except (ValueError, RuntimeError):
        pass

v = pd.DataFrame(rows)
v.to_csv(OUT / "validity_groups.csv", index=False)

rho_ab, p_ab = spearmanr(v.wb, v.Su_MPa)
tau_med = v.groupby("binder").tau_Ch.median()
n_bind = v.binder.value_counts()

summary = {
    "n_fits": int(len(v)),
    "spearman_Su_wb": round(float(rho_ab), 2),
    "p_Su_wb": float(p_ab),
    "tau_median_Ch": {k: round(float(x), 0) for k, x in tau_med.items()},
    "binder_counts": {k: int(x) for k, x in n_bind.items()},
}
(OUT / "validity_summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
print(json.dumps(summary, indent=2))


def fmt_p(p):
    return "$p<0.001$" if p < 0.001 else f"$p={p:.3f}$"


def comma(x):
    return f"{x:,.0f}"


M = {
    "ValNFits": str(summary["n_fits"]),
    "ValSpearSuWb": f"{summary['spearman_Su_wb']:.2f}",
    "ValSpearSuWbP": fmt_p(summary["p_Su_wb"]),
    "ValTauOPC": comma(summary["tau_median_Ch"].get("OPC", float("nan"))),
    "ValTauSlag": comma(summary["tau_median_Ch"].get("Slag", float("nan"))),
    "ValTauFA": comma(summary["tau_median_Ch"].get("FA", float("nan"))),
    "ValNOPC": str(summary["binder_counts"].get("OPC", 0)),
    "ValNSlag": str(summary["binder_counts"].get("Slag", 0)),
    "ValNFA": str(summary["binder_counts"].get("FA", 0)),
}
def tex_num(v):
    """macro value for LaTeX: a minus sign, not a hyphen, before a negative number"""
    return re.sub(r"(?<![-\d])-(?=\d)", r"\\ensuremath{-}", str(v))


(DST / "generated_validity_macros.tex").write_text(
    "\n".join(f"\\newcommand{{\\{k}}}{{{tex_num(vv)}}}" for k, vv in M.items()) + "\n",
    "utf-8")
print(f"wrote {len(M)} macros -> generated_validity_macros.tex")

# ------------------------------------------------------------------- figure
BLUE, ORANGE, INK, MUTED = "#2563EB", "#EA580C", "#1F2937", "#6B7280"
plt.rcParams.update({
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.edgecolor": "black", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#E5E7EB", "grid.linewidth": 0.6,
    "figure.dpi": 300, "savefig.dpi": 300,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})
W = 160 / 25.4


def panel_label(ax, text):
    """Panel title centred below the panel, under its x-axis label."""
    fig = ax.figure
    fig.canvas.draw()
    below = (ax.bbox.y0 - ax.get_tightbbox(fig.canvas.get_renderer()).y0) * 72 / fig.dpi
    ax.annotate(text, xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -(below + 6)),
                textcoords="offset points", ha="center", va="top",
                fontsize=plt.rcParams["axes.titlesize"])


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W, 6.2))

# (a) Abrams-type trend
ax1.scatter(v.wb, v.Su_MPa, s=26, color=BLUE, alpha=.75,
            edgecolor="white", linewidth=.5)
ax1.set_xlabel("Water--binder ratio (%)".replace("--", "–"))
ax1.set_ylabel("Fitted ultimate strength $S_u$ (MPa)")
panel_label(ax1, "(a) Fitted $S_u$ falls with W/B (Abrams-type trend)")
ax1.text(.98, .92, f"Spearman $r_s$ = {summary['spearman_Su_wb']:.2f}"
         f" ($n$ = {len(v)})",
         transform=ax1.transAxes, ha="right", fontsize=8.5, color=MUTED)

# (b) binder kinetics
order = [b for b in ["OPC", "Slag", "FA", "SF", "LowHeat"]
         if b in v.binder.unique()]
for i, b in enumerate(order):
    y = v.loc[v.binder == b, "tau_Ch"]
    ax2.scatter(np.full(len(y), i) + np.random.default_rng(1).uniform(
        -.12, .12, len(y)), y, s=24, color=BLUE, alpha=.6,
        edgecolor="white", linewidth=.5)
    ax2.plot([i - .22, i + .22], [y.median()] * 2, color=ORANGE, lw=2.4)
ax2.set_xticks(range(len(order)))
ax2.set_xticklabels([f"{b}\n(n={int(n_bind[b])})" for b in order])
ax2.set_yscale("log")
ax2.set_ylabel(r"Fitted time-scale $\tau$ ($^{\circ}$C$\cdot$h)")
panel_label(ax2, "(b) Slower fitted kinetics for slag and fly-ash binders"
                 " (medians in orange)")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig6_validity.{ext}", bbox_inches="tight")
plt.close(fig)
print("wrote fig6_validity")
