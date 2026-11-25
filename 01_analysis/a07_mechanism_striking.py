# -*- coding: utf-8 -*-
"""
a07_mechanism_striking.py — engineering-depth additions to Section 4
(2026-08-13): the structural origin of the mismatch cost and its translation
into formwork-striking time.

Analytic core (isothermal records): every convention scales time by a
temperature-dependent coefficient c(T):
    c_TTF(T)  = T - T0                       [°C]
    c_EA(T)   = exp[-Ea/R (1/(T+273.15) - 1/293.15)]   [-]
A curve calibrated on axis A but read with axis-B maturity is evaluated at
arguments scaled by rho(T) = c_B(T)/c_A(T); at a fixed curing temperature the
apparent time to reach any strength criterion is scaled by EXACTLY
t_mis/t_true = c_A(T)/c_B(T), independent of the fitted curve.

Empirical checks computed here (predictors chosen by exploration on
2026-08-13; the axis-ratio SPREAD hypothesis was tested first and is null,
rho=-0.06/-0.02 — Eq. (4) applies no recalibration, so the measured cost is
governed by the magnitude of rho, not its within-group variation; the spread
columns are kept in the CSV for the record):
  1. cross-family cost vs the group's strength scale S_max
     (analytic prediction: order-one RELATIVE error -> cost ~ strength);
  2. within-TTF cost / S_max vs the group's minimum curing temperature
     (analytic prediction: rho(T)=(T+10)/T diverges as T falls);
  3. the distribution of the time to reach 10 MPa at 20 °C on each group's
     TTF10 curve (the anchor that converts exact clock ratios into hours).

Outputs
    out/mechanism_summary.json, out/mechanism_groups.csv
    02_manuscript/generated_mechanism_macros.tex
    03_figures/fig5_mechanism.{png,pdf}
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

R_GAS, EA, TREF_K = 8.314, 33500.0, 293.15


def c_ttf(T, t0):
    return T - t0


def c_ea(T):
    return np.exp(-EA / R_GAS * (1.0 / (T + 273.15) - 1.0 / TREF_K))


groups = s1.load_groups()
cost = pd.read_csv(OUT / "mismatch_costs.csv")

WITHIN = ["TTF10_into_NS0", "NS0_into_TTF10"]
CROSS = ["NS0_into_EA335", "EA335_into_NS0", "TTF10_into_EA335", "EA335_into_TTF10"]

rows = []
for (source, wb, binder), g in groups:
    temps = np.sort(g.curing_temp_C.unique().astype(float))
    # axis-ratio spread over the group's temperatures
    r_w = (temps + 10.0) / np.where(temps == 0.0, np.nan, temps)  # NS0 vs TTF10
    r_c = c_ea(temps) / (temps + 10.0)                            # EA vs TTF10
    spread_w = np.nanmax(r_w) / np.nanmin(r_w)
    spread_c = np.nanmax(r_c) / np.nanmin(r_c)
    # time to reach 10 MPa at 20 °C on the TTF10 curve
    t10 = np.nan
    try:
        S = g.compressive_strength_MPa.to_numpy(float)
        M = (g.curing_temp_C.to_numpy(float) + 10.0) * g.age_hr.to_numpy(float)
        (su, tau, beta), _ = s1.fit(M, S)
        if su > 10.0:
            t10 = tau / (30.0 * np.log(su / 10.0) ** (1.0 / beta))  # hours
    except (ValueError, RuntimeError):
        pass
    rows.append({"source_no": source, "wb": wb, "binder": binder,
                 "T_min": temps.min(), "T_max": temps.max(),
                 "S_max": float(g.compressive_strength_MPa.max()),
                 "spread_within": spread_w, "spread_cross": spread_c,
                 "t10_h": t10})

mech = pd.DataFrame(rows)
mech = mech.merge(cost, on=["source_no", "wb", "binder"], how="left")
mech["cost_within"] = mech[[c for c in WITHIN if c in mech]].mean(axis=1)
mech["cost_cross"] = mech[[c for c in CROSS if c in mech]].mean(axis=1)
mech.to_csv(OUT / "mechanism_groups.csv", index=False)

w = mech.dropna(subset=["cost_within"]).copy()
x = mech.dropna(subset=["cost_cross"]).copy()
w["rel_cost_within"] = w.cost_within / w.S_max
rho_scale, p_scale = spearmanr(x.S_max, x.cost_cross)
rho_tmin, p_tmin = spearmanr(w.T_min, w.rel_cost_within)
t10 = mech.t10_h.dropna()

summary = {
    "n_within": int(len(w)),
    "n_cross": int(len(x)),
    "spearman_cross_vs_Smax": round(float(rho_scale), 2),
    "p_cross_vs_Smax": float(p_scale),
    "spearman_within_rel_vs_Tmin": round(float(rho_tmin), 2),
    "p_within_rel_vs_Tmin": float(p_tmin),
    "t10_n": int(len(t10)),
    "t10_median_h": round(float(t10.median()), 1),
    "t10_q1_h": round(float(t10.quantile(.25)), 1),
    "t10_q3_h": round(float(t10.quantile(.75)), 1),
    # exact clock ratios t_mis/t_true = c_A/c_B (curve-independent)
    "ratio_20C_NS0curve_TTFdata": round(20.0 / 30.0, 2),
    "ratio_20C_TTFcurve_NS0data": round(30.0 / 20.0, 2),
    "ratio_5C_NS0curve_TTFdata": round(5.0 / 15.0, 2),
    "ratio_20C_TTFcurve_EAdata": round(30.0 / float(c_ea(np.array([20.0]))[0]), 1),
    "ratio_20C_EAcurve_TTFdata": round(float(c_ea(np.array([20.0]))[0]) / 30.0, 3),
}
(OUT / "mechanism_summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
print(json.dumps(summary, indent=2))


def fmt_p(p):
    return "$p<0.001$" if p < 0.001 else f"$p={p:.3f}$"


M = {
    "MechSpearCrossScale": f"{summary['spearman_cross_vs_Smax']:.2f}",
    "MechSpearCrossScaleP": fmt_p(summary["p_cross_vs_Smax"]),
    "MechSpearWithinTmin": f"{summary['spearman_within_rel_vs_Tmin']:.2f}",
    "MechSpearWithinTminP": fmt_p(summary["p_within_rel_vs_Tmin"]),
    "MechNWithin": str(summary["n_within"]),
    "MechNCross": str(summary["n_cross"]),
    "TTenN": str(summary["t10_n"]),
    "TTenMedian": f"{summary['t10_median_h']:.0f}",
    "TTenQOne": f"{summary['t10_q1_h']:.0f}",
    "TTenQThree": f"{summary['t10_q3_h']:.0f}",
    "TTenEarly": f"{summary['t10_median_h'] * (1 - 20.0 / 30.0):.0f}",
    "TTenLate": f"{summary['t10_median_h'] * 0.5:.0f}",
}
def tex_num(v):
    """macro value for LaTeX: a minus sign, not a hyphen, before a negative number"""
    return re.sub(r"(?<![-\d])-(?=\d)", r"\\ensuremath{-}", str(v))


(DST / "generated_mechanism_macros.tex").write_text(
    "\n".join(f"\\newcommand{{\\{k}}}{{{tex_num(v)}}}" for k, v in M.items()) + "\n",
    "utf-8")
print(f"wrote {len(M)} macros -> generated_mechanism_macros.tex")

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


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(W, 9.0))

# (a) cross-family cost tracks the strength scale
ax1.scatter(x.S_max, x.cost_cross, s=26, color=ORANGE, alpha=.75,
            edgecolor="white", linewidth=.5)
ax1.set_xlabel("Maximum strength of the mixture group (MPa)")
ax1.set_ylabel("Mean cross-family\nΔRMSE (MPa)")
panel_label(ax1, "(a) Across families, the error scales with the strength")
ax1.text(.02, .92,
         f"Spearman $r_s$ = {summary['spearman_cross_vs_Smax']:.2f}"
         f" ($n$ = {len(x)})",
         transform=ax1.transAxes, ha="left", fontsize=8.5, color=MUTED)

# (b) within-family relative cost concentrates at low curing temperature
# (T_min takes only a few discrete values -> horizontal jitter for legibility)
jit = np.random.default_rng(2).uniform(-.35, .35, len(w))
ax2.scatter(w.T_min + jit, 100 * w.rel_cost_within, s=26, color=BLUE,
            alpha=.75, edgecolor="white", linewidth=.5)
ax2.set_xlabel(r"Minimum curing temperature of the group ($^{\circ}$C)")
ax2.set_ylabel("Datum-swap ΔRMSE /\nmaximum strength (%)")
panel_label(ax2, "(b) Within the TTF family, the relative error concentrates\n"
                 "in low-temperature curing")
ax2.text(.98, .92,
         f"Spearman $r_s$ = {summary['spearman_within_rel_vs_Tmin']:.2f}"
         f" ($n$ = {len(w)})",
         transform=ax2.transAxes, ha="right", fontsize=8.5, color=MUTED)

# (c) exact clock ratio vs curing temperature
# t_mis/t* = c_A/c_B = 1/rho with A = curve convention, B = data convention.
# Legend uses the "B data -> A curve" order of Table 3 and Eq. (4)
# (labels were swapped against Eq. (5) before 2026-08-19; fixed; relabelled
# to the B -> A notation in 2026-11, curves unchanged)
T = np.linspace(5, 40, 200)
ax3.plot(T, (T + 10) / T, color=BLUE, lw=2,
         label="datum-0 data → TTF $-$10 curve")
ax3.plot(T, T / (T + 10), color=BLUE, lw=2, ls="--",
         label="TTF $-$10 data → datum-0 curve")
ax3.plot(T, (T + 10) / c_ea(T), color=ORANGE, lw=2,
         label="equivalent-age data → TTF $-$10 curve")
ax3.plot(T, c_ea(T) / (T + 10), color=ORANGE, lw=2, ls="--",
         label="TTF $-$10 data → equivalent-age curve")
ax3.axhline(1.0, color=MUTED, lw=1, ls=":")
ax3.set_yscale("log")
ax3.set_xlabel(r"Curing temperature ($^{\circ}$C)")
ax3.set_ylabel("Apparent / true time\n" r"to criterion, $t_{\rm mis}/t^{*}$")
panel_label(ax3, "(c) Exact distortion of the time to a strength criterion"
                 " (curve-independent)")
ax3.legend(frameon=False, fontsize=8, loc="upper center",
           bbox_to_anchor=(0.55, 0.92))
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig5_mechanism.{ext}", bbox_inches="tight")
plt.close(fig)
print("wrote fig5_mechanism")
