# -*- coding: utf-8 -*-
"""
a09_r1_robustness.py — robustness analyses (2026-08-19), one block each:

  - source-level reporting statistics (unit-of-analysis check)
  - curve-form sensitivity of the mismatch cost (hyperbolic and
    logarithmic strength relations) and a scale-harmonized
    cross-family variant that separates numeric-scale mismatch
    from shape mismatch
  - hierarchical (source -> group) bootstrap CIs for the six
    Table-3 medians and the two Spearman correlations of Sect. 5.1
  - holdout results stratified by number of training temperatures;
    grid-boundary hits of the datum search
  - mismatch medians restricted to high-confidence (table-
    transcribed) rows
  - share of holdout tests contributed by the three largest sources
  - audit of rows with W/B > 100%

Outputs
    out/r1_robustness.json
    02_manuscript/generated_r1_macros.tex
"""
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DST = HERE.parent / "02_manuscript"

spec = importlib.util.spec_from_file_location("s1", HERE / "s1_holdout_archived.py")
s1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s1)

R_GAS, EA, TREF_K = 8.314, 33500.0, 293.15
RNG = np.random.default_rng(20260819)
REPS = 10000

PAIRS = {  # column in mismatch_costs.csv -> manuscript macro stem
    "TTF10_into_NS0": "TTFtoNS", "NS0_into_TTF10": "NStoTTF",
    "NS0_into_EA335": "NStoEA", "EA335_into_NS0": "EAtoNS",
    "TTF10_into_EA335": "TTFtoEA", "EA335_into_TTF10": "EAtoTTF",
}
WITHIN = ["TTF10_into_NS0", "NS0_into_TTF10"]
CROSS = ["NS0_into_EA335", "EA335_into_NS0", "TTF10_into_EA335", "EA335_into_TTF10"]
CONVS = ["NS0", "TTF10", "EA335"]


def m_conv(g, conv):
    T = g.curing_temp_C.to_numpy(float)
    t = g.age_hr.to_numpy(float)
    if conv == "NS0":
        return T * t
    if conv == "TTF10":
        return (T + 10.0) * t
    if conv == "EA335":
        return np.exp(-EA / R_GAS * (1.0 / (T + 273.15) - 1.0 / TREF_K)) * t
    raise ValueError(conv)


def hyp(m, su, tau):
    return su * m / (tau + m)


def fit_hyp(m, s):
    if np.any(m <= 0) or len(m) < 4 or len(np.unique(s)) < 3:
        raise ValueError("invalid fit")
    p, _ = curve_fit(hyp, m, s, p0=[max(s) * 1.1, np.median(m)],
                     bounds=([max(s) * .9, 1e-9], [max(s) * 3, 1e15]),
                     maxfev=50000)
    return p


def fit_log(m, s):
    if np.any(m <= 0) or len(m) < 4 or len(np.unique(s)) < 3:
        raise ValueError("invalid fit")
    b, a = np.polyfit(np.log(m), s, 1)
    return a, b


def log_pred(m, a, b):
    return a + b * np.log(m)


def rmse(y, yh):
    return float(np.sqrt(np.mean((y - yh) ** 2)))


def mismatch_costs(groups, form):
    """Per-group cost table for one curve form ('expo'|'hyp'|'log')."""
    rows = []
    for (source, wb, binder), g in groups:
        S = g.compressive_strength_MPa.to_numpy(float)
        fits = {}
        for c in CONVS:
            M = m_conv(g, c)
            try:
                if form == "expo":
                    p, _ = s1.fit(M, S)
                    fits[c] = (p, rmse(S, s1.expo(M, *p)))
                elif form == "hyp":
                    p = fit_hyp(M, S)
                    fits[c] = (p, rmse(S, hyp(M, *p)))
                else:
                    p = fit_log(M, S)
                    fits[c] = (p, rmse(S, log_pred(M, *p)))
            except (ValueError, RuntimeError):
                fits[c] = None
        row = {"source_no": source}
        got = False
        for a in CONVS:
            if not fits[a]:
                continue
            pa, rmse_aa = fits[a]
            for b in CONVS:
                if b == a or not fits[b]:
                    continue
                Mb = m_conv(g, b)
                if np.any(Mb <= 0):
                    continue
                pred = (s1.expo(Mb, *pa) if form == "expo"
                        else hyp(Mb, *pa) if form == "hyp"
                        else log_pred(Mb, *pa))
                row[f"{b}_into_{a}"] = rmse(S, pred) - rmse_aa
                got = True
        if got:
            rows.append(row)
    return pd.DataFrame(rows)


def med_range(df, cols):
    meds = [df[c].median() for c in cols if c in df]
    return round(float(min(meds)), 1), round(float(max(meds)), 1)


groups = s1.load_groups()
summary = {}

# ---------------------------------------------------- source level
conv_src = pd.read_csv(OUT / "conv_by_source.csv")
summary["source_level"] = {
    "n_sources": int(len(conv_src)),
    "src_with_unreported_rows": int((conv_src.pct_unreported > 0).sum()),
    "src_all_rows_unreported": int((conv_src.pct_unreported == 100).sum()),
    "src_datum_incomplete": int((conv_src.datum_stated_pct < 100).sum()),
    "src_datum_none": int((conv_src.datum_stated_pct == 0).sum()),
    "source_equal_unreported_pct": round(float(conv_src.pct_unreported.mean()), 1),
    "source_equal_datum_missing_pct":
        round(float((100 - conv_src.datum_stated_pct).mean()), 1),
}

# ------------------------------------------- curve-form sensitivity
forms = {}
for form in ("expo", "hyp", "log"):
    cdf = mismatch_costs(groups, form)
    w_lo, w_hi = med_range(cdf, WITHIN)
    c_lo, c_hi = med_range(cdf, CROSS)
    forms[form] = {"groups": int(len(cdf)),
                   "within_median_range": [w_lo, w_hi],
                   "cross_median_range": [c_lo, c_hi]}
summary["curve_form_sensitivity"] = forms

# scale-harmonized cross-family variant (exponential form): rescale the
# mis-read axis so its median matches the curve axis -> shape-only mismatch
rows = []
for (source, wb, binder), g in groups:
    S = g.compressive_strength_MPa.to_numpy(float)
    row = {}
    for a in CONVS:
        Ma = m_conv(g, a)
        try:
            pa, _ = s1.fit(Ma, S)
        except (ValueError, RuntimeError):
            continue
        rmse_aa = rmse(S, s1.expo(Ma, *pa))
        for b in CONVS:
            if b == a or (a, b) not in [("NS0", "EA335"), ("EA335", "NS0"),
                                        ("TTF10", "EA335"), ("EA335", "TTF10")]:
                continue
            Mb = m_conv(g, b)
            if np.any(Mb <= 0):
                continue
            Mb_s = Mb * np.median(Ma) / np.median(Mb)
            row[f"{b}_into_{a}"] = rmse(S, s1.expo(Mb_s, *pa)) - rmse_aa
    if row:
        rows.append(row)
scal = pd.DataFrame(rows)
s_lo, s_hi = med_range(scal, [c for c in scal.columns])
summary["scale_harmonized_cross"] = {"groups": int(len(scal)),
                                     "median_range": [s_lo, s_hi]}

# --------------------------------- hierarchical bootstrap for Sect.5
cost = pd.read_csv(OUT / "mismatch_costs.csv")
mech = pd.read_csv(OUT / "mechanism_groups.csv")
mech["rel_within"] = mech.cost_within / mech.S_max

def hier_resample(df):
    srcs = df.source_no.unique()
    pick = RNG.choice(srcs, size=len(srcs), replace=True)
    parts = []
    for s_ in pick:
        gsub = df[df.source_no == s_]
        idx = RNG.choice(gsub.index, size=len(gsub), replace=True)
        parts.append(df.loc[idx])
    return pd.concat(parts)

med_boot = {c: [] for c in PAIRS}
rho_scale_boot, rho_tmin_boot = [], []
for _ in range(REPS):
    b1 = hier_resample(cost)
    for c in PAIRS:
        med_boot[c].append(b1[c].median())
    b2 = hier_resample(mech)
    x2 = b2.dropna(subset=["cost_cross"])
    w2 = b2.dropna(subset=["rel_within"])
    if x2.S_max.nunique() > 2:
        rho_scale_boot.append(spearmanr(x2.S_max, x2.cost_cross)[0])
    if w2.T_min.nunique() > 1:
        rho_tmin_boot.append(spearmanr(w2.T_min, w2.rel_within)[0])

def ci(v, nd=1):
    lo, hi = np.nanpercentile(v, [2.5, 97.5])
    return [round(float(lo), nd), round(float(hi), nd)]

summary["hier_bootstrap"] = {
    "reps": REPS, "seed": 20260819,
    "median_ci_MPa": {PAIRS[c]: ci(med_boot[c]) for c in PAIRS},
    "spearman_cross_vs_Smax_ci": ci(rho_scale_boot, 2),
    "spearman_within_rel_vs_Tmin_ci": ci(rho_tmin_boot, 2),
}

# -------------------------- holdout stratification, bound hits
nest = pd.read_csv(OUT / "nested_leave_temperature_out.csv")
paired = pd.read_csv(OUT / "nested_leave_temperature_out_paired.csv")
# number of curing temperatures per group taken from the DATA, not from the
# count of successful held-out tests (some folds fail to fit and are absent)
tmap = {(src, wb, bind): g.curing_temp_C.nunique()
        for (src, wb, bind), g in groups}
paired["n_temps"] = [tmap[(r.source_no, r.wb, r.binder)]
                     for r in paired.itertuples()]
strata = {}
for label, sel in [("3_temps", paired.n_temps == 3),
                   ("4plus_temps", paired.n_temps >= 4)]:
    d = paired.loc[sel, "fixed_minus10_minus_nested_MPa"]
    strata[label] = {
        "tests": int(sel.sum()),
        "median_delta_MPa": round(float(d.median()), 3),
        "nested_better": int((d > 1e-12).sum()),
        "tie": int((d.abs() <= 1e-12).sum()),
        "fixed_better": int((d < -1e-12).sum()),
    }
# grid-boundary hits among the 251 PAIRED tests only
sel_nest = nest[nest.model == "nested_mixture_optimum"].merge(
    paired[["source_no", "wb", "binder", "held_temp_C"]],
    on=["source_no", "wb", "binder", "held_temp_C"])
hits = ((sel_nest.t0_selected <= -25.0) | (sel_nest.t0_selected >= 15.0)).sum()
top3 = paired.source_no.value_counts().head(3)
summary["holdout_strata"] = strata
summary["holdout_bound_hits"] = {
    "tests": int(len(sel_nest)), "grid_bound_hits": int(hits),
    "pct": round(100.0 * hits / len(sel_nest), 1),
}
summary["holdout_top3_sources"] = {
    "sources": [int(s_) for s_ in top3.index],
    "tests": int(top3.sum()),
    "pct_of_all": round(100.0 * top3.sum() / len(paired), 1),
}

# ----------------------------- high-confidence-only mismatch medians
d = pd.read_csv(HERE.parent / "00_dataset" / "maturity_corpus_v3.csv",
                encoding="utf-8-sig")
for c in ["curing_temp_C", "age_value", "compressive_strength_MPa", "wb_ratio_pct"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
d["age_day"] = np.where(d.age_unit.astype(str).str.strip().eq("hr"),
                        d.age_value / 24, d.age_value)
d["age_hr"] = d.age_day * 24
d = d[d.curing_condition.astype(str).str.contains(
    "isothermal|sealed|water|air-sealed", case=False, na=False)]
d = d.dropna(subset=["curing_temp_C", "age_hr", "compressive_strength_MPa",
                     "wb_ratio_pct"])
d = d[d.confidence.astype(str).str.strip().str.lower().eq("high")].copy()
d["binder"] = d.binder_type.map(s1.binder_class)
hi_groups = []
for key, g in d.groupby(["source_no", "wb_ratio_pct", "binder"]):
    if g.curing_temp_C.nunique() >= 2 and g.age_hr.nunique() >= 3 and len(g) >= 8:
        hi_groups.append((key, g.copy()))
hi_cost = mismatch_costs(hi_groups, "expo")
hw_lo, hw_hi = med_range(hi_cost, WITHIN)
hc_lo, hc_hi = med_range(hi_cost, CROSS)
summary["high_confidence_only"] = {
    "groups": int(len(hi_cost)),
    "within_median_range": [hw_lo, hw_hi],
    "cross_median_range": [hc_lo, hc_hi],
}

# ---------------------------------------------------------- R2 additions ----
# activation-energy sensitivity of the cross-family medians.
# EA is the module-level constant read by m_conv at call time.
ea_extremes = []
for ea_val in (30000.0, 40000.0, 45000.0):
    EA = ea_val
    cdf_ea = mismatch_costs(groups, "expo")
    lo, hi = med_range(cdf_ea, CROSS)
    ea_extremes += [lo, hi]
EA = 33500.0
summary["ea_sensitivity"] = {
    "grid_kJ_mol": [30, 40, 45],
    "cross_median_range": [round(min(ea_extremes), 1),
                           round(max(ea_extremes), 1)],
}

# sign of the mismatch cost
pair_cols = list(PAIRS)
vals = cost[pair_cols].to_numpy().ravel()
vals = vals[~np.isnan(vals)]
summary["cost_sign"] = {
    "min_MPa": round(float(vals.min()), 2),
    "n_negative": int((vals < 0).sum()),
    "n_total": int(len(vals)),
}

# unreported rows split by source type (series-specific vs source-wide)
conv_src["unrep_rows"] = (conv_src.n_rows * conv_src.pct_unreported
                          / 100).round().astype(int)
summary["unreported_split"] = {
    "rows_from_allunrep_sources":
        int(conv_src.loc[conv_src.pct_unreported == 100, "unrep_rows"].sum()),
    "rows_from_mixed_sources":
        int(conv_src.loc[(conv_src.pct_unreported > 0)
                         & (conv_src.pct_unreported < 100),
                         "unrep_rows"].sum()),
}

# ----------------------------------------------- W/B > 100 % audit
full = pd.read_csv(HERE.parent / "00_dataset" / "maturity_corpus_v3.csv",
                   encoding="utf-8-sig")
full["wb_ratio_pct"] = pd.to_numeric(full.wb_ratio_pct, errors="coerce")
over = full[full.wb_ratio_pct > 100]
summary["wb_over_100"] = {
    "rows": int(len(over)),
    "sources": sorted(int(s_) for s_ in over.source_no.unique()),
    "wb_values": sorted(float(v) for v in over.wb_ratio_pct.unique()),
    "in_eligible_subset": bool(any(
        wb > 100 for (_, wb, _), _ in groups)),
}

(OUT / "r1_robustness.json").write_text(json.dumps(summary, indent=2), "utf-8")
print(json.dumps(summary, indent=2))

# ------------------------------------------------------------------- macros
sl = summary["source_level"]
fh, fl = forms["hyp"], forms["log"]
hb = summary["hier_bootstrap"]
st, bh = summary["holdout_strata"], summary["holdout_bound_hits"]
t3, hc = summary["holdout_top3_sources"], summary["high_confidence_only"]

def rng_(pair):
    lo, hi = (round(x, 1) + 0.0 for x in pair)   # "+ 0.0" turns a rounded -0.0 into 0.0
    return f"{lo:.1f}" if lo == hi else f"{lo:.1f}--{hi:.1f}"

M = {
    "RSrcUnrepSome": sl["src_with_unreported_rows"],
    "RSrcUnrepAll": sl["src_all_rows_unreported"],
    "RSrcDatumInc": sl["src_datum_incomplete"],
    "RSrcDatumZero": sl["src_datum_none"],
    "RSrcEqUnrepPct": sl["source_equal_unreported_pct"],
    "RSrcEqDatumMissPct": sl["source_equal_datum_missing_pct"],
    "RHypWithinMed": rng_(fh["within_median_range"]),
    "RHypCrossMed": rng_(fh["cross_median_range"]),
    "RLogWithinMed": rng_(fl["within_median_range"]),
    "RLogCrossMed": rng_(fl["cross_median_range"]),
    "RScaleCrossMed": rng_(summary["scale_harmonized_cross"]["median_range"]),
    "RSpearCrossCiLow": hb["spearman_cross_vs_Smax_ci"][0],
    "RSpearCrossCiHigh": hb["spearman_cross_vs_Smax_ci"][1],
    "RSpearTminCiLow": hb["spearman_within_rel_vs_Tmin_ci"][0],
    "RSpearTminCiHigh": hb["spearman_within_rel_vs_Tmin_ci"][1],
    "RHoldThreeN": st["3_temps"]["tests"],
    "RHoldThreeMed": f"{st['3_temps']['median_delta_MPa']:.3f}",
    "RHoldThreeWin": st["3_temps"]["nested_better"],
    "RHoldThreeTie": st["3_temps"]["tie"],
    "RHoldThreeLoss": st["3_temps"]["fixed_better"],
    "RHoldFourN": st["4plus_temps"]["tests"],
    "RHoldFourMed": f"{st['4plus_temps']['median_delta_MPa']:.3f}",
    "RHoldFourWin": st["4plus_temps"]["nested_better"],
    "RHoldFourTie": st["4plus_temps"]["tie"],
    "RHoldFourLoss": st["4plus_temps"]["fixed_better"],
    "RBoundHits": bh["grid_bound_hits"],
    "RBoundHitPct": bh["pct"],
    "RTopThreeTests": t3["tests"],
    "RTopThreePct": t3["pct_of_all"],
    "RHighConfGroups": hc["groups"],
    "RHighConfWithinMed": rng_(hc["within_median_range"]),
    "RHighConfCrossMed": rng_(hc["cross_median_range"]),
    "RWbOverVals": (f"{min(summary['wb_over_100']['wb_values']):.1f}--"
                    f"{max(summary['wb_over_100']['wb_values']):.1f}\\%"),
    "REaCrossMed": rng_(summary["ea_sensitivity"]["cross_median_range"]),
    "RCostMinAll": f"{summary['cost_sign']['min_MPa']:.1f}",
    "RCostNegN": summary["cost_sign"]["n_negative"],
    "RCostAllN": summary["cost_sign"]["n_total"],
    "RUnrepRowsPureSrc": summary["unreported_split"]["rows_from_allunrep_sources"],
    "RUnrepRowsMixedSrc": summary["unreported_split"]["rows_from_mixed_sources"],
}
for c, stem in PAIRS.items():
    lo, hi = hb["median_ci_MPa"][stem]
    M[f"RCi{stem}"] = f"{lo:.1f} to {hi:.1f}"

def tex_num(v):
    """macro value for LaTeX: a minus sign, not a hyphen, before a negative number"""
    return re.sub(r"(?<![-\d])-(?=\d)", r"\\ensuremath{-}", str(v))


(DST / "generated_r1_macros.tex").write_text(
    "\n".join(f"\\newcommand{{\\{k}}}{{{tex_num(v)}}}" for k, v in M.items()) + "\n",
    "utf-8")
print(f"wrote {len(M)} macros -> generated_r1_macros.tex")
