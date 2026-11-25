# -*- coding: utf-8 -*-
"""
a10_minor_revision.py — additional analyses (2026-11). Read-only with respect
to all earlier outputs: it reuses the archived loader and fit
(s1_holdout_archived.py) and the paired hold-out table, and writes only the
files listed below. One block each:

  - attrition from the corpus to the eligible groups; subset sizes
  - convention shares; within-group R2 under NS0 and EA335
  - aligned and misaligned RMSE (definition of the added RMSE)
  - fit bounds and failures; S_u and tau details
  - datum-grid feasibility and ties; boundary tests
  - hold-out under alternative treatments of the boundary tests
  - groups pooling more than one published series; sensitivity;
    group map (Additional file 3)
  - analytic TTF10 / EA335 ratio range

Claims that the manuscript states as facts (no fit failures, no beta bound
hits, no exact datum ties, signs and zero-crossings of intervals) are asserted
below, so a change in the data cannot silently invalidate a sentence.

Outputs
    out/r3_minor_revision.json
    out/group_map.csv                      (Additional file 3)
    02_manuscript/generated_r3_macros.tex
"""
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DST = HERE.parent / "02_manuscript"
DATA = HERE.parent / "00_dataset" / "maturity_corpus_v3.csv"

spec = importlib.util.spec_from_file_location("s1", HERE / "s1_holdout_archived.py")
s1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s1)

R_GAS, EA, TREF_K = 8.314, 33500.0, 293.15
CONVS = ["NS0", "TTF10", "EA335"]
WITHIN = ["TTF10_into_NS0", "NS0_into_TTF10"]
CROSS = ["NS0_into_EA335", "EA335_into_NS0", "TTF10_into_EA335", "EA335_into_TTF10"]
REPS = 10000
SEED_HOLD = 20260718  # same design and seed as a03_bootstrap.py
SEED_CORR = 20260819  # same seed as the correlation CIs in a09_r1_robustness.py
EPS = 1e-12           # tie tolerance used throughout the hold-out analysis
BOUND_RTOL = 1e-4     # a fitted parameter within 0.01 % of a bound is "on" it

np.seterr(all="ignore")  # nonpositive maturity at a withheld T is handled explicitly


def c_ea(T):
    T = np.asarray(T, float)
    return np.exp(-EA / R_GAS * (1.0 / (T + 273.15) - 1.0 / TREF_K))


def m_conv(g, conv):
    T = g.curing_temp_C.to_numpy(float)
    t = g.age_hr.to_numpy(float)
    if conv == "NS0":
        return T * t
    if conv == "TTF10":
        return (T + 10.0) * t
    if conv == "EA335":
        return c_ea(T) * t
    raise ValueError(conv)


def comma(x):
    return f"{int(round(x)):,}"


def f2(x):
    """Two decimals; values that round to zero print as 0.00 (no '-0.00')."""
    return "0.00" if abs(x) < 0.005 else f"{x:.2f}"


def rng_txt(lo, hi, nd=1):
    return f"{lo:.{nd}f}" if round(lo, nd) == round(hi, nd) else f"{lo:.{nd}f}--{hi:.{nd}f}"


def half_up(x):
    return int(np.floor(x + 0.5))


summary = {}

# ============================================================ attrition
raw = pd.read_csv(DATA, encoding="utf-8-sig")
d = raw.copy()
for c in ["curing_temp_C", "age_value", "compressive_strength_MPa", "wb_ratio_pct"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
d["age_day"] = np.where(d.age_unit.astype(str).str.strip().eq("hr"),
                        d.age_value / 24, d.age_value)
d["age_hr"] = d.age_day * 24
cure = d[d.curing_condition.astype(str).str.contains(
    "isothermal|sealed|water|air-sealed", case=False, na=False)]
comp = cure.dropna(subset=["curing_temp_C", "age_hr", "compressive_strength_MPa",
                           "wb_ratio_pct"]).copy()
comp["binder"] = comp.binder_type.map(s1.binder_class)
cand = list(comp.groupby(["source_no", "wb_ratio_pct", "binder"]))
fail_t = sum(1 for _, g in cand if g.curing_temp_C.nunique() < 2)
fail_a = sum(1 for _, g in cand
             if g.curing_temp_C.nunique() >= 2 and g.age_hr.nunique() < 3)
fail_n = sum(1 for _, g in cand if g.curing_temp_C.nunique() >= 2
             and g.age_hr.nunique() >= 3 and len(g) < 8)

groups = s1.load_groups()
G = {k: g for k, g in groups}
KEYS = [k for k, _ in groups]
assert len(groups) == len(cand) - fail_t - fail_a - fail_n
summary["attrition"] = {
    "corpus_rows": int(len(d)), "corpus_sources": int(d.source_no.nunique()),
    "curing_rows": int(len(cure)), "curing_sources": int(cure.source_no.nunique()),
    "complete_rows": int(len(comp)), "complete_sources": int(comp.source_no.nunique()),
    "candidate_groups": len(cand), "fail_lt2_temps": fail_t,
    "fail_lt3_ages": fail_a, "fail_lt8_rows": fail_n,
    "eligible_groups": len(groups),
    "eligible_rows": int(sum(len(g) for _, g in groups)),
    "eligible_sources": int(len({k[0] for k in KEYS})),
}

# ================================================= convention shares
conv = json.loads((OUT / "conv_summary.json").read_text("utf-8"))
cr = conv["convention_rows"]
three = cr["TTF_-10"] + cr["TTF_NurseSaul"] + cr["equivalent_age"]
summary["three_conventions"] = {
    "rows": three,
    "pct_of_all_rows": round(100.0 * three / conv["corpus_rows"], 1),
    "pct_of_rows_with_stated_index":
        round(100.0 * three / (conv["corpus_rows"] - cr["unreported"]), 1),
}

# ============================== fits under the three conventions
fits, fit_failures = {}, 0
for k, g in groups:
    S = g.compressive_strength_MPa.to_numpy(float)
    smax = S.max()
    for c in CONVS:
        M = m_conv(g, c)
        try:
            p, r2 = s1.fit(M, S)
        except (ValueError, RuntimeError):
            fit_failures += 1
            continue
        fits[(k, c)] = {
            "p": p, "r2": float(r2), "rmse": s1.rmse(S, s1.expo(M, *p)),
            "su_lo": bool(abs(p[0] / (0.9 * smax) - 1) < BOUND_RTOL),
            "su_hi": bool(abs(p[0] / (3.0 * smax) - 1) < BOUND_RTOL),
            "beta_hit": bool(abs(p[2] / 0.1 - 1) < BOUND_RTOL
                             or abs(p[2] / 6.0 - 1) < BOUND_RTOL),
        }
assert fit_failures == 0, "text states that all group-convention fits converged"
beta_hits = sum(v["beta_hit"] for v in fits.values())
assert beta_hits == 0, "text states that beta never reached a bound"

bounds = {c: {"su_lower": int(sum(fits[(k, c)]["su_lo"] for k in KEYS)),
              "su_upper": int(sum(fits[(k, c)]["su_hi"] for k in KEYS))}
          for c in CONVS}
for c in CONVS:
    bounds[c]["su_any"] = bounds[c]["su_lower"] + bounds[c]["su_upper"]
r2_med = {c: float(np.median([fits[(k, c)]["r2"] for k in KEYS])) for c in CONVS}
aligned = {c: float(np.median([fits[(k, c)]["rmse"] for k in KEYS])) for c in CONVS}

# misaligned (total) RMSE and the added RMSE per ordered pair, B into A
mis, added = {}, {}
for k, g in groups:
    S = g.compressive_strength_MPa.to_numpy(float)
    for a in CONVS:
        for b in CONVS:
            if b == a:
                continue
            Mb = m_conv(g, b)
            if np.any(Mb <= 0):
                continue
            r_mis = s1.rmse(S, s1.expo(Mb, *fits[(k, a)]["p"]))
            mis.setdefault(f"{b}_into_{a}", []).append(r_mis)
            added.setdefault(f"{b}_into_{a}", []).append(r_mis - fits[(k, a)]["rmse"])
mis_med = {p: float(np.median(v)) for p, v in mis.items()}
# consistency with the published Table 3 medians (a02 output)
mm = json.loads((OUT / "mismatch_summary.json").read_text("utf-8"))
for p, v in added.items():
    assert abs(np.median(v) - mm["mismatch_median_MPa"][p]) < 0.011, p
assert abs(r2_med["TTF10"] - mm["withinmix_R2_median"]) < 0.0011

summary["fits"] = {
    "group_convention_fits": len(fits), "fit_failures": fit_failures,
    "beta_bound_hits": int(beta_hits), "su_bound_hits": bounds,
    "withinmix_R2_median": {c: round(r2_med[c], 3) for c in CONVS},
    "aligned_rmse_median_MPa": {c: round(aligned[c], 2) for c in CONVS},
    "misaligned_rmse_median_MPa": {p: round(v, 2) for p, v in mis_med.items()},
}

# ======================================================= S_u and tau
val = pd.DataFrame([{
    "source_no": int(k[0]), "wb": float(k[1]), "binder": k[2],
    "Su": fits[(k, "TTF10")]["p"][0], "tau": fits[(k, "TTF10")]["p"][1],
    "beta": fits[(k, "TTF10")]["p"][2],
    "su_bound": fits[(k, "TTF10")]["su_lo"] or fits[(k, "TTF10")]["su_hi"]}
    for k in KEYS])
rs_all = spearmanr(val.wb, val.Su)
ok = ~val.su_bound
rs_ex = spearmanr(val.wb[ok], val.Su[ok])
assert round(float(rs_all[0]), 2) == -0.74  # published value (a08)
rng = np.random.default_rng(SEED_CORR)
by_src = {s_: sub.index.to_numpy() for s_, sub in val.groupby("source_no")}
srcs = sorted(by_src)
wb_arr, su_arr = val.wb.to_numpy(), val.Su.to_numpy()
boot = []
for _ in range(REPS):
    idx = np.concatenate([rng.choice(by_src[s_], size=len(by_src[s_]), replace=True)
                          for s_ in rng.choice(srcs, size=len(srcs), replace=True)])
    boot.append(spearmanr(wb_arr[idx], su_arr[idx])[0])
ci_su = np.nanpercentile(boot, [2.5, 97.5])
assert ci_su[1] < 0, "text reads the S_u-W/B interval as excluding zero"
tau_med = val.groupby("binder").tau.median()
n_bind = val.binder.value_counts()
summary["validity"] = {
    "n_fits": int(len(val)),
    "spearman_Su_wb": round(float(rs_all[0]), 3), "p": float(rs_all[1]),
    "su_on_bound": int(val.su_bound.sum()),
    "spearman_Su_wb_excl_bound": round(float(rs_ex[0]), 3),
    "n_excl_bound": int(ok.sum()),
    "spearman_Su_wb_hier_ci": [round(float(ci_su[0]), 2), round(float(ci_su[1]), 2)],
    "tau_median_Ch": {b: round(float(v), 0) for b, v in tau_med.items()},
    "binder_counts": {b: int(v) for b, v in n_bind.items()},
    "beta_median": round(float(val.beta.median()), 2),
    "beta_range": [round(float(val.beta.min()), 2), round(float(val.beta.max()), 2)],
}

# ======================================== datum grid: feasibility, ties
nest = pd.read_csv(OUT / "nested_leave_temperature_out.csv", encoding="utf-8-sig")
nest = nest[nest.model == "nested_mixture_optimum"]
sel_pub = {(int(r.source_no), float(r.wb), r.binder, float(r.held_temp_C)): float(r.t0_selected)
           for r in nest.itertuples()}
tests = []
for k, g in groups:
    temps = sorted(g.curing_temp_C.unique())
    if len(temps) < 3:
        continue
    for held in temps:
        tr = g[g.curing_temp_C != held]
        S_tr = tr.compressive_strength_MPa.to_numpy(float)
        r2s = {}
        for t0 in s1.T0_GRID:
            try:
                _, r2 = s1.fit(s1.m_ttf(tr, t0), S_tr)
                r2s[float(t0)] = float(r2)
            except (ValueError, RuntimeError):
                pass
        best, sel = -np.inf, None
        for t0 in s1.T0_GRID:  # the strict '>' rule of s1.select_t0
            if float(t0) in r2s and r2s[float(t0)] > best:
                best, sel = r2s[float(t0)], float(t0)
        key = (int(k[0]), float(k[1]), k[2], float(held))
        assert sel == sel_pub[key], key  # reproduces the archived selection
        tests.append({
            "source_no": key[0], "wb": key[1], "binder": key[2], "held_temp_C": key[3],
            "n_temps": len(temps), "feasible": len(r2s), "selected": sel,
            "n_exact_ties": sum(abs(v - best) <= EPS for v in r2s.values()),
            "min_train_T": float(min(t for t in temps if t != held)),
            "max_feasible_t0": max(r2s)})
tests = pd.DataFrame(tests)
assert len(tests) == len(nest)
exact_ties = int((tests.n_exact_ties > 1).sum())
assert exact_ties == 0, "text states that no exact ties occurred"
assert (tests.max_feasible_t0 < tests.min_train_T).all()
bnd = tests[tests.selected >= tests.held_temp_C]
summary["datum_grid"] = {
    "candidate_tests": int(len(tests)),
    "groups_3_temps": int(tests[tests.n_temps == 3][["source_no", "wb", "binder"]]
                          .drop_duplicates().shape[0]),
    "groups_4plus_temps": int(tests[tests.n_temps >= 4][["source_no", "wb", "binder"]]
                              .drop_duplicates().shape[0]),
    "grid_size": int(len(s1.T0_GRID)),
    "feasible_min": int(tests.feasible.min()),
    "feasible_median": float(tests.feasible.median()),
    "exact_ties": exact_ties,
    "boundary_tests": int(len(bnd)),
    "boundary_negative": int((bnd.selected > bnd.held_temp_C).sum()),
    "boundary_zero": int((bnd.selected == bnd.held_temp_C).sum()),
}

# ============================ hold-out under alternative treatments
paired = pd.read_csv(OUT / "nested_leave_temperature_out_paired.csv", encoding="utf-8-sig")
base = paired.rename(columns={"fixed_minus10_minus_nested_MPa": "d"})[
    ["source_no", "wb", "binder", "held_temp_C", "d"]].copy()


def tkey(df):
    return list(zip(df.source_no.astype(int), df.wb.astype(float),
                    df.binder.astype(str), df.held_temp_C.astype(float)))


bset = set(tkey(bnd))
assert len(base) == len(tests) - summary["datum_grid"]["boundary_negative"]

# restricted search: candidates below the withheld temperature only
alt = []
for r in bnd.itertuples():
    g = G[(r.source_no, r.wb, r.binder)]
    tr, te = g[g.curing_temp_C != r.held_temp_C], g[g.curing_temp_C == r.held_temp_C]
    S_tr = tr.compressive_strength_MPa.to_numpy(float)
    best, sel = -np.inf, None
    for t0 in s1.T0_GRID[s1.T0_GRID < r.held_temp_C]:
        try:
            _, r2 = s1.fit(s1.m_ttf(tr, t0), S_tr)
        except (ValueError, RuntimeError):
            continue
        if r2 > best:
            best, sel = r2, float(t0)
    p_opt, _ = s1.fit(s1.m_ttf(tr, sel), S_tr)
    p_fix, _ = s1.fit(s1.m_ttf(tr, -10), S_tr)
    y = te.compressive_strength_MPa.to_numpy(float)
    alt.append({"source_no": r.source_no, "wb": r.wb, "binder": r.binder,
                "held_temp_C": r.held_temp_C, "t0_restricted": sel,
                "d": s1.rmse(y, s1.expo(s1.m_ttf(te, -10), *p_fix))
                     - s1.rmse(y, s1.expo(s1.m_ttf(te, sel), *p_opt))})
alt = pd.DataFrame(alt)
excl = base[[k not in bset for k in tkey(base)]]
restr = pd.concat([excl, alt[["source_no", "wb", "binder", "held_temp_C", "d"]]],
                  ignore_index=True)


def holdout_summary(h):
    """Counts, medians and the a03 hierarchical bootstrap (same draws)."""
    h = h.copy()
    h["gk"] = h.source_no.astype(str) + "|" + h.wb.astype(str) + "|" + h.binder
    delta = h.d.to_numpy()
    grp = {k_: v.d.to_numpy() for k_, v in h.groupby("gk")}
    srcg = {s_: sorted(v.gk.unique()) for s_, v in h.groupby("source_no")}
    srcs_ = sorted(srcg)
    rng_ = np.random.default_rng(SEED_HOLD)
    bg, bs = [], []
    for _ in range(REPS):
        gm, sm = [], []
        for s_ in rng_.choice(srcs_, size=len(srcs_), replace=True):
            this = [grp[gk].mean() for gk in rng_.choice(srcg[s_], size=len(srcg[s_]),
                                                         replace=True)]
            gm += this
            sm.append(np.mean(this))
        bg.append(np.mean(gm))
        bs.append(np.mean(sm))
    gmean = h.groupby(["source_no", "gk"]).d.mean()
    return {
        "tests": int(len(h)), "groups": int(h.gk.nunique()),
        "sources": int(h.source_no.nunique()),
        "nested_better": int((delta > EPS).sum()), "tie": int((np.abs(delta) <= EPS).sum()),
        "fixed_better": int((delta < -EPS).sum()),
        "median": round(float(np.median(delta)), 3),
        "group_equal": [round(float(gmean.mean()), 3),
                        round(float(np.quantile(bg, .025)), 3),
                        round(float(np.quantile(bg, .975)), 3)],
        "source_equal": [round(float(gmean.groupby(level=0).mean().mean()), 3),
                         round(float(np.quantile(bs, .025)), 3),
                         round(float(np.quantile(bs, .975)), 3)],
    }


hs_base = holdout_summary(base)
hs_excl = holdout_summary(excl)
hs_restr = holdout_summary(restr)
hold_pub = json.loads((OUT / "holdout_summary.json").read_text("utf-8"))
pub_ci = {b["estimand"]: b for b in hold_pub["bootstrap"]}
assert hs_base["group_equal"][0] == pub_ci["group_equal_mean"]["estimate_MPa"]
for hs in (hs_base, hs_excl, hs_restr):  # "in none of the variants ... an advantage"
    assert hs["group_equal"][0] < 0 and hs["source_equal"][0] < 0
assert hs_restr["group_equal"][2] < 0  # text: that interval lies below zero
summary["holdout_variants"] = {
    "published_251": hs_base, "exclude_boundary": hs_excl,
    "restricted_search": hs_restr,
    "boundary_tests": [{**{k_: (float(v) if isinstance(v, (np.floating, float)) else
                                 (int(v) if isinstance(v, (np.integer, int)) else v))
                           for k_, v in r._asdict().items() if k_ != "Index"}}
                       for r in bnd.itertuples()],
}

# ====================================== high-confidence subset (Table 2)
hi = comp[comp.confidence.astype(str).str.strip().str.lower().eq("high")]
hi_groups = [(k, g) for k, g in hi.groupby(["source_no", "wb_ratio_pct", "binder"])
             if g.curing_temp_C.nunique() >= 2 and g.age_hr.nunique() >= 3 and len(g) >= 8]
r1 = json.loads((OUT / "r1_robustness.json").read_text("utf-8"))
assert len(hi_groups) == r1["high_confidence_only"]["groups"]
summary["high_confidence"] = {
    "groups": len(hi_groups), "rows": int(sum(len(g) for _, g in hi_groups)),
    "sources": int(len({k[0] for k, _ in hi_groups}))}

# ============================== groups pooling more than one published series
def series_id(sid, temp, age):
    """Published specimen label with the row's own curing-temperature and age
    tokens removed, leaving the mixture/series part (e.g. '3C-W45-20C-x16' ->
    '3C-W45'). Only the row's own values are stripped, so mixture codes that
    merely look like temperatures (S09 '3C' = concrete) are kept."""
    s = str(sid)
    t_txt, a_txt = re.escape(f"{temp:g}"), re.escape(f"{age:g}")
    for pat in (rf"(?<![\d.]){t_txt}\s*(°C|℃|C)(?![A-Za-z])",  # curing temperature
                rf"(?<![A-Za-z])T{t_txt}(?![\d.])",            # 'T20' temperature tags
                rf"(?<![\d.]){a_txt}\s*(d|h|hr|day|days)(?![A-Za-z])",  # age
                r"x\d+",                                       # age as multiple of set time
                r"sealed",                                     # curing label
                r"(?<=-)[CRH]$"):                              # cold/room/hot suffix
        s = re.sub(pat, "", s)
    return re.sub(r"[_\-\s]+", "-", s).strip("-")


def series_ids(g):
    return sorted({series_id(r.specimen_id, r.curing_temp_C, r.age_value)
                   for r in g.itertuples()})


# attribute that differs between the pooled series, verified in the sources
POOLED_ATTR = {
    8: ("kiln", "cement-kiln-dust replacement level (0, 10, 20%)"),
    9: ("mortar", "concrete versus mortar specimens"),
    41: ("fiber", "fiber type (none, synthetic, metallic)"),
    44: ("water", "unit water content (170 vs 182 kg/m3) and water-reducing "
                  "admixture (high-range AE vs AE); Table 1/3 of the source"),
    45: ("cement", "ordinary versus moderate-heat Portland cement"),
    52: ("consist", "stiff versus soft consistency (slump about 8 vs 18 cm), with "
                    "sealed and standard water-cured series; published values are "
                    "inter-laboratory means (Tables 3.1-3.5 of the source)"),
}
pooled = {}
for k, g in groups:
    ids = series_ids(g)
    nb = g.binder_type.astype(str).nunique()
    if len(ids) > 1 or nb > 1:
        assert int(k[0]) in POOLED_ATTR, f"unlabelled pooled group {k}"
        pooled[k] = POOLED_ATTR[int(k[0])]
pooled_rows = int(sum(len(G[k]) for k in pooled))
per_attr = pd.Series([v[0] for v in pooled.values()]).value_counts().to_dict()


def is_pooled(s_, w_, b_):
    return (int(s_), float(w_), str(b_)) in {(int(a), float(b), c) for a, b, c in pooled}


cost = pd.read_csv(OUT / "mismatch_costs.csv")
keep = [not is_pooled(r.source_no, r.wb, r.binder) for r in cost.itertuples()]
ex_within = [float(cost.loc[keep, c].median()) for c in WITHIN]
ex_cross = [float(cost.loc[keep, c].median()) for c in CROSS]
keep_v = [not is_pooled(r.source_no, r.wb, r.binder) for r in val.itertuples()]
ex_r2 = float(np.median([fits[(k, "TTF10")]["r2"] for k in KEYS if k not in pooled]))
rs_pool = spearmanr(val.wb[keep_v], val.Su[keep_v])
keep_h = [not is_pooled(s_, w_, b_) for s_, w_, b_ in
          zip(base.source_no, base.wb, base.binder)]
hs_pool = holdout_summary(base[keep_h])
for est in ("group_equal", "source_equal"):  # text: "all intervals include zero"
    assert hs_pool[est][1] < 0 < hs_pool[est][2]
summary["pooled_groups"] = {
    "groups": len(pooled), "rows": pooled_rows,
    "sources": sorted(int(k[0]) for k in {(k[0],) for k in pooled}),
    "per_attribute": {k_: int(v) for k_, v in per_attr.items()},
    "excluded_sensitivity": {
        "groups_left": int(sum(keep)),
        "within_median_range": [round(min(ex_within), 2), round(max(ex_within), 2)],
        "cross_median_range": [round(min(ex_cross), 2), round(max(ex_cross), 2)],
        "withinmix_R2_median": round(ex_r2, 3),
        "spearman_Su_wb": round(float(rs_pool[0]), 3),
        "holdout": hs_pool},
}

# ===================================== analytic cross-family ratio
T_grid = np.round(np.arange(5.0, 40.0 + 1e-9, 0.1), 1)
ratio = (T_grid + 10.0) / c_ea(T_grid)
summary["cross_ratio_TTF10_EA335"] = {
    "at_20C": round(float(30.0 / c_ea(20.0)), 2),
    "range_5_40C": [round(float(ratio.min()), 1), round(float(ratio.max()), 1)],
    "NS0_EA335_at_20C": round(float(20.0 / c_ea(20.0)), 2)}

# ============================================ Additional file 3: group map
cnt = base.groupby(["source_no", "wb", "binder"]).size()
hi_keys = {(int(k[0]), float(k[1]), k[2]) for k, _ in hi_groups}
gm_rows = []
for i, (k, g) in enumerate(groups, 1):
    kk = (int(k[0]), float(k[1]), k[2])
    gm_rows.append({
        "group_id": f"G{i:02d}",
        "source_no": kk[0],
        "source_short": g.source_short.iloc[0],
        "wb_ratio_pct": kk[1],
        "binder_class": kk[2],
        "published_binder_types": "; ".join(sorted(g.binder_type.astype(str).unique())),
        "published_series_ids": "; ".join(series_ids(g)),
        "pools_multiple_series": "yes" if k in pooled else "no",
        "pooled_attribute": pooled[k][1] if k in pooled else "",
        "n_rows": int(len(g)),
        "curing_conditions": "; ".join(sorted(g.curing_condition.astype(str).unique())),
        "curing_temps_C": "; ".join(f"{t:g}" for t in sorted(g.curing_temp_C.unique())),
        "age_min_d": round(float(g.age_day.min()), 2),
        "age_max_d": round(float(g.age_day.max()), 2),
        "in_high_confidence_subset": "yes" if kk in hi_keys else "no",
        "holdout_paired_tests": int(cnt.get(kk, 0)),
        "row_ids": "; ".join(str(int(x)) for x in sorted(g.row_id)),
    })
group_map = pd.DataFrame(gm_rows)
group_map.to_csv(OUT / "group_map.csv", index=False, encoding="utf-8-sig")
assert group_map.n_rows.sum() == summary["attrition"]["eligible_rows"]
assert group_map.holdout_paired_tests.sum() == len(base)

(OUT / "r3_minor_revision.json").write_text(json.dumps(summary, indent=2), "utf-8")
print(json.dumps({k_: v for k_, v in summary.items() if k_ != "holdout_variants"},
                 indent=2))
print(json.dumps({k_: summary["holdout_variants"][k_] for k_ in
                  ("published_251", "exclude_boundary", "restricted_search")}, indent=2))

# ====================================================================== macros
at, fi, va = summary["attrition"], summary["fits"], summary["validity"]
dg, hv = summary["datum_grid"], summary["holdout_variants"]
pg, cx = summary["pooled_groups"], summary["cross_ratio_TTF10_EA335"]
ex = pg["excluded_sensitivity"]
w_mis = [fi["misaligned_rmse_median_MPa"][p] for p in WITHIN]
c_mis = [fi["misaligned_rmse_median_MPa"][p] for p in CROSS]
al = list(fi["aligned_rmse_median_MPa"].values())
ge_x, se_x = hv["exclude_boundary"]["group_equal"], hv["exclude_boundary"]["source_equal"]
ge_r, se_r = hv["restricted_search"]["group_equal"], hv["restricted_search"]["source_equal"]
M = {
    # attrition and subset sizes
    "RevRowsCuring": comma(at["curing_rows"]), "RevSrcCuring": at["curing_sources"],
    "RevRowsComplete": comma(at["complete_rows"]), "RevSrcComplete": at["complete_sources"],
    "RevCandGroups": at["candidate_groups"], "RevFailTemps": at["fail_lt2_temps"],
    "RevFailAges": at["fail_lt3_ages"], "RevEligSources": at["eligible_sources"],
    "RevHighConfRows": comma(summary["high_confidence"]["rows"]),
    "RevHighConfSources": summary["high_confidence"]["sources"],
    # convention shares; within-group R2
    # whole percent: the one-decimal value (67.9) would visibly disagree with
    # the sum of the rounded published shares (56.9 + 5.1 + 6.0 = 68.0)
    "RevPctThreeConv": f"{half_up(100.0 * three / conv['corpus_rows'])}",
    "RevPctThreeOfStated": f"{half_up(summary['three_conventions']['pct_of_rows_with_stated_index'])}",
    "RevWithinRTwoNS": f"{fi['withinmix_R2_median']['NS0']:.3f}",
    "RevWithinRTwoEA": f"{fi['withinmix_R2_median']['EA335']:.3f}",
    # aligned and misaligned RMSE
    "RevAlignedRMSE": rng_txt(min(al), max(al)),
    "RevMisRMSEWithin": rng_txt(min(w_mis), max(w_mis)),
    "RevMisRMSECross": rng_txt(min(c_mis), max(c_mis)),
    # fit settings
    "RevSuBoundNS": fi["su_bound_hits"]["NS0"]["su_any"],
    "RevSuBoundTTF": fi["su_bound_hits"]["TTF10"]["su_any"],
    "RevSuBoundEA": fi["su_bound_hits"]["EA335"]["su_any"],
    "RevSuBoundTTFLo": fi["su_bound_hits"]["TTF10"]["su_lower"],
    "RevSuBoundTTFHi": fi["su_bound_hits"]["TTF10"]["su_upper"],
    # S_u and tau
    "RevSpearSuWbExBound": f"{va['spearman_Su_wb_excl_bound']:.2f}",
    "RevNExBound": va["n_excl_bound"],
    "RevSpearSuWbCiLow": f"{va['spearman_Su_wb_hier_ci'][0]:.2f}",
    "RevSpearSuWbCiHigh": f"{va['spearman_Su_wb_hier_ci'][1]:.2f}",
    "RevTauSF": comma(va["tau_median_Ch"]["SF"]), "RevNSF": va["binder_counts"]["SF"],
    "RevTauLH": comma(va["tau_median_Ch"]["LowHeat"]),
    "RevNLH": va["binder_counts"]["LowHeat"],
    # datum grid
    "RevHoldCandTests": dg["candidate_tests"],
    "RevHoldGroupsThree": dg["groups_3_temps"], "RevHoldGroupsFour": dg["groups_4plus_temps"],
    "RevGridN": dg["grid_size"], "RevGridFeasMin": dg["feasible_min"],
    "RevGridFeasMed": f"{dg['feasible_median']:.0f}",
    "RevBoundaryN": dg["boundary_tests"], "RevBoundaryNeg": dg["boundary_negative"],
    "RevBoundaryZero": dg["boundary_zero"],
    # hold-out variants
    "RevExclN": hv["exclude_boundary"]["tests"],
    "RevExclGE": f2(ge_x[0]), "RevExclGELow": f2(ge_x[1]), "RevExclGEHigh": f2(ge_x[2]),
    "RevExclSE": f2(se_x[0]), "RevExclSELow": f2(se_x[1]), "RevExclSEHigh": f2(se_x[2]),
    "RevRestrN": hv["restricted_search"]["tests"],
    "RevRestrGE": f2(ge_r[0]), "RevRestrGELow": f2(ge_r[1]), "RevRestrGEHigh": f2(ge_r[2]),
    "RevRestrSE": f2(se_r[0]), "RevRestrSELow": f2(se_r[1]), "RevRestrSEHigh": f2(se_r[2]),
    # pooled groups
    "RevPooledGroups": pg["groups"], "RevPooledRows": pg["rows"],
    "RevPooledSources": len(pg["sources"]),
    "RevPooledKiln": pg["per_attribute"].get("kiln", 0),
    "RevPooledMortar": pg["per_attribute"].get("mortar", 0),
    "RevPooledFiber": pg["per_attribute"].get("fiber", 0),
    "RevPooledWater": pg["per_attribute"].get("water", 0),
    "RevPooledCement": pg["per_attribute"].get("cement", 0),
    "RevPooledConsist": pg["per_attribute"].get("consist", 0),
    "RevExPoolGroups": ex["groups_left"],
    "RevExPoolWithin": rng_txt(*ex["within_median_range"]),
    "RevExPoolCross": rng_txt(*ex["cross_median_range"]),
    "RevExPoolRTwo": f"{ex['withinmix_R2_median']:.3f}",
    "RevExPoolSpear": f"{ex['spearman_Su_wb']:.2f}",
    "RevExPoolHoldN": ex["holdout"]["tests"],
    # analytic cross-family ratio
    "RevRatioCrossLow": half_up(cx["range_5_40C"][0]),
    "RevRatioCrossHigh": half_up(cx["range_5_40C"][1]),
}
def tex_num(v):
    """macro value for LaTeX: a minus sign, not a hyphen, before a negative number"""
    return re.sub(r"(?<![-\d])-(?=\d)", r"\\ensuremath{-}", str(v))


(DST / "generated_r3_macros.tex").write_text(
    "\n".join(f"\\newcommand{{\\{k}}}{{{tex_num(v)}}}" for k, v in M.items()) + "\n", "utf-8")
print(f"wrote {len(M)} macros -> generated_r3_macros.tex; group map -> out/group_map.csv")
