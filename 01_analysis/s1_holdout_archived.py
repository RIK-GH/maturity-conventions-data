#!/usr/bin/env python
"""Nested held-out validation — ARCHIVED ALGORITHM, verbatim.

Copied 2026-08-13 from the Technical Note's analysis code
(validate_maturity_constants.py). Only the input/output paths were changed to
point at this folder; the algorithm, eligibility rules, fit bounds, and
binder normalisation are untouched. This is the code that produced the
Technical Note's 251 paired held-temperature tests, and S1_dataset.csv there
is byte-identical to maturity_corpus_v3.csv here, so this run doubles as the
reproduction check for the new manuscript.

Original docstring follows.
"""
"""Nested held-out validation for the maturity-constant manuscript.

Primary analysis: leave one curing temperature out within groups that have at
least three temperatures.  Constants are selected only on the training
temperatures; an exponential strength--maturity curve is then fitted on the
same training observations and evaluated at the withheld temperature.

Secondary analysis: leave one source study out.  The binder-class datum is the
median of in-sample training-study optima and is compared with -10 C on every
eligible test group.  This is intentionally a conservative transfer test, not
a claim of a universal material constant.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "00_dataset", "maturity_corpus_v3.csv")
OUT = os.path.join(HERE, "out")
R = 8.314
T0_GRID = np.arange(-25.0, 16.0, 1.0)


def expo(m, su, tau, beta):
    return su * np.exp(-(tau / m) ** beta)


def binder_class(x):
    x = str(x).lower()
    if any(k in x for k in ["low-heat", "low heat", "(lc)", "(l)"]): return "LowHeat"
    if any(k in x for k in ["slag", "bfs", "ggbfs", "(bb)", "blast", "bsc"]): return "Slag"
    if any(k in x for k in ["sf", "silica"]): return "SF"
    if any(k in x for k in ["fa", "fly", "fac"]): return "FA"
    return "OPC"


def fit(m, s):
    if np.any(m <= 0) or len(m) < 4 or len(np.unique(s)) < 3:
        raise ValueError("invalid fit")
    p, _ = curve_fit(expo, m, s, p0=[max(s)*1.1, np.median(m), .6],
                     bounds=([max(s)*.9, 1e-6, .1], [max(s)*3, 1e12, 6]),
                     maxfev=50000)
    y = expo(m, *p)
    ss = np.sum((s-s.mean())**2)
    return p, 1-np.sum((s-y)**2)/ss if ss else np.nan


def m_ttf(g, t0):
    return (g.curing_temp_C.to_numpy(float)-t0)*g.age_hr.to_numpy(float)


def select_t0(g):
    best = (float("-inf"), None)
    for t0 in T0_GRID:
        try:
            _, r2 = fit(m_ttf(g, t0), g.compressive_strength_MPa.to_numpy(float))
            if r2 > best[0]: best = (r2, t0)
        except (ValueError, RuntimeError):
            pass
    if best[1] is None: raise ValueError("no feasible datum")
    return best[1]


def load_groups():
    d = pd.read_csv(DATA, encoding="utf-8-sig")
    for c in ["curing_temp_C", "age_value", "compressive_strength_MPa", "wb_ratio_pct"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["age_day"] = np.where(d.age_unit.astype(str).str.strip().eq("hr"), d.age_value/24, d.age_value)
    d["age_hr"] = d.age_day*24
    d = d[d.curing_condition.astype(str).str.contains("isothermal|sealed|water|air-sealed", case=False, na=False)]
    d = d.dropna(subset=["curing_temp_C", "age_hr", "compressive_strength_MPa", "wb_ratio_pct"]).copy()
    d["binder"] = d.binder_type.map(binder_class)
    out = []
    for key, g in d.groupby(["source_no", "wb_ratio_pct", "binder"]):
        if g.curing_temp_C.nunique() >= 2 and g.age_hr.nunique() >= 3 and len(g) >= 8:
            out.append((key, g.copy()))
    return out


def rmse(y, yh): return float(np.sqrt(np.mean((y-yh)**2)))


def loto(groups):
    rows=[]
    for (source, wb, binder), g in groups:
        temps = sorted(g.curing_temp_C.unique())
        if len(temps) < 3: continue
        for held in temps:
            tr, te = g[g.curing_temp_C != held], g[g.curing_temp_C == held]
            try:
                t0 = select_t0(tr)
                p_opt, _ = fit(m_ttf(tr, t0), tr.compressive_strength_MPa.to_numpy(float))
                p_fix, _ = fit(m_ttf(tr, -10), tr.compressive_strength_MPa.to_numpy(float))
                y = te.compressive_strength_MPa.to_numpy(float)
                rows += [dict(source_no=source, wb=wb, binder=binder, held_temp_C=held,
                              model="nested_mixture_optimum", t0_selected=t0,
                              rmse_MPa=rmse(y, expo(m_ttf(te,t0),*p_opt)),
                              bias_MPa=float(np.mean(expo(m_ttf(te,t0),*p_opt)-y)), n_test=len(te)),
                         dict(source_no=source, wb=wb, binder=binder, held_temp_C=held,
                              model="fixed_minus10", t0_selected=-10,
                              rmse_MPa=rmse(y, expo(m_ttf(te,-10),*p_fix)),
                              bias_MPa=float(np.mean(expo(m_ttf(te,-10),*p_fix)-y)), n_test=len(te))]
            except (ValueError, RuntimeError, FloatingPointError):
                continue
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    g = load_groups()
    a = loto(g)
    a.to_csv(os.path.join(OUT,"nested_leave_temperature_out.csv"),index=False,encoding="utf-8-sig")
    paired = a.pivot_table(index=["source_no","wb","binder","held_temp_C"],columns="model",values="rmse_MPa").dropna()
    paired["fixed_minus10_minus_nested_MPa"] = paired.fixed_minus10-paired.nested_mixture_optimum
    paired.reset_index().to_csv(os.path.join(OUT,"nested_leave_temperature_out_paired.csv"),index=False,encoding="utf-8-sig")
    print("Eligible groups:",len(g),"; LOTO paired temperature tests:",len(paired))
    print("Overall median fixed-minus10 minus nested RMSE (MPa):",round(paired.iloc[:,-1].median(),3))
    print(paired.reset_index().groupby("binder").agg(n=("fixed_minus10_minus_nested_MPa","size"), median_delta=("fixed_minus10_minus_nested_MPa","median")).round(3))

if __name__ == "__main__": main()
