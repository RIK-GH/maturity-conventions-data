# -*- coding: utf-8 -*-
"""
a05_macros.py — turn the analysis outputs into LaTeX macros so that every
number in the manuscript is generated, never typed. Same discipline as the
Technical Note: if a value changes, rerun a01–a03 and this script; the text
follows.

Reads  01_analysis/out/{conv_summary,mismatch_summary,holdout_summary}.json
Writes 02_manuscript/generated_results_macros.tex
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DST = HERE.parent / "02_manuscript"
DST.mkdir(exist_ok=True)

conv = json.loads((OUT / "conv_summary.json").read_text("utf-8"))
mm = json.loads((OUT / "mismatch_summary.json").read_text("utf-8"))
hold = json.loads((OUT / "holdout_summary.json").read_text("utf-8"))

boot = {b["estimand"]: b for b in hold["bootstrap"]}
glob = {g["convention"]: g for g in mm["global_fits"]}


def f(x, nd=2):
    return f"{x:.{nd}f}"


def comma(x):
    return f"{x:,}"


M = {
    # corpus
    "NRows": comma(conv["corpus_rows"]),
    "NSources": str(conv["corpus_sources"]),
    # reporting profile
    "PctTTFTen": f(conv["convention_pct"]["TTF_-10"], 1),
    "RowsTTFTen": comma(conv["convention_rows"]["TTF_-10"]),
    "PctUnreported": f(conv["convention_pct"]["unreported"], 1),
    "RowsUnreported": comma(conv["convention_rows"]["unreported"]),
    "PctOther": f(conv["convention_pct"]["other"], 1),
    "PctEqAge": f(conv["convention_pct"]["equivalent_age"], 1),
    "PctNSZero": f(conv["convention_pct"]["TTF_NurseSaul"], 1),
    "PctDatumUnstated": f(conv["datum_unstated_pct"], 1),
    "RowsDatumUnstated": comma(conv["datum_unstated_rows"]),
    # groups / fits
    "NEligibleGroups": str(mm["eligible_groups"]),
    "NFitRows": comma(glob["TTF10"]["n"]),
    "GlobalRTwoNS": f(glob["NS0"]["R2"], 2),
    "GlobalRTwoTTF": f(glob["TTF10"]["R2"], 2),
    "GlobalRTwoEA": f(glob["EA335"]["R2"], 2),
    "GlobalRMSETTF": f(glob["TTF10"]["RMSE_MPa"], 0),
    "WithinRTwoMedian": f(mm["withinmix_R2_median"], 3),
    "WithinRTwoQOne": f(mm["withinmix_R2_q1"], 3),
    "WithinRTwoQThree": f(mm["withinmix_R2_q3"], 3),
    # mismatch costs
    "CostTTFtoNS": f(mm["mismatch_median_MPa"]["TTF10_into_NS0"], 1),
    "CostNStoTTF": f(mm["mismatch_median_MPa"]["NS0_into_TTF10"], 1),
    "CostNStoEA": f(mm["mismatch_median_MPa"]["NS0_into_EA335"], 1),
    "CostEAtoNS": f(mm["mismatch_median_MPa"]["EA335_into_NS0"], 1),
    "CostTTFtoEA": f(mm["mismatch_median_MPa"]["TTF10_into_EA335"], 1),
    "CostEAtoTTF": f(mm["mismatch_median_MPa"]["EA335_into_TTF10"], 1),
    "CostEAtoTTFQThree": f(mm["mismatch_q3_MPa"]["EA335_into_TTF10"], 1),
    "CostMax": f(max(mm["mismatch_max_MPa"].values()), 0),
    "QThreeTTFtoNS": f(mm["mismatch_q3_MPa"]["TTF10_into_NS0"], 1),
    "QThreeNStoTTF": f(mm["mismatch_q3_MPa"]["NS0_into_TTF10"], 1),
    "QThreeNStoEA": f(mm["mismatch_q3_MPa"]["NS0_into_EA335"], 1),
    "QThreeEAtoNS": f(mm["mismatch_q3_MPa"]["EA335_into_NS0"], 1),
    "QThreeTTFtoEA": f(mm["mismatch_q3_MPa"]["TTF10_into_EA335"], 1),
    "MaxTTFtoNS": f(mm["mismatch_max_MPa"]["TTF10_into_NS0"], 1),
    "MaxNStoTTF": f(mm["mismatch_max_MPa"]["NS0_into_TTF10"], 1),
    "MaxNStoEA": f(mm["mismatch_max_MPa"]["NS0_into_EA335"], 1),
    "MaxEAtoNS": f(mm["mismatch_max_MPa"]["EA335_into_NS0"], 1),
    "MaxTTFtoEA": f(mm["mismatch_max_MPa"]["TTF10_into_EA335"], 1),
    "MaxEAtoTTF": f(mm["mismatch_max_MPa"]["EA335_into_TTF10"], 1),
    # holdout
    "NHoldTests": str(hold["tests"]),
    "NHoldGroups": str(hold["groups"]),
    "NHoldSources": str(hold["sources"]),
    "HoldNestedBetter": str(hold["directions"]["nested_better"]),
    "HoldTie": str(hold["directions"]["tie"]),
    "HoldFixedBetter": str(hold["directions"]["fixed_better"]),
    "HoldMedian": f(hold["delta_median_MPa"], 3),
    "HoldQOne": f(hold["delta_q1_MPa"], 3),
    "HoldQThree": f(hold["delta_q3_MPa"], 3),
    "GroupEqualMean": f(boot["group_equal_mean"]["estimate_MPa"], 3),
    "GroupEqualLow": f(boot["group_equal_mean"]["ci95_low"], 3),
    "GroupEqualHigh": f(boot["group_equal_mean"]["ci95_high"], 3),
    "SourceEqualMean": f(boot["source_equal_mean"]["estimate_MPa"], 3),
    "SourceEqualLow": f(boot["source_equal_mean"]["ci95_low"], 3),
    "SourceEqualHigh": f(boot["source_equal_mean"]["ci95_high"], 3),
    "BootReps": comma(hold["reps"]),
    "BootSeed": str(hold["seed"]),
}

def tex_num(v):
    """macro value for LaTeX: a minus sign, not a hyphen, before a negative number"""
    return re.sub(r"(?<![-\d])-(?=\d)", r"\\ensuremath{-}", str(v))


lines = [f"\\newcommand{{\\{k}}}{{{tex_num(v)}}}" for k, v in M.items()]
(DST / "generated_results_macros.tex").write_text("\n".join(lines) + "\n",
                                                  "utf-8")
print(f"wrote {len(M)} macros -> {DST / 'generated_results_macros.tex'}")
