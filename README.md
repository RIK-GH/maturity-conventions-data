# Maturity-convention mismatch in cross-study reuse of strength–maturity data

Data, analysis code, derived results, and figures for the article *Quantifying the
strength-prediction errors of maturity-convention mismatch in cross-study reuse of
strength–maturity data* by Inkyu Rhee, Department of Civil Engineering, and Smart City
Research Center, Chonnam National University, Gwangju 61186, Republic of Korea
(rheei@jnu.ac.kr).

## Main findings

- Once the maturity convention is identified, the corpus is materially coherent:
  mixture-specific calibration gives a median within-group R² of 0.953 (pooled R² ≈ 0.35).
- The convention is among the least completely reported aspects of the data: 23.6% of the
  2,471 rows give no maturity index, and 34.4% lack the datum temperature or activation energy.
- Conventions are not interchangeable. Reading a calibration curve with maturity from the
  wrong convention increases the RMSE of predicted strength by a median of 2.5–3.0 MPa within
  the temperature–time-factor family and 16.9–22.3 MPa across families.
- Tuning the datum per mixture does not replace reporting the convention: in 251
  withheld-temperature tests, the tuned datum showed no reliable advantage over the fixed
  −10 °C datum.
- Reusable strength–maturity data therefore need five items: the convention and formula, the
  parameter values, the units, the underlying temperature–age–strength records, and a mixture
  identifier.

## Contents

| Path | Contents |
|---|---|
| `00_dataset/maturity_corpus_v3.csv` | Additional file 1. Row-level dataset: 2,471 compressive-strength observations from 37 published sources, with per-row provenance and the maturity convention as published |
| `00_dataset/sources_37.csv` | Additional file 2. Bibliographic register of the 37 sources |
| `01_analysis/out/group_map.csv` | Additional file 3. Mapping of the published mixture and series identifiers to the 73 analytical groups |
| `01_analysis/*.py` | Analysis code |
| `01_analysis/out/` | Derived results (CSV, JSON) |
| `03_figures/` | Figures of the article (PNG, PDF) |

## Reproducing the results

Python 3 with numpy, pandas, scipy, and matplotlib. Run the scripts from `01_analysis/` in
this order. Results are written to `01_analysis/out/`, figures to `03_figures/`, and the
article's LaTeX macros to `02_manuscript/`. Random seeds are fixed, so every run reproduces
the numbers in the article.

```
python a01_conventions.py
python s1_holdout_archived.py      # several minutes
python a02_mismatch_consistent.py  # several minutes
python a03_bootstrap.py
python a04_figures.py
python a05_macros.py
python a07_mechanism_striking.py
python a08_material_validity.py
python a06_descriptive_workflow.py
python a09_r1_robustness.py
python a10_minor_revision.py
```

| Script | Computes |
|---|---|
| `s1_holdout_archived.py` | Withheld-temperature validation; the group definition, curve fit, and datum search used by the other scripts |
| `a01_conventions.py` | Reporting profile of the maturity conventions |
| `a02_mismatch_consistent.py` | Errors from convention mismatch; global and within-mixture fits |
| `a03_bootstrap.py` | Hierarchical bootstrap of the withheld-temperature tests |
| `a04_figures.py` | Figs. 2, 3, and 5 |
| `a05_macros.py` | LaTeX macros for the main results |
| `a06_descriptive_workflow.py` | Descriptive statistics of the corpus |
| `a07_mechanism_striking.py` | Origin of the mismatch error and its effect on formwork-striking time; Fig. 6 |
| `a08_material_validity.py` | Material coherence of the corpus; Fig. 4 |
| `a09_r1_robustness.py` | Robustness checks: source-level statistics, curve form, bootstrap intervals, hold-out stratification, high-confidence subset |
| `a10_minor_revision.py` | Attrition, three-convention comparison, fit settings, datum grid, boundary tests, pooled groups; Additional file 3 |

## Figures

| Article | File in `03_figures/` |
|---|---|
| Fig. 1 | `fig0_workflow`, drawn with Mermaid from `fig0_workflow.mmd` (`a06_descriptive_workflow.py` writes an earlier matplotlib version under the same name) |
| Fig. 2 | `fig1_reporting_profile` |
| Fig. 3 | `fig3_calibration` |
| Fig. 4 | `fig6_validity` |
| Fig. 5 | `fig2_convention_cost` |
| Fig. 6 | `fig5_mechanism` |

## License and citation

- Code (`01_analysis/*.py`): MIT License, see `LICENSE`.
- Data, derived results, and figures: CC BY 4.0, see `LICENSE-DATA`. The values are numerical
  facts extracted from the 37 cited publications; no text, figure images, or table images of
  the original articles are included.
- When reusing the data, cite the article and, for individual rows, the original publication
  identified by `source_no` (details in `00_dataset/sources_37.csv`).
