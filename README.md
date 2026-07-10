# Claude-research

Quantitative research: **does a statistically significant trading edge exist
on XAUUSD 15-minute bars?**

**Answer: a real but small predictive signal exists (walk-forward AUC 0.529,
13/13 years above 0.5); it does not survive realistic retail transaction
costs (+0.114R/trade gross → −0.105R/trade at 2.5bp round-trip). Breakeven
≈ 1.3bp. No tradeable retail edge was found.**

**Operating system:** the research program is governed by
[FMROS v1.0](FMROS/00_README.md) — a research constitution, methodology,
experiment engine, and agent specification distilled from these cycles.
To continue the work with any coding agent, hand it
[FMROS/10_Master_Execution_Prompt.md](FMROS/10_Master_Execution_Prompt.md).

Full findings:
- Cycle 1 (edge existence): [report/REPORT.md](report/REPORT.md)
- Cycle 2 (new-hypothesis discovery, 10 tracks): [report/CYCLE2.md](report/CYCLE2.md) —
  4 clean rejections, 3 quantified levers (vol-gating, barrier geometry,
  passive execution), 1 exploratory composite (+0.128R net, unproven at
  year granularity).
- Cycle 3 (pre-registered confirmation): [report/CYCLE3.md](report/CYCLE3.md) —
  X1 on never-tested odd years: +0.079R, right-signed but p=0.11 →
  **WEAKENED**; verdict deferred to 2027+ data per standing registry entry.

## Layout

- `research/data/` — data acquisition (HistData M1 2009–2026), QC, 15m
  resample, FOMC/NFP calendar
- `research/features.py` — 205 features, triple-barrier + directional labels
- `research/eval_features.py` — MI / gain / permutation importance,
  redundancy, VIF (walk-forward)
- `research/models_compare.py` — 11 model families, purged walk-forward
- `research/targets_compare.py` — which label is most learnable
- `research/backtest.py` — cost-aware event simulation, bootstrap +
  placebo significance
- `research/results/` — all experiment outputs (CSV/JSON)
- `report/` — research report + figures

Large parquet artifacts are gitignored; rebuild via the reproduction steps
at the bottom of the report.
