# FMROS v1.0 — Financial Markets Research Operating System

**Purpose.** This is a research constitution and operating manual for an
autonomous or semi-autonomous quantitative research program. It is written
so that any competent research agent — Claude Code, another LLM agent, or a
human quant — can be handed this directory and told: *"Continue the work."*

**Provenance.** FMROS was not written from theory. It codifies the working
practices, statistical standards, failures, and validated findings of two
completed research cycles on XAUUSD 15-minute data (2010–2026, 407,749
bars, ~30 experiments), documented in `report/REPORT.md` and
`report/CYCLE2.md` of this repository. Where this manual asserts something
about markets, it cites that evidence; where it proposes something
untested, it says so.

**Evidence tiers.** Every substantive claim in FMROS carries one of three
labels:

* **[ESTABLISHED]** — standard practice in the quantitative research
  literature and consistent with our own evidence.
* **[TESTED-HERE]** — validated or refuted by a specific experiment in this
  repository (with a pointer to the result file).
* **[HYPOTHESIS]** — plausible, motivated, and **untested**; must pass the
  Experiment Engine (Volume 5) before it may inform any decision.

Anything not labeled is procedural instruction, not a claim about markets.

## Contents

| Volume | File | Role |
|---|---|---|
| 1 | `01_Constitution.md` | Immutable rules: philosophy, evidence, ethics, anti-overfitting, statistics, coding, documentation |
| 2 | `02_Research_Methodology.md` | Hypothesis → experiment → verdict lifecycle |
| 3 | `03_Research_Programs.md` | Ten standing research departments with agendas |
| 4 | `04_Agent_Specifications.md` | Eleven agent roles: responsibilities, I/O, failure conditions |
| 5 | `05_Experiment_Engine.md` | Registry, priority scoring, acceptance/rejection machinery |
| 6 | `06_Validation_Framework.md` | Walk-forward, purging, bootstrap, placebo, leakage protocols |
| 7 | `07_Risk_Engine.md` | Sizing, drawdown control, gap risk, ruin mathematics |
| 8 | `08_Execution_Engine.md` | Cost models, passive fills, adverse selection, geometry |
| 9 | `09_Production.md` | Paper → shadow → live gates; monitoring; automatic disable |
| 10 | `10_Master_Execution_Prompt.md` | The single prompt that boots an agent into this system |
| — | `appendix/` | Templates, checklists, experiment registry, agent prompts, coding standards |

## The one-paragraph state of knowledge (2026-07)

A real but small predictive signal exists on XAUUSD 15m (walk-forward AUC
0.529, 13/13 years above 0.5). It is volatility-flavored: magnitude is
more predictable (AUC ≈ 0.55) than direction (≈ 0.52). Gross expectancy of
the reference strategy is +0.114R/trade; retail friction (2.5bp) kills it
(−0.105R). Cycle 2 validated three partial levers — regime gating
(+0.16R effect), barrier geometry (+0.08R), passive execution (+0.09R) —
and rejected four other hypotheses cleanly. Their composite (X1) is the
first net-positive configuration (+0.128R, day-block p=0.02) but fails
year-block significance (p=0.10) with profits concentrated in two vol
regimes. Its pre-registered confirmation on odd years is the program's
current live experiment. No deployable edge is proven yet.
