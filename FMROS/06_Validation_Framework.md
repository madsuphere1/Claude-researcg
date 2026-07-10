# Volume 6 — Validation Framework

The exact protocols. Where Volume 2 says *what*, this volume says *how*,
with the parameter values in force.

## 6.1 Data partitions (current dataset: XAUUSD 15m, 2010–2026)

| Partition | Years | Allowed uses |
|---|---|---|
| Dev window | 2010–2013 | Screening, MI, redundancy, SSL pre-training, HMM diagnostics. Burned for confirmation forever. |
| Walk-forward test years | 2014–2026 | One confirmatory look per pre-registered experiment. Cycle 2 consumed even years for its fold subsets; cycle 3 consumed odd years for X1. |
| The future (2027+) | — | The only truly clean confirmation set. Accrues ~26k bars/year. |

Consumption is tracked in the registry: an experiment's entry lists which
partition-slices it burned. The Statistician refuses designs whose slice
was already influenced by the same hypothesis family.

## 6.2 Walk-forward protocol

* Expanding training window from 2010, purge 96 bars before test start
  (≥ 6× the 16-bar label horizon; also covers the 64-bar G2 horizon with
  margin ≥ 1.5×) [ESTABLISHED + TESTED-HERE: expanding > rolling].
* Fold-internal validation = last training year; ALL fold-level choices
  (thresholds, gate cutoffs) are functions of training+validation only.
* Fold subsets for compute (every-second-year) are legal if declared
  ex-ante; the subset choice itself is part of pre-registration.
* Retraining cadence: annual (tested optimum; see i1_drift.json).

## 6.3 Purged cross-validation

For non-sequential model questions (feature ablations, hyperparameters)
K-fold with purge+embargo (embargo = label horizon) is acceptable at
screening stage [ESTABLISHED]. Confirmation is always walk-forward — CV
folds mix regimes in ways that flatter adaptive models.

## 6.4 Bootstrap procedures

* **Day-block bootstrap** (minimum): resample trading days with
  replacement, B ≥ 3000–5000; report 2.5/97.5 percentiles and p(≤0).
* **Coarse-block bootstrap** (gates acceptance when clustering visible):
  year blocks, or regime blocks when a gate strategy is under test.
  Rule of force: *if profits' Herfindahl across years exceeds 0.4, or
  fewer than 3 years are individually positive, the coarse block gates.*
  (X1 precedent both cycles.)
* Paired designs (execution studies) bootstrap the per-signal
  *difference* — much tighter CIs, same blocks.

## 6.5 Placebo / reality check

* Timing claims: shuffle predictions within fold-year (preserves marginal
  distribution and threshold pass rates), rerun the FULL simulator,
  N ≥ 50; report observed percentile. [House standard since cycle 1.]
* Frequency-matched random gates for gating claims (bracket study
  pattern).
* Search claims: the declared-universe deflation of §5.5; where a
  max-statistic is the claim, White's reality check via stationary
  bootstrap on the candidate panel [ESTABLISHED; not yet needed here
  beyond Bonferroni].

## 6.6 Sensitivity analysis (mandatory grid per strategy claim)

* Costs: {1.5, 2.5, 4.0} bp (+ 0bp for mechanism display).
* Decision threshold: ±0.01, ±0.02 absolute.
* Key structural parameter (barrier width, gate P, δ): the declared grid.
* Verdict language must state the grid's worst corner, not just the
  primary spec.

## 6.7 Stress testing

* Regime slices: performance in each HMM state, ADX/vol quartiles,
  sessions (report table; no cherry-picks: all slices or none).
* Event windows: FOMC/NFP days in vs out.
* Worst-case sequencing: max consecutive losses at trade level; day-block
  resampled maxDD distribution (P50/P95); the ruin table of Volume 7.
* Crisis replay: 2020-03 and 2022 tightening episodes reported
  separately for any strategy that trades vol.

## 6.8 Leakage detection

Protocol of §2.6 (correlation screen at |ρ|>0.05, +1-bar shift test,
smoothed-state audit, boundary audit, too-good tripwire at AUC 0.60 /
+0.5R). Additionally: any new *data source* gets a timestamp-integrity
audit (are quotes stamped at event time or arrival time?) before use.

## 6.9 Monte Carlo

* Equity-path questions (ruin, DD distributions): day-block resampling of
  the R stream under the exact sizing rule, N ≥ 2000.
* Frequency-matched random-entry baselines through the full simulator for
  "is the entry signal worth anything" questions.

## 6.10 The anchor set

Registry-tagged experiments re-run on every engine/library change
(§5.7): cycle-1 baseline backtest; B2v2 managed G2; A1 gated baseline;
B1 δ=0.10; X1 pilot; C3 confirmation. Tolerances: ±0.005R expectancy,
±0.002 AUC. This is the regression-test suite of the research program.
