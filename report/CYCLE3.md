# Cycle 3 — X1 Confirmation (first experiment under FMROS v1.0)

One pre-registered experiment, run under the newly codified operating
system (`FMROS/`), with its decision rule frozen before execution.

**Question.** Does the X1 composite — the conjunction of cycle 2's three
validated levers, frozen exactly as specified in the pilot — earn positive
net expectancy on the odd test years (2015–2025) it had never been
evaluated on?

**Answer: WEAKENED** (per the pre-declared rule). It earned **+0.079R per
trade net** (334 trades, PF 1.15, Sharpe 0.50, maxDD −24%), right-signed
and ~60% of the pilot's magnitude — but day-block p = 0.108 misses the
0.05 confirmation gate, and year-block p = 0.187. Three of six years
positive, with profits again concentrated in vol-rich years (2017 +0.45R,
2021 +0.17R) as the strategy's design predicts.

| | Even years (pilot, exploratory) | Odd years (confirmation) |
|---|---|---|
| Trades | 414 | 334 |
| Expectancy | +0.128R | +0.079R |
| Day-block p(≤0) | 0.020 | 0.108 |
| Year-block p(≤0) | 0.102 | 0.187 |
| Max DD | −13.9% | −23.8% |

Descriptively, the pooled sample is +0.106R over 748 trades — but pooling
was not pre-registered and the even years are exploratory-tainted, so
this number carries no confirmatory weight.

**Interpretation.** The composite behaves like a real but modest edge
observed through too few volatility regimes: consistent sign across two
disjoint eras, magnitude shrinking from exploratory to confirmatory
sample exactly as winner's-curse arithmetic predicts, and significance
starved by regime concentration rather than by trade count. The two
honest paths to a verdict are calendar time (the standing 2027+ re-run,
registry C3-006, rule inherited verbatim) and cheaper measured execution
(registry C3-002: shadow-measure real spreads and passive fill rates —
the 1.25bp assumption is now the most verdict-sensitive number in the
program).

## Two further verdicts under FMROS (same day)

**C3-003 — cross-asset macro conditioning: REJECTED.** A pre-registered
12-feature block of 2-day-lagged daily macro state (10y real yield,
broad dollar, VIX, WTI from FRED) *reduced* walk-forward AUC in all
three test folds (mean −0.0087) while consuming 21–23% of the model's
training gain — the textbook noise-fit signature, and a validation of
the house rule that importance is not usefulness. Gold's macro anchors
operate at horizons of days-plus; they do not time 4-hour barrier odds
through lagged daily samples. The honest successor hypothesis
(intraday-timed macro series) is queued.

**C3-004 — M1 fidelity replay: IMMATERIAL.** After fixing a real schema
defect it exposed (barrier levels now persisted in trade records;
deterministic re-runs matched anchors exactly), minute-level replay of
all 360 stop-loss exits across both composite samples flipped **zero**
of them to take-profits. The composite's +0.128R / +0.079R carry no
tie-break bias; the binding fidelity question is the passive-fill cost
assumption (C3-002), not bar resolution.

## A fourth verdict: cross-asset textual conditioning

**C3-007 — GDELT media tone/theme signal: REJECTED.** Prompted by a
comparison to external LLM-trading projects (Fincept Terminal — a UI/
data terminal, no research methodology of its own; Tauric Research's
TradingAgents — a multi-agent LLM decision framework, no statistical
validation), the question worth actually testing was distilled to: does
*textual* reasoning about news carry timing information that C3-003's
*numeric* macro block lacked? Built on GDELT's automated, deterministic
tone/theme tagging (not a generative LLM — see contamination note
below), a frozen 6-feature weekly block (mean tone, tone negativity,
and inflation/policy-uncertainty/debt/armed-conflict theme shares,
2015-02 onward, 2-day lag, 52-week z-score) produced mean AUC uplift
**−0.0027**, negative in 2 of 3 folds (2018 −0.011, 2022 **+0.0045**,
2026 −0.0021) — REJECTED per the frozen rule. P8 (Cross-Asset) is now
0-for-2: neither numeric macro state nor weekly media tone conditions
XAUUSD's 4-hour barrier odds. Both failures share one mechanism: the
conditioning signal's native timescale (days, for a macro print or an
accumulating media narrative) doesn't match a 4-hour horizon — evidence
for a structural mismatch now, not a fluke of one data type.

**What was deliberately not built.** A true TradingAgents-style backtest
— a generative LLM "reasoning" retrospectively about historical news —
was not attempted. Evaluating that as a historical backtest risks the
model's own pretraining knowledge of what happened *after* a given news
item leaking into its "analysis": a lookahead bias invisible to the
standard leakage screens, because the leak lives in the model's weights,
not in any joined dataframe column. That variant stays queued as
forward/paper-only until a contamination-safe evaluation design exists.

**Program state.** FMROS v1.0 is now the governing system; the registry
(`FMROS/appendix/experiment_registry/registry.md`) carries 19 closed
verdicts and 3 open entries (one operator-blocked on a broker feed, one
calendar-blocked on 2027+ data, one ready: C3-005 regime-transition
prediction). No strategy qualifies for
any production gate. Nothing here is investment advice; simulated results
throughout.
