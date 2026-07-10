# How the Research Cycles Are Built, How Decisions Are Made, and What They Found

This document answers three questions in one place: **how** a research
cycle is created, **how** a verdict is decided (and how the process
defends against fooling itself), and **what** the four cycles actually
found. It is written so a competent agent or human can re-run any cycle
and reach the same numbers. Simulated results throughout; nothing here is
investment advice.

The governing system is FMROS v1.0 (`FMROS/`); this is the plain-language
tour of it. The authoritative record is the experiment registry
(`FMROS/appendix/experiment_registry/registry.md`).

---

## 1. What a "cycle" is

A cycle is a batch of pre-registered experiments run under one governing
question, closed with a written report. Not a backtest — a *research
program* whose object is the market and whose standard is out-of-sample
evidence.

| Cycle | Governing question | Outcome |
|---|---|---|
| 1 | Does any tradable edge exist on XAUUSD 15m? | Small real signal (AUC 0.529); dies at retail cost (breakeven ≈1.3bp vs 2.5bp) |
| 2 | Which levers move net expectancy? | 3 validated (vol-gate, wide barriers, passive entry); 4 refuted |
| 3 | Do the levers compose, and do exogenous signals help? | Composite WEAKENED; macro + news conditioning REJECTED |
| 4 | Does training-window choice change the answer? | Expanding wins decisively; rolling-2 has no signal at all |

---

## 2. How a cycle is created — the lifecycle

Every experiment walks the same path (FMROS Volume 5). The order is not
optional; several steps exist specifically to stop the researcher from
fooling themselves.

1. **Mechanism first.** Before any code, write *why* the effect should
   exist — an economic or microstructural reason. "The model likes it"
   is not a mechanism. Blind mining is barred (Constitution §1.6):
   H-D1's 2,000-formula search was allowed *only* because it declared its
   universe and Bonferroni-corrected for it, and it still found nothing
   the hand set lacked.

2. **Pre-register.** Write the hypothesis, the exact feature/data family,
   the multiplicity universe (how many things are being tried), and a
   **verbatim decision rule** — the accept/reject threshold — and commit
   that file to git *before the confirmatory run*. The commit timestamp
   is the proof the rule was fixed in advance. Every C3/C4 entry was
   committed before its run.

3. **Run** through the shared engine — walk-forward training (expanding,
   annual retrain, 96-bar purge), the event simulator with costs,
   position blocking, pessimistic ties, pre-gap flatten, and the leverage
   cap. No experiment gets its own private simulator.

4. **Dissect.** Never report a pooled number alone. Break every result
   down by year, by volatility regime, by exit reason, by tail. A pooled
   p-value with profits concentrated in two episodes confirms nothing
   (this rule has teeth — see §4).

5. **Adversarial review.** A second pass hunts for leakage, cost
   misapplication, and selection effects. Five leakage screens run; any
   engine change triggers an anchor re-check (§5).

6. **Verdict + lessons**, recorded in the registry, never edited after
   the fact. Negative results are written up at the same prominence as
   positive ones — the graveyard is part of the map.

---

## 3. How a decision is made — the rules that bind the verdict

A verdict is not a judgment call. It is the output of the pre-registered
decision rule applied to the run. The rules that make verdicts
trustworthy:

- **Near-efficiency null.** The prior is that no edge survives costs. The
  burden is on the hypothesis to beat that, out-of-sample, after costs.
- **One look per slice.** Each data slice can confirm a hypothesis
  family once. The 2010–2013 window is screening-only. Odd years
  2015–2025 were spent confirming the composite (C3-001) and cannot be
  re-spent. This is why new questions increasingly need *new data*
  (C3-008 waits for 2027+) rather than another pass over the same years.
- **Multiplicity is always declared.** If you try N things, the bar rises
  with N. Undeclared trying is how noise gets published.
- **Cost is reported in tiers.** Every P&L is shown at 1.25 / 1.5 / 2.5 /
  4 bp. A result that is positive at 1.25bp and negative at 2.5bp is
  reported as exactly that — not rounded to "profitable."
- **Aggregates are dissected (see §4).**
- **Errors cascade, never get silently fixed.** A wrong result is struck
  to INVALID with a diagnosis and its dependents are re-run.

Worked example of a decision rule (C3-003, verbatim, frozen before run):
*ACCEPT if mean AUC uplift ≥ +0.003 AND uplift negative in ≤ 1 of 3
folds; REJECT otherwise.* The run returned −0.0087, negative in 3/3 →
REJECTED. No interpretation required; the rule decided.

---

## 4. The anti-fallacy that matters most here: dissect every aggregate

The single rule this project leans on hardest, because it is the easiest
to violate by accident:

> A mean, a pooled p-value, or a total P&L is an **aggregate**. Edges in
> this market concentrate in a few volatility episodes (proven in
> C3-001). Therefore an aggregate can hide — or fake — an edge. Every
> aggregate must be broken down by regime before it is believed.

This rule was **exercised live in Cycle 4**, against the research team's
own work. The window study first concluded "rolling-2 has no signal"
from a full-history *mean* AUC of 0.5012. That was a fallacy — a flat
mean is precisely the aggregate the rule says to distrust. The correct
move was to dissect by regime:

| Window | Mean AUC (all bars) | AUC inside high-vol gate | Verdict inside edge-case regime |
|---|---|---|---|
| Expanding | 0.5195 | 0.5255 | real edge |
| Rolling-5 | 0.5121 | 0.5225 | marginal edge (sign p=0.046) |
| Rolling-2 | 0.5012 | 0.5112 | still a coin (sign p=0.500) |

The dissection *changed the finding's basis*: it revealed that the gate
concentrates signal for every window (the flat mean really was hiding
structure), and only then confirmed that rolling-2's edge-case years
(0.567 in 2018, etc.) do not persist — scattered good years are the
signature of noise, not a hidden regime. The weak claim ("mean is 0.50,
so dead") was a fallacy; the strong claim ("no edge survives regime
dissection") is earned. Same conclusion, honestly obtained instead of
assumed. This is what the rule is *for*.

---

## 5. How to run a cycle again (reproducibility)

The numbers are not narrative — they regenerate. Key entry points:

- `strategy/xauusd_x1_final.py` — the consolidated strategy. Retrains all
  walk-forward folds and **self-verifies against recorded anchors**: it
  must reproduce 414 trades / +0.127908R (even years) and 334 /
  +0.078835R (odd years) exactly, or it aborts with ANCHOR MISMATCH.
- `strategy/window_comparison.py` — end-to-end P&L for expanding /
  rolling-5 / rolling-2 across all years, full metrics + cost tiers.
- `research/cycle3/window_research_cycle.py` — the C1-BASE foundation
  (does a signal exist?) redone per window: per-year OOS AUC, sign test.
- `research/cycle3/window_edge_dissection.py` — the §4 regime dissection:
  gated vs ungated AUC per window.
- Anchors and their expected values live in each script's header;
  `FMROS/appendix/checklists/anchor_smoke.md` is the regression test.

Determinism is enforced by fixed seeds and cached, content-addressed
predictions; a changed engine invalidates the caches and forces a
re-anchor.

---

## 6. Consolidated results (registry, abridged)

| ID | Question | Verdict | Effect |
|---|---|---|---|
| C1-BASE | Signal exists? | CONFIRMED | WF AUC 0.529, 13/13 yrs>0.5 |
| C1-ECON | Survives retail cost? | REFUTED | +0.114R gross → −0.105R @2.5bp |
| H-A1 | Vol-gate helps? | WEAKENED | −0.105→+0.053R; regimes intraday |
| H-B2 | Wide barriers help? | mechanism CONFIRMED | cost-in-R halved (G0→G2) |
| H-B1 | Passive entry helps? | WEAKENED | +0.088R ITT, adverse selection quantified |
| H-F1 | Vol brackets profitable? | REFUTED | −0.457R, worse than random |
| H-I1 | Rolling/PSI retrain beats expanding? | REFUTED | expanding best everywhere |
| X1/C3-001 | Levers compose? | WEAKENED | +0.079R odd yrs, day-block p=0.108 |
| C3-003 | Daily macro conditions 15m? | REJECTED | uplift −0.0087, noise-fit |
| C3-007 | News tone conditions 15m? | REJECTED | uplift −0.0027, P8 0-for-2 |
| C4/C3-008 | Rolling window beats expanding? | REFUTED (screen); arbiter open on 2027+ | expanding +0.106R vs rolling-5 −0.037R vs rolling-2 −0.083R; rolling-2 no signal |

**State of knowledge.** A small, real, cost-fragile edge exists on
XAUUSD 15m, concentrated in high-volatility regimes, best extracted with
all available history and cheap passive execution. No configuration has
cleared a production gate. The open frontier is measured execution cost
(C3-002, needs a broker feed) and calendar time (C3-006, C3-008, need
2027+ data). Every negative result above is a fence that keeps the next
agent from re-paying for the same dead end.

---

## 7. How metrics must be chosen, evaluated, and combined (operator doctrine)

This section is standing doctrine, adopted after the operator correctly
rejected a Cycle 5 analysis that ranked performance metrics as flat,
independent columns. Metrics are **not** independent columns. They are a
coupled algebraic system — the operator's analogy is PV=nRT: you cannot
vary one and hold the others fixed, because an equation of state binds
them. Cycle 6 proved this on our own metrics and it is now a rule.

**7.1 There is a hierarchy: primitives vs derived.** A handful of
quantities are measured directly from the trade stream (primitives:
win-rate p, avg-win W, avg-loss L, dispersion σ, downside σ_d, max
drawdown, longest losing streak, skew, kurtosis, count N). Everything
else is a **deterministic function** of those:
expectancy = p·W+(1−p)·L; profit_factor = p·W/((1−p)|L|) ≡ omega;
sharpe = expectancy/σ; sortino = expectancy/σ_d;
calmar = total/|maxDD| ≡ recovery_factor. A derived metric **cannot
carry information its primitives lack**. Analyse at the primitive layer.

**7.2 Collapse redundancy before ranking.** On our data, 28 metrics
carry only ~6 independent dimensions (PCA: first component = 51% of all
variance), and **eleven** "return" metrics (expectancy, sharpe, sortino,
PF, omega, calmar, recovery, win-rate, total_R, downside_dev, skew) move
as a **single factor** at |ρ|≥0.90. Ranking collinear metrics separately
multiply-counts one signal and fabricates false agreement. Rule: cluster
by |ρ|, keep **one representative per independent axis**, then analyse.

**7.3 Some metrics are structurally blind.** Metrics pinned by the fixed
TP/SL barrier geometry (payoff ratio, avg-win, avg-loss, tail
percentiles, best/worst R) barely move when the model changes, so they
**cannot** discriminate between models or predict anything. Exclude them
from model/window comparison — they are constants in disguise.

**7.4 Evaluation must be out-of-sample, with a permutation null.**
In-sample correlation across many metrics is a false-positive generator.
Cycle 5 reported three metrics that "predict" forward returns (in-sample
ρ≈0.34); under leave-one-out cross-validation with a 500-draw
permutation null, the effect vanished (OOS R²≈0, permutation p=0.28) and
the claim was **withdrawn**. Rule: no forward-predictive claim without
(a) OOS cross-validation and (b) a permutation null that re-runs the
entire metric/combination search on shuffled targets.

**7.5 Combinations are tested, not assumed — and bounded by sample
size.** Patterns can live in joint structure, so search combinations
(permutations of primitives) — but honestly. All singles and pairs of 9
primitives were OOS-tested; the best pair did not beat the best single
and neither beat the null. Beyond pairs / beyond linear is **not
testable at n≈42** without overfitting, so it stays open, not claimed.
More power requires more data (C3-008 / 2027+).

**7.6 The frozen protocol for any future metric study.** (i) reduce to
independent axes; (ii) drop barrier-pinned metrics; (iii) analyse
primitives; (iv) OOS-CV + permutation null for every predictive claim;
(v) report combinations only if they beat both the best single and the
null. This is the standing method for evaluating metrics on 2027+ data.
