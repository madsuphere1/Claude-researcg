# Volume 2 — Research Methodology

How a hypothesis becomes a verdict. This volume operationalizes the
Constitution; nothing here may weaken it.

## 2.1 The research lifecycle

```
IDEA → REGISTRY ENTRY → DESIGN REVIEW → DEV SCREEN → PRE-REGISTRATION
     → CONFIRMATORY RUN → DIAGNOSIS → VERDICT → REPORT → (maybe) PRODUCTION PIPELINE
```

State transitions are recorded in the experiment registry
(`FMROS/appendix/experiment_registry/`). An experiment may be in exactly
one state. Skipping states is prohibited; in particular nothing goes from
DEV SCREEN to VERDICT without a frozen pre-registration.

## 2.2 Hypothesis generation

Sources, in order of historical productivity in this program:

1. **Decomposition of a known result.** Ask "which term of net expectancy
   does this failure/success live in?" Cycle 2's three validated levers
   all came from decomposing cycle 1's single number (−0.105R) into
   signal, cost-paid, and cost-denominator terms. [TESTED-HERE]
2. **Diagnostics of the last experiment.** Failure analysis is the best
   idea generator: the G2 tail diagnosis produced the risk-managed
   variant; the bracket whipsaw failure produced the options-payoff
   hypothesis (open).
3. **Mechanism transplants from literature** — with the discipline that
   literature effects usually die on our data (London-open reversion:
   documented in FX majors, rejected on gold [TESTED-HERE:
   `research/cycle2/results/c1_session_reversion.json`]). Treat literature
   as a hypothesis source, never as evidence.
4. **Systematic search** (symbolic features, architecture sweeps) — only
   inside declared-universe multiplicity control, and with the sobering
   prior that our 2,000-formula search recovered exactly one
   already-known factor.

Required fields for any new hypothesis (template in
`appendix/templates/hypothesis.md`): mechanism (whose behaviour, why not
arbitraged), the economic term attacked, falsification criterion,
expected effect size, data requirements, and the multiplicity universe it
belongs to.

**Prioritization** is by expected information value, not expected profit:
prefer the experiment that most reduces uncertainty about the program's
main open question (Volume 5 §5.3 gives the scoring rubric).

## 2.3 Experimental design rules

1. **Cohort thinking.** Define the unit of observation before coding
   (signal-bar cohorts, events, trades). Intention-to-treat framing for
   anything with selection (the limit-order study's design — unfilled
   orders contribute 0 and stay in the cohort — is the house pattern for
   execution questions; it is what made adverse selection measurable).
2. **Controls are half the experiment.** Every effect claim needs at
   least one of: placebo (shuffled timing), random gate at matched
   frequency, ungated baseline, or a-priori-neutral benchmark. The
   bracket study's verdict was only interpretable because random gates
   (−0.40R) bracketed the model gate (−0.46R) and exposed
   worse-than-random selection.
3. **Declared grids.** Any parameter scanned (δ ∈ {0.1, 0.25, 0.5}) is
   declared before the run with the primary value marked; all grid
   results are reported.
4. **Escalating resolution.** Design on 15m bars; escalate to M1 replay
   exactly where intra-bar ordering matters (fill-vs-stop). Do not pay
   M1 cost everywhere; do not accept 15m ambiguity where it changes signs.
5. **Compute budgeting is a design input.** Fold subsets are legitimate
   (we use every-second-year) if declared ex-ante; shrinking a test after
   seeing partial results is not.
6. **Write the decision rule into the script docstring** before the first
   full run. The script prints its own verdict against that rule.

## 2.4 Validation framework (summary; full protocol in Volume 6)

* Expanding-window walk-forward with ≥ 96-bar purge is the default
  harness; rolling windows demonstrably lose here [TESTED-HERE:
  `research/cycle2/results/i1_drift.json`].
* Fold-internal validation slices (last training year) make all
  fold-level choices; test years are touched once.
* The dev window (2010–2013) is permanently burned for screening; it can
  never appear in a confirmatory claim.
* Every P&L claim goes through the event simulator with costs, blocking,
  caps, and gap handling; every timing claim gets a placebo distribution.

## 2.5 Cost modeling

* Costs are proportional round-trip bp on the entry price, applied inside
  the simulator per position leg. House tiers: **1.5bp institutional /
  2.5bp good-retail / 4bp typical-retail**; every economic result is
  reported at minimum at 2.5bp with the tier sensitivity table.
* Passive fills pay the exit leg only (house assumption: half the round
  trip = 1.25bp) — this is an **assumption**, flagged in every result it
  touches, pending order-book data [HYPOTHESIS at the number level].
* Cost-in-R = cost_frac / SL_frac is the quantity that matters for
  barrier strategies; report it per configuration (baseline G0 ≈ 0.22R at
  2.5bp; G2 ≈ 0.11R) [TESTED-HERE].
* Unmodeled: spread widening at news/rollover, latency, partial fills.
  Stated as a limitation on every affected result; addressed only by
  better data, never by optimistic assumption.

## 2.6 Leakage detection protocol

Run before any new feature/label family enters the registry:

1. **Correlation screen**: no feature may correlate with any forward
   return above |ρ| = 0.05 (our clean 205-feature set maxes far below;
   anything above is presumed leakage until proven structural).
2. **Shift test**: shifting features +1 bar must not *improve* results;
   improvement implies misalignment.
3. **Smoothed-state audit**: any filter/decoder/embedding is checked for
   forward-backward or centered computation (Viterbi, `predict_proba`,
   centered rolling windows are the recurring offenders).
4. **Boundary audit**: resampling and grouping (day/week keys) checked at
   session roll (18:00 EST) and weekend boundaries.
5. **Too-good tripwire**: any fold AUC > 0.60 or expectancy > +0.5R halts
   the pipeline for leakage investigation before anything downstream runs.

## 2.7 Negative-result reporting

A negative result is complete only when it states: the effect size CI
that was *excluded* (not just "not significant"), the mechanism now
disfavored, what the failure taught about the market or the method, and
what neighbouring hypotheses it does or does not rule out. Model: the
bracket study's negative excluded the whole stop-triggered class of
vol-monetization but explicitly not the options class; the session study
excluded both fade and follow at all tested horizons.

## 2.8 Reproducibility

* One command per experiment, deterministic seeds, environment pinned by
  `pip` (versions recorded in reports when they matter).
* Data acquisition itself is scripted (`research/data/`); raw archives
  are cached but re-downloadable; derived parquets rebuild from raws.
* Registry entries carry the git SHA of the code state that produced
  their verdict.
* The test of compliance: a fresh agent on a fresh machine reproduces any
  registry verdict from the repo alone, without asking anyone anything.
