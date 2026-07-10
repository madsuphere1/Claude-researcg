# Volume 7 — Risk Engine

Risk rules are part of a strategy's specification, not an afterthought —
and equally, risk rules cannot create edge (proven by construction and
simulation [TESTED-HERE: gh_sizing.json]). This volume fixes the house
rules and the analyses every strategy owes.

## 7.1 House risk rules (current defaults; changes are registry events)

1. **Per-trade risk**: 1% of compounding equity, sized off the SL
   distance. R-multiples are reported against this risk unit.
2. **Leverage cap**: notional/equity ≤ 10×. Rationale: inverse-ATR sizing
   explodes leverage exactly when vol is low and gap-jump risk per dollar
   of stop-distance is highest [TESTED-HERE: the unmanaged-G2 blowup].
3. **Pre-gap flatten**: no position held across any >2h session gap
   (weekends, holidays). Exits at the last pre-gap bar close.
4. **No new entries** Friday ≥ 15:00 EST or into a detected upcoming gap.
5. **Daily loss stop**: −3% equity halts new entries for the trading day.
6. **Trade cap**: ≤ 5 entries per trading day (mandate constraint; the
   validated strategies use ~0.2–0.8).
7. **One position at a time** per instrument (portfolio rules will
   supersede when P9 matures).

## 7.2 Sizing policies (validated menu)

| Policy | Status | Evidence |
|---|---|---|
| Flat 1% | Default | Baseline everywhere |
| Probability-weighted (0.5%–2% by signal margin) | Validated improvement | MAR 0.99→1.06 on gross stream |
| DD-throttle (×0.5 below −10%, ×0.25 below −20%) | Validated improvement | MAR →1.07; big loss-slowing on negative streams |
| Quarter-Kelly (clipped 0.25%–2%, 300-trade window) | Validated but undersizes | CAGR 7% vs 26%; use only if tail-aversion dominates |

Kelly notes [ESTABLISHED + TESTED-HERE]: trade-level variance overstates
risk relative to day-block structure here, so raw Kelly fractions are
mis-calibrated; any Kelly use must estimate on block-resampled growth,
not i.i.d. trades.

## 7.3 Dynamic stops / targets

Current evidence: fixed ATR-multiples (1.5/1.0 and 3/2) are the tested
geometries; cost-in-R falls ∝ 1/SL [TESTED-HERE]. Trailing/break-even
logic is **untested** here [HYPOTHESIS] — registry entries welcome, with
the warning that trailing interacts with the same-bar ambiguity rules
(design must specify tie-breaks pessimistically or use M1 replay).

## 7.4 Gap risk (the house tail mechanism)

Every strategy spec must answer, before simulation results are believed:

* What is the maximum holding period, and how often does it cross
  weekends/holidays?
* What is position leverage at the low-ATR percentile (P5) of entries?
* What is the worst historical gap × that leverage? (Gold weekend gaps
  exceed 2% several times per decade.)

The Risk Manager signs this analysis. The pre-gap flatten rule makes it
moot for current strategies; any future overnight/持仓 strategy re-opens it.

## 7.5 Loss-streak and ruin analysis (owed by every strategy)

* Max consecutive losses observed + its day-block resampled P95.
* P(maxDD ≥ 20% / 50%) under the strategy's sizing rule, N ≥ 2000
  day-block resamples.
* Time-to-recover distribution.
* Regime concentration of losses (which state, which years).

Reference numbers: cycle-1 baseline (negative stream) hits 50% DD with
probability ~1; gross stream at flat 1% ~0.1%; X1-family strategies
publish theirs in the registry entries.

## 7.6 Regime-dependent risk

The gate probability from P2 is a legitimate sizing input (it is causal).
Validated so far only as a binary gate; graded sizing on P(state) is
[HYPOTHESIS] with a ready design (prob-weighted policy composed with the
gate).

## 7.7 Probability of ruin / liquidation

For leveraged CFD accounts, ruin is not equity→0 but margin-call at the
broker's maintenance level. The ruin analysis of §7.5 must therefore be
run at the *account's* leverage and margin parameters when a deployment
is proposed (Volume 9 gate G3 consumes this).

## 7.8 The negative-stream sanity check

Any proposed risk/sizing rule is run against the cycle-1 negative stream.
If it turns that stream profitable, the implementation is broken (fail
closed). This is a standing regression test, not a one-off.
