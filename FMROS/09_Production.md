# Volume 9 — Production Trading

Nothing in this volume is active. It becomes active only when a strategy
reaches CONFIRMED — which, as of this writing, none has (X1 is WEAKENED:
right-signed on both even-year pilot and odd-year confirmation, but
significance unmet; see registry C3-001). This volume exists so the path
is defined *before* anyone is excited enough to cut corners.

## 9.1 The gate sequence

```
CONFIRMED verdict (Vol. 5)
  → G1 Implementation audit
  → G2 Paper trading
  → G3 Risk & account audit
  → G4 Shadow trading
  → G5 Limited live
  → G6 Full allocation
```

No gate may be skipped; each gate's report is a registry artifact; the
operator (human sponsor) signs G3, G5, G6 personally.

### G1 — Implementation audit
Strategy re-implemented as a single deployable module (signal → order
intents) decoupled from research code; Reviewer + Auditor verify
tick-for-tick agreement with the research simulator on the confirmation
period (tolerance: identical trades, P&L within float noise).

### G2 — Paper trading (≥ 3 months or ≥ 50 trades, whichever is longer)
Live data feed, simulated fills under the house fill model. Pass criteria
(pre-declared here): trade frequency within 2× of backtest expectation;
expectancy's 80% CI overlapping the backtest estimate; zero causality
violations (any use of unavailable-at-decision-time data found = fail);
all risk rules observed mechanically.

### G3 — Risk & account audit
Volume 7 ruin analysis at the actual account's leverage/margin; kill
switch parameterization (below); capital allocation sized so the P95
resampled maxDD ≤ the operator's stated tolerance.

### G4 — Shadow trading (≥ 1 month)
Real orders at minimum size (or broker demo with real spread feed if
minimum size is unavailable). Purpose: measure implementation shortfall —
real spread paid, fill rates on passive orders vs the 1.25bp/trade-through
assumptions. Pass: measured costs ≤ 1.25× assumed; passive fill rate
≥ 0.8× assumed. Shortfall data feeds back to P5 regardless of pass/fail.

### G5 — Limited live (≥ 3 months, ≤ 20% of target allocation)
Full risk rules; monitoring below; monthly report to operator. Pass:
performance within the backtest's day-block 90% cone; no manual
interventions required.

### G6 — Full allocation
Operator decision, informed by G5 report. Never automatic.

## 9.2 Monitoring (from G2 onward)

Daily health check (automated, one page):
* Data feed integrity (gaps, stale quotes, clock skew).
* Positions/orders vs strategy intent reconciliation.
* Risk-rule compliance (leverage, daily stop, trade cap).
* Rolling 20-trade expectancy + CUSUM against backtest distribution.
* Fill quality vs assumption (passive fill rate, realized spread).
* Regime state + gate fraction (is the strategy trading at expected
  frequency for the regime?).

Weekly: equity vs backtest cone; slice table (session/regime) drift.

## 9.3 Performance-decay detection

Distribution-based drift (PSI) is **known-bad** as a trigger here
[TESTED-HERE: i1_drift.json]. The house method is outcome-based:
* CUSUM on trade R-multiples against the confirmation-period mean;
* alarm when the rolling 50-trade expectancy's 90% CI excludes the
  confirmation point estimate from below.
An alarm pauses new entries (positions close per rules) and files a
diagnosis task; resumption requires a Chief Scientist memo.

## 9.4 Automatic disable (kill switches)

Hard, unattended, no-override-without-operator:
1. Account drawdown from high-water ≥ the G3-declared budget (default 20%).
2. Daily loss ≥ 2× the daily stop (implies rule breach or gap event).
3. Data feed anomaly (no ticks > 5 min in-session; clock skew > 2s).
4. Order-reconciliation mismatch (unknown position or unacked order).
5. Fill-quality collapse: realized cost > 2× assumption over 20 trades.
6. Any exception in the decision path (fail closed, flat).

Every trigger event gets a post-mortem in the registry before re-enable.

## 9.5 Portfolio & capital controls

Single-strategy era: allocation per G3/G5/G6 above. Multi-strategy era
(when ≥ 2 CONFIRMED strategies exist): correlation-of-trade-streams
analysis (P9) gates combined allocation; combined P95 maxDD budgeted the
same way; per-strategy kill switches remain independent.

## 9.6 Retraining in production

Annual expanding-window retrain (the validated cadence), executed as a
G1-mini: new model must reproduce the walk-forward protocol exactly,
pass the anchor-set regression, and paper-trade for 2 weeks before
swapping. Mid-year retrains only via the §9.3 alarm path.

## 9.7 The current deployment position (honest statement)

As of 2026-07: **nothing qualifies for G1.** The X1 composite would need
either (a) 2027+ out-of-sample confirmation reaching its pre-declared
significance, or (b) a materially cheaper execution channel (≤1.5bp
all-in, at which the geometry lever alone is near water) plus a fresh
confirmation. Operating this volume's machinery on an unconfirmed
strategy would violate the Constitution, and the correct current use of
production infrastructure is G4-style *shadow measurement* of real
spreads and passive fill rates on XAUUSD — which is simultaneously the
data acquisition P5/P7 most need. That is the recommended first
production-track action, and it risks nothing.
