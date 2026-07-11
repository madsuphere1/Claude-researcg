# Pattern-Expert Router — 16 learned patterns with per-pattern TP/SL

Operator design: describe each candle by context, cluster into 16
patterns ("16 models"), route each bar to its pattern, and set direction
+ TP + SL from how that pattern historically behaved. Built rigorously:
KMeans(16) fit on train only, per-pattern rule grid-searched on train
(side × TP × SL in ATR units), applied walk-forward out-of-sample.
Instrument XAUUSD 15m, 2010–2026, cost 2.5 bp, pessimistic barrier ties.
`pattern_experts.py`.

## Result — REFUTED, and not by cost

| Selection | OOS gross | OOS net @2.5bp | notes |
|---|---|---|---|
| train net>0 (strict) | — | **−0.151R** | only 0–1 of 16 patterns qualify/yr; those lose |
| train gross-best (all 16) | **−0.047R** | −0.184R | win 41.7%, 340k trades |

The key number is **OOS gross −0.047R**: *before any cost*, the
train-selected pattern rules are already net-negative out-of-sample. So
this is **not** a cost problem. It is a **stationarity** problem: the
forward behavior of a pattern (its direction, its MFE/MAE, hence its
best TP/SL) measured on training data **does not persist** into the test
year. The setup that ran up last year fades this year.

## Why the idea failed here (and it's the operator's own earlier lesson)

The router learns **one behavior per pattern, averaged over all training
history** — precisely the "fit all history into one thing" move the
operator warned against. A pattern's follow-through is itself
regime-dependent: "breaks the 4h low" mean-reverts in a range and
continues in a downtrend. Collapsing that into a single TP/SL per
pattern throws away the regime that actually determines the outcome, so
the rule is right on average and wrong in every specific regime — which
nets to no OOS edge.

## What this does NOT claim
One implementation, one instrument, KMeans clusters, 16-bar horizon. It
refutes *this* pattern-router, not the concept universally. Honest
successors: (1) condition each pattern on the prevailing regime
(pattern × trend-state), so behavior is learned per regime not globally;
(2) test on BTC where order-flow adds real signal (B1-FLOW); (3) require
a pattern's train edge to be *stable across sub-periods* before trusting
it (a stationarity filter). All are queued, none are claims yet.
