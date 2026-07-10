# Volume 8 — Execution Engine

Execution is where this program found most of its economics. Net
expectancy = gross − cost-in-R, and both cost factors (bp paid, SL
denominator) proved more tractable than the signal itself
[TESTED-HERE, cycle 2].

## 8.1 Cost model

* Proportional round-trip cost on entry price, per position leg.
* Tiers: **1.5bp institutional / 2.5bp good-retail / 4.0bp typical
  retail** (XAUUSD CFD/spot, majors venues). Every economic claim reports
  ≥ the 2.5bp figure + the tier table.
* Passive fills: exit-leg-only cost, house figure 1.25bp — an assumption
  pending order-book evidence; flagged wherever it moves a verdict.
* Known unmodeled: spread widening at news/rollover (biases against
  event-window strategies being tested optimistically — treat any
  event-window profit with extra suspicion), latency, partial fills.
* Breakeven bookkeeping: cycle-1 spec breakeven ≈ 1.3bp; G2 geometry
  ≈ 2.0bp; X1 composite (passive fills) sits at the edge of viability at
  good-retail costs. These three numbers are the program's economic map.

## 8.2 Barrier geometry

Cost-in-R = cost_frac / SL_frac. Widening SL from 1× to 2×ATR halves the
friction share; gross edge fell slower (monotone G0→G1→G2)
[TESTED-HERE]. Constraints discovered with it: longer horizons cross
weekends (Volume 7 rules now bind) and slow trade frequency (~0.25/day
at X1 spec). Wider-still geometries (SL 3×ATR+, multi-day) are untested
[HYPOTHESIS] and interact with the overnight-risk reopening of §7.4.

## 8.3 Passive entry (limit orders)

The validated spec: limit at signal close − δ·ATR (mirrored short),
δ = 0.10 primary; cancel after 4 bars; fill requires trade-through by
≥ 1 tick (no queue-position credit — conservative); same-bar fill-vs-stop
ordering resolved on M1; intention-to-treat accounting.

Measured economics [TESTED-HERE: b1_limit_entry.json]:

| δ | Fill rate | ITT uplift | Adverse selection (filled vs missed counterfactual) |
|---|---|---|---|
| 0.10 | 88% | +0.088R | −0.24R vs +0.85R |
| 0.25 | 77% | +0.083R | −0.37R vs +0.79R |
| 0.50 | 58% | +0.069R | −0.61R vs +0.64R |

Lessons: adverse selection is real, large, and *measurable* with the ITT
design; shallow limits dominate (missed-winner cost grows faster than
price improvement as δ deepens). Queue modeling, partial fills, iceberg
behaviour: open, need better data (P7 agenda).

## 8.4 Order types in simulation

* Market: fills next bar open, full spread cost.
* Limit: trade-through rule above.
* Stop (bracket entries): fills at stop level + spread; the bracket
  study's whipsaw evidence [TESTED-HERE: 35% double-trigger on expansion
  bars] is a standing warning for any stop-entry design at 15m.
* All simulations: one position, pessimistic ties, pre-gap flatten.

## 8.5 VWAP/TWAP, scale-in/out, adaptive trailing

All [HYPOTHESIS] here: no volume data for true VWAP (TWAP proxy exists as
a feature); scale-in/out designs are welcome registry entries (note: they
multiply ticket costs — the ITT framework extends naturally); trailing
stops interact with tie-break ambiguity (see §7.3).

## 8.6 Execution research protocol

Any execution claim uses: intention-to-treat cohorts over identical
signal sets; paired bootstrap on per-signal differences; the market-entry
counterfactual computed per signal; M1 escalation where ordering matters;
and reports fill rate + adverse-selection decomposition, not just the
headline uplift. (This is the H-B1 design, promoted to protocol.)

## 8.7 Simulator custody

The event engine (`research/backtest.py` + cycle-2/3 variants) is owned
by the Execution Engineer: assumption changelog in-file, anchor-set
re-runs on every change (§5.7), and a standing TODO list of fidelity
upgrades ranked by sign-flipping potential (current top: full-M1 replay
of composite trades; spread-widening model).
