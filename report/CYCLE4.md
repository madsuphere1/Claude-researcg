# Cycle 4 — The Window Question

One operator hypothesis, tested as a proper cycle: **does a rolling
(recent-only) training window beat the house expanding (all-history)
window, because recent structural change makes old data harmful?**

This cycle exists because the operator pushed back on the house rule
with a specific, legitimate argument: the strategy has lost money for
two straight years (see the trailing-window analysis), and maybe that's
because models trained on 2010–2021 data no longer fit a changed 2024–
2026 market. That is a real, testable claim, and H-I1 (cycle 2) had only
tested it at the AUC level, not at the dollar/expectancy level of the
full X1 strategy. So it earned a cycle.

## What was run

`research/cycle3/screen_rolling_x1.py` — the frozen X1 composite spec,
changed in exactly one place: the classifier's training window. Three
variants on the **exploratory even-year slice** (2014–2026 even years;
the odd-year confirmation slice was deliberately left untouched):

* **Expanding** (house rule): train on all data from 2010 to year−1.
* **Rolling-5**: train on the trailing 5 years only.
* **Rolling-2**: train on the trailing 2 years only (the operator's
  "keep it to ~2 years" instinct, made concrete).

Everything else — G2 barriers, HMM vol-gate, limit entry, risk caps,
thresholds protocol — held identical. Multiplicity (2 rolling variants)
declared before running. This is a **screen**: it generates a hypothesis
verdict on the exploratory slice; it does not confirm anything. The
confirmatory arbiter is registry **C3-008**, frozen for 2027+ data.

## Result: the shorter the window, the worse the strategy

| Training window | Trades | Net expectancy | Win rate |
|---|---|---|---|
| **Expanding (house)** | 414 | **+0.128R** | 47.6% |
| Rolling-5 years | 358 | **−0.058R** | 39.9% |
| Rolling-2 years | 389 | **−0.145R** | 36.2% |

Cutting history didn't rescue the strategy — it **inverted** it. A
configuration that earns +0.128R on this slice becomes a −0.058R loser
at a 5-year window and a −0.145R loser at 2 years, monotonically. On the
headline question, the house rule wins decisively and the operator
hypothesis is **refuted on the exploratory slice**.

**The mechanism is visible in the thresholds.** Rolling models had to
raise their probability threshold to 0.76–0.90 to trade at all (expanding
sits at 0.55–0.72). A short window can't find the faint AUC-0.53 signal,
so the model is uncertain everywhere and only fires at extreme
confidence — fewer, noisier trades. This is textbook **overfitting to a
short recent sample**, the exact opposite failure from the "underfitting
by using too much data" intuition. At this signal-to-noise, data is the
scarce resource, not recency.

## Where the operator was right — and why it still doesn't carry

The honest part. The claim was specifically about *recent* years, so
here are only the recent even years, which is where the argument lives:

| Year | Expanding | Rolling-5 (n) | Rolling-2 (n) |
|---|---|---|---|
| 2024 | −0.063R | **+0.351R** (21) | −0.519R (11) |
| 2026 | −0.069R | −0.111R (169) | −0.341R (101) |

In **2024**, rolling-5 genuinely beat expanding, and by a lot
(+0.351R vs −0.063R). That is the operator's thesis showing up in real
data — not noise-free, but real. If 2024 were the whole story, the
house rule would be in trouble.

It isn't the whole story. **2026 — the most recent year and an 8×
larger sample (169 trades vs 21) — went the other way**: rolling-5
(−0.111R) was *worse* than expanding (−0.069R), and rolling-2 was worse
still (−0.341R). So the recent-regime argument scores one win on 21
trades and one loss on 169 trades. Weighted by evidence, it does not
survive — but it is not zero, and that is exactly why C3-008 exists
rather than closing the question here.

## A second finding, unavoidable once dissected

Expanding's +0.128R even-year edge is **almost entirely 2016**
(+0.413R). Every other even year is near zero or negative (2022 alone is
−0.543R). This is the regime-concentration lesson (C3-001) again: the
edge is not spread across time, it clusters in a few volatility
episodes. It reframes the operator's original worry correctly — the
problem with recent performance is **not** that old data poisons the
model; it's that recent years simply haven't contained the volatility
regime this strategy monetizes. A shorter window doesn't manufacture
that regime; it just fits noise while waiting for it.

## End-to-end, all years, full metrics (the complete run)

The screen above used even years only. The operator then asked for the
*whole* thing end-to-end. `strategy/window_comparison.py` runs all three
window policies over the full 2014–2026 span as one continuous
simulation, identical treatment, complete metrics suite:

| Window | Trades | Exp/R | Win% | PF | Sharpe | maxDD | p(day) | @2.5bp |
|---|---|---|---|---|---|---|---|---|
| **Expanding** | 748 | **+0.106** | 46.7% | 1.20 | +0.68 | −26.2% | **0.008** | +0.068 |
| Rolling-5 | 598 | −0.037 | 41.0% | 0.94 | −0.21 | −43.3% | 0.775 | −0.072 |
| Rolling-2 | 628 | −0.083 | 39.2% | 0.86 | −0.49 | −51.4% | 0.961 | −0.121 |

Expanding is the *only* profitable policy, and the only one that is
statistically distinguishable from zero (day-block p=0.008). Rolling
windows are negative-expectancy on **every** metric at once —
expectancy, win rate, profit factor below 1, negative Sharpe, and
catastrophic drawdowns (−43% and −51% vs expanding's −26%; those alone
would end most accounts). The degradation is strictly monotone in how
much history you throw away. At realistic 2.5bp costs the rolling
variants sit at −0.072R and −0.121R — bleeding money on every trade.

Per-year expectancy (R/trade, 1.25bp, n in parens):

| Year | Expanding | Rolling-5 | Rolling-2 |
|---|---|---|---|
| 2020 | +0.149 (200) | −0.135 (97) | −0.016 (32) |
| 2021 | +0.172 (114) | +0.047 (65) | +0.038 (148) |
| 2023 | −0.105 (118) | −0.075 (83) | +0.050 (19) |
| 2024 | −0.063 (46) | **+0.351 (21)** | −0.519 (11) |
| 2025 | −0.037 (43) | −0.193 (37) | −0.090 (42) |
| 2026 | −0.069 (34) | −0.111 (169) | −0.341 (101) |

The recent-years verdict is now complete and it holds against the
operator hypothesis on the weight of evidence: of the last three years,
rolling-5 beat expanding in **one** (2024, 21 trades) and lost the other
two, including 2026 where it traded 169 times and still lost more than
expanding. Rolling didn't just fail to help the recent slump — in 2026
it manufactured 5× more trades and lost money on them, turning a
−0.069R drift into a −0.111R bleed. That is the short-window failure
mode in one cell: more confident, more active, more wrong.

## The research cycle redone per window (the root cause, upstream of P&L)

The P&L tables above answer "does the traded money differ by window."
The operator then asked the deeper question: redo the *research cycle*
itself — the foundational C1-BASE test ("does a predictive signal exist
at all?") — for each window. `research/cycle3/window_research_cycle.py`
computes out-of-sample walk-forward AUC by year on the G2 label, per
window (same folds, same labels, only the training window differs):

| Window | Mean AUC | Years > 0.5 | Sign-test p | Worst yr | Best yr |
|---|---|---|---|---|---|
| **Expanding** | **0.5195** | 11/13 | **0.011** | 0.486 | 0.548 |
| Rolling-5 | 0.5121 | 10/13 | 0.046 | 0.486 | 0.534 |
| Rolling-2 | **0.5012** | **6/13** | **0.709** | 0.468 | 0.523 |

This is the cleanest result in the cycle. The signal doesn't just earn
less money on shorter windows — **it ceases to exist**. Expanding shows
a real, significant edge (AUC 0.5195, positive in 11 of 13 years,
sign-test p=0.011). Rolling-5 is a degraded but still-detectable edge
(p=0.046, right at the margin). Rolling-2 is **statistically
indistinguishable from a coin** — mean AUC 0.5012, positive in only 6 of
13 years, sign-test p=0.709 (i.e. this pattern is exactly what pure
noise produces). The degradation is monotone: 0.5195 → 0.5121 → 0.5012,
marching straight to 0.5.

That is *why* the rolling P&L was negative — not bad luck, not costs, but
the absence of any signal to trade. A 2-year window cannot resolve an
AUC-0.52 edge because there isn't enough data in two years to separate a
signal that faint from noise; the model fits the noise and trades it.
This is the definitive refutation of the "too much history = underfit"
premise: at this signal-to-noise, **history is what makes the signal
visible at all**, and starving the model of it erases the very thing the
research cycle was built to detect.

(Note: mean AUC here is 0.5195 for expanding vs the 0.529 headline in
C1-BASE because this uses the G2 barrier label — TP 3×ATR / SL 2×ATR /
64-bar — not C1-BASE's baseline 1.5×/1× / 16-bar label. The cross-window
comparison is apples-to-apples; the small absolute offset is the
different target, and is not the point.)

### Correction: the flat mean is itself an aggregate — dissect it

The operator caught a real fallacy in the paragraph above. Declaring
rolling-2 "no signal" from a *full-history mean* AUC of 0.5012 is exactly
the aggregate-hides-concentration error the Constitution forbids: C3-001
proved this edge lives in volatility episodes, so a window could look
dead on average yet carry a real edge inside its own edge-case regime.
The mean was not enough; the claim had to be tested by regime.

`research/cycle3/window_edge_dissection.py` splits each window's test-year
AUC by the HMM high-vol gate (the edge-case selector the strategy
actually trades; the gate is identical across windows, so the split is
clean):

| Window | Gated AUC | Gated yrs>0.5 | Sign p | Ungated AUC | Best gated yr |
|---|---|---|---|---|---|
| Expanding | 0.5255 | 9/13 | 0.133 | 0.5172 | 0.571 (2017) |
| Rolling-5 | 0.5225 | 10/13 | **0.046** | 0.5106 | 0.579 (2017) |
| Rolling-2 | 0.5112 | 7/13 | **0.500** | 0.4984 | 0.567 (2018) |

Two things come out of the dissection, and the first vindicates the
operator:

1. **The gate concentrates signal for every window** — gated AUC exceeds
   ungated AUC in all three rows. The regime structure is real; the flat
   mean *was* washing it out, exactly as warned. Individual edge-case
   years exist even for rolling-2 (0.567 in 2018, 0.534 in 2015, 0.554 in
   2020).

2. **But the conclusion survives the correct test.** Even inside its
   edge-case regime, rolling-2's good years do not persist: 7/13 above
   0.5, sign-test p = 0.500 — a coin. Its high years are scattered, not a
   stable exploitable regime. Rolling-5, by contrast, keeps a marginal
   real signal in the gate (10/13, p = 0.046), which is why C3-008 stays
   open for rolling-5 and not for rolling-2.

So the corrected, properly-earned statement is *not* "rolling-2 is dead
because its average is 0.50." It is: **rolling-2 has no edge that
survives regime dissection — its edge-case years exist but do not repeat,
which is the signature of noise, not a hidden regime.** The stronger
claim needed the dissection to stand; the weaker mean-only claim was a
fallacy, and the operator was right to reject it.

## Verdict and what changes

* **Screen verdict: rolling windows REFUTED on the exploratory slice.**
  Monotone degradation with shorter windows; the house expanding/annual-
  retrain rule stands. Recorded in registry alongside H-I1.
* **C3-008 remains open and standing** — expanding vs rolling-5 on 2027+
  data, decision rule frozen before that data exists. The one 2024 data
  point in the operator's favor is enough to keep the question alive on
  genuinely new data, which is the only place it can be settled without
  either of us fitting to what we already saw.
* **Nothing about the strategy's economics changed.** It was WEAKENED
  before this cycle and is WEAKENED after. Cycle 4 closed a methodology
  question (how to train), not the open economic question (does the edge
  survive real costs in the current regime). Registry now: 20 closed
  verdicts, 2 standing/open. Simulated results; not investment advice.
