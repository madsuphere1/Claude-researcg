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
