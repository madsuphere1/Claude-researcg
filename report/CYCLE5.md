# Cycle 5 — Window × Interval × Metric: which numbers actually inform?

> **CORRECTION (see CYCLE6.md).** Finding 3's "HELPS" ranking below
> (max_dd_R, max_consec_losses, tail_ratio as forward predictors) was
> computed from **in-sample** Spearman ρ and is **WITHDRAWN**. Cycle 6
> re-tested it out-of-sample (leave-one-out CV + permutation null): OOS
> R² ≈ 0, permutation p = 0.28 — indistinguishable from noise. The
> metric-informativeness section is superseded by Cycle 6. The
> structural findings (redundancy, sliding-interval decay) stand.
> Cycle 6 also shows the 28 metrics were not independent columns — 11 of
> them are one factor — which invalidates the flat ranking method used
> in Finding 3. Read CYCLE6.md for the corrected treatment.

The operator asked for the thing behind the thing: not "what did each
window earn" (Cycle 4 answered that) but **map every window and every
sliding interval against a full quant-metric suite, then work out which
metrics carry real information and which are decorative.** This is a
mapping study (registry C5-001, exploratory) — it characterises a
surface and generates hypotheses; it confirms nothing. Small-sample
caveats are load-bearing here and are stated where they bite.

## The grid

* **Training windows** (how the model is fit): expanding, rolling-5,
  rolling-4, rolling-3, rolling-2 — the full ladder.
* **Evaluation intervals** (what slice performance is measured over):
  full history + sliding 2-, 3-, 5-year spans stepped by 1 year.
* **28 metrics** per cell (`research/cycle5/metrics.py`), spanning
  return, risk, tail, and path families.

Trade streams: expanding 748, rolling-5 598, rolling-4 807, rolling-3
602, rolling-2 628.

## Finding 1 — full-history suite: only expanding is alive, and the ordering isn't clean

| Metric | Expanding | Roll-5 | Roll-4 | Roll-3 | Roll-2 |
|---|---|---|---|---|---|
| Expectancy R | **+0.106** | −0.037 | −0.022 | −0.032 | −0.083 |
| Win rate | 0.467 | 0.410 | 0.413 | 0.415 | 0.392 |
| Profit factor | 1.20 | 0.94 | 0.96 | 0.95 | 0.86 |
| Sharpe (per-trade) | +0.089 | −0.031 | −0.018 | −0.027 | −0.071 |
| Max DD (R) | **−29.3** | −53.7 | −50.6 | −33.3 | −69.0 |
| Ulcer index | **10.3** | 26.3 | 24.4 | 12.6 | 23.5 |

Expanding is the only profitable window on every metric. Among the
rolling windows the ordering is **not** cleanly monotone (rolling-4
−0.022 edges rolling-5 −0.037), which is itself honest evidence: below
"all history" the differences between rolling lengths are inside the
noise. The one clean monotone signal is **risk**, not return — max
drawdown and ulcer index blow out the moment you leave expanding
(−29R → −50R+), because shorter windows trade noise with conviction and
string together deeper losing runs.

**A structural observation that matters for metric selection:** several
metrics are *nearly identical across all five windows* — avg_win_R
(~1.34), payoff_ratio (~1.35), var95 (~−1.05), cvar95 (~−1.06),
tail_ratio (~1.40), best_R/worst_R. These are pinned by the TP/SL
barrier geometry, which is frozen across windows. A metric that cannot
move when the model changes **cannot carry information about which model
is better** — it is structurally blind. Any dashboard that leans on
payoff ratio or average win to compare models is reading a constant.

## Finding 2 — sliding intervals: the edge decayed for EVERY window

The 2-, 3-, 5-year sliding grids (full tables in
`research/cycle5/results/c5_intervals.json`) all tell one story. Early
intervals are strongly positive for expanding and mixed-positive for
rolling; **late intervals are negative for everyone.** The 3-year grid,
recent rows:

| Interval | Expanding | Roll-5 | Roll-4 | Roll-3 | Roll-2 |
|---|---|---|---|---|---|
| 2019-2021 | +0.159 | −0.065 | −0.012 | −0.006 | −0.006 |
| 2020-2022 | +0.146 | −0.084 | +0.005 | −0.028 | −0.034 |
| 2021-2023 | +0.019 | −0.052 | +0.185 | −0.005 | −0.031 |
| 2022-2024 | −0.107 | −0.037 | −0.128 | −0.052 | −0.105 |
| 2023-2025 | −0.082 | −0.043 | −0.122 | −0.090 | −0.119 |
| **2024-2026** | **−0.056** | −0.082 | −0.076 | −0.177 | −0.285 |

Two conclusions the operator's original hypothesis has to face:

1. **The recent slump is a time/regime problem, not a window problem.**
   Every window is negative in 2022-2026. No training-window choice fixes
   it, because the market simply hasn't presented the volatility regime
   this strategy monetises. Expanding is merely the *least* negative.
2. **Rolling never rescues the recent period at any length.** In the
   headline 2024-2026 window, expanding (−0.056) beats every rolling
   variant, and the shorter the window the worse it gets
   (rolling-2 −0.285). The "recent data fits recent market better"
   intuition is refuted interval-by-interval, not just on average.

## Finding 3 — which metrics actually inform (the core question)

For each metric, pooled across all windows, over 42 (trailing-2yr →
next-year expectancy) pairs: does the trailing value **predict** next
year (forward ρ)? Is it **stable** enough to act on (lag-1 autocorr)?
Does it **separate** the profitable window from the dead ones
(cross-policy discrimination)?

| Metric | Forward ρ | Persistence | Discrimination | Class |
|---|---|---|---|---|
| max_consec_losses | −0.352 | +0.62 | +0.41 | **HELPS** |
| max_dd_R | +0.337 | +0.51 | +0.90 | **HELPS** |
| tail_ratio | +0.307 | +0.30 | +0.21 | **HELPS** |
| var95 / cvar95 | +0.32 / +0.31 | ~0.27 | ~−0.1 | weak |
| kurtosis | −0.307 | +0.30 | −0.90 | weak |
| win_rate | +0.216 | +0.49 | +0.90 | weak |
| expectancy / PF / Sharpe / Sortino / omega | **+0.140** | +0.47 | **+1.00** | weak |
| median_R, avg_win_R, payoff_ratio, downside_dev, max_consec_wins | ≈0 | mixed | mixed | **DECORATIVE** |

This is the result worth the whole cycle:

**Return metrics describe the past perfectly and predict the future
barely.** Expectancy, profit factor, Sharpe, Sortino, omega all have
discrimination = +1.00 (they rank the five windows by profitability
flawlessly — of course, they *are* profitability) but forward ρ of only
+0.14. Knowing a window's past Sharpe tells you almost nothing about its
next-year return. This is the entire program's thesis, quantified: **past
performance is not a forward signal here.**

**Risk and tail metrics are the ones that carry forward information.**
The three HELPS metrics are all about *losing*: maximum drawdown,
longest losing streak, and tail ratio. Trailing drawdown predicts
forward expectancy (ρ=+0.34) *and* is persistent (0.51) *and*
discriminates (0.90) — the only metric strong on all three. Notably,
max_consec_**losses** informs (ρ=−0.35) while max_consec_**wins** is
decorative (ρ=−0.01): downside clustering carries signal, upside
clustering is noise. Risk is more forecastable than return.

**Decorative metrics** — median_R (always ≈−1 because win rate < 50%, so
the typical trade is a full loss and the median never moves),
payoff_ratio and avg_win_R (barrier-geometry constants), max_consec_wins
— should be dropped from any model-selection dashboard. They neither
predict nor discriminate.

### The load-bearing caveat
42 pooled pairs, 23 metrics scanned. A forward ρ of ~0.33 is suggestive,
not established; with this much multiplicity the three HELPS metrics are
*hypotheses*, not verdicts. The honest, pre-registered expectation was
"most metrics won't predict forward" and that is exactly what happened —
20 of 23 are weak or decorative. The three survivors (drawdown, losing
streak, tail ratio) become the pre-registered watch-list for out-of-
sample data (C3-008 / 2027+), not a validated selection rule.

## Action items (carried forward)

1. **Model/window selection should be driven by risk metrics, not return
   metrics.** Rank candidates by trailing max-drawdown and losing-streak
   behaviour; treat Sharpe/PF as description, not forecast. [→ P10, feeds
   the production-monitoring design in Volume 9.]
2. **Drop structurally-constant metrics** (payoff ratio, avg win, tail
   percentiles fixed by barrier geometry) from any window-comparison
   dashboard — they are blind by construction.
3. **The recent-regime drag is universal**, so the lever is regime
   *detection/entry*, not window length — reinforces P2's standing
   regime-transition hypothesis (C3-005) over any further window search.
4. **C3-008 gains a metric watch-list:** when 2027+ data lands, evaluate
   drawdown / consec-losses / tail-ratio as forward predictors, pre-
   registered, alongside the expanding-vs-rolling P&L test.

Registry: C5-001 closed (exploratory, mapping). No production gate
touched. Simulated results; not investment advice.
