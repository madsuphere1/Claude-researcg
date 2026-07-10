# XAUUSD 15-Minute Edge Research

**Question under test:** does a statistically significant, economically exploitable
trading edge exist on XAUUSD (gold) 15-minute bars, tradeable at ≤5 trades/day
with 1% risk per trade?

**Methodology:** hypothesis-driven, walk-forward, cost-aware. Nothing is assumed
to work; every claim below is backed by an out-of-sample experiment in this
repository, and failed experiments are reported alongside successful ones.

---

## 1. Executive Summary

**Verdict: a real but small predictive signal exists; it does not survive
realistic retail transaction costs. No tradeable retail edge was found.**

1. **A small statistical edge exists.** Gradient-boosted trees predicting
   "TP before SL" (1.5R/1R, 4-hour horizon) achieve walk-forward AUC above
   0.50 in **all 13 test years (2014–2026)**, mean **0.529**. This is far
   too consistent to be luck (see §6), but it is *small* — the market is
   nearly efficient at this frequency.
2. **The edge is mostly volatility knowledge, not directional foresight.**
   Excursion-magnitude targets (MFE/MAE) are the most predictable
   (AUC ≈ 0.55); pure direction is barely predictable (≈ 0.52–0.53).
3. **Half the classic indicator canon adds nothing.** 101 of 205 candidate
   features have zero-or-negative permutation importance out-of-sample —
   including ADX, Aroon, most Donchian/candle-pattern features, and
   day-of-week flags.
4. **The signal decays over time.** AUC falls from ≈0.55 (2014–16) to
   ≈0.51 (2024–25) — consistent with broad market efficiency gains.
5. **Costs are the binding constraint.** Frictionless the strategy earns
   +0.114R/trade (Sharpe 1.38, 18× in 12.5 years); at 2.5bp round-trip it
   loses −0.105R/trade (95% CI [−0.150, −0.061]). Breakeven ≈ 1.3bp — 
   institutional execution territory. The model's timing beats a
   frequency-matched placebo by +0.13R/trade (p<0.01): real skill,
   insufficient magnitude.

---

## 2. Dataset

| Item | Value |
|---|---|
| Source | HistData.com free ASCII M1 archives (bid-quote bars) |
| Raw granularity | 1-minute OHLC, 2009-03-15 → 2026-06-26 |
| Rows (M1) | 6,052,168 after dedup |
| Research bars | 407,749 15-minute bars |
| Clock | US Eastern Standard Time, fixed UTC-5 (trading day rolls 18:00) |
| QC | 0 OHLC violations, 0 non-positive prices, no month with <55% coverage, 5 one-minute bars with \|return\|>2% (kept; verified real events) |

Known data limitations (all handled explicitly):

* **No true tick volume** — HistData M1 zeroes the volume field. We built
  activity proxies from intra-bar M1 movement instead: realized variance,
  absolute-move sum, up/down-minute imbalance, minute-count per 15m bar.
* **No historical spread series** — costs are modelled as proportional
  round-trip bp with sensitivity analysis (§7).
* **Economic calendar**: FOMC decision days scraped from federalreserve.gov
  (≈8/yr; a few meetings in 2017/2023/2024 are missed by parsing-variant
  noise; 2020 includes the two real emergency meetings). NFP approximated
  as first-Friday-of-month 08:30 ET. Other releases are captured only
  through time-of-day features — a stated limitation.
* 2009 data is discarded as indicator warm-up; research window is
  **2010-01 → 2026-06**.

## 3. Feature Engineering

205 candidate features across nine families (`research/features.py`), all
computed strictly from information available at bar close, ATR- or
price-normalised for stationarity across a 4× price range:

| Family | Count (approx) | Examples |
|---|---|---|
| Returns & distribution | 20 | log-returns 1–480 bars, rolling skew/kurtosis, autocorrelation, variance-ratio |
| Candle anatomy | 17 | body/wick fractions, gap size, engulfing, inside/outside, close streaks |
| Trend | 22 | EMA(8–200) distances & slopes, ribbon alignment, ADX, Aroon, LR slope/R² |
| Momentum | 13 | RSI(7/14/28), MACD, stochastics, Williams %R, CCI, divergence proxies |
| Volatility | 21 | ATR ratios & 1-yr rank, Bollinger/Keltner position, squeeze, Donchian width, Parkinson, realized-vol z-scores, compression |
| Activity (volume proxies) | 8 | minute-count, realized-variance z, up/down-minute imbalance, Kaufman efficiency |
| Market structure (SMC) | 24 | lag-confirmed swings, BOS/CHoCH, liquidity sweeps, fair-value gaps (count + distance), order-block distances |
| Levels | 26 | day/week/month opens, previous-day/week/month H/L, classic pivots, round numbers ($10/$50), day-anchored TWAP ("VWAP" proxy) |
| Time / session / events | 30 | EST hour (incl. sin/cos), sessions (Tokyo/London/NY, DST-correct), NY open/close distances, 08:30/10:00/14:00 windows, FOMC/NFP day & distance, weekend-gap size |
| Short lags | 40 | t−1, t−2 of the 20 fastest-moving state features |

Leakage screens: no feature correlates with next-bar return above |ρ|=0.05;
swing/FVG features are confirmed with explicit lag; labels use only bars
t+1 onward with entry at next open.

### Labels researched

| Label | Definition | Notes |
|---|---|---|
| `y_dir_1/5/10` | direction of 1/5/10-bar forward return | balanced (49.9% up) |
| `y_tp_long/short` | TP 1.5×ATR before SL 1.0×ATR within 16 bars, entry next open | base rate ≈ 39% vs 40% breakeven; same-bar tie → SL (pessimistic); trades never span weekends |
| `mfe_16` / `mae_16` | max favorable/adverse excursion in ATR units over 16 bars | magnitude targets |

## 4. Feature Evaluation (walk-forward, never in-sample)

Method: mutual information on the 2010–13 development window only; LightGBM
gain across 13 walk-forward folds; permutation importance on the 2024–26
folds; Spearman correlation clustering; ridge-stabilised VIF.

**What matters (consistent across MI, gain, and permutation):**

1. **Volatility-regime state** — `vol_of_vol` (top of *all three* rankings),
   `rv_rank_1y`, `atr96_p`, `rv_z_96`.
2. **Long-horizon return distribution** — rolling skew/kurtosis (96 bars),
   2–5-day returns (`ret_192`, `ret_480`), return autocorrelation.
3. **Distance to reference levels** — month/week opens, previous-month and
   previous-week highs/lows, classic pivots (R1/S1), the **$50 round-number
   grid** (`dist_r50` ranks 5th by gain).
4. **Calendar/event position** — `days_to_event`, `hrs_to_fomc`,
   `day_of_month`, minutes since NY open.

**What does not matter** (permutation importance ≤ 0 out-of-sample, 101 of
205): ADX, Aroon, Donchian position/width, most single-candle patterns,
day-of-week flags, Parkinson vol, most EMA distances, short-lag copies —
i.e., the bulk of the classic indicator canon is redundant or noise here.

**Redundancy:** 69 features fall into 25 clusters with |ρ|≥0.9 (EMA family,
RSI family, lag copies…); 104 features have VIF>10. Tree models tolerate
this; linear models received the full standardized set (their weaker
results partly reflect it).

## 5. Model Comparison

Identical data per fold (last 5 years training, purged, test years 2018,
2020, 2022, 2024, 2026), target `y_tp_long`. Sequence models consume raw
32-bar windows of 18 dynamic channels. Mean across folds:

| Model | AUC | Log-loss | Brier | Top-5% gross expectancy (R) |
|---|---|---|---|---|
| Random Forest | **0.5213** | **0.6668** | **0.2370** | **+0.078** |
| Rank ensemble (LGB+XGB+LR) | 0.5198 | 0.8429* | 0.2982* | +0.047 |
| XGBoost | 0.5194 | 0.6781 | 0.2419 | +0.049 |
| LightGBM | 0.5165 | 0.6891 | 0.2466 | +0.024 |
| MLP (64,32) | 0.5154 | 0.9240* | 0.3051* | +0.050 |
| LSTM | 0.5138 | 0.6687 | 0.2378 | +0.026 |
| Logistic regression | 0.5128 | 0.6699 | 0.2384 | +0.042 |
| Transformer (2-layer) | 0.5122 | 0.6684 | 0.2377 | +0.052 |
| GRU | 0.5096 | 0.6683 | 0.2376 | +0.044 |
| Temporal CNN | 0.5084 | 0.6686 | 0.2378 | +0.013 |
| Linear SVM (SGD) | 0.5019 | — | — | +0.034 |

\* rank-averaged / uncalibrated probabilities inflate log-loss & Brier; AUC is the comparable metric.

Findings:

* **Tree ensembles > deep sequence models** on this tabular problem — the
  engineered features already capture the useful temporal structure;
  32-bar raw windows add nothing (LSTM/GRU/TCN/Transformer all ≤ 0.514).
  TFT/N-BEATS/TabNet were not run on this CPU-only box (future work);
  their nearest proxies (Transformer/TCN/MLP) all underperform trees.
* **Non-linearity is real but modest**: logistic regression (0.513) trails
  boosted trees (0.517–0.521) by ~0.5–0.8 AUC points.
* LightGBM was retained for the trading pipeline (its walk-forward AUC on
  the full expanding window is 0.529; RF's advantage on 5-yr windows does
  not survive expanding-window training and costs 30× the training time —
  both were validated).

### Which target is most learnable? (same protocol)

| Target | Mean AUC | Interpretation |
|---|---|---|
| `mfe_16` (magnitude) | **0.5526** | volatility is forecastable |
| `mae_16` (magnitude) | 0.5474 | volatility is forecastable |
| `y_dir_1` | 0.5273 | weak short-horizon directional signal |
| `y_dir_5` | 0.5262 | fades with horizon |
| `y_dir_10` | 0.5183 | " |
| `y_tp_short` | 0.5179 | barrier outcome = direction × vol (hardest) |
| `y_tp_long` | 0.5165 | " |

The market's *magnitude* is predictable; its *sign* barely is. Any viable
strategy must therefore lean on volatility timing, level context, and
asymmetric payoff geometry rather than raw direction calls.

### Walk-forward AUC by year (y_tp_long, LightGBM, expanding window)

2014: 0.551 · 2015: 0.553 · 2016: 0.538 · 2017: 0.520 · 2018: 0.518 ·
2019: 0.521 · 2020: 0.535 · 2021: 0.530 · 2022: 0.529 · 2023: 0.529 ·
2024: 0.507 · 2025: 0.515 · 2026H1: 0.527 — **mean 0.529, 13/13 above 0.5**,
with visible decay after 2016.

## 6. Statistical Significance of the Statistical Edge

*(see §8 for the economic significance of the traded strategy)*

Under H₀ (no predictability), each yearly AUC is symmetric around 0.5.
Observing 13/13 years above 0.5 has sign-test probability 2⁻¹³ ≈ 1.2×10⁻⁴.
Each yearly AUC also individually rejects 0.5 (n_test ≈ 20–30k bars/year;
the 95% CI half-width on AUC at that n is ≈ 0.006–0.008, well below the
observed 0.507–0.553 excesses). The *statistical* signal is real.

## 7. Backtest (walk-forward, cost-aware)

**Specification** (`research/backtest.py`): two LightGBM models (long/short,
`y_tp_long`/`y_tp_short`) trained expanding-window per test year 2014–2026;
the entry threshold is chosen **only on the last year of each training
window** (max expectancy subject to ≤5 trades/day). Entries at next bar
open; TP 1.5×ATR / SL 1.0×ATR / 16-bar timeout; same-bar TP+SL counted as
SL; positions force-flattened across weekend gaps; no entries Friday
≥15:00 EST; one position at a time; max 5 entries/day; −3% daily loss stop;
1% of compounding equity risked per trade; proportional round-trip cost.

**Result: 2,671 trades over 12.5 test years (0.82/day)** — the ≤5/day
constraint never binds; the model rarely finds qualifying bars.

| Round-trip cost | Expectancy | Win rate | PF | Sharpe | Max DD | Final equity |
|---|---|---|---|---|---|---|
| 0.0 bp (frictionless) | **+0.114 R** | 45.5% | 1.21 | **+1.38** | −26% | **18.0×** |
| 1.5 bp (institutional) | −0.017 R | 45.2% | 0.97 | −0.21 | −75% | 0.51× |
| 2.5 bp (good retail CFD) | −0.105 R | 45.0% | 0.84 | −1.23 | −95% | 0.05× |
| 4.0 bp (typical retail) | −0.237 R | 44.5% | 0.67 | −2.66 | −99.8% | ~0 |

Reference (2.5bp case): avg win +1.22R, avg loss −1.19R, median hold 4.4
bars (~1.1h), exits 52% SL / 41% TP / 6% timeout / 1% weekend-gap flatten;
longs 1,322 / shorts 1,349; max 9 consecutive wins, 12 consecutive losses.
Threshold robustness: shifting every fold's threshold ±0.01/±0.02 moves
net expectancy only within [−0.111, −0.088] R — the negative result is not
a threshold artifact.

**The average cost burden is ≈0.22R per trade at 2.5bp** (SL = 1×ATR ≈
6–12bp of price, so 2.5bp round-trip ≈ 0.2–0.4R depending on regime).
The gross edge (+0.114R) is real but roughly half of realistic retail
friction. Breakeven cost is ≈1.3bp round-trip — institutional-grade
execution territory.

## 8. Economic Significance

* **Gross edge vs. placebo:** 100 simulations with fold-year-shuffled
  probabilities (identical marginal distribution & threshold, destroyed
  timing) average **−0.233R ± 0.015** net, vs the model's −0.105R
  (p < 0.01, model > all 100 placebos). Model timing adds **≈ +0.13R per
  trade** of genuine skill — also visible gross: +0.114R vs placebo-implied
  ≈ −0.01R gross.
* **Net edge:** day-block bootstrap 95% CI on net expectancy at 2.5bp is
  **[−0.150, −0.061] R**, P(≤0) = 100%. The traded strategy is
  *significantly losing* at retail costs.
* **Statistical vs. economic significance:** the AUC signal (13/13 years,
  §6) is real; it is simply too small to clear the cost hurdle at this
  trade frequency and holding period (~1 hour).

## 9. Failure Analysis

**What kills the strategy: costs, not wrongness.** At 0bp the equity curve
compounds 18× in 12.5 years; every basis point of friction removes ≈0.09R
per trade across ~2,700 short-duration trades.

**Where losses concentrate (net, 2.5bp, 95% CIs in figure
`conditional_expectancy.png`):**

| Condition | n | Net expectancy |
|---|---|---|
| High-vol regime (top ATR quartile) | 718 | **+0.03R ± 0.09** (≈breakeven) |
| Trend (ADX>25) | 1,459 | −0.04R ± 0.06 |
| NY session | 684 | −0.06R ± 0.09 |
| London session | 635 | −0.12R ± 0.10 |
| Tokyo session | 1,261 | −0.15R ± 0.07 |
| Range (ADX<20) | 682 | −0.23R ± 0.09 |
| Low-vol regime (bottom quartile) | 766 | **−0.28R ± 0.09** |
| FOMC / NFP days | 85 / 65 | +0.01R / −0.09R (CIs ±0.26/±0.29 — inconclusive) |

Two mechanisms: (i) cost-in-R is inversely proportional to ATR, so low-vol
trades pay 2–3× the friction; (ii) the signal itself is stronger in
expansion/trend regimes. *Caveat:* these splits are made on test data — a
"high-vol only" filter is a **new hypothesis generated by this analysis**,
not a validated result, and would need fresh out-of-sample confirmation.

**Losing streaks** (max 12) cluster in low-vol chop and in 2025, the year
with both the highest trade count (656 — threshold transferred poorly from
its validation year) and near-zero AUC. Worst months: Feb/May/Jul/Sep 2025,
Sep 2021.

**Alpha decay:** walk-forward AUC fell from ≈0.55 (2014–16) to ≈0.51
(2024–25). Yearly net expectancy is negative in 10 of 13 test years; the
two positive years (2016 +0.10R, 2022 +0.06R) are high-vol years.

**Answers to the standing research questions:**

* *Which features matter?* Volatility-regime state, long-horizon return
  distribution shape, distance-to-levels, calendar position (§4).
* *Which indicators add nothing?* ADX, Aroon, Donchian, most candle
  patterns, day-of-week, most EMA distances — 101 of 205 features (§4).
* *Trend vs range?* The edge lives in trend/expansion; ranging and low-vol
  markets are net-negative in every specification tested.
* *News?* FOMC/NFP-day trades are too few for inference; event-distance
  features carry real importance (calendar position matters more than the
  event day itself).
* *Most profitable session?* NY (least unprofitable net; highest gross).
* *What causes losing streaks?* Low-vol chop + threshold mis-transfer
  across regime shifts.
* *Regime detectability?* Yes — vol regimes are the *most* predictable
  structure in the data (MFE/MAE AUC 0.55) and the strongest features.
* *How many trades/day maximize expectancy?* Fewer than allowed: the
  validation-optimal threshold yields 0.8/day, and lowering it (more
  trades) only degrades expectancy (§7 threshold sensitivity).

## 10. Limitations

1. **Bid-quote bars, no true volume/spread history** — activity proxies
   substitute for tick volume; costs are modelled, not observed. Real
   spreads widen precisely at news/rollover moments the strategy trades.
2. **15m barrier resolution** — same-bar TP+SL ties are counted as losses
   (pessimistic); M1-resolution replay would tighten this bound but the
   sign of the net result is robust to it (the 0→1.5bp cliff dominates).
3. **Slippage/latency/partial fills** are folded into the proportional cost
   assumption rather than modelled microstructurally.
4. **Calendar coverage**: only FOMC (scraped) and NFP (rule-based); CPI and
   other releases enter only via time-of-day features. A few FOMC meetings
   (2017/2023/2024 parsing variants) are missing.
5. **Model scope**: TFT, N-BEATS, TabNet, CatBoost and kernel SVM were not
   run (CPU budget); given that every model family tested clusters within
   0.02 AUC, a materially different outcome from these variants is
   unlikely but unproven.
6. **Single instrument, single timeframe** per the mandate.

## 11. Future Research

1. **Volatility-conditional strategy**: trade only top-quartile ATR-rank
   regimes (the one net-≈breakeven slice). Requires fresh OOS validation —
   2027+ data or M1-replay on held-out years — before any claim.
2. **Longer holding periods**: cost-in-R falls linearly with barrier width;
   4h–1d horizons with 2–3×ATR barriers re-run through this pipeline.
3. **Volatility targets instead of direction**: the market's forecastable
   dimension is magnitude (AUC 0.55). Options-adjacent expressions (or
   vol-breakout entries) monetise vol forecasts better than fixed-barrier
   direction bets.
4. **Execution research**: limit-order entries (earn, don't pay, the
   spread) change the cost sign; requires order-book data.
5. **M1-resolution barrier replay** to remove the pessimistic tie-break.
6. **Cross-asset spillover features** (DXY, yields, SPX) — absent here.

## 12. Final Conclusions

1. **A statistically significant predictive signal exists** on XAUUSD 15m:
   walk-forward AUC 0.529 (13/13 years > 0.5, sign-test p ≈ 1.2×10⁻⁴),
   and the traded gross edge (+0.114R/trade, Sharpe 1.38 frictionless,
   +0.13R/trade above placebo timing) is genuine model skill.
2. **No reliable edge exists after realistic retail costs.** Breakeven is
   ≈1.3bp round-trip; at 2.5bp the strategy loses −0.105R/trade with 95%
   CI entirely below zero. This conclusion is robust to threshold choice,
   and is reported exactly as measured — the honest answer to the primary
   research question is **negative at retail frictions, marginal at
   institutional frictions**.
3. **The predictable structure is volatility, levels, and calendar — not
   direction.** Strategies that monetise magnitude forecasts, or that
   confine directional bets to high-vol/trend regimes with cheap
   execution, are the defensible next hypotheses; both remain unvalidated.
4. **Feature and model folklore did not survive testing**: half the
   indicator canon is noise here, deep sequence models underperform
   gradient-boosted trees, and win-rate maximisation would have pointed
   in exactly the wrong direction (the frictionless-profitable system wins
   only 45.5% of trades).

---

*Reproduction: `python3 research/data/download_histdata.py && python3
research/data/build_dataset.py && python3 research/data/build_calendar.py
&& python3 research/features.py`, then `eval_features.py`,
`models_compare.py`, `targets_compare.py`, `backtest.py`,
`make_figures.py`. Figures in `report/figures/`.*
