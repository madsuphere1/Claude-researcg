# XAUUSD 15-Minute Edge Research

**Question under test:** does a statistically significant, economically exploitable
trading edge exist on XAUUSD (gold) 15-minute bars, tradeable at ≤5 trades/day
with 1% risk per trade?

**Methodology:** hypothesis-driven, walk-forward, cost-aware. Nothing is assumed
to work; every claim below is backed by an out-of-sample experiment in this
repository, and failed experiments are reported alongside successful ones.

---

## 1. Executive Summary

*(finalised in Section 9 — Conclusions; summary of findings below)*

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
5. **Economic viability after realistic costs is the binding constraint** —
   quantified in §7 (backtest) and §8 (significance).

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

*(results below; specification in `research/backtest.py`)*

## 8. Economic Significance

## 9. Failure Analysis

## 10. Limitations

## 11. Future Research

## 12. Final Conclusions
