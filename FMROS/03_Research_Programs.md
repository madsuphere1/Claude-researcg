# Volume 3 — Research Programs

Ten standing departments. Each has a mandate, the current evidence base
inherited from cycles 1–2, standing questions ranked by information
value, and known dead ends (which are protected knowledge: re-litigating
them requires new mechanism or data per Constitution §1.6).

Each program runs as an autonomous track: it maintains its own section of
the experiment registry, proposes hypotheses to the Chief Scientist
(Volume 4), and inherits all Constitution/Methodology constraints.

---

## P1 — Market Structure & Levels

**Mandate.** Price interaction with reference levels: prior highs/lows,
opens, pivots, round numbers, swing structure, BOS/CHoCH, FVGs, sweeps.

**Evidence base.** Distance-to-levels features are a top-4 importance
family ([TESTED-HERE]: `dist_month_open`, `dist_r50` ($50 grid!),
prior-week/month extremes all rank high in gain and permutation).
Classic SMC event *flags* (BOS, sweep, FVG counts) carry little value
in comparison; it is the *distances* that matter.

**Standing questions.**
1. Are level distances predictive because of genuine order clustering, or
   as proxies for mean-reversion scale? Design: orthogonalize against
   z-score features; test incremental value. [HYPOTHESIS]
2. Does the $50-grid effect strengthen with price (grid density falls as
   gold rises)? A structural break test across the 4× price range.
3. Event-study of first touch of prior-day/week extremes with
   cost-honest fade/breakout cohorts (the H-C1 design pattern applies).

**Dead ends.** None yet closed at program level.

---

## P2 — Regime Detection

**Mandate.** Identify market states in real time; feed gates to every
other program.

**Evidence base.** The strongest single lever found so far. 3-state HMM
on [r, log RV], forward-filtered: gating to the high-vol state moved net
expectancy −0.105 → +0.053R and *gross* +0.114 → +0.190R [TESTED-HERE:
`research/cycle2/results/a1_hmm_regimes.json`]. Regimes are intraday:
median dwell 1–2.5h, transition diagonals 0.84–0.90 — the "daily regime"
mental model is wrong for this market. Vol magnitude is the most
predictable target (MFE/MAE AUC ≈ 0.55).

**Standing questions.**
1. State-count and observable ablation: 2 vs 3 vs 4 states; add
   activity/imbalance channels; does gate quality improve out-of-fold?
2. Gate probability threshold as a risk dial (P>0.5 vs 0.6 vs 0.7):
   trade count vs concentration trade-off, chosen on validation slices.
3. Regime-*transition* prediction (entering expansion within k bars) as a
   target — plausibly more valuable than state identification.
   [HYPOTHESIS]
4. Cross-regime model specialization (separate models per state) vs the
   current gate-only use.

**Dead ends.** Daily-persistence assumption (dwell ≥ 1 day) — refuted.

---

## P3 — Feature Discovery

**Mandate.** Grow the feature set; police its quality.

**Evidence base.** 205 hand features built; 101 worthless out-of-sample;
winners are vol-regime state, long-horizon return distribution shape,
level distances, calendar position [TESTED-HERE]. A 2,000-formula
symbolic search (declared universe, Bonferroni×20) statistically
confirmed exactly one factor — short-horizon mean reversion — which the
hand set already contained; joint model uplift was −0.003 AUC. Deep/SSL
representations: masked-GRU embeddings +0.0022 AUC (met its bar,
economically nil).

**Standing questions.**
1. **Cross-asset features (highest information value in this program):**
   DXY, real yields, SPX, oil — nothing cross-asset has ever been tested
   here. Requires a data-acquisition project first. [HYPOTHESIS]
2. Feature *interactions* with regime (P2): do level distances only work
   in trends? (Cycle-1 conditional analysis suggests yes but was
   post-hoc.)
3. Retire the 101 dead features from production models — measure the
   simplicity dividend.

**Dead ends.** Blind formula mining for *new* alpha (universe of simple
compositions over price/activity series is mined out); adding redundant
copies of known factors (negative uplift).

---

## P4 — Representation Learning

**Mandate.** Learned features from raw sequences; model architectures.

**Evidence base.** Consistently humbling [TESTED-HERE]: LSTM/GRU/TCN/
Transformer all below tree ensembles on identical folds; masked-GRU
embeddings economically nil; trees on engineered tabular features remain
the production standard. TFT/N-BEATS/TabNet remain formally untested
(CPU constraint), with a documented prior that they won't matter.

**Standing questions.**
1. One honest GPU-budget test of TFT/TabNet to close the question.
2. SSL on *cross-asset* multichannel input once P3.1 data exists —
   the one place representation learning might still surprise.
   [HYPOTHESIS]

**Dead ends.** Small sequence models on single-asset OHLC-derived
channels — three architectures, two cycles, zero value.

---

## P5 — Execution Science

**Mandate.** The cost side: fill mechanics, order types, geometry.

**Evidence base.** The program's most productive department.
Cost-in-R mechanism confirmed monotonically (G0 −0.105 → G1 −0.055 →
G2-managed −0.027R at 2.5bp; +0.019R at 1.5bp). Passive limit entry:
ITT uplift +0.088R (CI [0.06, 0.12]) at δ=0.1×ATR with adverse selection
fully quantified (fills' market-counterfactual −0.24R; misses would have
made +0.85R) [TESTED-HERE: `research/cycle2/results/b1_limit_entry.json`].

**Standing questions.**
1. δ/K joint surface for limit entries, chosen on validation folds only.
2. Partial-fill and queue modeling — currently assumption-level; needs
   order-book or broker-fill data to firm the 1.25bp passive-cost figure.
3. Scale-in/scale-out (split entries across the limit ladder). [HYPOTHESIS]
4. Time-of-day-conditional spread model (news/rollover widening) to
   de-optimism the cost model.

**Dead ends.** Market entry at signal close +1 bar as anything but the
pessimistic baseline.

---

## P6 — Risk Science

**Mandate.** Sizing, drawdown control, tail risk, gap risk.

**Evidence base.** Sizing cannot create edge (confirmed by construction
and simulation: nothing rescues the negative stream). On positive
streams, prob-weighted sizing and DD-throttles raise MAR ~8%; clipped
quarter-Kelly undersizes. Weekend gaps × inverse-ATR sizing is the
dominant tail mechanism found (single-handedly destroyed unmanaged G2);
pre-weekend flatten + 10× leverage cap is now house standard
[TESTED-HERE: `research/cycle2/results/gh_sizing.json`,
`b2v2_gap_managed.json`].

**Standing questions.**
1. Vol-scaled risk (fixed *dollar* vol per trade instead of fixed 1%) —
   interacts with the cost-in-R denominator; needs careful design.
2. Regime-conditional sizing driven by P2's gate probability. [HYPOTHESIS]
3. Ruin surfaces for the composite family under regime-block resampling
   (day blocks understate; the X1 lesson).

**Dead ends.** Unclipped Kelly; any sizing pitch claiming to fix a
negative-expectancy stream.

---

## P7 — Market Microstructure

**Mandate.** Sub-bar dynamics: order flow, imbalance, activity, spreads.

**Evidence base.** Limited by data (no true volume/spread in HistData).
M1-derived proxies (realized variance, up/down-minute imbalance,
efficiency) carry modest but real importance; the M1 replay
infrastructure (fill-vs-stop ordering) is built and validated. The one
event-structure hypothesis tested (Asia→London reversion) is dead in
both directions [TESTED-HERE].

**Standing questions.**
1. Acquire true tick/volume data (Dukascopy ticks are blocked from this
   environment; broker exports or purchased data are the route) — the
   program's single most valuable data upgrade, unlocking real order-flow
   features and honest spread history.
2. M1-resolution replay of the *entire* composite (removes the residual
   pessimistic-tie assumptions).
3. Intraday seasonality map of the mean-reversion factor (P3's confirmed
   factor, localized by hour) — screening design ready. [HYPOTHESIS]

**Dead ends.** Session-boundary reversion trades at 15m granularity.

---

## P8 — Cross-Asset & Economic Research

**Mandate.** Everything exogenous: DXY, rates, equities, calendars, news.

**Evidence base.** Calendar *position* features (days-to/since event)
matter more than event-day flags; FOMC/NFP-day trades were too few for
inference. Two independent conditioning-signal types now tested and
both closed negative [TESTED-HERE]: lagged daily macro levels (real
yield/DXY/VIX/WTI, `research/cycle3/results/c3_003_macro.json`, mean
uplift −0.0087, 3/3 folds negative) and weekly GDELT media tone/theme
tags (`research/cycle3/results/c3_007_gdelt.json`, mean uplift −0.0027,
2/3 folds negative). Both fail for the same declared mechanism: the
conditioning signal's native timescale (days, for a macro print or an
accumulating narrative) is a poor match for a 4-hour barrier horizon —
this is now a pattern across two data types, not an isolated result.

**Standing questions.**
1. Real-yield and DXY conditioning of the gold signal — CLOSED (C3-003,
   REJECTED).
2. High-impact-news spread/vol dynamics as execution inputs to P5.
   [HYPOTHESIS]
3. Cross-asset lead-lag at 15m–4h horizons. [HYPOTHESIS]
4. Daily- or intraday-resolution GDELT re-test — C3-007 sampled only
   weekly as a bandwidth-bounded pilot; a finer sample is a fairer test
   of the textual-conditioning mechanism before concluding it's dead at
   this program's timescale. [HYPOTHESIS]

**Dead ends.** Lagged daily numeric macro (C3-003); weekly-sampled
media tone/theme tags (C3-007). Any generative-LLM "reasoning" over
historical news evaluated as a backtest — the model's own training-data
knowledge of subsequent events risks silent lookahead contamination that
standard leakage screens do not catch; constrained to forward/paper
evaluation only until a contamination-safe design exists.

---

## P9 — Portfolio Construction

**Mandate.** Multi-signal, multi-config, eventually multi-asset capital
allocation.

**Evidence base.** Single-instrument so far; trade-level ranking by model
probability adds MAR. The portfolio question becomes real the moment two
validated configs exist (e.g., X1-family variants across gates).

**Standing questions.**
1. Correlation structure of the three levers' trade streams (do gate,
   geometry, execution variants fire on the same bars?).
2. Kelly-fraction allocation across configs under regime-block bootstrap.
   [HYPOTHESIS]

---

## P10 — Adaptation & Monitoring

**Mandate.** Model lifecycle: when to retrain, when to distrust.

**Evidence base.** Annual expanding retrain beats rolling windows
everywhere; PSI-style drift statistics are anti-correlated with what
matters (they track training-set maturity, not alpha decay) and would
misfire as retrain triggers [TESTED-HERE:
`research/cycle2/results/i1_drift.json`]. Static models decay slowly
(2013 model still ≈ 0.52 in 2026). Cycle 4 re-tested rolling windows at
the *full-strategy expectancy* level (not just AUC), on operator
challenge: expanding +0.128R vs rolling-5 −0.058R vs rolling-2 −0.145R
on the even-year slice — monotone, shorter window = worse, driven by
short windows being unable to resolve the AUC-0.53 signal (thresholds
forced to 0.76–0.90). One large-sample-contradicted recent-year point
(2024 rolling-5 +0.351R on 21 trades) keeps the *new-data* arbiter
C3-008 open [TESTED-HERE: `research/cycle3/results/screen_rolling_x1.json`,
`report/CYCLE4.md`].

**Standing questions.**
1. Performance-based (not distribution-based) decay detection:
   sequential probability ratio tests on rolling trade outcomes — the
   production-monitoring question of Volume 9. [HYPOTHESIS]
2. Does the vol-gate reduce decay exposure (is gated alpha younger)?
3. Expanding vs rolling-5 on 2027+ data (C3-008, standing) — the only
   clean arbiter of the window question; rule frozen before data exists.

**Dead ends.** Rolling training windows (refuted twice: H-I1 at AUC
level, C4 screen at expectancy level); PSI-triggered retraining.
