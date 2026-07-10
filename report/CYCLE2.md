# XAUUSD Research — Cycle 2: New-Edge Discovery

Baseline (cycle 1, `report/REPORT.md`): real but small signal (WF AUC 0.529),
gross +0.114R/trade, dead at retail costs (−0.105R at 2.5bp), breakeven
≈1.3bp, volatility more predictable than direction, alpha decaying.

Cycle 2 ran **ten pre-declared, falsifiable hypotheses** across the
mandated tracks (registry with decision rules frozen before each run:
`research/cycle2/REGISTRY.md`). Every verdict below — including the
failures — is reported at the pre-declared criterion.

## Scoreboard

| ID | Track | Hypothesis (short) | Verdict | Key number |
|----|-------|--------------------|---------|-----------|
| H-C1 | Microstructure | Fade Asia drift at London open | **REJECTED** | OOS −2.8bp/trade, CI [−4.3, −1.4]; gross effect ≈ 0 |
| H-B2 | Execution | Wider barriers beat cost drag | **PARTIAL** — mechanism confirmed, edge insufficient | G0→G1→G2 monotone (−0.105→−0.055→−0.027R managed); +0.019R at 1.5bp |
| H-A1 | Regimes | HMM vol-state gate improves net expectancy | **PARTIAL** — large improvement, not significant; persistence criterion failed | −0.105R → +0.053R (CI [−0.03, +0.14], p=0.12); dwell 1–2.5h not ≥1d |
| H-B1 | Execution | Passive limit entry ≥ +0.10R ITT uplift | **PARTIAL** — uplift real but below bar | +0.088R (CI [0.06, 0.12]) at δ=0.1; total still −0.02R |
| H-F1 | Targets | Vol forecast + OCO stop-brackets | **REJECTED** (strongly) | −0.457R, CI < 0; 35% whipsaws; worse than random gates |
| H-D1 | Feature discovery | Symbolic search finds new signal | **STATISTICAL YES / ECONOMIC NO** — one factor (short-horizon mean reversion), already in the model | 19/20 survive Bonferroni, OOS IC ≈ −0.03; joint LGB uplift −0.0031 → REJECT |
| H-I1 | Adaptation | Rolling windows / PSI-triggered retraining | **REJECTED** (both parts) | Expanding ≥ rolling everywhere; PSI shrinks while decay continues |
| H-G1 | Sizing/Risk | Sizing reshapes, never creates | **CONFIRMED** | MAR 0.99→1.07 on gross stream; nothing rescues the negative stream |
| H-E1 | Representation | Masked-GRU embeddings add AUC | **ACCEPTED (marginally)** | +0.0022 mean uplift (bar: 0.002), positive all 3 folds; economically nil |
| X1 | Composite (exploratory) | Gate + geometry + limit entry | **PROMISING, UNPROVEN** — needs cycle-3 confirmation | +0.128R net; day-block p=0.02 but year-block p=0.10; 2016+2020 = 118% of profits |

---

## 1. H-C1 — London-open reversion of the Asia move (Track C)

**Theory.** Thin Asian liquidity lets gold drift on inventory pressure;
London depth arrival corrects it (documented in FX for majors).

**Experiment.** 1,895 OOS events (first London bar per day, 2014–26); fade
the overnight move, horizons 8/16/32 bars; day-block bootstrap; 2.5bp cost.

**Result. Rejected.** Dev-window effect already absent (CI includes 0);
OOS fade loses −2.5 to −4.5bp per trade across all horizons, and the
big-move quintile is *worse* (−4.8bp). Following the move also loses
(−2.2bp): the Asia move carries **no exploitable information** either way
at London open — net returns ≈ −cost. Gold's 23h CME-linked liquidity
apparently never gets thin enough for the FX-style inventory effect.

## 2. H-B2 — Barrier geometry vs the cost wall (Track B)

**Theory.** Cost is fixed in price terms; cost-in-R = cost/SL-distance.
Cycle-1 features showed multi-hour persistence, so gross edge should decay
slower than 1/SL when barriers widen.

**Experiment.** Three geometries (same 1.5 payoff ratio): G0 = 1.5/1.0×ATR
/16 bars (cycle 1), G1 = 2/1.33/32, G2 = 3/2/64. Full walk-forward retrain
(even test years, declared), same threshold protocol, full simulation.

**Result — the mechanism is real:** net expectancy at 2.5bp improves
monotonically −0.105R (G0) → −0.055R (G1) → +0.104R (G2 unmanaged).

**…but the first G2 number was an artifact of unmanaged tail risk.** Its
bootstrap CI was [−0.23, +0.62] and the equity path lost 96% *even at zero
cost*: 64-bar holds cross weekend closes, where gaps blow through stops and
inverse-ATR sizing turns low-vol gap losses into >10% equity hits. With
standard risk controls declared in a follow-up variant (flatten before
weekend close, 10× leverage cap — risk rules only, no signal change):

| G2 managed | 0bp | 1.5bp | 2.5bp | 4bp |
|---|---|---|---|---|
| Expectancy (R) | +0.084 | **+0.019** | **−0.027** | −0.097 |
| Sharpe | +1.00 | +0.23 | −0.31 | −1.12 |
| Max DD | −34% | −43% | −57% | −79% |

Bootstrap CI at 2.5bp: [−0.090, +0.035] (p₀=0.80). Placebo timing:
−0.111R ± 0.017 → the model contributes **+0.084R of genuine timing skill**
(p=0.00). **Verdict: partial.** Wider geometry moved breakeven cost from
≈1.3bp to ≈2.0bp — a real improvement that is still short of retail
friction, and the honest headline is that half the unmanaged "profit" was
symmetric weekend-gap luck that risk control correctly removes.

## 3. H-A1 — Hidden-regime gating (Track A)

**Theory.** Vol clustering is the strongest structure in this market
(cycle-1: MFE/MAE AUC 0.55). If regimes are identifiable in real time,
confining trading to the high-vol state simultaneously (i) concentrates
the signal and (ii) shrinks cost-in-R.

**Experiment.** 3-state Gaussian HMM on [r, log RV] refit per fold on
training data only; test-year state probabilities via **forward-only
filtering** (hmmlearn's decoders are smoothed = lookahead; a manual causal
filter was implemented). Gate: P(high-vol state) > 0.6. Same predictions,
thresholds, and engine as cycle 1.

**Result.**

* Regime structure exists but is fast: transition-matrix diagonals
  0.84–0.90, median dwell 1–2.5h. The declared persistence criterion
  (calm-state dwell ≥ 1 day) **fails** — gold's 15m regimes are
  intraday-scale, not day-scale.
* The gate keeps 17% of bars and 26% of trades (693) and moves net
  expectancy at 2.5bp from −0.105R to **+0.053R** (CI [−0.032, +0.136],
  p₀=0.12); Sharpe −1.23 → +0.33; maxDD −95% → −23%. Gross expectancy
  rises +0.114R → +0.190R, so this is signal concentration, not merely
  cost relief.

**Verdict: partial.** The single most effective lever found in cycle 2,
directionally confirming the cycle-1 conjecture under a proper pre-declared
causal test — but the point estimate is not yet statistically distinguishable
from zero at retail costs.

## 4. H-B1 — Passive limit-order entry (Track B)

**Theory.** Market entries pay the spread; resting limits earn price
improvement instead — if adverse selection doesn't eat the difference.

**Experiment.** Intention-to-treat over ~2,700 signal cohorts: limit at
signal close −δ·ATR (mirrored short), cancel after 4 bars, fills require
price to trade one tick *through* the limit (no queue credit); intra-bar
fill-vs-stop ordering resolved on minute data; filled trades pay 1.25bp
(exit leg only) vs 2.5bp for market entries.

**Result.** Adverse selection is enormous and now quantified: signals that
fill would have lost −0.24R even at market (the limit selects the losers);
signals that *miss* would have made **+0.85R** (the limit misses the
runners). Price improvement still wins on net:

| δ (×ATR) | Fill rate | ITT uplift vs market | 95% CI |
|---|---|---|---|
| 0.10 | 88% | **+0.088R** | [+0.060, +0.117] |
| 0.25 | 77% | +0.083R | [+0.044, +0.121] |
| 0.50 | 58% | +0.069R | [+0.022, +0.116] |

**Verdict: partial** (uplift significant and >+0.05R floor, but below the
+0.10R success bar; cohort remains net-negative at −0.02R). Execution alone
does not rescue the baseline strategy; it recovers roughly 0.09R of the
0.22R cost burden.

## 5. H-F1 — Monetizing vol forecasts with stop-brackets (Track F)

**Theory.** Magnitude is the predictable dimension; a symmetric OCO
bracket (buy-stop +0.5·ATR / sell-stop −0.5·ATR, TP 2·ATR, SL 1·ATR)
converts a magnitude forecast into P&L without a direction call.

**Result: strongly rejected — and diagnostically valuable.** Gated
brackets: −0.457R (CI [−0.54, −0.37]), *worse* than ungated (−0.355R) and
worse than random gates at matched frequency (−0.40R ± 0.10). Cause: 35%
of triggered brackets whipsaw (both stops trade within a bar). The vol
model works — it correctly finds expansion bars — but expansion bars are
exactly where two-sided violence kills stop-triggered entries. **Magnitude
knowledge cannot be monetized through linear knock-in instruments at this
timescale; it requires curved payoffs (options) or a direction overlay.**

## 6. H-D1 — Automated symbolic feature discovery (Track D)

2,000 seeded random formulas (typed grammar over 8 base series); dev-window
screen 2010-11 vs 2012-13 with sign agreement; top 20 validated once on
2014–26 yearly ICs, Bonferroni ×20.

**Result: statistically accepted, economically thin.** 19/20 survive
(p < 0.0025 each), but they all encode **one factor**: smoothed 0.5–8h
returns with negative forward IC ≈ −0.03 — short-horizon mean reversion.
Predicted move per σ of signal ≈ 1bp per 75min — far below friction as a
standalone strategy. Joint model test (declared step 4): **−0.0031 mean
AUC** when the accepted formulas are appended to the cycle-1 feature set
(folds 2018/2022/2026; negative in 2 of 3) → **REJECT for incremental
model value: the factor was already captured by hand-engineered features,
and duplicating it slightly degrades the model.**

## 7. H-I1 — Drift and adaptation (Track I)

**Both sub-hypotheses rejected.**

* Rolling windows lose to expanding everywhere: post-2020 AUC 0.5246
  (expanding) vs 0.5156 (5y) vs 0.5115 (3y). Old data still carries
  weight-worthy signal; the decay is not concept replacement.
* PSI-based retraining triggers would misfire: prediction PSI *shrinks*
  monotonically (0.29 → 0.02) as training sets mature, while AUC decay
  continues — drift statistics track training-set growth, not alpha loss.
  (The PSI–AUC correlation of +0.70 is this same artifact: early
  high-PSI folds were the high-AUC years.)
* A static 2013 model still averages ≈0.52 AUC through 2026 — annual
  expanding retraining is better and cheap; nothing fancier is justified.

## 8. H-G1 — Sizing and risk engine (Tracks G/H)

On the gross-positive stream: probability-weighted sizing and a drawdown
throttle raise MAR from 0.99 to ≈1.07 with P(50% DD) ≤ 0.2%; quarter-Kelly
(clipped) undersizes (CAGR 7% vs 26%) because the R-stream's variance
overstates risk relative to its day-block structure. On the cycle-1
negative stream, **no rule produces profit** (best case: lose slower).
Confirms the declared position: sizing reshapes the growth/risk profile
of an existing edge and cannot create one.

## 9. H-E1 — Self-supervised representations (Track E)

Masked-reconstruction GRU (dev-window-trained, 32-bar windows, 6 channels)
embeddings appended to the feature set: +0.0022 mean AUC (positive in all
three folds), just clearing the +0.002 declared bar; embeddings alone are
far below baseline (0.50–0.51). **Formally accepted; practically
negligible** — consistent with cycle-1's finding that sequence models add
little once hand features exist.

## 10. Answers this cycle adds to the research questions

* **Can regimes be detected before entry? Yes** — but they are intraday
  (hours), not daily, and detection is causal-filterable (H-A1).
* **Does execution create profitability? Partially** — passive entry
  recovers ~0.09R of the 0.22R cost burden with adverse selection fully
  priced (H-B1); geometry recovers another ~0.08R (H-B2); neither alone
  crosses zero at 2.5bp.
* **Is the mean-reversion factor real? Yes** (H-D1: 19 Bonferroni-surviving
  formulas, stable sign) — **and already in the model** (no AUC uplift).
* **Should we retrain adaptively? No** — annual expanding retrain is the
  best tested policy; drift statistics mislead (H-I1).
* **Can vol predictability be monetized without direction? Not with
  stop-brackets** (H-F1's decisive whipsaw failure). Options-style payoffs
  remain the untested route.

## 11. X1 composite pilot (exploratory — NOT a validated result)

The three partial levers attack different terms of net expectancy, so
their conjunction is the natural candidate strategy: **G2 geometry
(TP 3×ATR / SL 2×ATR / 64 bars) + HMM vol-gate (causal filter, P>0.6)
+ limit entries (δ=0.1×ATR, 4-bar cancel, 1.25bp cost) + pre-weekend
flatten + 10× leverage cap.** All components were individually
pre-declared and separately validated; **their conjunction was formed
after seeing cycle-2 results**, so this pilot is hypothesis-generating,
not confirmatory.

Result (even test years 2014–2026, 414 trades, 0.25/day):

| Metric | Value |
|---|---|
| Net expectancy | **+0.128R/trade** |
| Day-block bootstrap CI | [+0.006, +0.247], p(≤0) = 0.02 |
| **Year-block bootstrap CI** | **[−0.084, +0.258], p(≤0) = 0.10** |
| Win rate / PF | 47.6% / 1.24 |
| Sharpe / max DD | 0.82 / −13.9% |
| Exits | 205 SL · 178 TP · 20 timeout · 8 pre-gap · 3 same-bar |

**The critical caveat is regime concentration**: 5 of 7 test years are
individually negative; 2016 and 2020 — the two high-vol years — contribute
118% of total R (the other five years sum to a net loss). A vol-gated
strategy is *designed* to earn in vol-rich regimes, but with effectively
2 profitable regime episodes the sample is far too thin to claim
significance at the year level (p=0.10), and a year-level sign test cannot
reject zero. **Status: promising, unproven.** The pre-registered cycle-3
confirmation protocol: freeze this exact specification (no parameter may
move), run nested re-training on odd test years 2015–2025 as a first
robustness pass, and treat 2027+ data as the true out-of-sample test.
Deployment assumptions if it confirms: ≈1.25bp effective cost on filled
limits (maker-style execution), tolerance for multi-year flat/negative
stretches between vol regimes, and ~0.25 trades/day utilisation.

## 12. Limitations

1. All cycle-2 experiments share cycle-1's data limitations (bid-quote M1
   bars, modeled costs, no order-book).
2. H-B1's fill model is conservative (trade-through required, no queue
   position) but its cost asymmetry (1.25 vs 2.5bp) is an assumption.
3. Even-year fold subsets (declared) halve the test sample for B2/F1.
4. The HMM gate and G2 predictions were each validated once; their
   conjunction (X1) is exploratory by construction.
5. Multiple partial results (A1 p=0.12; B1 near its bar) are individually
   suggestive; jointly they motivate — but do not constitute — a validated
   strategy.

## 13. Future work (cycle 3 candidates)

1. **Confirmatory test of the composite** on genuinely unseen data
   (2027+ live-sim, or full nested re-training on odd years with
   pre-registered spec).
2. Options-based vol monetization (H-F1's lesson).
3. Order-book/queue data to firm up H-B1's cost asymmetry.
4. Cross-asset conditioning (DXY, real yields) — still untouched.

## 14. Conclusions

Cycle 2 rejected four hypotheses outright (session reversion, stop-bracket
vol monetization, rolling-window adaptation, drift-triggered retraining),
confirmed two structural claims (sizing cannot create edge; symbolic
search re-finds known mean reversion), and validated three *levers* that
each attack a different term of net expectancy without any one of them
crossing zero at retail costs:

| Lever | Effect on net R (2.5bp) |
|---|---|
| HMM vol-gate (H-A1) | −0.105 → +0.053R (p=0.12) |
| Wide geometry, gap-managed (H-B2) | −0.105 → −0.027R |
| Passive limit entry (H-B1) | +0.088R ITT uplift (CI > 0) |

The market's message across both cycles is coherent: **the predictive
signal is real, small, volatility-flavored, and the fight is against
friction.** Every economically meaningful gain this cycle came from cost
engineering (geometry, passive fills) or risk concentration (vol gating) —
none from better prediction (D1 and E1 added ≈nothing; D1's joint test was
negative). Their conjunction (X1) is the first configuration in two cycles
whose net-of-cost expectancy is positive with a sub-3% p-value at trade
granularity — and the year-level analysis immediately disciplines that
excitement: profits live in two vol regimes, and p=0.10 at year blocks.

**Honest bottom line: no deployable edge is proven.** What cycle 2
produced is (a) four clean rejections that close whole avenues (session
reversion, stop-bracket vol monetization, rolling-window adaptation,
drift-triggered retraining), (b) three quantified levers with mechanisms
that replicate monotonically, and (c) one frozen composite specification
whose confirmation or refutation is a single, pre-registered cycle-3
experiment away.
