# Cycle-2 Hypothesis Registry

All hypotheses below were declared BEFORE their experiments were run.
Decision rules and success criteria are fixed here, in advance, to prevent
test-set optimization. Failed hypotheses stay in this file.

Baseline (cycle 1, treated as established):
walk-forward AUC ≈ 0.529 on y_tp_long; gross expectancy +0.114R/trade;
net at 2.5bp −0.105R; breakeven cost ≈1.3bp; volatility more predictable
than direction (AUC 0.55 vs 0.52); alpha decaying over time.

| ID | Track | Hypothesis | Theory | Falsification criterion |
|----|-------|-----------|--------|------------------------|
| H-C1 | C Microstructure | The net-of-costs return of fading the Asia-session drift at London open is positive | Inventory/liquidity effects: thin Asian liquidity lets price drift; London depth arrival corrects it (documented in FX literature) | Dev-window (2010–13) effect absent, or 2014+ walk-forward net expectancy CI includes 0 |
| H-B2 | B Execution | Widening barriers to 2–3×ATR with proportionally longer horizon raises NET expectancy vs cycle-1 1.5/1.0×ATR/16-bar spec, because cost-in-R falls ∝1/SL while gross edge falls slower | Cost is fixed in price terms; signal has multi-hour persistence (ret_192/480 importance) | Net expectancy at 2.5bp not better than cycle-1 −0.105R, or gross edge vanishes at wider geometry |
| H-A1 | A Regimes | (i) A 3-state Gaussian HMM on (r, |r|, rv) fitted walk-forward yields persistent states (median dwell ≥ 1 day for the calm state); (ii) gating cycle-1 signal to the highest-vol state yields net expectancy > cycle-1 baseline AND ≥0 at 2.5bp | Vol clustering is the strongest structure in the data; cost-in-R shrinks with ATR; cycle-1 conditional analysis independently pointed here (flagged then as untested) | States not persistent (dwell <4h), or gated net expectancy CI below 0 |
| H-B1 | B Execution | Passive limit entry at signal−δ (δ≈0.25×ATR), cancel-after-4-bars, changes cohort NET expectancy by ≥ +0.10R vs market entry, net of adverse selection and missed winners | Earn-vs-pay the spread swings ~2×cost; limit placement in mean-reverting micro-noise buys better prices | Intention-to-treat cohort expectancy improves < +0.05R, or fill rate <40% with missed-trade opportunity cost eating the gain. Fill rule: M1 low must trade ≥1 tick THROUGH the limit (conservative, no queue credit) |
| H-F1 | F Targets | A vol-expansion forecast (top-decile predicted MFE) gating symmetric OCO stop-brackets (buy-stop +0.5×ATR / sell-stop −0.5×ATR, TP 2×ATR, SL 1×ATR) produces positive net expectancy | Magnitude is the predictable dimension (AUC .55); a bracket monetizes magnitude without a direction call; double ticket cost is offset by 2R:1R geometry in true expansions | Walk-forward net expectancy CI includes 0 at 2.5bp/leg, or trades <0.05/day (vacuous) |
| H-D1 | D Feature discovery | Among ~2000 randomly generated symbolic features (operator grammar over base series), at least one adds ≥0.003 AUC to the cycle-1 LightGBM feature set on 2014+ walk-forward, surviving a top-20-screen Bonferroni-style deflation | Hand-engineering explores a tiny corner of formula space | No screened candidate's OOS AUC uplift ≥0.003, or uplift not significant under stationary bootstrap of fold AUCs |
| H-I1 | I Adaptation | (i) Rolling 3–5y training beats expanding window on post-2020 folds (adaptation to decay); (ii) PSI-triggered retraining ≈ calendar retraining with fewer retrains | Alpha decay means old data mis-weights current regime | Expanding ≥ rolling on post-2020 AUC; PSI trigger misses decay episodes |
| H-G1 | G/H Sizing | On any positive-expectancy trade stream from this cycle: probability-weighted fractional-Kelly sizing with a drawdown throttle raises geometric growth per unit max-DD vs flat 1%; on the cycle-1 (negative) stream, no sizing rule produces profit | Sizing cannot create edge, only reshape it; Kelly optimizes log growth | If any sizing rule turns the negative-expectancy stream profitable, the simulator is broken (sanity check), fail closed |
| H-E1 | E Representation | A small masked-reconstruction encoder (dev-window trained) yields 32-d embeddings that raise LightGBM OOS AUC by ≥0.002 when appended to hand features | Self-supervision may capture shapes hand features miss | Uplift <0.002 or negative on 2014+ folds (cycle-1 prior: sequence models added nothing) |

Execution notes:
* Dev window for any screening: 2010–2013. Confirmation: walk-forward 2014–2026 (or a stated subset for CPU reasons, chosen ex-ante as every second year).
* All net figures at 2.5bp round-trip per position leg unless stated; sensitivity at 1.5/4bp.
* Significance: day-block bootstrap for expectancy; stationary bootstrap across folds for AUC uplifts; placebo shuffles where timing is the claim.
