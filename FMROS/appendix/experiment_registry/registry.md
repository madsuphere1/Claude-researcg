# Experiment Registry (append-only)

Format per Volume 5. Verdicts are never edited; superseding entries link
back. Result paths are relative to repo root. Git SHAs identify the code
state that produced the verdict (see git log around each result commit).

| ID | Program | Hypothesis (short) | State | Verdict | Effect (primary) | Results |
|----|---------|--------------------|-------|---------|------------------|---------|
| C1-BASE | — | A predictive signal exists on XAUUSD 15m (y_tp_long learnable) | closed | CONFIRMED | WF AUC 0.529, 13/13 yrs > 0.5, sign-test p≈1.2e-4 | `research/results/wf_auc_y_tp_long.json` |
| C1-FEAT | P3 | ≥ some of 205 engineered features carry OOS value | closed | CONFIRMED (partially) | 104 useful / 101 dead by permutation | `research/results/perm_importance.csv` |
| C1-MODEL | P4 | Deep sequence models beat trees | closed | REFUTED | trees 0.517–0.521 vs deep 0.508–0.514 AUC | `research/results/model_comparison_summary.csv` |
| C1-TGT | P2 | Magnitude more predictable than direction | closed | CONFIRMED | MFE/MAE AUC ≈0.55 vs dir 0.52 | `research/results/target_comparison.csv` |
| C1-ECON | P5 | Baseline strategy survives retail costs | closed | REFUTED | +0.114R gross → −0.105R at 2.5bp; breakeven ≈1.3bp | `research/results/backtest_main.json` |
| H-C1 | P7 | London-open reversion of Asia move | closed | REFUTED (at screen + OOS) | OOS −2.8bp/trade net, CI [−4.3,−1.4]; gross ≈0 | `research/cycle2/results/c1_session_reversion.json` |
| H-B2 | P5 | Wider barriers beat cost drag | closed | WEAKENED (mechanism CONFIRMED) | G0→G1→G2 monotone; G2-managed −0.027R @2.5bp, +0.019R @1.5bp | `research/cycle2/results/b2_barriers.json`, `b2v2_gap_managed.json` |
| H-A1 | P2 | HMM vol-gate improves net expectancy; states persist ≥1d | closed | WEAKENED (gate) / REFUTED (persistence) | −0.105→+0.053R (p=0.12); dwell 1–2.5h | `research/cycle2/results/a1_hmm_regimes.json` |
| H-B1 | P5 | Passive limit entry ITT uplift ≥ +0.10R | closed | WEAKENED | +0.088R CI [0.06,0.12] at δ=0.10; adverse selection quantified | `research/cycle2/results/b1_limit_entry.json` |
| H-F1 | P2/P5 | Vol forecast + OCO stop-brackets profitable | closed | REFUTED | −0.457R CI<0; 35% whipsaw; worse than random gate | `research/cycle2/results/f1_vol_bracket.json` |
| H-D1 | P3 | Symbolic search finds new factor | closed | CONFIRMED (stat) / REFUTED (economic) | 19/20 Bonferroni-survive one MR factor; joint AUC −0.003 | `research/cycle2/results/d1_feature_search.json`, `d1b_joint_test.json` |
| H-I1 | P10 | Rolling windows / PSI-triggered retrain beat expanding/calendar | closed | REFUTED (both) | expanding best everywhere; PSI anti-signal | `research/cycle2/results/i1_drift.json` |
| H-G1 | P6 | Sizing reshapes, never creates | closed | CONFIRMED | MAR 0.99→1.07 (gross); negative stream unrescuable | `research/cycle2/results/gh_sizing.json` |
| H-E1 | P4 | Masked-GRU embeddings add ≥0.002 AUC | closed | CONFIRMED (marginal) | +0.0022, positive 3/3 folds; economically nil | `research/cycle2/results/e1_representation.json` |
| X1 | P2+P5+P6 | Composite of three levers is net-positive (EXPLORATORY) | closed | promising-unproven → superseded by C3-001 | +0.128R even yrs; day p=0.02, year p=0.10; 2016+2020 = 118% of R | `research/cycle2/results/x1_composite_pilot.json` |
| C3-001 | P2+P5+P6 | X1 frozen spec confirms on odd years 2015–2025 | closed | **WEAKENED** | +0.079R, 334 trades; day-block p=0.108; year-block p=0.187; 3/6 yrs positive | `research/cycle3/results/c3_x1_confirmation.json` |
| C3-002 | P5/P7 | Shadow-measure real spreads + passive fill rates (data project) | PROPOSED | — | unblocks the 1.25bp assumption; risk-free | — |
| C3-003 | P8 | Lagged daily macro block (real yield/DXY/VIX/WTI) adds ≥0.003 AUC | closed | **REJECTED** | uplift −0.0087, negative all 3 folds despite 21–23% gain share (noise-fit signature) | `research/cycle3/results/c3_003_macro.json` |
| C3-004 | P7 | 15m tie-break pessimism materially biases composite expectancy | closed | **IMMATERIAL** | 0/360 SL exits flip at M1 resolution; recorded expectancies stand; trade schema v2 (levels persisted) | `research/cycle3/results/c3_004_m1_replay.json` |
| C3-005 | P2 | Regime-transition prediction (entering expansion) as target | PROPOSED | — | mechanism-motivated successor to H-A1 | — |
| C3-006 | — | Re-run C3-001 decision rule on 2027+ data as it accrues | PRE-REGISTERED (standing) | — | the clean arbiter of the X1 family; rule inherited verbatim from C3-001 | — |
| C3-007 | P8 | GDELT weekly tone/theme block (textual successor to C3-003's numeric macro) adds ≥0.003 AUC | closed | **REJECTED** | uplift −0.0027, negative 2/3 folds; P8 now 0-for-2 on cross-asset conditioning | `research/cycle3/results/c3_007_gdelt.json` |
| C3-008 | P10 | Rolling-5y models beat expanding models on 2027+ data (operator hypothesis; successor to H-I1 via new data + 2024/2026 AUC sliver) | PRE-REGISTERED (standing) | — | rule frozen before any test data exists; evaluate after 2029 | `FMROS/appendix/experiment_registry/entries/C3-008.md` |
| C4-SCREEN | P10 | (exploratory screen) rolling vs expanding window, full X1 strategy, even years | closed | **REFUTED (screen)** | expanding +0.128R vs rolling-5 −0.058R vs rolling-2 −0.145R; monotone, shorter=worse; recent-year sliver (2024 rolling5 +0.351R/n21) survives only to keep C3-008 alive | `research/cycle3/results/screen_rolling_x1.json` |
| C5-001 | P10 | (exploratory mapping) window×interval×28-metric; which metrics predict forward? | closed | mapping (no verdict) | return metrics discriminate (ρ=1.0) but don't predict (ρ_fwd≈0.14); only risk/tail predict (max-DD, consec-losses, tail-ratio); edge decayed for ALL windows post-2022 | `research/cycle5/results/`, `report/CYCLE5.md` |

## Notes

* X1/C3-001 lesson (program-level): a vol-gated strategy's profits
  concentrate in vol regimes *by design*; per-year significance therefore
  needs many regime episodes, i.e., calendar time. This is a structural
  patience requirement, not a fixable analysis defect.
* The INVALID mechanism has been exercised twice pre-registry (cycle-2
  d1b feature-name crash; x1 stats-column crash) — both were run
  failures, fixed and re-run; no silent-wrong-number incident is known.
