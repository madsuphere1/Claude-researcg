# Anchor Set Smoke Test

Run when: any change to `research/backtest.py`, `research/features.py`,
`research/wf.py`, dataset parquets, or library versions.

| Anchor | Command (repo root) | Expected (tolerance ±0.005R / ±0.002 AUC) |
|---|---|---|
| Cycle-1 baseline | `cd research && python3 backtest.py` (folds only for smoke: check fold 2014 thr≈0.792) | exp −0.105R @2.5bp; 2,671 trades |
| B2v2 managed G2 | `cd research/cycle2 && python3 exp_b2v2_gap_managed.py` | −0.027R @2.5bp; 1,413 trades |
| A1 gate | `cd research/cycle2 && python3 exp_a1_hmm_regimes.py` | gated +0.053R; 693 trades |
| B1 δ=0.10 | `cd research/cycle2 && python3 exp_b1_limit_entry.py` | ITT uplift +0.088R; fill 88% |
| X1 pilot | `cd research/cycle2 && python3 exp_x1_composite_pilot.py` | +0.128R; 414 trades |
| C3 confirmation | `cd research/cycle3 && python3 exp_c3_x1_confirmation.py` | +0.079R; 334 trades; verdict WEAKEN |

Full runs are hours; for smoke purposes the first fold's threshold and
the final trade count/expectancy are the checked invariants. Any
unexplained delta blocks the change (Volume 5 §5.7).
