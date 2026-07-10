"""CYCLE-3 CONFIRMATION: X1 composite on odd test years (pre-registered).

Registered BEFORE running (see FMROS/appendix/experiment_registry/):

  Specification is FROZEN from cycle-2 X1. No parameter may move:
    G2 barriers TP 3xATR / SL 2xATR / 64-bar horizon;
    LightGBM params & threshold protocol identical to cycle-2;
    HMM 3-state gate, causal forward filter, P(high-vol) > 0.6;
    limit entry delta = 0.10xATR, cancel after 4 bars, trade-through fill,
    1.25bp cost on fills; pre-weekend flatten; 10x leverage cap;
    max 5 signals/day; no Friday >= 15:00 arming.

  Test sample: ODD years 2015-2025 (never used as test years in cycle 2).
  Models are re-trained per fold on all data strictly before the test year
  (nested walk-forward; purge 96 bars), so no odd-year observation is used
  to evaluate a model that saw it in training.

  DECLARED DECISION RULE (fixed in advance):
    CONFIRM  if net expectancy > 0 AND day-block bootstrap p(<=0) < 0.05
             on the pooled odd-year trades;
    WEAKEN   if expectancy > 0 but p >= 0.05;
    REFUTE   if expectancy <= 0.
  Secondary (reported, not gating): year-block bootstrap; per-year table;
  comparison of even-year (pilot) vs odd-year (confirmation) expectancy.

  Known residual caveat (stated up front): odd-year data influenced
  cycle-2 only through TRAINING sets of later folds and through the
  composite's design being motivated by pooled cycle-2 behaviour. This is
  the strongest confirmation available without post-2026 data; 2027+
  remains the final arbiter.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
import backtest  # noqa: E402
import wf  # noqa: E402
from features import atr  # noqa: E402

CY2 = Path(__file__).parents[1] / "cycle2"
sys.path.insert(0, str(CY2))
from exp_b2_barriers import barrier_labels, GEOMETRIES  # noqa: E402
from exp_x1_composite_pilot import hmm_gate, simulate_composite  # noqa: E402
import exp_x1_composite_pilot as x1  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
ODD_YEARS = list(range(2015, 2026, 2))


def main() -> None:
    df, feats = wf.load()
    bars = pd.read_parquet(HERE.parents[1] / "research" / "data" / "xauusd_15m.parquet")
    bars = bars.loc[df.index]
    a14 = atr(bars, 14)

    g = GEOMETRIES["G2"]
    labs = barrier_labels(bars, a14, g["tp"], g["sl"], g["h"])
    d2 = df.copy()
    d2["y_tp_long"] = labs.y_tp_long
    d2["y_tp_short"] = labs.y_tp_short

    backtest.TP_R, backtest.SL_R, backtest.HORIZON = g["tp"] / g["sl"], 1.0, g["h"]
    backtest.atr14 = lambda x, _a=a14: _a.reindex(x.index) * g["sl"]
    wf.TEST_YEARS = ODD_YEARS
    x1.FOLD_YEARS = ODD_YEARS

    folds = backtest.walk_forward_predictions(d2, feats)
    pred = pd.concat([f.pred for f in folds])
    thr = {f.test_year: f.threshold for f in folds}

    gate = hmm_gate(df, bars)          # x1.FOLD_YEARS now = odd years
    m1arrs = x1.load_m1_arrays()
    a14s = a14 * g["sl"]

    trades, daily = simulate_composite(d2, pred, thr, gate, a14s, m1arrs)
    out = {"spec": "frozen cycle-2 X1", "test_years": ODD_YEARS,
           "n_trades": int(len(trades))}
    if len(trades):
        out["stats"] = backtest.perf_stats(trades, daily)
        out["bootstrap_dayblock"] = backtest.block_bootstrap_ci(trades)
        rng = np.random.default_rng(7)
        gs = [v.r_mult.to_numpy() for _, v in trades.groupby("year")]
        means = np.array([np.concatenate([gs[j] for j in rng.integers(0, len(gs), len(gs))]).mean()
                          for _ in range(5000)])
        out["bootstrap_yearblock"] = dict(
            ci_lo=float(np.quantile(means, .025)),
            ci_hi=float(np.quantile(means, .975)),
            p_leq_0=float((means <= 0).mean()))
        out["by_year"] = {str(y): dict(n=int(len(v)), exp=float(v.r_mult.mean()))
                          for y, v in trades.groupby("year")}
        exp = out["stats"]["expectancy_R"]
        p = out["bootstrap_dayblock"]["p_leq_0"]
        out["verdict"] = ("CONFIRM" if exp > 0 and p < 0.05
                          else "WEAKEN" if exp > 0 else "REFUTE")
        trades.to_parquet(RESULTS / "c3_trades.parquet")
        daily.rename("equity").to_frame().to_parquet(RESULTS / "c3_equity.parquet")
    else:
        out["verdict"] = "REFUTE (no trades)"
    (RESULTS / "c3_x1_confirmation.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "by_year"},
                     indent=1, default=float))
    print("by_year:", json.dumps(out.get("by_year", {}), default=float))


if __name__ == "__main__":
    main()
