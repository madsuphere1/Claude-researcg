"""End-to-end window comparison: expanding vs rolling-5 vs rolling-2,
full X1 strategy, ALL years 2014-2026, identical treatment.

This is a COMPLETE DESCRIPTIVE backtest of the three training-window
policies on all available history — the "whole end to end" the operator
asked for. It is NOT a new confirmatory test: the confirmatory arbiter
for the window question is registry C3-008 (expanding vs rolling-5 on
2027+ data, rule frozen before that data exists). Cycle 4's screen
already refuted rolling on the even-year slice; this run extends the
same picture to every year with the full metrics suite so the operator
can see it head-to-head.

All three variants are simulated the SAME way: one continuous pass over
2014-2026 (unlike the anchor runs, which sliced even/odd separately — so
the expanding number here may differ trivially from the pooled 748-trade
figure because positions are no longer cut at the even/odd boundary).
Only the classifier training window differs between variants.

Run:  python3 strategy/window_comparison.py
First run generates rolling predictions for 2x13 folds (~15-25 min) and
caches them under strategy/artifacts/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO / "strategy"))
sys.path.insert(0, str(REPO / "research"))
sys.path.insert(0, str(REPO / "research" / "cycle2"))
sys.path.insert(0, str(REPO / "research" / "cycle3"))

import wf  # noqa: E402
from xauusd_x1_final import (ART, build_engine, full_report,  # noqa: E402
                             simulate)
from screen_rolling_x1 import rolling_predictions  # noqa: E402

ALL_YEARS = list(range(2014, 2027))
WINDOWS = [5, 2]


def get_rolling(d2, feats, W):
    """Rolling predictions for ALL years, cached."""
    pf = ART / f"pred_rolling{W}_allyears.parquet"
    tf = ART / f"thr_rolling{W}_allyears.json"
    if pf.exists():
        return pd.read_parquet(pf), {int(k): v for k, v in json.loads(tf.read_text()).items()}
    folds = rolling_predictions(d2, feats, W, ALL_YEARS)
    pred = pd.concat([f.pred for f in folds])
    thr = {f.test_year: f.threshold for f in folds}
    pred.to_parquet(pf)
    tf.write_text(json.dumps(thr))
    return pred, thr


def summary_row(name, rep):
    s = rep["stats"]
    return dict(variant=name, trades=rep["n_trades"],
                expectancy_R=round(s["expectancy_R"], 4),
                win_rate=round(s["win_rate"], 4),
                profit_factor=round(s["profit_factor"], 3),
                sharpe=round(s["sharpe"], 3),
                max_dd=round(s["max_dd"], 4),
                p_dayblock=round(rep["bootstrap_dayblock"]["p_leq_0"], 4),
                p_yearblock=round(rep["bootstrap_yearblock"]["p_leq_0"], 4),
                exp_at_2_5bp=round(rep["cost_tiers"]["2.5bp"]["expectancy_R"], 4))


def main() -> None:
    print("building engine (expanding predictions cached)...")
    d2, bars, a14s, pred_exp, thr_exp, gate, m1arrs = build_engine()
    _, feats = wf.load()

    reports, rows = {}, []

    # --- expanding, continuous all-years pass ---
    tr, dl = simulate(d2, pred_exp, thr_exp, gate, a14s, m1arrs)
    reports["expanding"] = full_report("EXPANDING (all years, continuous)", tr, dl)
    tr.to_parquet(ART / "trades_expanding_allyears.parquet")
    rows.append(summary_row("expanding", reports["expanding"]))

    # --- rolling variants ---
    for W in WINDOWS:
        pred, thr = get_rolling(d2, feats, W)
        tr, dl = simulate(d2, pred, thr, gate, a14s, m1arrs)
        reports[f"rolling{W}"] = full_report(f"ROLLING-{W}Y (all years, continuous)", tr, dl)
        tr.to_parquet(ART / f"trades_rolling{W}_allyears.parquet")
        rows.append(summary_row(f"rolling{W}", reports[f"rolling{W}"]))

    # --- head-to-head, all years and recent-only ---
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD (all years 2014-2026, continuous, 1.25bp)")
    print("=" * 70)
    hdr = f"{'variant':>11} {'trades':>7} {'exp/R':>8} {'win%':>6} {'PF':>5} {'Sharpe':>7} {'maxDD':>7} {'p_day':>6} {'@2.5bp':>7}"
    print(hdr)
    for r in rows:
        print(f"{r['variant']:>11} {r['trades']:>7} {r['expectancy_R']:>+8.3f} "
              f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
              f"{r['sharpe']:>7.2f} {r['max_dd']*100:>6.1f}% "
              f"{r['p_dayblock']:>6.3f} {r['exp_at_2_5bp']:>+7.3f}")

    # per-year expectancy matrix
    print("\nPER-YEAR NET EXPECTANCY (R/trade, 1.25bp), n in parens")
    years = ALL_YEARS
    print(f"{'year':>6} " + " ".join(f"{v:>16}" for v in ("expanding", "rolling5", "rolling2")))
    for y in years:
        cells = []
        for v in ("expanding", "rolling5", "rolling2"):
            by = reports[v]["by_year"].get(str(y))
            cells.append(f"{by['exp']:>+8.3f}({by['n']:>3})" if by else f"{'—':>13}")
        print(f"{y:>6} " + " ".join(f"{c:>16}" for c in cells))

    out = {"declared": "descriptive end-to-end, all years; arbiter is C3-008 on 2027+",
           "summary": rows,
           "per_year": {v: reports[v]["by_year"] for v in reports},
           "full": reports}
    (ART / "window_comparison.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\nsaved {ART / 'window_comparison.json'}")


if __name__ == "__main__":
    main()
