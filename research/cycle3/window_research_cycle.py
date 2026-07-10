"""The RESEARCH CYCLE (not the strategy backtest) redone per window.

The strategy backtest (strategy/window_comparison.py) answers "does the
traded P&L differ by window." This answers the deeper, upstream question
each research cycle opens with — C1-BASE: *does a predictive signal
exist at all, and how strong is it* — separately for expanding,
rolling-5, and rolling-2. It re-derives the foundational finding
("WF AUC 0.529, 13/13 years > 0.5, sign-test p") for each window so the
degradation can be located in the SIGNAL, not just the dollars.

Uses the cached walk-forward predictions already produced by
window_comparison.py (no retraining) and the G2 labels from the engine.
Metric is out-of-sample test-year AUC on y_tp_long / y_tp_short.
"""

from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO / "strategy"))
sys.path.insert(0, str(REPO / "research"))
sys.path.insert(0, str(REPO / "research" / "cycle2"))

from xauusd_x1_final import ART, build_engine  # noqa: E402

RESULTS = Path(__file__).parent / "results"
YEARS = list(range(2014, 2027))
WINDOWS = {
    "expanding": ART / "predictions.parquet",
    "rolling5": ART / "pred_rolling5_allyears.parquet",
    "rolling2": ART / "pred_rolling2_allyears.parquet",
}


def sign_test_p(k: int, n: int) -> float:
    """One-sided binomial P(X >= k | p=0.5): probability of >= k years
    above AUC 0.5 by chance if the true signal were absent."""
    return float(sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n)


def main() -> None:
    d2, *_ = build_engine()          # d2 carries G2 y_tp_long / y_tp_short
    yl = d2["y_tp_long"]
    ys = d2["y_tp_short"]

    out = {}
    for name, path in WINDOWS.items():
        pred = pd.read_parquet(path)
        rows = {}
        auc_long, auc_short = [], []
        for y in YEARS:
            m = pred.index.year == y
            idx = pred.index[m]
            pl, ps = pred.p_long[m], pred.p_short[m]
            # align labels, drop NaN (barrier undefined near series end)
            ll, ls = yl.reindex(idx), ys.reindex(idx)
            okl, oks = ll.notna(), ls.notna()
            al = roc_auc_score(ll[okl].astype(int), pl[okl]) if okl.sum() > 100 else float("nan")
            as_ = roc_auc_score(ls[oks].astype(int), ps[oks]) if oks.sum() > 100 else float("nan")
            rows[str(y)] = dict(auc_long=round(al, 4), auc_short=round(as_, 4),
                                n=int(okl.sum()))
            if np.isfinite(al):
                auc_long.append(al)
            if np.isfinite(as_):
                auc_short.append(as_)

        n = len(auc_long)
        k = sum(a > 0.5 for a in auc_long)
        out[name] = dict(
            mean_auc_long=round(float(np.mean(auc_long)), 4),
            mean_auc_short=round(float(np.mean(auc_short)), 4),
            years_above_0p5_long=f"{k}/{n}",
            sign_test_p_long=sign_test_p(k, n),
            worst_year_auc_long=round(float(np.min(auc_long)), 4),
            best_year_auc_long=round(float(np.max(auc_long)), 4),
            by_year=rows,
        )

    # -------- report --------
    print("\nRESEARCH-CYCLE FOUNDATION PER WINDOW (C1-BASE redone)")
    print("out-of-sample test-year AUC on y_tp_long\n")
    print(f"{'window':>11} {'mean AUC':>9} {'yrs>0.5':>8} {'sign p':>9} {'worst':>7} {'best':>7}")
    for name, r in out.items():
        print(f"{name:>11} {r['mean_auc_long']:>9.4f} {r['years_above_0p5_long']:>8} "
              f"{r['sign_test_p_long']:>9.2e} {r['worst_year_auc_long']:>7.4f} "
              f"{r['best_year_auc_long']:>7.4f}")

    print("\nPER-YEAR AUC (y_tp_long)")
    print(f"{'year':>6} " + " ".join(f"{w:>11}" for w in WINDOWS))
    for y in YEARS:
        cells = [out[w]["by_year"][str(y)]["auc_long"] for w in WINDOWS]
        print(f"{y:>6} " + " ".join(f"{c:>11.4f}" for c in cells))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "window_research_cycle.json").write_text(json.dumps(out, indent=1))
    print(f"\nsaved {RESULTS / 'window_research_cycle.json'}")


if __name__ == "__main__":
    main()
