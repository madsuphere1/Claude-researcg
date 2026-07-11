"""C7-005 — Regime Rider (C7-004 picked config) over sliding 2/3/5-year
intervals. Shows what the strategy did in every multi-year stretch, so
consistency (not just the full-history total) is visible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from regime_rider import indicators, run  # noqa

RESULTS = Path(__file__).parent / "results"
YEARS = list(range(2010, 2027))
SPANS = [2, 3, 5]


def interval_stats(T, a, b):
    sub = T[(T.year >= a) & (T.year <= b)]
    if len(sub) < 5:
        return None
    eqc = (1 + sub.acct).cumprod()
    dd = (eqc / eqc.cummax() - 1).min()
    tot = eqc.iloc[-1] - 1
    return dict(n=int(len(sub)), ret=round(tot * 100, 1),
                maxdd=round(dd * 100, 1),
                ret_dd=round(tot / abs(dd), 2) if dd < 0 else None,
                win=round(float((sub.acct > 0).mean()) * 100, 0))


def main() -> None:
    b = pd.read_parquet(REPO / "research" / "data" / "xauusd_15m.parquet")
    b = b[b.index.year >= 2010]
    df = b.join(indicators(b)).dropna()
    c = df.close
    sma = c.rolling(1920).mean().values
    up = c.values > sma
    dn = c.values < sma
    up[:1920] = dn[:1920] = False
    T = run(df, up, dn, df.ema26.values, 0.03)   # picked config

    out = {}
    for s in SPANS:
        print(f"\n{'='*58}\nSLIDING {s}-YEAR WINDOWS — Regime Rider\n{'='*58}")
        print(f"{'window':>11}{'trades':>7}{'return':>9}{'maxDD':>8}{'ret/DD':>8}{'win%':>6}")
        grid = {}
        rets = []
        for a in range(YEARS[0], YEARS[-1] - s + 2):
            r = interval_stats(T, a, a + s - 1)
            if r:
                grid[f"{a}-{a+s-1}"] = r
                rets.append(r["ret"])
                rd = f"{r['ret_dd']:>8.2f}" if r["ret_dd"] is not None else f"{'—':>8}"
                print(f"{a}-{a+s-1:>6}{r['n']:>7}{r['ret']:>+8.1f}%{r['maxdd']:>7.1f}%{rd}{r['win']:>5.0f}%")
        pos = sum(x > 0 for x in rets)
        print(f"  -> positive {s}-yr windows: {pos}/{len(rets)}  "
              f"(worst {min(rets):+.1f}%, best {max(rets):+.1f}%, median {np.median(rets):+.1f}%)")
        out[f"span{s}"] = {"grid": grid, "positive": f"{pos}/{len(rets)}",
                           "worst": min(rets), "best": max(rets),
                           "median": float(np.median(rets))}
    RESULTS.joinpath("c7_rider_intervals.json").write_text(json.dumps(out, indent=1))
    print(f"\nsaved {RESULTS/'c7_rider_intervals.json'}")


if __name__ == "__main__":
    main()
