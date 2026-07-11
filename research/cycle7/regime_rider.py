"""C7-004 — Regime Trend Rider: beat the operator EA properly.

Principle (operator's): don't fit all history into one static strategy —
trade the current regime. Design from evidence:
* entry timing has no alpha (placebo) -> enter simply, in regime direction
* long-only is the EA's fatal flaw -> dual-side via a trend regime filter
* FIXED +5% TP throws away trends (gold +65% in 2025) -> NO fixed TP,
  exit only on a trailing stop that rides the trend to its end
* wide stops -> cost-robust

Regime: price vs long trend MA. Entry: buy dips in uptrend / sell rallies
in downtrend (frequent). Exit: initial 2% stop, then a % trailing stop
from the best price reached. Same 0.5% risk, session, guardrails.
Real XAUUSD 15m 2010-2026, M15 intrabar, gross of spread.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from replicate_ea import H_END, H_START, RISK, indicators  # noqa

RESULTS = Path(__file__).parent / "results"
SL_PCT = 0.02
DAILY_STOP = -0.05
MAX_CONSEC = 2


def run(df, up, dn, ema, trail_buf):
    o, h, l, c = df.open.values, df.high.values, df.low.values, df.close.values
    idx = df.index
    day = idx.date
    eq, trades, pos = 1.0, [], None
    cd, consec, eq0, halt = None, 0, 1.0, False
    for i in range(len(df)):
        if day[i] != cd:
            cd, consec, eq0, halt = day[i], 0, eq, False
        if pos is not None:
            s = pos["side"]
            # update best price + trailing stop
            if s == 1:
                pos["peak"] = max(pos["peak"], h[i])
                if pos["peak"] >= pos["entry"] * (1 + trail_buf):
                    pos["sl"] = max(pos["sl"], pos["peak"] * (1 - trail_buf))
                ex = pos["sl"] if l[i] <= pos["sl"] else None
            else:
                pos["peak"] = min(pos["peak"], l[i])
                if pos["peak"] <= pos["entry"] * (1 - trail_buf):
                    pos["sl"] = min(pos["sl"], pos["peak"] * (1 + trail_buf))
                ex = pos["sl"] if h[i] >= pos["sl"] else None
            if ex is not None:
                pret = (ex / pos["entry"] - 1) * s
                acct = pret / SL_PCT * RISK
                eq *= 1 + acct
                trades.append(dict(year=int(idx[i].year), side=s, ret=pret, acct=acct))
                consec = consec + 1 if acct < 0 else 0
                pos = None
        if eq / eq0 - 1 <= DAILY_STOP or consec >= MAX_CONSEC:
            halt = True
        if pos is None and not halt:
            hh, dw = idx[i].hour, idx[i].dayofweek
            if not ((hh == 23) or (hh < H_START) or (hh >= H_END) or (dw == 3)):
                if up[i] and c[i] <= ema[i]:          # buy dip in uptrend
                    pos = dict(side=1, entry=c[i], sl=c[i] * (1 - SL_PCT), peak=c[i])
                elif dn[i] and c[i] >= ema[i]:        # sell rally in downtrend
                    pos = dict(side=-1, entry=c[i], sl=c[i] * (1 + SL_PCT), peak=c[i])
    return pd.DataFrame(trades)


def summ(T):
    eqc = (1 + T.acct).cumprod()
    dd = (eqc / eqc.cummax() - 1).min()
    return dict(n=int(len(T)), total_pct=round((eqc.iloc[-1] - 1) * 100, 2),
                win=round(float((T.acct > 0).mean()), 3), maxdd_pct=round(dd * 100, 2),
                ret_dd=round((eqc.iloc[-1] - 1) / abs(dd), 2),
                bull=round((np.prod(1 + T[T.year.isin([2023, 2024, 2025])].acct) - 1) * 100, 1),
                bear=round((np.prod(1 + T[T.year.isin([2013, 2014, 2015])].acct) - 1) * 100, 1),
                by_year={str(y): round((np.prod(1 + g.acct) - 1) * 100, 1)
                         for y, g in T.groupby("year")})


def main() -> None:
    b = pd.read_parquet(REPO / "research" / "data" / "xauusd_15m.parquet")
    b = b[b.index.year >= 2010]
    df = b.join(indicators(b)).dropna()
    c = df.close
    ema = df.ema26.values
    ea = json.loads((RESULTS / "c7_ea_replication.json").read_text())

    print("REGIME TREND RIDER — config scan (trend window x trail buffer)\n")
    print(f"{'trendW':>7}{'trail':>7}{'trades':>8}{'return%':>9}{'maxDD%':>8}{'ret/DD':>7}{'bull%':>7}{'bear%':>7}")
    results = {}
    for W in (1920, 3840):
        sma = c.rolling(W).mean().values
        up = c.values > sma
        dn = c.values < sma
        up[:W] = dn[:W] = False
        for tb in (0.02, 0.03):
            T = run(df, up, dn, ema, tb)
            r = summ(T)
            results[f"W{W}_tb{tb}"] = r
            print(f"{W:>7}{tb:>7.2f}{r['n']:>8}{r['total_pct']:>+8.1f}%{r['maxdd_pct']:>7.1f}%"
                  f"{r['ret_dd']:>7.2f}{r['bull']:>+6.1f}%{r['bear']:>+6.1f}%")

    print(f"\n{'EA baseline':>21}{ea['total_return_pct']:>+8.1f}%{ea['max_drawdown_pct']:>7.1f}%"
          f"{ea['total_return_pct']/abs(ea['max_drawdown_pct']):>7.2f}"
          f"{ea['bull_2023_2025']:>+6.1f}%{ea['bear_2013_2015']:>+6.1f}%")

    # pick the median-robust config (not the single max) — W=1920, tb=0.03
    pick = results["W1920_tb0.03"]
    out = {"pick": "W1920_tb0.03", "picked_result": pick,
           "ea_baseline": {k: ea[k] for k in ("total_return_pct", "max_drawdown_pct",
                                              "bull_2023_2025", "bear_2013_2015")},
           "all_configs": results}
    RESULTS.joinpath("c7_regime_rider.json").write_text(json.dumps(out, indent=1))
    print(f"\nPICK W1920/tb0.03  per-year vs gold:")
    gold = {2010:'+29',2011:'+10',2012:'+7',2013:'-28',2014:'-2',2015:'-11',2016:'+8',
            2017:'+13',2018:'-2',2019:'+18',2020:'+25',2021:'-4',2022:'-0',2023:'+13',
            2024:'+27',2025:'+65',2026:'-6'}
    for y in range(2010, 2027):
        v = pick["by_year"].get(str(y))
        if v is not None:
            print(f"  {y}: {v:+6.1f}%   gold {gold.get(y,''):>4}%")
    print(f"\nsaved {RESULTS/'c7_regime_rider.json'}")


if __name__ == "__main__":
    main()
