"""C7-003 — My counter-strategy vs the operator's fund EA.

Evidence-driven design (not more indicators):
* Entry timing adds no alpha (C7-002 placebo) -> compete on DIRECTION.
* Operator EA is long-only -> loses every gold bear (2013/2015).
* Fix: REGIME-ALIGNED DUAL SIDE. Long when price > long trend MA, short
  when below. Keep the proven wide-stop / trailing / guardrail structure.

Same harness as replicate_ea: real XAUUSD 15m 2010-2026, M15 intrabar,
pessimistic ordering, 0.5% risk, 2% stop / 2.5R TP, break-even, trailing,
session filter, 2-consec-loss halt, daily equity stop. Head-to-head +
placebo on THIS strategy too (honesty).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from replicate_ea import (BE_SL, BE_TRIG, H_END, H_START, RISK, SL_PCT,  # noqa
                          TP_R, TRAIL_BUF, TRAIL_TRIG, indicators)

RESULTS = Path(__file__).parent / "results"
TREND_W = 1920            # ~20 trading days: the secular-trend filter
DAILY_STOP = -0.05
MAX_CONSEC = 2


def backtest(df, entries_long, entries_short, seedless=True):
    o, h, l, c = df.open.values, df.high.values, df.low.values, df.close.values
    idx = df.index
    day = idx.date
    equity, trades, pos = 1.0, [], None
    cur_day, consec, day_eq0, halted = None, 0, 1.0, False
    for i in range(len(df)):
        if day[i] != cur_day:
            cur_day, consec, day_eq0, halted = day[i], 0, equity, False
        if pos is not None:
            side = pos["side"]
            ex = None
            if side == 1:
                if l[i] <= pos["sl"]:
                    ex = pos["sl"]
                elif h[i] >= pos["tp"]:
                    ex = pos["tp"]
                else:
                    if h[i] >= pos["entry"] * (1 + TRAIL_TRIG):
                        pos["sl"] = max(pos["sl"], c[i] * (1 - TRAIL_BUF))
                    elif h[i] >= pos["entry"] * (1 + BE_TRIG) and pos["sl"] < pos["entry"]:
                        pos["sl"] = pos["entry"] * (1 + BE_SL)
            else:  # short
                if h[i] >= pos["sl"]:
                    ex = pos["sl"]
                elif l[i] <= pos["tp"]:
                    ex = pos["tp"]
                else:
                    if l[i] <= pos["entry"] * (1 - TRAIL_TRIG):
                        pos["sl"] = min(pos["sl"], c[i] * (1 + TRAIL_BUF))
                    elif l[i] <= pos["entry"] * (1 - BE_TRIG) and pos["sl"] > pos["entry"]:
                        pos["sl"] = pos["entry"] * (1 - BE_SL)
            if ex is not None:
                pret = (ex / pos["entry"] - 1) * side
                acct = pret / SL_PCT * RISK
                equity *= 1 + acct
                trades.append(dict(year=int(idx[i].year), side=side, ret=pret, acct=acct))
                consec = consec + 1 if acct < 0 else 0
                pos = None
        if equity / day_eq0 - 1 <= DAILY_STOP or consec >= MAX_CONSEC:
            halted = True
        if pos is None and not halted:
            hh, dw = idx[i].hour, idx[i].dayofweek
            if not ((hh == 23) or (hh < H_START) or (hh >= H_END) or (dw == 3)):
                if entries_long[i]:
                    e = c[i]
                    pos = dict(side=1, entry=e, sl=e * (1 - SL_PCT), tp=e * (1 + SL_PCT * TP_R))
                elif entries_short[i]:
                    e = c[i]
                    pos = dict(side=-1, entry=e, sl=e * (1 + SL_PCT), tp=e * (1 - SL_PCT * TP_R))
    return pd.DataFrame(trades), equity


def summarize(T):
    eqc = (1 + T.acct).cumprod()
    dd = (eqc / eqc.cummax() - 1).min()
    return dict(n=int(len(T)), total_pct=round((eqc.iloc[-1] - 1) * 100, 2),
                win=round(float((T.acct > 0).mean()), 3),
                maxdd_pct=round(dd * 100, 2),
                avgR=round(float((T.ret / SL_PCT).mean()), 4),
                bull_23_25=round((np.prod(1 + T[T.year.isin([2023, 2024, 2025])].acct) - 1) * 100, 2),
                bear_13_15=round((np.prod(1 + T[T.year.isin([2013, 2014, 2015])].acct) - 1) * 100, 2))


def main() -> None:
    b = pd.read_parquet(REPO / "research" / "data" / "xauusd_15m.parquet")
    b = b[b.index.year >= 2010]
    d = indicators(b)
    df = b.join(d).dropna()
    c = df.close
    trend = c.rolling(TREND_W).mean()
    up = (c > trend).values
    dn = (c < trend).values
    ema12 = df.ema12.values
    cprev = np.r_[c.values[0], c.values[:-1]]
    eprev = np.r_[ema12[0], ema12[:-1]]
    # trend-aligned pullback entries (cross of price & ema12 in trend dir)
    long_e = up & (c.values > ema12) & (cprev <= eprev)
    short_e = dn & (c.values < ema12) & (cprev >= eprev)
    long_e[:TREND_W] = short_e[:TREND_W] = False

    T, _ = backtest(df, long_e, short_e)
    mine = summarize(T)
    # side split
    mine["long_share"] = round(float((T.side == 1).mean()), 3)
    mine["by_year"] = {str(y): dict(n=int(len(g)), ret=round((np.prod(1 + g.acct) - 1) * 100, 2),
                                    win=round(float((g.acct > 0).mean()), 3))
                       for y, g in T.groupby("year")}

    # placebo on MY strategy: random entries in the SAME trend direction
    rng = np.random.default_rng(0)
    plc = []
    for s in range(15):
        rng = np.random.default_rng(s)
        rl = up & (rng.random(len(df)) < 0.004)
        rs = dn & (rng.random(len(df)) < 0.004)
        rl[:TREND_W] = rs[:TREND_W] = False
        Tp, _ = backtest(df, rl, rs)
        if len(Tp):
            plc.append((Tp.ret / SL_PCT).mean())
    mine["placebo_random_same_regime_avgR"] = round(float(np.mean(plc)), 4)
    mine["placebo_std"] = round(float(np.std(plc)), 4)

    ea = json.loads((RESULTS / "c7_ea_replication.json").read_text())
    RESULTS.joinpath("c7_superior.json").write_text(json.dumps(mine, indent=1))

    print("HEAD-TO-HEAD  (real XAUUSD 15m 2010-2026, gross of spread)\n")
    print(f"{'':>22}{'operator EA':>14}{'mine (dual-trend)':>20}")
    print(f"{'total return':>22}{ea['total_return_pct']:>13.1f}%{mine['total_pct']:>19.1f}%")
    print(f"{'max drawdown':>22}{ea['max_drawdown_pct']:>13.1f}%{mine['maxdd_pct']:>19.1f}%")
    print(f"{'return / |DD|':>22}{ea['total_return_pct']/abs(ea['max_drawdown_pct']):>14.2f}{mine['total_pct']/abs(mine['maxdd_pct']):>20.2f}")
    print(f"{'win rate':>22}{ea['win_rate']*100:>13.1f}%{mine['win']*100:>19.1f}%")
    print(f"{'trades':>22}{ea['n_trades']:>14}{mine['n']:>20}")
    print(f"{'BULL 2023-2025':>22}{ea['bull_2023_2025']:>13.1f}%{mine['bull_23_25']:>19.1f}%")
    print(f"{'BEAR 2013-2015':>22}{ea['bear_2013_2015']:>13.1f}%{mine['bear_13_15']:>19.1f}%")
    print(f"\nmine long share: {mine['long_share']:.0%}  |  "
          f"placebo (random same-regime) avgR {mine['placebo_random_same_regime_avgR']:+.3f} "
          f"vs mine avgR {mine['avgR']:+.3f}")
    print(f"\n{'year':>6}{'mine ret%':>11}{'mine win':>9}{'  gold'}")
    gold = {2010:'+29%',2011:'+10%',2012:'+7%',2013:'-28%',2014:'-2%',2015:'-11%',
            2016:'+8%',2017:'+13%',2018:'-2%',2019:'+18%',2020:'+25%',2021:'-4%',
            2022:'-0%',2023:'+13%',2024:'+27%',2025:'+65%',2026:'-6%'}
    for y in range(2010, 2027):
        r = mine["by_year"].get(str(y))
        if r:
            print(f"{y:>6}{r['ret']:>+10.1f}%{r['win']*100:>8.0f}%  {gold.get(y,'')}")
    print(f"\nsaved {RESULTS/'c7_superior.json'}")


if __name__ == "__main__":
    main()
