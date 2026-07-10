"""C7-002 — Faithful replication of the operator's MT5 EA (ba4eff4f-fund.mq5)
backtested on real XAUUSD 15m, 2010-2026, all years.

The EA is LONG-ONLY momentum: enter long when a 5-condition trend score
>= 0.5; SL -2%, TP +5% (2.5R); break-even at +1%; trailing stop from +2%;
one position; London-session hours; prop guardrails (2 consec-loss halt,
daily equity stop). No sell logic.

Purpose: show what a long-only strategy does across gold's real regimes —
the 2013-2015 bear vs the 2023-2025 bull — so "it works" can be located
in beta (gold went up) vs alpha (timing edge).

Intrabar via M15 high/low, pessimistic ordering (SL checked before TP).
Timezone caveat: session filter applied on the data's hour; broker
server time may differ, which shifts *which* bars qualify but not the
long-only regime dependence that dominates the result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).parents[2]
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

SL_PCT, TP_R = 0.02, 2.5
BE_TRIG, BE_SL = 0.01, 0.0005
TRAIL_TRIG, TRAIL_BUF = 0.02, 0.01
RISK = 0.005                       # 0.5% per trade
DAILY_STOP = -0.05                 # ~prop daily drawdown proxy (% of equity)
MAX_CONSEC = 2
H_START, H_END = 2, 10


def wilder(s, n):
    return s.ewm(alpha=1 / n, adjust=False).mean()


def indicators(b):
    c, h, l = b.close, b.high, b.low
    d = pd.DataFrame(index=b.index)
    # RSI
    dl = c.diff()
    d["rsi"] = 100 - 100 / (1 + wilder(dl.clip(lower=0), 14) / wilder(-dl.clip(upper=0), 14))
    # MACD
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["macd"] = macd
    d["macd_sig"] = macd.ewm(span=9, adjust=False).mean()
    # MAs
    d["sma20"] = c.rolling(20).mean()
    d["ema12"] = c.ewm(span=12, adjust=False).mean()
    d["ema26"] = c.ewm(span=26, adjust=False).mean()
    d["bb_mid"] = c.rolling(20).mean()
    # ADX
    up, dn = h.diff(), -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = wilder(tr, 14)
    plus_di = 100 * wilder(pd.Series(plus_dm, index=b.index), 14) / atr
    minus_di = 100 * wilder(pd.Series(minus_dm, index=b.index), 14) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    d["adx"] = wilder(dx.fillna(0), 14)
    return d


def signal(row, price):
    s = 0.0
    if 35 < row.rsi < 70:
        s += 0.15
    if row.macd > row.macd_sig:
        s += 0.20
    if price > row.sma20 and row.ema12 > row.ema26:
        s += 0.25
    if row.adx > 25:
        s += 0.15
    if price > row.bb_mid:
        s += 0.25
    return s


def main() -> None:
    b = pd.read_parquet(REPO / "research" / "data" / "xauusd_15m.parquet")
    b = b[b.index.year >= 2010]
    d = indicators(b)
    df = b.join(d).dropna()
    o, h, l, c = (df.open.to_numpy(), df.high.to_numpy(),
                  df.low.to_numpy(), df.close.to_numpy())
    idx = df.index
    hour, dow = idx.hour.to_numpy(), idx.dayofweek.to_numpy()
    day = idx.date
    rows = df[["rsi", "macd", "macd_sig", "sma20", "ema12", "ema26",
               "bb_mid", "adx"]]

    equity = 1.0
    trades = []
    pos = None
    cur_day = None
    day_consec = 0
    day_start_eq = 1.0
    halted = False
    n = len(df)
    for i in range(n):
        if day[i] != cur_day:
            cur_day, day_consec, day_start_eq, halted = day[i], 0, equity, False

        # ---- manage open position (intrabar, pessimistic) ----
        if pos is not None:
            exit_px = reason = None
            if l[i] <= pos["sl"]:
                exit_px, reason = pos["sl"], "sl/trail"
            elif h[i] >= pos["tp"]:
                exit_px, reason = pos["tp"], "tp"
            else:
                hi = h[i]
                if hi >= pos["entry"] * (1 + TRAIL_TRIG):
                    pos["sl"] = max(pos["sl"], c[i] * (1 - TRAIL_BUF))
                elif hi >= pos["entry"] * (1 + BE_TRIG) and pos["sl"] < pos["entry"]:
                    pos["sl"] = pos["entry"] * (1 + BE_SL)
            if exit_px is not None:
                pret = exit_px / pos["entry"] - 1
                acct = pret / SL_PCT * RISK          # 0.5% risk at -2% stop
                equity *= 1 + acct
                trades.append(dict(t=idx[i], year=int(idx[i].year),
                                   ret=pret, acct=acct, reason=reason))
                day_consec = day_consec + 1 if acct < 0 else 0
                pos = None

        # ---- daily guardrails ----
        if equity / day_start_eq - 1 <= DAILY_STOP:
            halted = True
        if day_consec >= MAX_CONSEC:
            halted = True

        # ---- entry ----
        if pos is None and not halted:
            skip = (hour[i] == 23) or (hour[i] < H_START) or (hour[i] >= H_END) or (dow[i] == 3)
            if not skip and signal(rows.iloc[i], c[i]) >= 0.50:
                entry = c[i]
                pos = dict(entry=entry, sl=entry * (1 - SL_PCT),
                           tp=entry * (1 + SL_PCT * TP_R))
    T = pd.DataFrame(trades)
    by_year = {}
    for y, g in T.groupby("year"):
        eq = (1 + g.acct).prod()
        by_year[str(y)] = dict(n=int(len(g)), win=round(float((g.acct > 0).mean()), 3),
                               ret_pct=round((eq - 1) * 100, 2),
                               avg_R=round(float((g.ret / SL_PCT).mean()), 3))
    # full-period equity / maxDD
    eqc = (1 + T.acct).cumprod()
    dd = (eqc / eqc.cummax() - 1).min()
    out = dict(
        n_trades=int(len(T)),
        total_return_pct=round((eqc.iloc[-1] - 1) * 100, 2),
        win_rate=round(float((T.acct > 0).mean()), 3),
        max_drawdown_pct=round(dd * 100, 2),
        avg_R=round(float((T.ret / SL_PCT).mean()), 4),
        bull_2023_2025=round((np.prod([1 + T[T.year.isin([2023, 2024, 2025])].acct]) - 1) * 100, 2)
        if len(T[T.year.isin([2023, 2024, 2025])]) else None,
        bear_2013_2015=round((np.prod([1 + T[T.year.isin([2013, 2014, 2015])].acct]) - 1) * 100, 2)
        if len(T[T.year.isin([2013, 2014, 2015])]) else None,
        by_year=by_year,
    )
    (RESULTS / "c7_ea_replication.json").write_text(json.dumps(out, indent=1))

    print(f"EA replication: {out['n_trades']} trades 2010-2026")
    print(f"total return {out['total_return_pct']:+.1f}%  win {out['win_rate']:.1%}  "
          f"maxDD {out['max_drawdown_pct']:.1f}%  avgR {out['avg_R']:+.3f}")
    print(f"\nBULL 2023-2025: {out['bull_2023_2025']:+.1f}%    "
          f"BEAR 2013-2015: {out['bear_2013_2015']:+.1f}%\n")
    print(f"{'year':>6} {'trades':>7} {'win%':>6} {'return%':>9} {'avgR':>7}  gold")
    goldchg = {2010: "+29%", 2011: "+10%", 2012: "+7%", 2013: "-28%", 2014: "-2%",
               2015: "-11%", 2016: "+8%", 2017: "+13%", 2018: "-2%", 2019: "+18%",
               2020: "+25%", 2021: "-4%", 2022: "-0%", 2023: "+13%", 2024: "+27%",
               2025: "+65%", 2026: "-6%"}
    for y in range(2010, 2027):
        r = by_year.get(str(y))
        if r:
            print(f"{y:>6} {r['n']:>7} {r['win']*100:>5.0f}% {r['ret_pct']:>+8.1f}% "
                  f"{r['avg_R']:>+7.2f}  {goldchg.get(y,'')}")
    print(f"\nsaved {RESULTS/'c7_ea_replication.json'}")


if __name__ == "__main__":
    main()
