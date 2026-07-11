"""C7-006 — Side-by-side: operator EA vs Regime Rider over sliding
2/3/5-year windows. Same data, same harness, same guardrails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from replicate_ea import (BE_SL, BE_TRIG, DAILY_STOP, H_END, H_START,  # noqa
                          MAX_CONSEC, RISK, SL_PCT, TP_R, TRAIL_BUF,
                          TRAIL_TRIG, indicators, signal)
from regime_rider import run as rider_run  # noqa

RESULTS = Path(__file__).parent / "results"
YEARS = list(range(2010, 2027))
SPANS = [2, 3, 5]


def ea_trades(df):
    o, h, l, c = df.open.values, df.high.values, df.low.values, df.close.values
    idx = df.index
    day = idx.date
    rows = df[["rsi", "macd", "macd_sig", "sma20", "ema12", "ema26", "bb_mid", "adx"]]
    eq, trades, pos = 1.0, [], None
    cd, consec, eq0, halt = None, 0, 1.0, False
    for i in range(len(df)):
        if day[i] != cd:
            cd, consec, eq0, halt = day[i], 0, eq, False
        if pos is not None:
            ex = None
            if l[i] <= pos["sl"]:
                ex = pos["sl"]
            elif h[i] >= pos["tp"]:
                ex = pos["tp"]
            else:
                hi = h[i]
                if hi >= pos["entry"] * (1 + TRAIL_TRIG):
                    pos["sl"] = max(pos["sl"], c[i] * (1 - TRAIL_BUF))
                elif hi >= pos["entry"] * (1 + BE_TRIG) and pos["sl"] < pos["entry"]:
                    pos["sl"] = pos["entry"] * (1 + BE_SL)
            if ex is not None:
                acct = (ex / pos["entry"] - 1) / SL_PCT * RISK
                eq *= 1 + acct
                trades.append(dict(year=int(idx[i].year), acct=acct))
                consec = consec + 1 if acct < 0 else 0
                pos = None
        if eq / eq0 - 1 <= DAILY_STOP or consec >= MAX_CONSEC:
            halt = True
        if pos is None and not halt:
            hh, dw = idx[i].hour, idx[i].dayofweek
            if not ((hh == 23) or (hh < H_START) or (hh >= H_END) or (dw == 3)):
                if signal(rows.iloc[i], c[i]) >= 0.50:
                    e = c[i]
                    pos = dict(entry=e, sl=e * (1 - SL_PCT), tp=e * (1 + SL_PCT * TP_R))
    return pd.DataFrame(trades)


def istats(T, a, b):
    sub = T[(T.year >= a) & (T.year <= b)]
    if len(sub) < 3:
        return None
    eqc = (1 + sub.acct).cumprod()
    dd = (eqc / eqc.cummax() - 1).min()
    tot = eqc.iloc[-1] - 1
    return dict(ret=tot * 100, dd=dd * 100,
                rdd=(tot / abs(dd)) if dd < 0 else float("nan"))


def main() -> None:
    b = pd.read_parquet(REPO / "research" / "data" / "xauusd_15m.parquet")
    b = b[b.index.year >= 2010]
    df = b.join(indicators(b)).dropna()
    c = df.close
    sma = c.rolling(1920).mean().values
    up, dn = c.values > sma, c.values < sma
    up[:1920] = dn[:1920] = False

    EA = ea_trades(df)
    RD = rider_run(df, up, dn, df.ema26.values, 0.03)

    out = {}
    for s in SPANS:
        print(f"\n{'='*70}\nSLIDING {s}-YEAR WINDOWS  —  EA  vs  REGIME RIDER\n{'='*70}")
        print(f"{'window':>11} | {'EA ret':>7} {'EA DD':>7} {'EA r/DD':>7} | "
              f"{'RD ret':>7} {'RD DD':>7} {'RD r/DD':>7} | winner")
        g = {}
        ea_wins = rd_wins = 0
        for a in range(YEARS[0], YEARS[-1] - s + 2):
            e, r = istats(EA, a, a + s - 1), istats(RD, a, a + s - 1)
            if not e or not r:
                continue
            win = "RIDER" if r["rdd"] > e["rdd"] or (np.isnan(e["rdd"])) else "EA"
            # prefer higher ret/DD; if EA negative dd inf handling
            better_rd = (r["ret"] / abs(r["dd"])) if r["dd"] < 0 else 99
            better_ea = (e["ret"] / abs(e["dd"])) if e["dd"] < 0 else 99
            win = "RIDER" if better_rd >= better_ea else "EA"
            if win == "RIDER":
                rd_wins += 1
            else:
                ea_wins += 1
            g[f"{a}-{a+s-1}"] = dict(ea=e, rd=r, winner=win)
            print(f"{a}-{a+s-1:>6} | {e['ret']:>+6.1f}% {e['dd']:>6.1f}% {better_ea:>7.2f} | "
                  f"{r['ret']:>+6.1f}% {r['dd']:>6.1f}% {better_rd:>7.2f} | {win}")
        print(f"  windows won -> RIDER {rd_wins}, EA {ea_wins}")
        out[f"span{s}"] = {"grid": g, "rider_wins": rd_wins, "ea_wins": ea_wins}

    RESULTS.joinpath("c7_side_by_side.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\nsaved {RESULTS/'c7_side_by_side.json'}")


if __name__ == "__main__":
    main()
