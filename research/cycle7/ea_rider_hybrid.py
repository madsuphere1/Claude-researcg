"""C7-007 — Beat the EA in the recent bull windows with its OWN entries.

The EA caps every winner at +5% (fixed TP). In a trend (gold +65% in
2025) that throws the move away. Hybrid = EA's exact long-only entries +
NO fixed TP, trailing-stop-only exit (rides the trend). Everything else
identical (session, guardrails, 2% stop, 0.5% risk). Head-to-head on the
recent windows the EA won.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from replicate_ea import (DAILY_STOP, H_END, H_START, MAX_CONSEC, RISK,  # noqa
                          SL_PCT, indicators, signal)
from side_by_side import ea_trades, istats  # noqa

RESULTS = Path(__file__).parent / "results"
YEARS = list(range(2010, 2027))
TRAIL = 0.03


def hybrid_trades(df):
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
            pos["peak"] = max(pos["peak"], h[i])
            if pos["peak"] >= pos["entry"] * (1 + TRAIL):
                pos["sl"] = max(pos["sl"], pos["peak"] * (1 - TRAIL))
            if l[i] <= pos["sl"]:                       # trailing/initial stop only
                acct = (pos["sl"] / pos["entry"] - 1) / SL_PCT * RISK
                eq *= 1 + acct
                trades.append(dict(year=int(idx[i].year), acct=acct))
                consec = consec + 1 if acct < 0 else 0
                pos = None
        if eq / eq0 - 1 <= DAILY_STOP or consec >= MAX_CONSEC:
            halt = True
        if pos is None and not halt:
            hh, dw = idx[i].hour, idx[i].dayofweek
            if not ((hh == 23) or (hh < H_START) or (hh >= H_END) or (dw == 3)):
                if signal(rows.iloc[i], c[i]) >= 0.50:   # EA's exact entry
                    e = c[i]
                    pos = dict(entry=e, sl=e * (1 - SL_PCT), peak=e)
    return pd.DataFrame(trades)


def main() -> None:
    b = pd.read_parquet(REPO / "research" / "data" / "xauusd_15m.parquet")
    b = b[b.index.year >= 2010]
    df = b.join(indicators(b)).dropna()
    EA = ea_trades(df)
    HY = hybrid_trades(df)

    recent = {2: ["2024-2025", "2025-2026"], 3: ["2023-2025", "2024-2026"],
              5: ["2021-2025", "2022-2026"]}
    out = {}
    print("RECENT WINDOWS the EA won  —  EA  vs  HYBRID (EA entries, no TP cap)\n")
    print(f"{'window':>11} | {'EA ret':>7} {'EA DD':>7} | {'HY ret':>7} {'HY DD':>7} | winner")
    wins = {"EA": 0, "HYBRID": 0}
    for s, labels in recent.items():
        for lab in labels:
            a, bb = map(int, lab.split("-"))
            e, hy = istats(EA, a, bb), istats(HY, a, bb)
            w = "HYBRID" if hy["ret"] > e["ret"] else "EA"
            wins[w] += 1
            out[lab] = dict(ea=e, hybrid=hy, winner=w)
            print(f"{lab:>11} | {e['ret']:>+6.1f}% {e['dd']:>6.1f}% | "
                  f"{hy['ret']:>+6.1f}% {hy['dd']:>6.1f}% | {w}")
    print(f"\nrecent-window score -> HYBRID {wins['HYBRID']}, EA {wins['EA']}")

    # full picture too
    print(f"\n{'ALL windows (return%): EA vs HYBRID':>40}")
    allres = {}
    for s in (2, 3, 5):
        line = []
        for a in range(YEARS[0], YEARS[-1] - s + 2):
            e, hy = istats(EA, a, a + s - 1), istats(HY, a, a + s - 1)
            if e and hy:
                allres[f"{a}-{a+s-1}"] = dict(span=s, ea=round(e["ret"], 1),
                                              hy=round(hy["ret"], 1),
                                              ea_dd=round(e["dd"], 1), hy_dd=round(hy["dd"], 1))
    out["_all"] = allres
    RESULTS.joinpath("c7_hybrid.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"saved {RESULTS/'c7_hybrid.json'}")


if __name__ == "__main__":
    main()
