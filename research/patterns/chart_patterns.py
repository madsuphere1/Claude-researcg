"""16 chart patterns a trader SEES on the candles, with multi-timeframe
(1h/4h/1day) trend confirmation. Detection is deterministic (structural
rules on raw OHLC). For each pattern we then measure the HONEST forward
behavior: directional hit rate and triple-barrier expectancy over the
next H bars — no fitting, no cherry-picking, all of 2010-2026.

Also demonstrates the win-rate trap: what TP/SL gives "90% win rate" and
what its expectancy actually is.
Instrument: XAUUSD 15m.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).parents[2]
RESULTS = Path(__file__).parent / "results"
H = 16                       # forward horizon (4 hours)
COST = 2.5e-4


def atr(df, n=14):
    pc = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def detect(df):
    o, h, l, c = df.open, df.high, df.low, df.close
    o1, h1, l1, c1 = o.shift(1), h.shift(1), l.shift(1), c.shift(1)
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    hi4, lo4 = h.rolling(4).max().shift(1), l.rolling(4).min().shift(1)      # 1h
    hi16, lo16 = h.rolling(16).max().shift(1), l.rolling(16).min().shift(1)  # 4h
    hi96, lo96 = h.rolling(96).max().shift(1), l.rolling(96).min().shift(1)  # 1day
    avg_rng = rng.rolling(20).mean()
    up = (c > ema50) & (ema20 > ema50)          # 1h-scale trend
    dn = (c < ema50) & (ema20 < ema50)
    lowick = (np.minimum(o, c) - l)
    upwick = (h - np.maximum(o, c))

    P = {}
    P["01_uptrend"] = (up, +1)
    P["02_downtrend"] = (dn, -1)
    P["03_break_4h_high"] = (c > hi16, +1)
    P["04_break_4h_low"] = (c < lo16, -1)
    P["05_bull_engulf"] = ((c > o) & (c1 < o1) & (c >= o1) & (o <= c1), +1)
    P["06_bear_engulf"] = ((c < o) & (c1 > o1) & (c <= o1) & (o >= c1), -1)
    P["07_hammer"] = ((lowick > 2 * body) & (lowick / rng > 0.5), +1)
    P["08_shooting_star"] = ((upwick > 2 * body) & (upwick / rng > 0.5), -1)
    P["09_inside_bar"] = ((h < h1) & (l > l1), 0)
    P["10_range_compress"] = ((rng < 0.6 * avg_rng) & (atr(df) / c < (atr(df) / c).rolling(96).median()), 0)
    P["11_momentum_up"] = ((c > o) & (body > 1.5 * avg_rng), +1)
    P["12_momentum_down"] = ((c < o) & (body > 1.5 * avg_rng), -1)
    P["13_pullback_in_uptrend"] = (up & (c < ema20) & (c > ema50), +1)
    P["14_pullback_in_downtrend"] = (dn & (c > ema20) & (c < ema50), -1)
    P["15_break_4h_high_+1d_up"] = ((c > hi16) & (c > hi96 * 0.999) & up, +1)   # MTF-confirmed
    P["16_break_4h_low_+1d_down"] = ((c < lo16) & (c < lo96 * 1.001) & dn, -1)  # MTF-confirmed
    return P


def forward(df):
    hi, lo, cl = df.high.values, df.low.values, df.close.values
    n = len(df)
    from numpy.lib.stride_tricks import sliding_window_view
    mfe = np.full(n, np.nan); mae = np.full(n, np.nan); term = np.full(n, np.nan)
    fh = sliding_window_view(hi[1:], H).max(1); fl = sliding_window_view(lo[1:], H).min(1)
    m = len(fh)
    mfe[:m] = fh / cl[:m] - 1; mae[:m] = fl / cl[:m] - 1; term[:m] = cl[H:H + m] / cl[:m] - 1
    return mfe, mae, term


def barrier_R(mfe, mae, term, atr_p, side, tp=1.5, sl=1.0):
    tpm, slm = tp * atr_p, sl * atr_p
    if side == 1:
        R = np.where(mae <= -slm, -1.0, np.where(mfe >= tpm, tp / sl, term / slm))
    else:
        R = np.where(mfe >= slm, -1.0, np.where(mae <= -tpm, tp / sl, -term / slm))
    return R


def main():
    df = pd.read_parquet(REPO / "research" / "data" / "xauusd_15m.parquet")
    df = df[df.index.year >= 2010]
    atr_p = (atr(df) / df.close).values
    mfe, mae, term = forward(df)
    pats = detect(df)
    ok = np.isfinite(mfe) & np.isfinite(atr_p) & (atr_p > 0)

    rows = {}
    for name, (mask, side) in pats.items():
        m = mask.fillna(False).values & ok
        n = int(m.sum())
        if n < 200:
            rows[name] = dict(n=n, note="too rare"); continue
        # directional hit rate (does price close in the pattern's direction after H?)
        if side != 0:
            dir_hit = float(((term[m] > 0) == (side == 1)).mean())
            R = barrier_R(mfe[m], mae[m], term[m], atr_p[m], side, 1.5, 1.0)
            netR = R - COST / (1.0 * atr_p[m])
            rows[name] = dict(n=n, freq_pct=round(100 * n / ok.sum(), 2),
                              dir=("long" if side == 1 else "short"),
                              dir_hit_rate=round(dir_hit, 3),
                              barrier_win=round(float((R > 0).mean()), 3),
                              gross_R=round(float(R.mean()), 4),
                              net_R=round(float(netR.mean()), 4))
        else:  # neutral patterns: report both sides' hit for context
            rows[name] = dict(n=n, freq_pct=round(100 * n / ok.sum(), 2),
                              dir="neutral",
                              up_after=round(float((term[m] > 0).mean()), 3))

    # the win-rate trap demo on the best directional pattern
    best = max((k for k in rows if rows[k].get("net_R") is not None),
               key=lambda k: rows[k]["net_R"])
    bm = pats[best][0].fillna(False).values & ok
    side = pats[best][1]
    trap = {}
    for tp, sl in [(1.5, 1.0), (0.5, 3.0), (0.25, 5.0)]:
        R = barrier_R(mfe[bm], mae[bm], term[bm], atr_p[bm], side, tp, sl)
        trap[f"TP{tp}_SL{sl}"] = dict(win=round(float((R > 0).mean()), 3),
                                      net_R=round(float((R - COST / (sl * atr_p[bm])).mean()), 4))

    out = {"instrument": "XAUUSD 15m", "H_bars": H, "cost_bp": 2.5,
           "patterns": rows, "best_pattern": best, "win_rate_trap": trap}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "chart_patterns.json").write_text(json.dumps(out, indent=1, default=float))

    print(f"{'pattern':>28}{'n':>8}{'dir':>7}{'dir_hit':>8}{'barr_win':>9}{'net_R':>8}")
    for k, v in rows.items():
        if "dir_hit_rate" in v:
            print(f"{k:>28}{v['n']:>8}{v['dir']:>7}{v['dir_hit_rate']:>8.3f}"
                  f"{v['barrier_win']:>9.3f}{v['net_R']:>+8.3f}")
        elif v.get('dir') == 'neutral':
            print(f"{k:>28}{v['n']:>8}{'neut':>7}{'up '+str(v['up_after']):>17}")
    print(f"\nWIN-RATE TRAP on best pattern ({best}):")
    for kk, vv in trap.items():
        print(f"  {kk:>14}: win {vv['win']:.0%}  net {vv['net_R']:+.4f}R")
    print(f"saved {RESULTS/'chart_patterns.json'}")


if __name__ == "__main__":
    main()
