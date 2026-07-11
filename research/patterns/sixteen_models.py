"""16 v6-style models: each = a chart condition (recognition) + a direction
+ its own TP/SL, tuned so that IN its condition it wins >=50% of the time,
with the best net expectancy. Barriers chosen on TRAIN (2010-2019), reported
out-of-sample (2020-2026). Honest: shows win rate AND net expectancy, so
">=50% win rate" is not confused with "profitable".
XAUUSD 15m, cost 2.5bp, pessimistic ties.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from chart_patterns import atr, barrier_R, detect, forward, COST, H  # noqa

REPO = Path(__file__).parents[2]
RESULTS = Path(__file__).parent / "results"

MEANING = {
    "01_uptrend": "Trend up (price>50EMA, 20>50EMA) — ride long",
    "02_downtrend": "Trend down (price<50EMA, 20<50EMA) — ride short",
    "03_break_4h_high": "Closes above prior 4h high — breakout long",
    "04_break_4h_low": "Closes below prior 4h low — breakdown short",
    "05_bull_engulf": "Green body engulfs prior red — reversal long",
    "06_bear_engulf": "Red body engulfs prior green — reversal short",
    "07_hammer": "Long lower wick, sellers rejected — bounce long",
    "08_shooting_star": "Long upper wick, buyers rejected — fade short",
    "09_inside_bar": "Range inside prior bar (compression) — breakout",
    "10_range_compress": "Narrow range + low vol — coiled breakout",
    "11_momentum_up": "Large green body — continuation long",
    "12_momentum_down": "Large red body — continuation short",
    "13_pullback_in_uptrend": "Dip to 20EMA within uptrend — buy the dip",
    "14_pullback_in_downtrend": "Rally to 20EMA within downtrend — sell rally",
    "15_break_4h_high_+1d_up": "4h breakout confirmed by daily uptrend (MTF)",
    "16_break_4h_low_+1d_down": "4h breakdown confirmed by daily downtrend (MTF)",
}
TP_GRID = [0.5, 1.0, 1.5, 2.0]
SL_GRID = [1.0, 1.5, 2.0, 3.0]


def main():
    df = pd.read_parquet(REPO / "research" / "data" / "xauusd_15m.parquet")
    df = df[df.index.year >= 2010]
    atr_p = (atr(df) / df.close).values
    mfe, mae, term = forward(df)
    pats = detect(df)
    yrs = df.index.year.values
    ok = np.isfinite(mfe) & np.isfinite(atr_p) & (atr_p > 0)
    tr = ok & (yrs <= 2019)
    te = ok & (yrs >= 2020)

    models = {}
    for name, (mask, side0) in pats.items():
        m = mask.fillna(False).values
        mtr, mte = m & tr, m & te
        if mtr.sum() < 500 or mte.sum() < 200:
            models[name] = {"meaning": MEANING[name], "note": "too rare"}
            continue
        sides = [side0] if side0 != 0 else [1, -1]
        best = None
        for side in sides:
            for tp in TP_GRID:
                for sl in SL_GRID:
                    Rtr = barrier_R(mfe[mtr], mae[mtr], term[mtr], atr_p[mtr], side, tp, sl)
                    win = float((Rtr > 0).mean())
                    net = float((Rtr - COST / (sl * atr_p[mtr])).mean())
                    if win >= 0.50 and (best is None or net > best[0]):
                        best = (net, side, tp, sl, win)
        if best is None:                       # no >=50% win config; take highest-win
            for side in sides:
                Rtr = barrier_R(mfe[mtr], mae[mtr], term[mtr], atr_p[mtr], side, 1.0, 3.0)
                win = float((Rtr > 0).mean())
                if best is None or win > best[4]:
                    net = float((Rtr - COST / (3.0 * atr_p[mtr])).mean())
                    best = (net, side, 1.0, 3.0, win)
        _, side, tp, sl, _ = best
        Rte = barrier_R(mfe[mte], mae[mte], term[mte], atr_p[mte], side, tp, sl)
        net_te = Rte - COST / (sl * atr_p[mte])
        models[name] = dict(
            meaning=MEANING[name],
            recognition_freq_pct=round(100 * m[ok].mean(), 2),
            direction="long" if side == 1 else "short",
            TP_atr=tp, SL_atr=sl,
            oos_win_rate=round(float((Rte > 0).mean()), 3),
            oos_net_R=round(float(net_te.mean()), 4),
            profitable=bool(net_te.mean() > 0),
        )

    prof = [k for k, v in models.items() if v.get("profitable")]
    win50 = [k for k, v in models.items() if v.get("oos_win_rate", 0) >= 0.50]
    out = {"instrument": "XAUUSD 15m", "train": "2010-2019", "test": "2020-2026",
           "cost_bp": 2.5, "models": models,
           "n_win_ge_50pct": len(win50), "n_profitable_net": len(prof),
           "profitable_models": prof}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "sixteen_models.json").write_text(json.dumps(out, indent=1, default=float))

    print(f"{'model':>27}{'dir':>6}{'TP/SL':>8}{'OOS win':>8}{'OOS netR':>9}{'  profit?'}")
    for k, v in models.items():
        if "oos_win_rate" in v:
            print(f"{k:>27}{v['direction']:>6}{str(v['TP_atr'])+'/'+str(v['SL_atr']):>8}"
                  f"{v['oos_win_rate']:>8.3f}{v['oos_net_R']:>+9.4f}   {'YES' if v['profitable'] else 'no'}")
    print(f"\nmodels with OOS win-rate >=50%: {len(win50)}/16")
    print(f"models net-PROFITABLE OOS @2.5bp: {len(prof)}/16  {prof}")
    print(f"saved {RESULTS/'sixteen_models.json'}")


if __name__ == "__main__":
    main()
