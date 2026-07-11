"""16 chart-pattern models on BTC + flow-confirmed variants (BTC's edge).
Expectancy-optimized per model, walk-forward (train 2017-2022, test
2023-2026), reported GROSS and NET at 2/5/10 bp. Tests whether Bitcoin's
order-flow lets pattern-models escape the "just the trend" collapse gold
suffered.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO / "research" / "patterns"))
from chart_patterns import atr, barrier_R, detect, forward  # noqa

RESULTS = Path(__file__).parent / "results"
TPG = [1, 1.5, 2, 3, 4]
SLG = [1, 1.5, 2]


def flow_patterns(df):
    vol = df.volume.replace(0, np.nan)
    imb = (df.taker_buy_base / vol - 0.5)
    volz = (np.log(vol) - np.log(vol).rolling(96).mean()) / (np.log(vol).rolling(96).std() + 1e-9)
    h16 = df.high.rolling(16).max().shift(1)
    l16 = df.low.rolling(16).min().shift(1)
    return {
        "F1_taker_buy_surge": ((imb > 0.15) & (volz > 1), +1),
        "F2_taker_sell_surge": ((imb < -0.15) & (volz > 1), -1),
        "F3_flow_confirmed_breakup": ((df.close > h16) & (imb > 0.10), +1),
        "F4_flow_confirmed_breakdn": ((df.close < l16) & (imb < -0.10), -1),
    }


def optimize(mask, side0, mfe, mae, term, atr_p, tr, te, cost):
    m = mask.fillna(False).values
    mtr, mte = m & tr, m & te
    if mtr.sum() < 500 or mte.sum() < 200:
        return None
    sides = [side0] if side0 != 0 else [1, -1]
    best = None
    for side in sides:
        for tp in TPG:
            for sl in SLG:
                R = barrier_R(mfe[mtr], mae[mtr], term[mtr], atr_p[mtr], side, tp, sl)
                net = float((R - cost / (sl * atr_p[mtr])).mean())
                if best is None or net > best[0]:
                    best = (net, side, tp, sl)
    _, side, tp, sl = best
    R = barrier_R(mfe[mte], mae[mte], term[mte], atr_p[mte], side, tp, sl)
    gross = float(R.mean())
    nets = {f"{bp}bp": round(float((R - bp * 1e-4 / (sl * atr_p[mte])).mean()), 4)
            for bp in (2, 5, 10)}
    return dict(direction="long" if side == 1 else "short", TP=tp, SL=sl,
                oos_win=round(float((R > 0).mean()), 3), oos_gross_R=round(gross, 4),
                oos_net=nets, n_test=int(mte.sum()))


def main():
    df = pd.read_parquet(REPO / "research" / "btc" / "data" / "btcusdt_15m.parquet")
    atr_p = (atr(df) / df.close).values
    mfe, mae, term = forward(df)
    yrs = df.index.year.values
    ok = np.isfinite(mfe) & np.isfinite(atr_p) & (atr_p > 0)
    tr = ok & (yrs <= 2022)
    te = ok & (yrs >= 2023)

    pats = detect(df)
    pats.update(flow_patterns(df))
    models = {}
    for name, (mask, side) in pats.items():
        r = optimize(mask, side, mfe, mae, term, atr_p, tr, te, 5e-4)  # tune on 5bp
        if r:
            models[name] = r

    gross_pos = [k for k, v in models.items() if v["oos_gross_R"] > 0]
    net10 = [k for k, v in models.items() if v["oos_net"]["10bp"] > 0]
    net5 = [k for k, v in models.items() if v["oos_net"]["5bp"] > 0]
    net2 = [k for k, v in models.items() if v["oos_net"]["2bp"] > 0]
    out = {"instrument": "BTCUSDT 15m", "train": "2017-2022", "test": "2023-2026",
           "models": models,
           "gross_positive": gross_pos, "net_pos_2bp": net2,
           "net_pos_5bp": net5, "net_pos_10bp": net10}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "sixteen_models_btc.json").write_text(json.dumps(out, indent=1, default=float))

    print(f"{'model':>27}{'dir':>6}{'TP/SL':>7}{'win':>6}{'gross':>8}{'net2':>8}{'net5':>8}{'net10':>8}")
    for k, v in models.items():
        print(f"{k:>27}{v['direction']:>6}{str(v['TP'])+'/'+str(v['SL']):>7}{v['oos_win']:>6.2f}"
              f"{v['oos_gross_R']:>+8.3f}{v['oos_net']['2bp']:>+8.3f}{v['oos_net']['5bp']:>+8.3f}{v['oos_net']['10bp']:>+8.3f}")
    print(f"\ngross-positive OOS: {len(gross_pos)}/{len(models)}  {gross_pos}")
    print(f"net-positive @2bp: {len(net2)}  @5bp: {len(net5)}  @10bp: {len(net10)}")
    print(f"  @10bp survivors: {net10}")
    print(f"saved {RESULTS/'sixteen_models_btc.json'}")


if __name__ == "__main__":
    main()
