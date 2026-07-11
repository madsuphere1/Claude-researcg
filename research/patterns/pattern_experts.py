"""Pattern-Expert router: 16 discovered candle patterns, each with its own
learned direction + TP/SL, routed per bar. Walk-forward, cost-honest.

Design (operator's idea, made rigorous):
  1. Describe each bar by context (trend, position-in-4h-range, breakout,
     vol, candle shape, previous candle).
  2. KMeans -> 16 patterns ("16 models"), fit on TRAIN only.
  3. Per pattern, on TRAIN: measure forward MFE/MAE/terminal over H bars,
     grid-search (side, TP, SL) in ATR units for best net expectancy.
     Patterns with train net<=0 are marked NO-TRADE.
  4. On TEST: assign each bar to its pattern, apply that pattern's learned
     rule, record net R (pessimistic triple-barrier ties, cost in R).

Expectancy is per-signal (overlapping trades allowed) — a capacity-blind
edge estimate, not a sequenced backtest. Instrument: XAUUSD 15m.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).parents[2]
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
K = 16
H = 16                      # forward horizon (bars) for behavior + barriers
COST = 2.5e-4              # 2.5 bp round-trip (gold retail)
PURGE = 96
TP_GRID = [1.0, 1.5, 2.0, 3.0]
SL_GRID = [1.0, 1.5, 2.0]


def atr(df, n=14):
    pc = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def pattern_features(df):
    o, h, l, c = df.open, df.high, df.low, df.close
    r1 = np.log(c / c.shift(1))
    hi16 = h.rolling(16).max()
    lo16 = l.rolling(16).min()
    rng = (hi16 - lo16).replace(0, np.nan)
    f = pd.DataFrame(index=df.index)
    f["ret1"] = r1
    f["ret4"] = np.log(c / c.shift(4))
    f["ret16"] = np.log(c / c.shift(16))
    f["pos_in_range"] = (c - lo16) / rng                    # 0=at 4h low, 1=at 4h high
    f["dist_ema200"] = c / c.ewm(span=200, adjust=False).mean() - 1
    f["atr_p"] = atr(df, 14) / c
    f["rsi"] = rsi(c, 14) / 100
    body = (c - o) / (h - l + 1e-9)
    f["body"] = body
    f["upwick"] = (h - np.maximum(o, c)) / (h - l + 1e-9)
    f["lowick"] = (np.minimum(o, c) - l) / (h - l + 1e-9)
    f["brk_hi"] = (c > hi16.shift(1)).astype(float)          # breaks prior 4h high
    f["brk_lo"] = (c < lo16.shift(1)).astype(float)          # breaks prior 4h low
    f["prev_ret"] = r1.shift(1)
    f["prev_body"] = body.shift(1)
    return f


def forward_excursions(df):
    hi, lo, cl = df.high.values, df.low.values, df.close.values
    n = len(df)
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    term = np.full(n, np.nan)
    fut_hi = sliding_window_view(hi[1:], H).max(axis=1)      # [i+1..i+H], i=0..n-H-1
    fut_lo = sliding_window_view(lo[1:], H).min(axis=1)
    m = len(fut_hi)
    mfe[:m] = fut_hi / cl[:m] - 1
    mae[:m] = fut_lo / cl[:m] - 1
    term[:m] = cl[H:H + m] / cl[:m] - 1
    return mfe, mae, term


def eval_rule(mfe, mae, term, atr_p, side, tp, sl):
    """Vectorized pessimistic triple-barrier net R for a rule over bars."""
    tpm, slm = tp * atr_p, sl * atr_p
    cost_R = COST / slm
    if side == 1:
        hit_sl = mae <= -slm
        hit_tp = mfe >= tpm
        R = np.where(hit_sl, -1.0, np.where(hit_tp, tp / sl, term / slm))
    else:
        hit_sl = mfe >= slm
        hit_tp = mae <= -tpm
        R = np.where(hit_sl, -1.0, np.where(hit_tp, tp / sl, -term / slm))
    return R - cost_R


def main():
    df = pd.read_parquet(REPO / "research" / "data" / "xauusd_15m.parquet")
    df = df[df.index.year >= 2010]
    F = pattern_features(df)
    atr_p = (atr(df, 14) / df.close).values
    mfe, mae, term = forward_excursions(df)
    yrs = df.index.year.values
    feat_cols = list(F.columns)
    Fv = F.values
    valid = np.isfinite(Fv).all(1) & np.isfinite(mfe) & np.isfinite(atr_p) & (atr_p > 0)

    test_years = [t for t in sorted(set(int(x) for x in yrs)) if (yrs < t).sum() > 40000]
    all_net, all_year, all_pat = [], [], []
    pattern_log = {}

    for ty in test_years:
        tr = np.flatnonzero((yrs < ty) & valid)
        tr = tr[tr < np.flatnonzero(yrs == ty)[0] - PURGE]
        te = np.flatnonzero((yrs == ty) & valid)
        if len(tr) < 20000 or len(te) < 2000:
            continue
        sc = StandardScaler().fit(Fv[tr])
        km = MiniBatchKMeans(K, random_state=7, n_init=3, batch_size=4096).fit(sc.transform(Fv[tr]))
        lab_tr = km.labels_
        lab_te = km.predict(sc.transform(Fv[te]))
        # learn per-pattern rule on TRAIN
        rules = {}
        for k in range(K):
            idx = tr[lab_tr == k]
            if len(idx) < 300:
                rules[k] = None
                continue
            best = (0.0, None)
            for side in (1, -1):
                for tp in TP_GRID:
                    for sl in SL_GRID:
                        net = eval_rule(mfe[idx], mae[idx], term[idx], atr_p[idx], side, tp, sl).mean()
                        if net > best[0]:
                            best = (net, (side, tp, sl))
            rules[k] = best[1]            # None if nothing beat 0
        # apply on TEST
        for k in range(K):
            if rules[k] is None:
                continue
            side, tp, sl = rules[k]
            idx = te[lab_te == k]
            if not len(idx):
                continue
            net = eval_rule(mfe[idx], mae[idx], term[idx], atr_p[idx], side, tp, sl)
            all_net.append(net)
            all_year.append(np.full(len(net), ty))
            all_pat.append(np.full(len(net), k))
        print(f"fold {ty}: {sum(r is not None for r in rules.values())}/{K} patterns traded", flush=True)

    net = np.concatenate(all_net)
    year = np.concatenate(all_year)
    pat = np.concatenate(all_pat)
    out = {
        "instrument": "XAUUSD 15m", "K": K, "H": H, "cost_bp": 2.5,
        "n_trades": int(len(net)),
        "net_expectancy_R": round(float(net.mean()), 4),
        "win_rate": round(float((net > 0).mean()), 4),
        "by_year": {int(y): dict(n=int((year == y).sum()),
                                 expR=round(float(net[year == y].mean()), 4))
                    for y in sorted(set(year))},
        "by_pattern": {int(k): dict(n=int((pat == k).sum()),
                                    expR=round(float(net[pat == k].mean()), 4))
                       for k in sorted(set(pat))},
    }
    # bootstrap CI (day-independent trade-level)
    rng = np.random.default_rng(7)
    bs = [net[rng.integers(0, len(net), len(net))].mean() for _ in range(3000)]
    out["ci95"] = [round(float(np.quantile(bs, .025)), 4), round(float(np.quantile(bs, .975)), 4)]
    out["p_leq_0"] = round(float((np.array(bs) <= 0).mean()), 4)
    (RESULTS / "pattern_experts.json").write_text(json.dumps(out, indent=1, default=float))

    print(f"\nPATTERN-EXPERT ROUTER (XAUUSD 15m, 2.5bp, OOS)")
    print(f"  trades {out['n_trades']:,}  net expectancy {out['net_expectancy_R']:+.4f}R  "
          f"win {out['win_rate']:.1%}")
    print(f"  95% CI {out['ci95']}  p(<=0) {out['p_leq_0']}")
    print(f"  by year: {out['by_year']}")
    print(f"  top patterns: " + ", ".join(
        f"P{k}:{v['expR']:+.3f}(n{v['n']})" for k, v in
        sorted(out['by_pattern'].items(), key=lambda kv: -kv[1]['expR'])[:6]))
    print(f"saved {RESULTS/'pattern_experts.json'}")


if __name__ == "__main__":
    main()
