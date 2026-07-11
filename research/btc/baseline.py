"""B1-BASE + B1-FLOW + B1-ECON — the BTCUSD founding cycle.

Three questions in one walk-forward pass:
  BASE: does a predictive signal exist on BTC 15m? (per-year OOS AUC)
  FLOW: do crypto-only order-flow features add AUC beyond price alone?
  ECON: does the signal survive crypto costs (~10bp)? cost-tier table.

Triple-barrier labels (TP 1.5xATR / SL 1xATR / 16-bar horizon), LightGBM,
expanding walk-forward, annual retrain, 96-bar purge. 24/7 -> no weekend
handling; year folds only.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
PURGE = 96
LGB = dict(objective="binary", learning_rate=0.05, num_leaves=63,
           min_data_in_leaf=500, feature_fraction=0.7, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=4, seed=7)


def atr(df, n=14):
    pc = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def features(df):
    c = df.close
    r1 = np.log(c / c.shift(1))
    f = pd.DataFrame(index=df.index)
    # price / momentum
    for k in (1, 4, 16, 96):
        f[f"ret{k}"] = np.log(c / c.shift(k))
    f["vol16"] = r1.rolling(16).std()
    f["vol96"] = r1.rolling(96).std()
    f["rsi14"] = _rsi(c, 14)
    f["dist_ema50"] = c / c.ewm(span=50, adjust=False).mean() - 1
    f["dist_ema200"] = c / c.ewm(span=200, adjust=False).mean() - 1
    f["atr_p"] = atr(df, 14) / c
    # PRICE range/skew
    f["hl_range"] = (df.high - df.low) / c
    # --- crypto-only order-flow (the new stuff) ---
    vol = df.volume.replace(0, np.nan)
    f["taker_imb"] = (df.taker_buy_base / vol - 0.5)          # aggressive-buy imbalance
    f["vol_z"] = (np.log(vol) - np.log(vol).rolling(96).mean()) / (np.log(vol).rolling(96).std() + 1e-9)
    f["trades_z"] = (np.log(df.trades.replace(0, np.nan)) - np.log(df.trades.replace(0, np.nan)).rolling(96).mean()) / (np.log(df.trades.replace(0, np.nan)).rolling(96).std() + 1e-9)
    f["avg_trade_sz"] = df.quote_volume / df.trades.replace(0, np.nan)
    f["taker_imb_ma16"] = f["taker_imb"].rolling(16).mean()
    # time (24/7)
    f["hour"] = df.index.hour
    f["dow"] = df.index.dayofweek
    price_cols = [c for c in f.columns if not c.startswith(("taker", "vol_z", "trades_z", "avg_trade"))]
    flow_cols = ["taker_imb", "vol_z", "trades_z", "avg_trade_sz", "taker_imb_ma16"]
    return f, price_cols, flow_cols


def _rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def barrier_labels(df, a, tp=1.5, sl=1.0, h=16):
    hi, lo, cl = df.high.values, df.low.values, df.close.values
    av = a.values
    n = len(df)
    yl = np.full(n, np.nan)
    for i in range(n - h):
        if not np.isfinite(av[i]) or av[i] <= 0:
            continue
        up = cl[i] + tp * av[i]
        dn = cl[i] - sl * av[i]
        yl[i] = 0
        for j in range(i + 1, i + h + 1):
            if lo[j] <= dn:
                yl[i] = 0
                break
            if hi[j] >= up:
                yl[i] = 1
                break
    return yl


def sign_p(k, n):
    return float(sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n)


def main():
    df = pd.read_parquet(HERE / "data" / "btcusdt_15m.parquet")
    print(f"bars {len(df):,}  {df.index.min()} .. {df.index.max()}", flush=True)
    a = atr(df, 14)
    f, price_cols, flow_cols = features(df)
    y = barrier_labels(df, a)
    d = pd.concat([f], axis=1)
    yrs = df.index.year.values
    years = sorted(set(yrs))
    test_years = [t for t in years if (yrs < t).sum() > 20000]

    def wf(cols):
        aucs, exps = {}, {}
        for ty in test_years:
            te = np.flatnonzero(yrs == ty)
            tr = np.flatnonzero(yrs < ty)
            tr = tr[tr < te[0] - PURGE]
            tr = tr[np.isfinite(y[tr])]
            tev = te[np.isfinite(y[te])]
            if len(tr) < 5000 or len(tev) < 500:
                continue
            m = lgb.train(LGB, lgb.Dataset(d[cols].iloc[tr], y[tr].astype(int)), num_boost_round=400)
            p = m.predict(d[cols].iloc[tev])
            aucs[ty] = float(roc_auc_score(y[tev].astype(int), p))
            # gross expectancy of a simple long-if-p>thr strategy at various costs
            exps[ty] = (p, y[tev], a.values[tev], df.close.values[tev])
        return aucs, exps

    price_auc, price_exp = wf(price_cols)
    allf_auc, _ = wf(price_cols + flow_cols)

    pv = list(price_auc.values())
    av_ = list(allf_auc.values())
    k = sum(x > 0.5 for x in pv)
    out = {
        "n_bars": int(len(df)),
        "span": [str(df.index.min()), str(df.index.max())],
        "test_years": test_years,
        "BASE_price_only": {"mean_auc": round(float(np.mean(pv)), 4),
                            "years_above_0.5": f"{k}/{len(pv)}",
                            "sign_p": sign_p(k, len(pv)),
                            "by_year": {str(t): round(v, 4) for t, v in price_auc.items()}},
        "FLOW_price_plus_flow": {"mean_auc": round(float(np.mean(av_)), 4),
                                 "uplift": round(float(np.mean(av_) - np.mean(pv)), 4),
                                 "by_year": {str(t): round(v, 4) for t, v in allf_auc.items()}},
    }

    # ECON: cost tiers on a threshold strategy (pooled OOS)
    TP_R, SL_R = 1.5, 1.0
    allp = np.concatenate([e[0] for e in price_exp.values()])
    ally = np.concatenate([e[1] for e in price_exp.values()])
    alla = np.concatenate([e[2] for e in price_exp.values()])
    allc = np.concatenate([e[3] for e in price_exp.values()])
    thr = np.quantile(allp, 0.90)                       # top-decile conviction longs
    sel = allp >= thr
    sl_dist = SL_R * alla[sel] / allc[sel]              # SL distance as fraction of price
    gross_R = np.where(ally[sel] == 1, TP_R, -SL_R)
    econ = {}
    for bp in (2, 5, 10):
        cost_R = (bp * 1e-4) / sl_dist                  # cost-in-R = cost / SL distance
        netR = gross_R - cost_R
        econ[f"{bp}bp"] = dict(n=int(sel.sum()), gross_R=round(float(gross_R.mean()), 4),
                               net_R=round(float(netR.mean()), 4),
                               win_rate=round(float((ally[sel] == 1).mean()), 4))
    out["ECON_top_decile_longs"] = econ

    (RESULTS / "b1_baseline.json").write_text(json.dumps(out, indent=1))
    print("\nBASE (price-only) mean AUC", out["BASE_price_only"]["mean_auc"],
          out["BASE_price_only"]["years_above_0.5"], "sign_p", round(out["BASE_price_only"]["sign_p"], 4))
    print("FLOW uplift from order-flow features:", out["FLOW_price_plus_flow"]["uplift"])
    print("ECON top-decile longs net R by cost:", {k: v["net_R"] for k, v in econ.items()})
    print("per-year price AUC:", out["BASE_price_only"]["by_year"])
    print(f"saved {RESULTS/'b1_baseline.json'}")


if __name__ == "__main__":
    main()
