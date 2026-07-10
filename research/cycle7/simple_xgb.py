"""C7-001 — Operator's challenge: does raw-data + simple-features + XGBoost
beat / match the elaborate 205-feature LightGBM pipeline?

Head-to-head on identical folds and identical G2 labels, walk-forward,
out-of-sample. Eight dead-simple causal features from raw OHLC vs the
full 205-feature set. If simple wins or even ties, the elaborate pipeline
is over-engineered and the operator is right.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO / "research"))
sys.path.insert(0, str(REPO / "research" / "cycle2"))
import wf  # noqa: E402
from features import atr  # noqa: E402
from exp_b2_barriers import GEOMETRIES, barrier_labels  # noqa: E402

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
YEARS = list(range(2014, 2027))
PURGE = 96


def simple_features(bars: pd.DataFrame) -> pd.DataFrame:
    c = bars.close
    r1 = np.log(c / c.shift(1))
    f = pd.DataFrame(index=bars.index)
    f["ret1"] = r1
    f["ret4"] = np.log(c / c.shift(4))
    f["ret16"] = np.log(c / c.shift(16))
    f["vol16"] = r1.rolling(16).std()
    f["mom16"] = r1.rolling(16).mean()
    f["dist_sma20"] = c / c.rolling(20).mean() - 1
    f["range"] = (bars.high - bars.low) / c
    f["hour"] = bars.index.hour
    return f


def main() -> None:
    df, feats205 = wf.load()
    bars = pd.read_parquet(REPO / "research" / "data" / "xauusd_15m.parquet").loc[df.index]
    a14 = atr(bars, 14)
    g = GEOMETRIES["G2"]
    labs = barrier_labels(bars, a14, g["tp"], g["sl"], g["h"])
    y = labs.y_tp_long.to_numpy()

    simple = simple_features(bars)
    fsimple = list(simple.columns)
    d = pd.concat([df[feats205], simple], axis=1)
    yrs = df.index.year

    xgb_params = dict(max_depth=4, eta=0.05, subsample=0.8,
                      colsample_bytree=0.7, min_child_weight=50,
                      objective="binary:logistic", eval_metric="auc",
                      tree_method="hist", nthread=4)

    def wf_auc(cols, use_xgb):
        aucs = {}
        for ty in YEARS:
            te = np.flatnonzero(yrs == ty)
            tr = np.flatnonzero(yrs < ty)
            tr = tr[tr < te[0] - PURGE]
            tr = tr[np.isfinite(y[tr])]
            tev = te[np.isfinite(y[te])]
            if len(tr) < 500 or len(tev) < 100:
                continue
            Xtr, Xte = d[cols].iloc[tr], d[cols].iloc[tev]
            if use_xgb:
                dm = xgb.DMatrix(Xtr, label=y[tr].astype(int))
                m = xgb.train(xgb_params, dm, num_boost_round=300)
                p = m.predict(xgb.DMatrix(Xte))
            else:
                import lightgbm as lgb
                m = lgb.train(dict(objective="binary", learning_rate=0.05,
                                   num_leaves=63, min_data_in_leaf=500,
                                   feature_fraction=0.7, bagging_fraction=0.8,
                                   bagging_freq=1, lambda_l2=5.0, verbose=-1,
                                   num_threads=4, seed=7),
                              lgb.Dataset(Xtr, y[tr].astype(int)),
                              num_boost_round=400)
                p = m.predict(Xte)
            aucs[ty] = float(roc_auc_score(y[tev].astype(int), p))
        return aucs

    runs = {
        "simple8_xgb": wf_auc(fsimple, True),
        "simple8_lgb": wf_auc(fsimple, False),
        "full205_lgb": wf_auc(feats205, False),
    }
    out = {}
    for name, a in runs.items():
        vals = list(a.values())
        out[name] = dict(mean_auc=round(float(np.mean(vals)), 4),
                         years_above_0p5=f"{sum(v>0.5 for v in vals)}/{len(vals)}",
                         by_year={str(k): round(v, 4) for k, v in a.items()})
    (RESULTS / "c7_simple_xgb.json").write_text(json.dumps(out, indent=1))

    print(f"{'model':>14} {'n_feats':>8} {'mean AUC':>9} {'yrs>0.5':>8}")
    for name, nf in (("simple8_xgb", 8), ("simple8_lgb", 8), ("full205_lgb", 205)):
        r = out[name]
        print(f"{name:>14} {nf:>8} {r['mean_auc']:>9.4f} {r['years_above_0p5']:>8}")
    print("\nper-year AUC:")
    print(f"{'year':>6} {'simple_xgb':>11} {'simple_lgb':>11} {'full_lgb':>11}")
    for ty in YEARS:
        row = [runs[m].get(ty) for m in ("simple8_xgb", "simple8_lgb", "full205_lgb")]
        if any(v is not None for v in row):
            print(f"{ty:>6} " + " ".join(f"{v:>11.4f}" if v else f"{'—':>11}" for v in row))
    print(f"\nsaved {RESULTS/'c7_simple_xgb.json'}")


if __name__ == "__main__":
    main()
