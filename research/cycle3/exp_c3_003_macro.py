"""C3-003: cross-asset macro conditioning (see registry entry for the
frozen design and decision rule).

DECISION RULE (verbatim from registry, frozen before run):
ACCEPT if mean AUC uplift >= +0.003 AND uplift negative in <= 1 of 3
folds (2018/2022/2026, y_tp_long); REJECT otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parents[1]))
import wf  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "results"
FOLDS = [2018, 2022, 2026]
SERIES = ["DFII10", "DTWEXBGS", "VIXCLS", "DCOILWTICO"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=500, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              verbose=-1, num_threads=4, seed=7)


def macro_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    daily = pd.read_parquet(HERE.parents[1] / "research" / "data" / "fred_daily.parquet")
    daily = daily.shift(2)                      # 2-business-day availability lag
    feats = {}
    for s in SERIES:
        v = daily[s]
        m = v.rolling(252, min_periods=100).mean()
        sd = v.rolling(252, min_periods=100).std()
        feats[f"{s}_z252"] = (v - m) / (sd + 1e-12)
        feats[f"{s}_chg5"] = v.diff(5)
        feats[f"{s}_chg21"] = v.diff(21)
    F = pd.DataFrame(feats)
    F = F[~F.index.duplicated(keep="last")].sort_index()
    # staleness cap: forward-fill max 5 business days
    F = F.ffill(limit=5)
    # join: each 15m bar (EST date) takes the most recent daily row at or
    # before its date; reindex with ffill onto the monotone bar-date vector
    bar_dates = pd.DatetimeIndex(index.date)
    out = F.reindex(bar_dates, method="ffill")
    out.index = index
    return out.astype(np.float32)


def main() -> None:
    df, feats = wf.load()
    macro = macro_features(df.index)
    dfe = pd.concat([df, macro], axis=1)
    macro_cols = list(macro.columns)
    print(f"macro features: {len(macro_cols)}, coverage "
          f"{macro.notna().all(axis=1).mean():.1%} of bars")

    lab = df["y_tp_long"].to_numpy()
    yrs = df.index.year
    out = {"coverage": float(macro.notna().all(axis=1).mean())}
    ups = []
    for ty in FOLDS:
        te = np.flatnonzero(yrs == ty)
        tr = np.flatnonzero(yrs < ty)
        tr = tr[tr < te[0] - wf.PURGE_BARS]
        tr = tr[np.isfinite(lab[tr])]
        tev = te[np.isfinite(lab[te])]
        r = {}
        gain_share = None
        for name, cols in (("baseline", feats),
                           ("baseline+macro", feats + macro_cols)):
            mdl = lgb.train(PARAMS, lgb.Dataset(dfe[cols].iloc[tr],
                                                lab[tr].astype(int)),
                            num_boost_round=400)
            r[name] = float(roc_auc_score(lab[tev].astype(int),
                                          mdl.predict(dfe[cols].iloc[tev])))
            if name == "baseline+macro":
                g = pd.Series(mdl.feature_importance("gain"), index=cols)
                gain_share = float(g[macro_cols].sum() / g.sum())
        r["uplift"] = r["baseline+macro"] - r["baseline"]
        r["macro_gain_share"] = gain_share
        ups.append(r["uplift"])
        out[str(ty)] = r
        print(ty, {k: round(v, 4) for k, v in r.items()}, flush=True)

    out["mean_uplift"] = float(np.mean(ups))
    out["n_negative_folds"] = int(sum(u < 0 for u in ups))
    out["verdict"] = ("ACCEPT" if out["mean_uplift"] >= 0.003
                      and out["n_negative_folds"] <= 1 else "REJECT")
    (RESULTS / "c3_003_macro.json").write_text(json.dumps(out, indent=1))
    print("mean uplift:", round(out["mean_uplift"], 4), out["verdict"])


if __name__ == "__main__":
    main()
