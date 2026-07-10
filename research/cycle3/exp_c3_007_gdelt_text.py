"""C3-007: GDELT textual/tone signal (see registry entry for the frozen
design and decision rule).

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
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=500, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              verbose=-1, num_threads=4, seed=7)


def gdelt_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    weekly = pd.read_parquet(HERE.parents[1] / "research" / "data" / "gdelt_weekly.parquet")
    weekly = weekly[["avg_tone", "tone_negativity", "econ_inflation_share",
                     "epu_policy_share", "econ_debt_share",
                     "armedconflict_share"]]
    weekly.index = pd.to_datetime(weekly.index).tz_localize(None)
    weekly = weekly[~weekly.index.duplicated(keep="last")].sort_index()
    weekly = weekly.shift(2)  # 2-business-day availability conservatism
    m = weekly.rolling(52, min_periods=20).mean()
    sd = weekly.rolling(52, min_periods=20).std()
    z = (weekly - m) / (sd + 1e-12)
    z.columns = [f"{c}_z52" for c in z.columns]
    z = z.ffill(limit=10)  # staleness cap: weekly sampling, wider cap than C3-003
    bar_dates = pd.DatetimeIndex(index.date)
    out = z.reindex(bar_dates, method="ffill")
    out.index = index
    return out.astype(np.float32)


def main() -> None:
    df, feats = wf.load()
    gdelt = gdelt_features(df.index)
    dfe = pd.concat([df, gdelt], axis=1)
    gdelt_cols = list(gdelt.columns)
    coverage = gdelt.notna().all(axis=1).mean()
    print(f"gdelt features: {len(gdelt_cols)}, coverage {coverage:.1%} of bars "
          f"(expected partial: no data before 2015-02)")

    lab = df["y_tp_long"].to_numpy()
    yrs = df.index.year
    out = {"coverage_all_bars": float(coverage)}
    ups = []
    for ty in FOLDS:
        te = np.flatnonzero(yrs == ty)
        tr = np.flatnonzero(yrs < ty)
        tr = tr[tr < te[0] - wf.PURGE_BARS]
        tr = tr[np.isfinite(lab[tr])]
        tev = te[np.isfinite(lab[te])]
        fold_cov = gdelt.iloc[te].notna().all(axis=1).mean()
        r = {"fold_gdelt_coverage": float(fold_cov)}
        gain_share = None
        for name, cols in (("baseline", feats),
                           ("baseline+gdelt", feats + gdelt_cols)):
            mdl = lgb.train(PARAMS, lgb.Dataset(dfe[cols].iloc[tr],
                                                lab[tr].astype(int)),
                            num_boost_round=400)
            r[name] = float(roc_auc_score(lab[tev].astype(int),
                                          mdl.predict(dfe[cols].iloc[tev])))
            if name == "baseline+gdelt":
                g = pd.Series(mdl.feature_importance("gain"), index=cols)
                gain_share = float(g[gdelt_cols].sum() / g.sum())
        r["uplift"] = r["baseline+gdelt"] - r["baseline"]
        r["gdelt_gain_share"] = gain_share
        ups.append(r["uplift"])
        out[str(ty)] = r
        print(ty, {k: round(v, 4) if isinstance(v, float) else v
                   for k, v in r.items()}, flush=True)

    out["mean_uplift"] = float(np.mean(ups))
    out["n_negative_folds"] = int(sum(u < 0 for u in ups))
    out["verdict"] = ("ACCEPT" if out["mean_uplift"] >= 0.003
                      and out["n_negative_folds"] <= 1 else "REJECT")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "c3_007_gdelt.json").write_text(json.dumps(out, indent=1))
    print("mean uplift:", round(out["mean_uplift"], 4), out["verdict"])


if __name__ == "__main__":
    main()
