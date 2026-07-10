"""H-I1: adaptation study - training windows, drift metrics, retrain policy.

(i) Does a rolling window beat the expanding window after the alpha decay
    seen post-2020?  Expanding vs rolling-5y vs rolling-3y, LightGBM,
    y_tp_long, all 13 folds.
(ii) Drift measurement: PSI of the top-20 features and of model predictions
     between each training window and its test year; does PSI anticipate
     AUC decay (does high drift predict worse folds)?
(iii) Static-model decay: train once on 2010-2013, evaluate every year
      through 2026 - the case for retraining at all.
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
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=500, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              verbose=-1, num_threads=4, seed=7)
LABEL = "y_tp_long"
TOP20 = ["vol_of_vol", "rv_rank_1y", "ret_kurt_96", "dist_month_open",
         "dist_r50", "atr96_p", "dist_pmh", "dist_pml", "ret_skew_96",
         "ret_480", "ret_autocorr1_96", "day_of_month", "dist_pwl",
         "var_ratio_4_96", "ret_skew_32", "days_since_event", "dist_pwh",
         "dist_week_open", "ret_192", "days_to_event"]


def psi(a: np.ndarray, b: np.ndarray, bins=10) -> float:
    qs = np.nanquantile(a, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    pa, _ = np.histogram(a, qs); pb, _ = np.histogram(b, qs)
    pa = pa / max(1, pa.sum()) + 1e-6
    pb = pb / max(1, pb.sum()) + 1e-6
    return float(np.sum((pa - pb) * np.log(pa / pb)))


def main() -> None:
    df, feats = wf.load()
    yrs = df.index.year
    lab = df[LABEL].to_numpy()
    out = {"windows": {}, "drift": {}, "static_decay": {}}

    for wname, wyears in (("expanding", None), ("rolling5", 5), ("rolling3", 3)):
        fold_auc = {}
        for ty in wf.TEST_YEARS:
            te = np.flatnonzero(yrs == ty)
            if not len(te):
                continue
            lo = 2010 if wyears is None else ty - wyears
            tr = np.flatnonzero((yrs >= lo) & (yrs < ty))
            tr = tr[tr < te[0] - wf.PURGE_BARS]
            tr = tr[np.isfinite(lab[tr])]
            tev = te[np.isfinite(lab[te])]
            mdl = lgb.train(PARAMS, lgb.Dataset(df[feats].iloc[tr],
                                                lab[tr].astype(int)),
                            num_boost_round=400)
            p = mdl.predict(df[feats].iloc[tev])
            fold_auc[str(ty)] = float(roc_auc_score(lab[tev].astype(int), p))
            if wname == "expanding":
                # drift diagnostics on the expanding config
                f_psi = {f: psi(df[f].iloc[tr].to_numpy(),
                                df[f].iloc[te].to_numpy()) for f in TOP20}
                p_tr = mdl.predict(df[feats].iloc[tr[-30000:]])
                out["drift"][str(ty)] = dict(
                    mean_feature_psi=float(np.mean(list(f_psi.values()))),
                    max_feature_psi=float(np.max(list(f_psi.values()))),
                    worst_feature=max(f_psi, key=f_psi.get),
                    prediction_psi=psi(p_tr, p))
            print(wname, ty, round(fold_auc[str(ty)], 4), flush=True)
        out["windows"][wname] = fold_auc

    # static model decay
    tr = np.flatnonzero(yrs < 2014)
    tr = tr[np.isfinite(lab[tr])]
    mdl = lgb.train(PARAMS, lgb.Dataset(df[feats].iloc[tr], lab[tr].astype(int)),
                    num_boost_round=400)
    for ty in wf.TEST_YEARS:
        te = np.flatnonzero(yrs == ty)
        te = te[np.isfinite(lab[te])]
        if len(te):
            out["static_decay"][str(ty)] = float(
                roc_auc_score(lab[te].astype(int), mdl.predict(df[feats].iloc[te])))

    # summary comparisons
    def era_mean(d, lo, hi):
        v = [a for y, a in d.items() if lo <= int(y) <= hi]
        return float(np.mean(v)) if v else np.nan
    out["summary"] = {
        w: {"pre2020": era_mean(d, 2014, 2019), "post2020": era_mean(d, 2020, 2026),
            "all": era_mean(d, 2014, 2026)}
        for w, d in out["windows"].items()}
    # drift vs AUC correlation
    dy = [y for y in out["drift"]]
    x = [out["drift"][y]["prediction_psi"] for y in dy]
    a = [out["windows"]["expanding"][y] for y in dy]
    out["psi_auc_corr"] = float(np.corrcoef(x, a)[0, 1])

    (RESULTS / "i1_drift.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1))
    print("corr(prediction PSI, fold AUC):", round(out["psi_auc_corr"], 3))


if __name__ == "__main__":
    main()
