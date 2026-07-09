"""Which prediction target is most learnable?

Same LightGBM config and walk-forward folds as models_compare, applied to
every candidate label. Classification targets are compared on AUC;
regression-style targets (forward returns, MFE/MAE) are binarised at their
training-set median so AUC remains comparable.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import wf

RESULTS = Path(__file__).parent / "results"
YEARS = [2018, 2020, 2022, 2024, 2026]
TRAIN_YEARS = 5

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=500, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              verbose=-1, num_threads=4, seed=7)

TARGETS = {
    "y_dir_1": ("cls", "next-bar direction"),
    "y_dir_5": ("cls", "5-bar direction"),
    "y_dir_10": ("cls", "10-bar direction"),
    "y_tp_long": ("cls", "TP-before-SL long 1.5R/1R"),
    "y_tp_short": ("cls", "TP-before-SL short 1.5R/1R"),
    "mfe_16": ("median", "16-bar MFE above train median"),
    "mae_16": ("median", "16-bar MAE above train median"),
}


def main() -> None:
    df, feats = wf.load()
    yr = df.index.year
    rows = []
    for ty in YEARS:
        test_mask = yr == ty
        test_start = np.flatnonzero(test_mask)[0]
        train_pos = np.flatnonzero((yr >= ty - TRAIN_YEARS) & (yr < ty))
        train_pos = train_pos[train_pos < test_start - wf.PURGE_BARS]
        test_pos = np.flatnonzero(test_mask)
        for tgt, (kind, desc) in TARGETS.items():
            v = df[tgt].to_numpy()
            tp = train_pos[np.isfinite(v[train_pos])]
            te = test_pos[np.isfinite(v[test_pos])]
            if kind == "median":
                thr = np.median(v[tp])
                ytr, yte = (v[tp] > thr).astype(int), (v[te] > thr).astype(int)
            else:
                ytr, yte = v[tp].astype(int), v[te].astype(int)
            mdl = lgb.train(PARAMS, lgb.Dataset(df[feats].iloc[tp], ytr),
                            num_boost_round=400)
            auc = roc_auc_score(yte, mdl.predict(df[feats].iloc[te]))
            rows.append(dict(target=tgt, desc=desc, test_year=ty,
                             auc=float(auc), n_test=len(te)))
            print(f"{ty} {tgt:<12} auc={auc:.4f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "target_comparison.csv", index=False)
    print("\nmean AUC by target:\n",
          res.groupby("target").auc.mean().sort_values(ascending=False)
          .round(4).to_string())


if __name__ == "__main__":
    main()
