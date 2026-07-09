"""Feature evaluation: MI, model importance, permutation importance,
correlation redundancy, VIF.

Primary target: y_tp_long (TP-before-SL for a hypothetical long, 1.5R/1R,
16-bar horizon); secondary: y_dir_1. All statistics are computed inside a
walk-forward structure - never on the full sample - so rankings reflect
out-of-sample relevance, not in-sample fit.

Outputs land in research/results/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

import wf

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
RNG = np.random.default_rng(7)

LGB_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=63,
    min_data_in_leaf=500, feature_fraction=0.7, bagging_fraction=0.8,
    bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=8, seed=7,
)


def mutual_information(df: pd.DataFrame, feats: list[str]) -> pd.Series:
    """MI on the pre-2014 development window only (never test years)."""
    dev = df[df.index < "2014-01-01"]
    dev = dev.dropna(subset=["y_tp_long"])
    sub = dev.sample(n=min(60_000, len(dev)), random_state=7)
    X = sub[feats].fillna(sub[feats].median())
    mi = mutual_info_classif(X, sub.y_tp_long.astype(int), random_state=7,
                             n_neighbors=5)
    return pd.Series(mi, index=feats).sort_values(ascending=False)


def wf_importance(df: pd.DataFrame, feats: list[str], label: str
                  ) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """LightGBM gain importance + AUC per walk-forward fold, plus
    permutation importance on the last 3 folds."""
    folds = wf.year_folds(df.index)
    gains, perms, fold_stats = [], [], []
    for fold in folds:
        tr = df.iloc[fold.train_idx].dropna(subset=[label])
        te = df.iloc[fold.test_idx].dropna(subset=[label])
        if len(tr) < 50_000 or len(te) < 2_000:
            continue
        t0 = time.time()
        ds = lgb.Dataset(tr[feats], tr[label].astype(int))
        model = lgb.train(LGB_PARAMS, ds, num_boost_round=400)
        p = model.predict(te[feats])
        auc = roc_auc_score(te[label].astype(int), p)
        fold_stats.append({"test_year": fold.test_year, "auc": float(auc),
                           "n_train": len(tr), "n_test": len(te),
                           "secs": round(time.time() - t0, 1)})
        gains.append(pd.Series(model.feature_importance("gain"), index=feats,
                               name=fold.test_year))
        if fold.test_year >= 2024:
            base = auc
            drops = {}
            X = te[feats].copy()
            for f in feats:
                saved = X[f].to_numpy().copy()
                X[f] = RNG.permutation(saved)
                drops[f] = base - roc_auc_score(te[label].astype(int),
                                                model.predict(X))
                X[f] = saved
            perms.append(pd.Series(drops, name=fold.test_year))
        print(f"fold {fold.test_year}: auc={auc:.4f} ({fold_stats[-1]['secs']}s)",
              flush=True)
    return pd.concat(gains, axis=1), (pd.concat(perms, axis=1) if perms else pd.DataFrame()), fold_stats


def redundancy(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    dev = df[df.index < "2014-01-01"]
    sub = dev.sample(n=min(80_000, len(dev)), random_state=7)[feats]
    sub = sub.fillna(sub.median())
    corr = sub.corr(method="spearman").fillna(0.0)
    dist = 1 - corr.abs()
    np.fill_diagonal(dist.values, 0.0)
    link = linkage(squareform(dist.values, checks=False), method="average")
    clusters = fcluster(link, t=0.1, criterion="distance")  # |rho| >= 0.9
    out = pd.DataFrame({"feature": feats, "cluster": clusters})
    sizes = out.groupby("cluster").transform("size")
    out["redundant_group"] = (sizes > 1).to_numpy()
    return out.sort_values(["cluster", "feature"]), corr


def vif_table(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """VIF via R^2 of each feature on the others (ridge-stabilised)."""
    dev = df[df.index < "2014-01-01"]
    sub = dev.sample(n=min(40_000, len(dev)), random_state=7)[feats]
    sub = sub.fillna(sub.median())
    X = (sub - sub.mean()) / (sub.std() + 1e-12)
    X = X.loc[:, X.std() > 0]
    XtX = X.T @ X / len(X)
    prec = np.linalg.pinv(XtX + 1e-6 * np.eye(len(XtX)))
    vif = pd.Series(np.diag(prec) * np.diag(XtX.values), index=X.columns)
    return vif.sort_values(ascending=False).rename("vif").to_frame()


def main() -> None:
    df, feats = wf.load()
    print(f"{len(df):,} rows, {len(feats)} features")

    mi = mutual_information(df, feats)
    mi.rename("mutual_info").to_frame().to_csv(RESULTS / "mutual_info.csv")
    print("MI top10:\n", mi.head(10).to_string())

    gains, perms, fold_stats = wf_importance(df, feats, "y_tp_long")
    gains.to_csv(RESULTS / "lgbm_gain_by_fold.csv")
    norm = gains / gains.sum(axis=0)
    rank = norm.mean(axis=1).sort_values(ascending=False)
    rank.rename("mean_gain_share").to_frame().to_csv(RESULTS / "lgbm_gain_rank.csv")
    if not perms.empty:
        perms.mean(axis=1).sort_values(ascending=False).rename(
            "mean_auc_drop").to_frame().to_csv(RESULTS / "perm_importance.csv")
    with open(RESULTS / "wf_auc_y_tp_long.json", "w") as fh:
        json.dump(fold_stats, fh, indent=1)

    clusters, corr = redundancy(df, feats)
    clusters.to_csv(RESULTS / "corr_clusters.csv", index=False)
    corr.to_csv(RESULTS / "spearman_corr.csv")
    vif = vif_table(df, feats)
    vif.to_csv(RESULTS / "vif.csv")

    print("\nWF AUC by year:", {d["test_year"]: round(d["auc"], 4) for d in fold_stats})
    print("gain top15:\n", rank.head(15).to_string())
    n_red = clusters.redundant_group.sum()
    print(f"features in redundant clusters (|rho|>=0.9): {n_red}")
    print("VIF>10 count:", int((vif.vif > 10).sum()))


if __name__ == "__main__":
    main()
