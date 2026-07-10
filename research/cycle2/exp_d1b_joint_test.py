"""H-D1 step 4 (declared in exp_d1_feature_search docstring): joint
economic test - do the accepted symbolic features add AUC to the full
cycle-1 LightGBM feature set?  Folds 2018/2022/2026, y_tp_long.
Success: mean uplift >= +0.003 AUC."""

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
from exp_d1_feature_search import build_bases, evaluate  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "results"
FOLDS = [2018, 2022, 2026]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=500, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              verbose=-1, num_threads=4, seed=7)


def main() -> None:
    df, feats = wf.load()
    bars = pd.read_parquet(HERE.parents[1] / "research" / "data" / "xauusd_15m.parquet")
    bars = bars.loc[df.index]
    bases = build_bases(bars)

    top = json.load(open(RESULTS / "d1_feature_search.json"))["top"]
    seen, exprs = set(), []
    for t in top:
        if t["accepted"] and t["expr"] not in seen:
            seen.add(t["expr"])
            exprs.append(t)
    exprs = exprs[:10]
    # re-evaluate their series (deterministic from stored expr? we saved the
    # node only implicitly; regenerate with same seed sequence instead)
    # -> simplest reliable route: rebuild all formulas with the same RNG seed
    from exp_d1_feature_search import rand_expr, RNG, N_FORMULAS, expr_str  # noqa: E402
    want = {t["expr"] for t in exprs}
    new_cols = {}
    for i in range(N_FORMULAS):
        node = rand_expr(3)
        s = expr_str(node)
        if s in want and s not in new_cols:
            try:
                new_cols[s] = evaluate(node, bases).astype(np.float32)
            except Exception:  # noqa: BLE001
                pass
    print(f"rebuilt {len(new_cols)}/{len(want)} accepted formulas", flush=True)

    # LightGBM forbids JSON-special characters in feature names
    renamed = {f"sym_{i}": v for i, v in enumerate(new_cols.values())}
    out_names = dict(zip(renamed, new_cols))
    add = pd.DataFrame(renamed, index=df.index)
    dfe = pd.concat([df, add], axis=1)
    lab = df["y_tp_long"].to_numpy()
    yrs = df.index.year
    out = {"exprs": out_names}
    ups = []
    for ty in FOLDS:
        te = np.flatnonzero(yrs == ty)
        tr = np.flatnonzero(yrs < ty)
        tr = tr[tr < te[0] - wf.PURGE_BARS]
        tr = tr[np.isfinite(lab[tr])]
        tev = te[np.isfinite(lab[te])]
        r = {}
        for name, cols in (("baseline", feats),
                           ("baseline+sym", feats + list(renamed))):
            mdl = lgb.train(PARAMS, lgb.Dataset(dfe[cols].iloc[tr], lab[tr].astype(int)),
                            num_boost_round=400)
            r[name] = float(roc_auc_score(lab[tev].astype(int),
                                          mdl.predict(dfe[cols].iloc[tev])))
        r["uplift"] = r["baseline+sym"] - r["baseline"]
        ups.append(r["uplift"])
        out[str(ty)] = r
        print(ty, {k: round(v, 4) for k, v in r.items()}, flush=True)
    out["mean_uplift"] = float(np.mean(ups))
    out["verdict"] = "ACCEPT" if out["mean_uplift"] >= 0.003 else "REJECT"
    (RESULTS / "d1b_joint_test.json").write_text(json.dumps(out, indent=1))
    print("mean uplift:", round(out["mean_uplift"], 4), out["verdict"])


if __name__ == "__main__":
    main()
