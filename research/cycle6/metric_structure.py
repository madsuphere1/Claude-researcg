"""C6-001 — Metric STRUCTURE: hierarchy, redundancy, and combinations.

Corrects a real fallacy in C5: metrics were ranked as flat independent
columns. They are not independent — they are a coupled algebraic system
(like PV=nRT), with a few primitives and many derived quantities, some
literally identical. This cycle:

  1. Proves the redundancy algebraically (identities that must hold).
  2. Measures it empirically (correlation clustering + effective
     dimensionality via PCA) — how many INDEPENDENT axes exist.
  3. Defines the metric hierarchy: primitives vs derived.
  4. Searches COMBINATIONS of the true primitives for forward-predictive
     structure that no single metric shows — with leave-one-out CV and a
     permutation null so a small (n~42) panel cannot fake a pattern.

No retraining; reuses the five cached trade streams.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO / "research" / "cycle5"))
from metrics import METRIC_KEYS, compute  # noqa: E402

ART = REPO / "strategy" / "artifacts"
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
POLICIES = ["expanding", "rolling5", "rolling4", "rolling3", "rolling2"]
YEARS = list(range(2014, 2027))
SPANS = [2, 3, 5]

# ---- the metric hierarchy, declared (formula = how it is built) ----
PRIMITIVES = {
    "win_rate": "p — fraction of trades with R>0",
    "avg_win_R": "W — mean R of winners",
    "avg_loss_R": "L — mean R of losers",
    "std_R": "sigma — dispersion of R",
    "downside_dev": "sigma_d — dispersion of losses only",
    "max_dd_R": "deepest cumulative-R drawdown (path)",
    "max_consec_losses": "longest losing run (path)",
    "skew": "3rd moment of R",
    "kurtosis": "4th moment of R",
    "n_trades": "count (scale)",
}
DERIVED = {
    "expectancy_R": "= p*W + (1-p)*L",
    "payoff_ratio": "= W / |L|",
    "profit_factor": "= p*W / ((1-p)*|L|)",
    "omega": "== profit_factor at threshold 0 (DUPLICATE)",
    "sharpe": "= expectancy_R / std_R",
    "sortino": "= expectancy_R / downside_dev",
    "calmar": "= total_R / |max_dd_R|",
    "recovery_factor": "== calmar (DUPLICATE)",
    "total_R": "= n_trades * expectancy_R",
    "median_R": "quantile (≈avg_loss while p<0.5)",
    "ulcer_index": "RMS of drawdown path (max_dd family)",
    "var95": "5th percentile (tail, ~barrier-pinned)",
    "cvar95": "mean of worst 5% (tail, ~barrier-pinned)",
    "tail_ratio": "= |p95| / |p5|",
    "max_consec_wins": "longest winning run (path)",
    "trades_per_year": "scale",
    "best_R": "max (barrier-pinned)",
    "worst_R": "min (barrier-pinned)",
}


def load_streams():
    out = {}
    for p in POLICIES:
        f = ART / f"trades_{p}_allyears.parquet"
        t = pd.read_parquet(f)
        t["entry_time"] = pd.to_datetime(t["entry_time"])
        out[p] = t
    return out


def interval_table(streams) -> pd.DataFrame:
    rows = []
    for p, tr in streams.items():
        for s in SPANS:
            for a in range(YEARS[0], YEARS[-1] - s + 2):
                w = tr[(tr.year >= a) & (tr.year <= a + s - 1)]
                if len(w) < 15:
                    continue
                m = compute(w.r_mult.to_numpy(), years_span=s)
                m["_policy"], m["_span"], m["_start"] = p, s, a
                rows.append(m)
    return pd.DataFrame(rows)


def forward_panel(streams, trail=2):
    """Trailing-`trail`yr primitive metrics -> next-year expectancy."""
    rows = []
    for p, tr in streams.items():
        for t in range(YEARS[0] + trail - 1, YEARS[-1]):
            w = tr[(tr.year >= t - trail + 1) & (tr.year <= t)]
            fwd = tr[tr.year == t + 1].r_mult
            if len(w) < 20 or len(fwd) < 10:
                continue
            m = compute(w.r_mult.to_numpy(), years_span=trail)
            m["_fwd"] = float(fwd.mean())
            rows.append(m)
    return pd.DataFrame(rows)


def main() -> None:
    streams = load_streams()
    tbl = interval_table(streams)
    num = tbl[METRIC_KEYS].apply(pd.to_numeric, errors="coerce")
    num = num.loc[:, num.std() > 1e-9]           # drop zero-variance (structural constants)
    dropped = [k for k in METRIC_KEYS if k not in num.columns]
    print(f"interval cells: {len(tbl)}; metrics with variance: {num.shape[1]}; "
          f"zero-variance dropped: {dropped}\n")

    out = {"n_cells": len(tbl), "zero_variance_metrics": dropped}

    # ---- 1. algebraic identities (prove the redundancy) ----
    ident = {}
    fh = {p: compute(streams[p].r_mult.to_numpy(), years_span=13) for p in POLICIES}
    checks = []
    for p in POLICIES:
        m = fh[p]
        exp_hat = m["win_rate"] * m["avg_win_R"] + (1 - m["win_rate"]) * m["avg_loss_R"]
        checks.append(abs(exp_hat - m["expectancy_R"]))
        assert abs(m["omega"] - m["profit_factor"]) < 1e-9
        assert abs(m["calmar"] - m["recovery_factor"]) < 1e-9
    ident["expectancy_identity_max_err"] = float(max(checks))
    ident["omega_equals_profit_factor"] = True
    ident["calmar_equals_recovery_factor"] = True
    out["algebraic_identities"] = ident
    print("ALGEBRAIC IDENTITIES (redundancy is exact, not incidental):")
    print(f"  expectancy = p*W+(1-p)*L   max error {ident['expectancy_identity_max_err']:.2e}")
    print("  omega == profit_factor      exact (literal duplicate column)")
    print("  calmar == recovery_factor   exact (literal duplicate column)\n")

    # ---- 2. empirical redundancy: correlation clusters + effective dim ----
    corr = num.corr(method="spearman").abs()
    # greedy clustering at |rho| >= 0.9
    thr = 0.90
    cols = list(corr.columns)
    unassigned = set(cols)
    clusters = []
    for c in cols:
        if c not in unassigned:
            continue
        grp = {d for d in unassigned if corr.loc[c, d] >= thr}
        clusters.append(sorted(grp))
        unassigned -= grp
    out["redundancy_clusters_rho0.90"] = clusters

    Xz = StandardScaler().fit_transform(num.values)
    pca = PCA().fit(Xz)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    n95 = int(np.searchsorted(cum, 0.95) + 1)
    out["effective_dim_90pct"] = n90
    out["effective_dim_95pct"] = n95
    out["pc_variance_ratio"] = [round(float(v), 3) for v in pca.explained_variance_ratio_[:6]]
    print(f"EFFECTIVE DIMENSIONALITY: {num.shape[1]} metrics collapse to "
          f"{n90} axes (90% var), {n95} axes (95%).")
    print(f"  top PC variance shares: {out['pc_variance_ratio']}")
    print(f"  correlation clusters (|rho|>=0.90), {len(clusters)} groups:")
    for g in clusters:
        print(f"    - {g}")
    print()

    # ---- 3. combination search for forward prediction ----
    panel = forward_panel(streams, trail=2)
    prim = [k for k in PRIMITIVES if k in panel.columns and panel[k].std() > 1e-9
            and k != "n_trades"]
    y = panel["_fwd"].values
    Xall = StandardScaler().fit_transform(panel[prim].values)
    loo = LeaveOneOut()

    def cv_r2(cols_idx):
        X = Xall[:, cols_idx]
        preds = np.empty(len(y))
        for tr_i, te_i in loo.split(X):
            preds[te_i] = LinearRegression().fit(X[tr_i], y[tr_i]).predict(X[te_i])
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot            # OOS R² (can be negative)

    singles = {prim[i]: cv_r2([i]) for i in range(len(prim))}
    best_single = max(singles, key=singles.get)
    pairs = {}
    for i, j in itertools.combinations(range(len(prim)), 2):
        pairs[(prim[i], prim[j])] = cv_r2([i, j])
    best_pair = max(pairs, key=pairs.get)

    # permutation null: shuffle y, redo the ENTIRE best-of-single+pair search
    rng = np.random.default_rng(7)
    obs_best = max(singles[best_single], pairs[best_pair])
    null = []
    for _ in range(500):
        yp = rng.permutation(y)

        def cvp(cols_idx):
            X = Xall[:, cols_idx]
            pr = np.empty(len(yp))
            for tr_i, te_i in loo.split(X):
                pr[te_i] = LinearRegression().fit(X[tr_i], yp[tr_i]).predict(X[te_i])
            return 1 - np.sum((yp - pr) ** 2) / np.sum((yp - yp.mean()) ** 2)
        bs = max(cvp([i]) for i in range(len(prim)))
        bp = max(cvp([i, j]) for i, j in itertools.combinations(range(len(prim)), 2))
        null.append(max(bs, bp))
    perm_p = float((np.array(null) >= obs_best).mean())

    out["combination_search"] = {
        "n_pairs": len(panel), "primitives": prim,
        "best_single": [best_single, round(singles[best_single], 3)],
        "best_pair": [list(best_pair), round(pairs[best_pair], 3)],
        "top5_singles": sorted(({k: round(v, 3) for k, v in singles.items()}).items(),
                               key=lambda kv: -kv[1])[:5],
        "top5_pairs": [[list(k), round(v, 3)] for k, v in
                       sorted(pairs.items(), key=lambda kv: -kv[1])[:5]],
        "permutation_p_best": perm_p,
        "null_best_mean": round(float(np.mean(null)), 3),
        "null_best_p95": round(float(np.quantile(null, 0.95)), 3),
    }
    print(f"COMBINATION SEARCH (forward prediction, OOS leave-one-out R², n={len(panel)})")
    print(f"  best single primitive: {best_single}  OOS R²={singles[best_single]:+.3f}")
    print(f"  best PAIR: {best_pair}  OOS R²={pairs[best_pair]:+.3f}")
    print(f"  permutation null: best-search R² mean {out['combination_search']['null_best_mean']:+.3f}, "
          f"95th pct {out['combination_search']['null_best_p95']:+.3f}")
    print(f"  permutation p(best >= observed) = {perm_p:.3f}  "
          f"-> {'REAL structure' if perm_p < 0.05 else 'NOT distinguishable from noise'}")

    (RESULTS / "c6_metric_structure.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\nsaved {RESULTS/'c6_metric_structure.json'}")


if __name__ == "__main__":
    main()
