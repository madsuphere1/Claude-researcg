"""Dissect each window's signal by REGIME — do not trust the flat mean.

The operator's fair critique of window_research_cycle.py: declaring
rolling-2 "no signal" from a full-history mean AUC of 0.5012 is exactly
the aggregate-hides-concentration fallacy the Constitution forbids
(C3-001 proved this edge lives in volatility episodes, not uniformly in
time). A window can look dead on average yet carry a real edge inside
its own edge-case regime.

So: for each window, compute test-year AUC SEPARATELY within the HMM
high-volatility gate (the bars the X1 strategy actually trades — the
program's established edge-case selector) vs outside it. If a window is
truly dead, it is dead in the gated bars too. If a window has a hidden
edge case, gated AUC reveals it. The gate is identical across windows
(same HMM), so the comparison is clean.

Multiplicity: 3 windows x 2 regime cells, declared. Diagnostic, not a
new confirmatory claim; C3-008 remains the arbiter on 2027+ data.
"""

from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO / "strategy"))
sys.path.insert(0, str(REPO / "research"))
sys.path.insert(0, str(REPO / "research" / "cycle2"))

from xauusd_x1_final import ART, build_engine  # noqa: E402

RESULTS = Path(__file__).parent / "results"
YEARS = list(range(2014, 2027))
WINDOWS = {
    "expanding": ART / "predictions.parquet",
    "rolling5": ART / "pred_rolling5_allyears.parquet",
    "rolling2": ART / "pred_rolling2_allyears.parquet",
}


def sign_p(k: int, n: int) -> float:
    return float(sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n)


def auc_over(mask, y, p):
    yy, pp = y[mask], p[mask]
    ok = yy.notna()
    if ok.sum() < 100 or yy[ok].nunique() < 2:
        return float("nan"), int(ok.sum())
    return float(roc_auc_score(yy[ok].astype(int), pp[ok])), int(ok.sum())


def main() -> None:
    d2, bars, a14s, pred_exp, thr_exp, gate, m1arrs = build_engine()
    yl = d2["y_tp_long"]
    g = gate.reindex(pred_exp.index).fillna(False)      # high-vol gate, same for all windows
    print(f"gate coverage overall: {g.mean():.1%} of bars\n")

    out = {"gate_coverage": float(g.mean())}
    for name, path in WINDOWS.items():
        pred = pd.read_parquet(path)
        gg = g.reindex(pred.index).fillna(False)
        gated_aucs, ungated_aucs = [], []
        rows = {}
        for y in YEARS:
            ym = pred.index.year == y
            idx = pred.index[ym]
            yly = yl.reindex(idx)
            ply = pred.p_long[ym]
            gy = gg.reindex(idx)
            ag, ng = auc_over(gy.to_numpy(), yly, ply)          # gated (edge-case regime)
            au, nu = auc_over(~gy.to_numpy(), yly, ply)         # ungated
            rows[str(y)] = dict(auc_gated=None if np.isnan(ag) else round(ag, 4),
                                n_gated=ng,
                                auc_ungated=None if np.isnan(au) else round(au, 4))
            if not np.isnan(ag):
                gated_aucs.append(ag)
            if not np.isnan(au):
                ungated_aucs.append(au)
        ng_years = len(gated_aucs)
        kg = sum(a > 0.5 for a in gated_aucs)
        out[name] = dict(
            mean_auc_gated=round(float(np.mean(gated_aucs)), 4),
            mean_auc_ungated=round(float(np.mean(ungated_aucs)), 4),
            gated_years_above_0p5=f"{kg}/{ng_years}",
            gated_sign_test_p=sign_p(kg, ng_years),
            gated_best_year=round(float(np.max(gated_aucs)), 4),
            gated_worst_year=round(float(np.min(gated_aucs)), 4),
            by_year=rows,
        )

    print("SIGNAL DISSECTED BY REGIME — AUC inside vs outside the high-vol gate")
    print("(the gate is the edge-case selector the X1 strategy trades)\n")
    print(f"{'window':>11} {'gated AUC':>10} {'gated yrs>0.5':>14} {'sign p':>9} "
          f"{'ungated AUC':>12} {'gated best yr':>13}")
    for name in WINDOWS:
        r = out[name]
        print(f"{name:>11} {r['mean_auc_gated']:>10.4f} {r['gated_years_above_0p5']:>14} "
              f"{r['gated_sign_test_p']:>9.2e} {r['mean_auc_ungated']:>12.4f} "
              f"{r['gated_best_year']:>13.4f}")

    print("\nPER-YEAR GATED AUC (the edge-case bars only)")
    print(f"{'year':>6} " + " ".join(f"{w:>11}" for w in WINDOWS))
    for y in YEARS:
        cells = []
        for w in WINDOWS:
            v = out[w]["by_year"][str(y)]["auc_gated"]
            cells.append(f"{v:>11.4f}" if v is not None else f"{'—':>11}")
        print(f"{y:>6} " + " ".join(cells))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "window_edge_dissection.json").write_text(json.dumps(out, indent=1))
    print(f"\nsaved {RESULTS / 'window_edge_dissection.json'}")


if __name__ == "__main__":
    main()
