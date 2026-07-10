"""EXPLORATORY SCREEN (not confirmatory): X1 composite driven by
rolling-window models instead of expanding-window models.

Motivation (operator hypothesis, 2026-07-10): recent structural change
makes long history harmful; rolling windows should adapt better. H-I1
refuted this at AUC level across 2014-2026 overall (expanding 0.529 vs
rolling5 0.523, rolling3 0.519), BUT rolling5 won the AUC head-to-head
in 2024 and 2026 — 2 of the last 3 years. That sliver is the declared
mechanism for this screen; it earns a *screen on the exploratory
slice*, not a re-litigation of H-I1 on confirmation data.

Design (declared before run):
* Slice: EVEN years 2014-2026 only — the exploratory pilot slice. The
  odd-year confirmation slice is NOT touched (one-look rule).
* Variants: rolling window W=5 and W=2 years (multiplicity: 2,
  declared). Classifier training window only; HMM gate, thresholds
  protocol, and every other X1 parameter stay frozen.
* Readout: per-year net expectancy vs the recorded expanding-window
  even-year results, with specific attention to 2024/2026 (the recent
  losing stretch). NO decision rule — screens generate hypotheses.
  The confirmatory arbiter is registry C3-008 (expanding vs rolling on
  2027+ data, rule frozen before that data exists).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO / "strategy"))
sys.path.insert(0, str(REPO / "research"))
sys.path.insert(0, str(REPO / "research" / "cycle2"))

import backtest  # noqa: E402
import wf  # noqa: E402
from xauusd_x1_final import (ART, EVEN_YEARS, GEOM, build_engine,  # noqa: E402
                             simulate)

HERE = Path(__file__).parent
RESULTS = HERE / "results"
WINDOWS = [5, 2]


def rolling_predictions(df, feats, window: int, test_years):
    """backtest.walk_forward_predictions with the training mask limited
    to the trailing `window` years (the ONLY change vs the house
    expanding protocol)."""
    out = []
    years = df.index.year
    for ty in test_years:
        test_mask = years == ty
        if not test_mask.any():
            continue
        test_start = np.flatnonzero(test_mask)[0]
        train_mask = (years >= ty - window) & (years < ty)     # <- rolling
        train_pos = np.flatnonzero(train_mask)
        train_pos = train_pos[train_pos < test_start - wf.PURGE_BARS]
        val_mask = years[train_pos] >= ty - backtest.VAL_YEARS
        fit_pos, val_pos = train_pos[~val_mask], train_pos[val_mask]

        preds, val_stats = {}, {}
        for side, label in (("long", "y_tp_long"), ("short", "y_tp_short")):
            lab = df[label].to_numpy()
            fp = fit_pos[np.isfinite(lab[fit_pos])]
            vp = val_pos[np.isfinite(lab[val_pos])]
            mdl = lgb.train(backtest.LGB_PARAMS,
                            lgb.Dataset(df[feats].iloc[fp], lab[fp].astype(int)),
                            num_boost_round=backtest.N_ROUNDS)
            preds[f"p_{side}"] = mdl.predict(df[feats].iloc[np.flatnonzero(test_mask)])
            val_stats[side] = (mdl.predict(df[feats].iloc[vp]), lab[vp])

        pv = np.concatenate([val_stats["long"][0], val_stats["short"][0]])
        yv = np.concatenate([val_stats["long"][1], val_stats["short"][1]])
        n_days = max(1, len(np.unique(df.index[val_pos].date)))
        thr, val_e = backtest.choose_threshold(
            pv, yv, len(val_pos) / n_days,
            np.quantile(pv, np.linspace(0.5, 0.995, 60)))
        out.append(backtest.FoldPred(ty, thr, val_e,
                                     pd.DataFrame(preds, index=df.index[test_mask])))
        print(f"W={window} fold {ty}: thr={thr:.3f} val_exp={val_e:+.3f}R",
              flush=True)
    return out


def main() -> None:
    d2, bars, a14s, pred_exp, thr_exp, gate, m1arrs = build_engine()
    _, feats = wf.load()

    # recorded expanding-window even-year baseline (anchor-verified)
    base = pd.read_parquet(ART / "trades_even.parquet")
    by_year_base = {int(y): float(v.r_mult.mean())
                    for y, v in base.groupby("year")}

    out = {"baseline_expanding_by_year": by_year_base,
           "baseline_expanding_exp": float(base.r_mult.mean()),
           "baseline_n_trades": int(len(base)),
           "declared": "exploratory screen, even years only, multiplicity 2"}
    for W in WINDOWS:
        folds = rolling_predictions(d2, feats, W, EVEN_YEARS)
        pred = pd.concat([f.pred for f in folds])
        thr = {f.test_year: f.threshold for f in folds}
        trades, daily = simulate(d2, pred, thr, gate, a14s, m1arrs)
        key = f"rolling{W}"
        if len(trades):
            out[key] = dict(
                n_trades=int(len(trades)),
                expectancy_R=float(trades.r_mult.mean()),
                win_rate=float((trades.r_mult > 0).mean()),
                by_year={str(y): dict(n=int(len(v)), exp=float(v.r_mult.mean()))
                         for y, v in trades.groupby("year")})
            trades.to_parquet(RESULTS / f"screen_rolling{W}_trades.parquet")
        else:
            out[key] = dict(n_trades=0)
        print(f"\n=== rolling{W}: {out[key].get('n_trades', 0)} trades, "
              f"{out[key].get('expectancy_R', float('nan')):+.3f}R "
              f"(expanding baseline {out['baseline_expanding_exp']:+.3f}R, "
              f"{out['baseline_n_trades']} trades) ===", flush=True)

    (RESULTS / "screen_rolling_x1.json").write_text(
        json.dumps(out, indent=1, default=float))
    print("\nsaved", RESULTS / "screen_rolling_x1.json")


if __name__ == "__main__":
    main()
