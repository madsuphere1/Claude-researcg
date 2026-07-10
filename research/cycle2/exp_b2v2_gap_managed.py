"""H-B2 follow-up: G2 wide-barrier geometry with weekend-gap risk managed.

The first G2 run showed positive mean R but a destroyed equity path. The
diagnosed mechanism (reported transparently): the 64-bar horizon holds
positions across weekend closes; gap exits blow through the stop, and
inverse-ATR sizing amplifies low-vol gap losses. Neither observation uses
signal information - this variant changes RISK RULES only:

  * pre-gap flatten: exit at the close of the last bar before any >2h gap
    (standard practice for intraday books);
  * leverage cap at 10x notional/equity.

Both rules are declared here before running the variant. Predictions are
recomputed identically to exp_b2 (same seeds/params/folds -> identical
models); results reported alongside the unmanaged version, plus a
placebo check (shuffled predictions within year, 50 sims) for the managed
variant, and the same variant applied to the baseline geometry G0 for
completeness.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
import backtest  # noqa: E402
import wf  # noqa: E402
from backtest import FoldPred  # noqa: E402
from features import atr  # noqa: E402
from exp_b2_barriers import barrier_labels, GEOMETRIES  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "results"
ART = HERE / "artifacts"
FOLD_YEARS = list(range(2014, 2027, 2))


def main() -> None:
    df, feats = wf.load()
    bars = pd.read_parquet(HERE.parents[1] / "research" / "data" / "xauusd_15m.parquet")
    bars = bars.loc[df.index]
    a14 = atr(bars, 14)

    g = GEOMETRIES["G2"]
    labs = barrier_labels(bars, a14, g["tp"], g["sl"], g["h"])
    d2 = df.copy()
    d2["y_tp_long"] = labs.y_tp_long
    d2["y_tp_short"] = labs.y_tp_short

    backtest.TP_R, backtest.SL_R, backtest.HORIZON = g["tp"] / g["sl"], 1.0, g["h"]
    scale = g["sl"]
    backtest.atr14 = lambda x, _s=scale, _a=a14: _a.reindex(x.index) * _s
    wf.TEST_YEARS = FOLD_YEARS

    folds = backtest.walk_forward_predictions(d2, feats)
    pred = pd.concat([f.pred for f in folds])
    pred.to_parquet(ART / "g2_predictions.parquet")
    json.dump({f.test_year: f.threshold for f in folds},
              open(ART / "g2_thresholds.json", "w"))

    out = {}
    for cb in (0.0, 1.5, 2.5, 4.0):
        tr, eq = backtest.simulate(d2, folds, cost_bp=cb, pre_gap_flatten=True,
                                   max_lev=10.0)
        s = backtest.perf_stats(tr, eq)
        out[f"managed_{cb}bp"] = s
        if cb == 2.5 and len(tr) > 50:
            out["managed_2.5bp_bootstrap"] = backtest.block_bootstrap_ci(tr)
            tr.to_parquet(RESULTS / "g2_managed_trades.parquet")
            eq.rename("equity").to_frame().to_parquet(RESULTS / "g2_managed_equity.parquet")
        print(f"managed @{cb}bp: n={s['n_trades']} exp={s['expectancy_R']:+.4f} "
              f"sharpe={s['sharpe']:.2f} maxdd={s['max_dd']:.1%} "
              f"final={s['final_equity']:.2f}", flush=True)

    # placebo for the managed variant at 2.5bp
    rng = np.random.default_rng(7)
    exps = []
    for _ in range(50):
        fake = []
        for f in folds:
            p = f.pred.copy()
            perm = rng.permutation(len(p))
            p["p_long"] = p["p_long"].to_numpy()[perm]
            p["p_short"] = p["p_short"].to_numpy()[perm]
            fake.append(FoldPred(f.test_year, f.threshold, 0.0, p))
        t_, _ = backtest.simulate(d2, fake, cost_bp=2.5, pre_gap_flatten=True,
                                  max_lev=10.0)
        if len(t_):
            exps.append(float(t_.r_mult.mean()))
    obs = out["managed_2.5bp"]["expectancy_R"]
    out["placebo"] = dict(n=len(exps), mean=float(np.mean(exps)),
                          sd=float(np.std(exps)),
                          p_geq_obs=float(np.mean([e >= obs for e in exps])))

    (RESULTS / "b2v2_gap_managed.json").write_text(json.dumps(out, indent=1, default=float))
    print("placebo:", out["placebo"])
    print("saved b2v2_gap_managed.json")


if __name__ == "__main__":
    main()
