"""H-B2: barrier geometry study - does widening barriers beat the cost drag?

Cost is fixed in price terms, so cost-in-R = cost / SL_distance falls as
1/SL. If the signal has multi-hour persistence, gross edge should fall
slower than cost when barriers widen. Geometries (declared in REGISTRY.md):

  G0 baseline : TP 1.5×ATR / SL 1.0×ATR / 16 bars (cycle-1 result reused)
  G1 medium   : TP 2.0×ATR / SL 1.33×ATR / 32 bars
  G2 wide     : TP 3.0×ATR / SL 2.0×ATR  / 64 bars

Identical payoff ratio (1.5) across geometries so win-probability targets
are comparable. Folds: test years 2014, 2016, ..., 2026 (pre-declared
subset), expanding training window, threshold chosen on the last training
year only. Full event-driven simulation with 2.5bp costs, 1%% risk.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
import backtest  # noqa: E402
import wf  # noqa: E402
from features import atr  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

GEOMETRIES = {"G1": dict(tp=2.0, sl=4 / 3, h=32),
              "G2": dict(tp=3.0, sl=2.0, h=64)}
FOLD_YEARS = list(range(2014, 2027, 2))


def barrier_labels(bars: pd.DataFrame, a14: pd.Series, tp: float, sl: float,
                   h: int) -> pd.DataFrame:
    """Vectorised-enough triple-barrier labels for both sides."""
    n = len(bars)
    hi = bars.high.to_numpy(); lo = bars.low.to_numpy()
    entry = bars.open.shift(-1).to_numpy()
    a = a14.to_numpy()
    gap_min = bars.index.to_series().diff().dt.total_seconds().to_numpy() / 60
    y_long = np.full(n, np.nan); y_short = np.full(n, np.nan)
    for i in range(n - h - 1):
        e, av = entry[i], a[i]
        if not (np.isfinite(e) and np.isfinite(av)) or av <= 0:
            continue
        tp_l, sl_l = e + tp * av, e - sl * av
        tp_s, sl_s = e - tp * av, e + sl * av
        done_l = done_s = False
        for j in range(1, h + 1):
            if gap_min[i + j] > 120:
                break
            hj, lj = hi[i + j], lo[i + j]
            if not done_l:
                if lj <= sl_l:
                    y_long[i], done_l = 0.0, True
                elif hj >= tp_l:
                    y_long[i], done_l = 1.0, True
            if not done_s:
                if hj >= sl_s:
                    y_short[i], done_s = 0.0, True
                elif lj <= tp_s:
                    y_short[i], done_s = 1.0, True
            if done_l and done_s:
                break
    return pd.DataFrame({"y_tp_long": y_long, "y_tp_short": y_short},
                        index=bars.index)


def main() -> None:
    df, feats = wf.load()
    bars = pd.read_parquet(HERE.parents[1] / "research" / "data" / "xauusd_15m.parquet")
    bars = bars.loc[df.index]
    a14 = atr(bars, 14)

    out = {}
    for gname, g in GEOMETRIES.items():
        t0 = time.time()
        labs = barrier_labels(bars, a14, g["tp"], g["sl"], g["h"])
        base = labs.y_tp_long.mean()
        print(f"{gname}: labels done ({time.time()-t0:.0f}s) "
              f"P(tp first)={base:.4f} resolved={labs.y_tp_long.notna().mean():.3f}",
              flush=True)

        d2 = df.copy()
        d2["y_tp_long"] = labs.y_tp_long
        d2["y_tp_short"] = labs.y_tp_short

        backtest.TP_R, backtest.SL_R, backtest.HORIZON = g["tp"] / g["sl"], 1.0, g["h"]
        # NOTE: risk sizing uses SL distance = SL_R*atr inside simulate; we
        # express barriers in units of (sl*ATR) so SL_R=1, TP_R=payoff ratio
        # -> need ATR column scaled: pass scaled atr via monkeypatched atr14
        scale = g["sl"]
        backtest.atr14 = lambda x, _s=scale, _a=a14: _a.reindex(x.index) * _s

        wf.TEST_YEARS = FOLD_YEARS
        folds = backtest.walk_forward_predictions(d2, feats)
        res_g = {"label_base_rate": float(base),
                 "fold_thresholds": {f.test_year: f.threshold for f in folds}}
        for cb in (0.0, 1.5, 2.5):
            trades, eq = backtest.simulate(d2, folds, cost_bp=cb)
            s = backtest.perf_stats(trades, eq)
            res_g[f"cost_{cb}"] = s
            if cb == 2.5 and len(trades):
                res_g["bootstrap_2.5"] = backtest.block_bootstrap_ci(trades)
            print(f"  {gname} @{cb}bp: n={s.get('n_trades')} "
                  f"exp={s.get('expectancy_R', float('nan')):+.4f}R "
                  f"sharpe={s.get('sharpe', float('nan')):.2f}", flush=True)
        out[gname] = res_g
        (RESULTS / "b2_barriers.json").write_text(json.dumps(out, indent=1, default=float))
    print("saved b2_barriers.json")


if __name__ == "__main__":
    main()
