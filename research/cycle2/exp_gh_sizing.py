"""H-G1: sizing & risk engine study (Tracks G+H).

Sizing cannot create edge - it reshapes a given trade stream's growth/risk
profile. We test four rules on:
  (a) the cycle-1 GROSS stream (0bp costs; positive expectancy +0.11R) -
      the "if execution were solved" scenario;
  (b) the cycle-1 NET stream (2.5bp; negative) - sanity: nothing should
      turn this profitable.

Rules (declared):
  flat1      : 1% risk always (baseline)
  prob       : risk = 0.5% + 1.5% * clip((p - thr)/0.1, 0, 1)
  kelly025   : quarter-Kelly from a trailing 300-trade estimate, clipped [0.25%, 2%]
  ddthrottle : 1% nominal; x0.5 below 10% drawdown, x0.25 below 20%

Metrics: geometric growth (CAGR over the simulated span), max drawdown,
MAR (CAGR/|maxDD|), and Monte-Carlo probability of >=50% drawdown
(10k day-block resamples of the R stream under each rule).
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

HERE = Path(__file__).parent
RESULTS = HERE / "results"
ART = HERE / "artifacts"


def get_streams():
    df, _ = wf.load()
    pred = pd.read_parquet(ART / "wf_predictions.parquet")
    thr = {int(k): v["threshold"] for k, v in
           json.loads((ART / "wf_thresholds.json").read_text()).items()}
    folds = [FoldPred(ty, thr[ty], 0.0, pred[pred.index.year == ty])
             for ty in sorted(thr) if (pred.index.year == ty).any()]
    streams = {}
    for name, cb in (("gross_0bp", 0.0), ("net_2.5bp", 2.5)):
        tr, _ = backtest.simulate(df, folds, cost_bp=cb)
        tr["thr"] = tr.entry_time.dt.year.map(thr)
        streams[name] = tr
    return streams


def run_rule(tr: pd.DataFrame, rule: str) -> pd.Series:
    """Returns equity curve indexed by trade; risk fraction chosen per rule
    using only information available before each trade."""
    eq = 1.0
    peak = 1.0
    curve = np.empty(len(tr))
    r_hist: list[float] = []
    p = tr.p.to_numpy(); thr = tr.thr.to_numpy(); r_mult = tr.r_mult.to_numpy()
    for i in range(len(tr)):
        if rule == "flat1":
            f = 0.01
        elif rule == "prob":
            f = 0.005 + 0.015 * min(max((p[i] - thr[i]) / 0.1, 0), 1)
        elif rule == "kelly025":
            if len(r_hist) >= 100:
                arr = np.array(r_hist[-300:])
                k = arr.mean() / (arr.var() + 1e-12)
                f = float(np.clip(0.25 * k * 0.01, 0.0025, 0.02)) if k > 0 else 0.0025
            else:
                f = 0.01
        elif rule == "ddthrottle":
            dd = eq / peak - 1
            f = 0.01 * (0.25 if dd < -0.20 else 0.5 if dd < -0.10 else 1.0)
        else:
            raise ValueError(rule)
        eq *= 1 + f * r_mult[i]
        peak = max(peak, eq)
        curve[i] = eq
        r_hist.append(r_mult[i])
    return pd.Series(curve, index=tr.entry_time.values)


def stats_from_curve(curve: pd.Series) -> dict:
    eq = curve.to_numpy()
    yrs = (curve.index[-1] - curve.index[0]) / np.timedelta64(365, "D")
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    return dict(final=float(eq[-1]),
                cagr=float(eq[-1] ** (1 / yrs) - 1),
                max_dd=float(dd.min()),
                mar=float((eq[-1] ** (1 / yrs) - 1) / abs(dd.min()) if dd.min() < 0 else np.nan))


def ruin_mc(tr: pd.DataFrame, rule: str, n=2000, seed=7) -> float:
    """P(max drawdown >= 50%) under day-block resampling."""
    rng = np.random.default_rng(seed)
    days = tr.entry_time.dt.date
    groups = [g for _, g in tr.groupby(days.values)]
    nD = len(groups)
    hits = 0
    for _ in range(n):
        sample = pd.concat([groups[j] for j in rng.integers(0, nD, nD)],
                           ignore_index=True)
        sample = sample.assign(entry_time=pd.date_range(
            "2014-01-01", periods=len(sample), freq="4h"))
        curve = run_rule(sample, rule)
        eq = curve.to_numpy()
        dd = (eq / np.maximum.accumulate(eq) - 1).min()
        hits += dd <= -0.5
    return hits / n


def main() -> None:
    streams = get_streams()
    out = {}
    for sname, tr in streams.items():
        res = {}
        for rule in ("flat1", "prob", "kelly025", "ddthrottle"):
            curve = run_rule(tr, rule)
            s = stats_from_curve(curve)
            s["p_ruin50"] = ruin_mc(tr, rule)
            res[rule] = s
            print(sname, rule, {k: round(v, 4) for k, v in s.items()}, flush=True)
        res["expectancy_R"] = float(tr.r_mult.mean())
        res["n_trades"] = len(tr)
        out[sname] = res
    (RESULTS / "gh_sizing.json").write_text(json.dumps(out, indent=1))
    print("saved gh_sizing.json")


if __name__ == "__main__":
    main()
