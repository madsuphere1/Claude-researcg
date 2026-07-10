"""X1 composite pilot - EXPLORATORY, requires cycle-3 confirmation.

Cycle-2 validated three independent levers, each attacking a different
term of the net-expectancy identity:
  net_R = gross_R(signal quality) - cost_bp / sl_bp(geometry & fill type)

  * H-A1 vol-gate: concentrates gross edge (+0.19R gross vs +0.11R)
  * H-B2 G2 geometry: halves cost-in-R (SL 2xATR)
  * H-B1 limit entry: halves the cost paid (delta=0.10, ITT +0.09R)

This pilot mechanically conjoins them: G2 walk-forward predictions
(cached by exp_b2v2), HMM vol-gate (recomputed with the same seeds as
exp_a1), limit entries at delta=0.10xATR with K=4 cancel, pre-gap flatten
and 10x leverage cap. Costs: 1.25bp on limit-filled trades (exit leg only).

HONESTY NOTE: the composite itself was formed after seeing cycle-2 test
results. Every component was individually pre-declared, but their
conjunction was not. Results are reported as hypothesis-generating only;
the confirmation protocol (cycle 3) is: freeze this exact spec, evaluate
on ODD test years 2015-2025 which none of the cycle-2 fold models were
tested on... note models were trained expanding so odd years appear in
TRAINING of later folds -> a truly clean confirmation needs new data
(2027+) or nested re-training. Both caveats are stated in the report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

sys.path.insert(0, str(Path(__file__).parents[1]))
import backtest  # noqa: E402
import wf  # noqa: E402
from features import atr  # noqa: E402
from exp_a1_hmm_regimes import forward_filter  # noqa: E402
from exp_b1_limit_entry import load_m1_arrays, m1_order  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "results"
ART = HERE / "artifacts"
FOLD_YEARS = list(range(2014, 2027, 2))
DELTA, K_BARS = 0.10, 4
TP_R_G2, SL_R_G2, HORIZON = 1.5, 1.0, 64   # in units of 2xATR (scale below)
SCALE = 2.0
GATE_P = 0.6
COST_LIMIT_BP = 1.25
RISK = 0.01
MAX_LEV = 10.0
TICK = 0.01


def hmm_gate(df: pd.DataFrame, bars: pd.DataFrame) -> pd.Series:
    r = np.log(bars.close).diff().fillna(0.0).to_numpy()
    lrv = np.log(bars.rv.replace(0, np.nan)).ffill().fillna(-20).to_numpy()
    X = np.column_stack([r * 1e3, lrv])
    yrs = df.index.year
    gate = pd.Series(False, index=df.index)
    for ty in FOLD_YEARS:
        te = np.flatnonzero(yrs == ty)
        if not len(te):
            continue
        tr = np.flatnonzero(yrs < ty)
        tr = tr[tr < te[0] - wf.PURGE_BARS]
        m = GaussianHMM(3, covariance_type="diag", n_iter=60, random_state=7)
        m.fit(X[tr])
        hi = int(np.argsort(m.means_[:, 1])[-1])
        seq = np.concatenate([tr[-500:], te])
        alpha = forward_filter(m, X[seq])[500:]
        gate.iloc[te] = alpha[:, hi] > GATE_P
        print(f"gate {ty}: {gate.iloc[te].mean():.2%}", flush=True)
    return gate


def simulate_composite(df, pred, thr_by_year, gate, a14s, m1arrs):
    """G2 geometry + vol gate + limit entry + pre-gap flatten + lev cap."""
    mi, mh, ml = m1arrs
    sub = df.loc[pred.index]
    a = a14s.reindex(pred.index).to_numpy()      # ALREADY scaled (2xATR)
    o = sub.open.to_numpy(); h = sub.high.to_numpy()
    l = sub.low.to_numpy(); c = sub.close.to_numpy()
    idx = sub.index
    yrs = idx.year.to_numpy()
    p_long = pred.p_long.to_numpy(); p_short = pred.p_short.to_numpy()
    g = gate.reindex(pred.index).fillna(False).to_numpy()
    gap_min = idx.to_series().diff().dt.total_seconds().to_numpy() / 60
    next_gap = np.roll(gap_min, -1); next_gap[-1] = 0
    tday = (idx - pd.Timedelta(hours=18)).date
    fri_pm = (idx.dayofweek == 4) & (idx.hour >= 15)
    n = len(sub)

    trades = []
    equity = 1.0
    eq_curve = {}
    day = None; day_n = 0
    pending = None; pos = None

    def close_trade(i, exit_px, reason):
        nonlocal pos, equity
        gross = pos["side"] * (exit_px - pos["entry"]) / pos["entry"]
        pnl = (gross - COST_LIMIT_BP * 1e-4) * pos["lev"]
        equity *= 1 + pnl
        trades.append(dict(entry_time=pos["t_fill"], exit_time=idx[i],
                           side=pos["side"], reason=reason,
                           pnl=pnl, r_mult=pnl / RISK, year=int(yrs[i]),
                           bars=int(i - pos["i_fill"]), p=np.nan))
        pos = None

    for i in range(n):
        if tday[i] != day:
            day, day_n = tday[i], 0
        if pos is not None:
            if next_gap[i] > 120 and i + 1 < n:
                if pos["side"] == 1 and l[i] <= pos["sl"]:
                    close_trade(i, pos["sl"], "sl")
                elif pos["side"] == 1 and h[i] >= pos["tp"]:
                    close_trade(i, pos["tp"], "tp")
                elif pos["side"] == -1 and h[i] >= pos["sl"]:
                    close_trade(i, pos["sl"], "sl")
                elif pos["side"] == -1 and l[i] <= pos["tp"]:
                    close_trade(i, pos["tp"], "tp")
                else:
                    close_trade(i, c[i], "pre_gap")
            elif gap_min[i] > 120:
                close_trade(i, o[i], "gap")
            else:
                hit_sl = (l[i] <= pos["sl"]) if pos["side"] == 1 else (h[i] >= pos["sl"])
                hit_tp = (h[i] >= pos["tp"]) if pos["side"] == 1 else (l[i] <= pos["tp"])
                if hit_sl:
                    close_trade(i, pos["sl"], "sl")
                elif hit_tp:
                    close_trade(i, pos["tp"], "tp")
                elif i >= pos["expiry"]:
                    close_trade(i, c[i], "timeout")
        if pos is None and pending is not None:
            if gap_min[i] > 120 or i > pending["exp_i"] or next_gap[i] > 120:
                pending = None
            else:
                side, lim = pending["side"], pending["limit"]
                if (l[i] <= lim - TICK) if side == 1 else (h[i] >= lim + TICK):
                    av = pending["atr"]
                    entry = lim
                    sl_px = entry - side * SL_R_G2 * av
                    tp_px = entry + side * TP_R_G2 * av
                    lev = min(RISK / (SL_R_G2 * av / entry), MAX_LEV)
                    pos = dict(side=side, entry=entry, sl=sl_px, tp=tp_px,
                               lev=lev, expiry=i + HORIZON, t_fill=idx[i], i_fill=i)
                    same_bar_sl = (l[i] <= sl_px) if side == 1 else (h[i] >= sl_px)
                    pending = None
                    if same_bar_sl:
                        first = m1_order(mi, mh, ml, idx[i],
                                         idx[i] + pd.Timedelta(minutes=15),
                                         lim, -side, sl_px, -side)
                        if first in ("both_same_min", "b"):
                            close_trade(i, sl_px, "sl_samebar")
        if pos is None and pending is None and i + 1 < n \
                and gap_min[i + 1] <= 120 and not fri_pm[i] and g[i] \
                and day_n < backtest.MAX_TRADES_DAY \
                and np.isfinite(a[i]) and a[i] > 0:
            thr = thr_by_year.get(yrs[i])
            if thr is None:
                continue
            side = 0
            if p_long[i] >= thr and p_long[i] >= p_short[i]:
                side = 1
            elif p_short[i] >= thr:
                side = -1
            if side != 0:
                pending = dict(side=side, limit=c[i] - side * DELTA * a[i] / SCALE,
                               atr=a[i], exp_i=i + K_BARS)
                day_n += 1
        eq_curve[idx[i]] = equity
    eq = pd.Series(eq_curve)
    daily = eq.groupby(pd.Series(tday, index=eq.index)).last()
    return pd.DataFrame(trades), daily


def main() -> None:
    df, _ = wf.load()
    bars = pd.read_parquet(HERE.parents[1] / "research" / "data" / "xauusd_15m.parquet")
    bars = bars.loc[df.index]
    a14 = atr(bars, 14) * SCALE          # G2 units: SL = 2xATR
    pred = pd.read_parquet(ART / "g2_predictions.parquet")
    thr = {int(k): v for k, v in
           json.loads((ART / "g2_thresholds.json").read_text()).items()}
    gate = hmm_gate(df, bars)
    m1arrs = load_m1_arrays()

    trades, daily = simulate_composite(df, pred, thr, gate, a14, m1arrs)
    stats = backtest.perf_stats(trades, daily) if len(trades) else {}
    out = {"stats": stats}
    if len(trades) > 50:
        out["bootstrap"] = backtest.block_bootstrap_ci(trades)
        out["by_year"] = {str(y): dict(n=int(len(g)), exp=float(g.r_mult.mean()))
                          for y, g in trades.groupby("year")}
        trades.to_parquet(RESULTS / "x1_composite_trades.parquet")
        daily.rename("equity").to_frame().to_parquet(RESULTS / "x1_composite_equity.parquet")
    (RESULTS / "x1_composite_pilot.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:2000])


if __name__ == "__main__":
    main()
