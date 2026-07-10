"""H-B1: passive limit-order entry vs market entry.

Cycle-1 market-entry strategy pays the full spread. A resting limit order
entered *below* the signal price (for longs) earns queue price improvement
and avoids paying the spread - IF it fills, and fills are adversely
selected (limits fill exactly when price moves against the signal).
This experiment quantifies that trade-off with an intention-to-treat
design: the cohort is ALL signal bars; unfilled orders contribute 0.

Declared spec (REGISTRY.md):
* Same signal bars and thresholds as cycle-1 backtest (cached predictions),
  same one-position/max-5-per-day/no-Friday-PM rules.
* Limit price = signal close - delta*ATR (long; mirrored short),
  delta in {0.10, 0.25, 0.50}; primary 0.25. Working time K = 4 bars,
  then cancel.
* Fill requires the 15m low to trade >= 1 tick (0.01) THROUGH the limit.
  If the fill bar also breaches the trade's SL, M1 data decides the
  intra-bar ordering: we walk minute bars; if SL level is traded through
  after (or in the same minute as) the fill minute, the trade exits SL in
  that bar (pessimistic on ties).
* Costs: market entry 2.5bp round-trip; limit entry 1.25bp (exit leg only).
  Sensitivity: limit at 1.5bp, market at 1.5bp.
* TP/SL = +1.5/-1.0 x ATR(signal) from FILL price; 16-bar horizon from fill.

Success criterion: intention-to-treat net expectancy per SIGNAL improves
by >= +0.10R over market entry; reject if < +0.05R.
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

HERE = Path(__file__).parent
RESULTS = HERE / "results"
ART = HERE / "artifacts"
TICK = 0.01
K_BARS = 4
HORIZON = 16
TP_R, SL_R = 1.5, 1.0
RISK = 0.01


def load_m1_arrays():
    m1 = pd.read_parquet(HERE.parents[1] / "research" / "data" / "xauusd_m1.parquet",
                         columns=["high", "low"])
    return (m1.index.values.astype("datetime64[ns]"),
            m1.high.to_numpy(), m1.low.to_numpy())


def m1_order(mi, mh, ml, t0, t1, level_a, dir_a, level_b, dir_b):
    """Between t0..t1, which level is traded first? dir=+1 means 'high >=
    level', -1 means 'low <= level'. Returns 'a', 'b', or 'both_same_min'
    /'none'."""
    i0, i1 = np.searchsorted(mi, np.datetime64(t0)), np.searchsorted(mi, np.datetime64(t1))
    for k in range(i0, min(i1, len(mi))):
        hit_a = (mh[k] >= level_a) if dir_a > 0 else (ml[k] <= level_a)
        hit_b = (mh[k] >= level_b) if dir_b > 0 else (ml[k] <= level_b)
        if hit_a and hit_b:
            return "both_same_min"
        if hit_a:
            return "a"
        if hit_b:
            return "b"
    return "none"


def simulate_limit(df, pred, thr_by_year, delta, cost_bp_limit, cost_bp_mkt,
                   m1arrs):
    """Event simulation with pending limit orders. Also runs the matched
    market-entry counterfactual per signal for cohort comparison."""
    mi, mh, ml = m1arrs
    sub = df.loc[pred.index]
    a = backtest.atr14(df).loc[pred.index].to_numpy()
    o = sub.open.to_numpy(); h = sub.high.to_numpy()
    l = sub.low.to_numpy(); c = sub.close.to_numpy()
    idx = sub.index
    yrs = idx.year.to_numpy()
    p_long = pred.p_long.to_numpy(); p_short = pred.p_short.to_numpy()
    gap_min = idx.to_series().diff().dt.total_seconds().to_numpy() / 60
    tday = (idx - pd.Timedelta(hours=18)).date
    fri_pm = (idx.dayofweek == 4) & (idx.hour >= 15)
    n = len(sub)

    signals = []          # per-signal records (intention to treat)
    day = None; day_trades = 0
    pending = None        # dict(side, limit, atr, placed_i, expires_i)
    pos = None
    equity = 1.0

    def close_trade(i, exit_px, reason):
        nonlocal pos, equity
        gross = pos["side"] * (exit_px - pos["entry"]) / pos["entry"]
        net = gross - cost_bp_limit * 1e-4
        pnl = net * pos["lev"]
        equity *= (1 + pnl)
        pos["rec"].update(filled=True, r_mult=pnl / RISK, reason=reason,
                          exit_time=str(idx[i]))
        signals.append(pos["rec"])
        pos = None

    for i in range(n):
        if tday[i] != day:
            day, day_trades = tday[i], 0
        # manage open position
        if pos is not None:
            if gap_min[i] > 120:
                close_trade(i, o[i], "gap")
            else:
                if pos["side"] == 1:
                    hit_sl = l[i] <= pos["sl"]; hit_tp = h[i] >= pos["tp"]
                else:
                    hit_sl = h[i] >= pos["sl"]; hit_tp = l[i] <= pos["tp"]
                if hit_sl:
                    close_trade(i, pos["sl"], "sl")
                elif hit_tp:
                    close_trade(i, pos["tp"], "tp")
                elif i >= pos["expiry"]:
                    close_trade(i, c[i], "timeout")
        # manage pending limit
        if pos is None and pending is not None:
            if gap_min[i] > 120 or i > pending["expires_i"]:
                pending["rec"].update(filled=False, r_mult=0.0, reason="cancel")
                signals.append(pending["rec"]); pending = None
            else:
                side, lim = pending["side"], pending["limit"]
                touched = (l[i] <= lim - TICK) if side == 1 else (h[i] >= lim + TICK)
                if touched:
                    entry = lim
                    av = pending["atr"]
                    sl_px = entry - side * SL_R * av
                    tp_px = entry + side * TP_R * av
                    # same-bar SL conflict -> M1 ordering
                    same_bar_sl = (l[i] <= sl_px) if side == 1 else (h[i] >= sl_px)
                    lev = RISK / (SL_R * av / entry)
                    pos = dict(side=side, entry=entry, sl=sl_px, tp=tp_px,
                               lev=lev, expiry=i + HORIZON, rec=pending["rec"])
                    pos["rec"]["fill_time"] = str(idx[i])
                    pending = None
                    if same_bar_sl:
                        # order within the bar decided on minute data
                        first = m1_order(mi, mh, ml, idx[i], idx[i] + pd.Timedelta(minutes=15),
                                         lim, -side, sl_px, -side)
                        # both levels are on the adverse side; limit is nearer,
                        # so 'a' (limit) normally trades first; if SL traded
                        # in the same minute -> pessimistic stop-out
                        if first in ("both_same_min", "b"):
                            close_trade(i, sl_px, "sl_samebar")
        # new signal
        if pos is None and pending is None and i + 1 < n \
                and gap_min[i + 1] <= 120 and not fri_pm[i] \
                and day_trades < backtest.MAX_TRADES_DAY \
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
                av = a[i]
                lim = c[i] - side * delta * av
                # matched market-entry counterfactual for this signal
                mkt = market_counterfactual(i, side, o, h, l, c, gap_min, av,
                                            cost_bp_mkt)
                rec = dict(signal_time=str(idx[i]), side=side, year=int(yrs[i]),
                           atr_frac=float(av / c[i]), mkt_r=mkt)
                pending = dict(side=side, limit=lim, atr=av, placed_i=i,
                               expires_i=i + K_BARS, rec=rec)
                day_trades += 1
    return pd.DataFrame(signals)


def market_counterfactual(i, side, o, h, l, c, gap_min, av, cost_bp):
    """R-multiple of the cycle-1 market entry for the same signal."""
    n = len(o)
    if i + 1 >= n:
        return 0.0
    entry = o[i + 1]
    sl_px = entry - side * SL_R * av
    tp_px = entry + side * TP_R * av
    lev = RISK / (SL_R * av / entry)
    for j in range(i + 1, min(i + 1 + HORIZON, n)):
        if gap_min[j] > 120:
            exit_px = o[j]; break
        hit_sl = (l[j] <= sl_px) if side == 1 else (h[j] >= sl_px)
        hit_tp = (h[j] >= tp_px) if side == 1 else (l[j] <= tp_px)
        if hit_sl:
            exit_px = sl_px; break
        if hit_tp:
            exit_px = tp_px; break
    else:
        exit_px = c[min(i + HORIZON, n - 1)]
    gross = side * (exit_px - entry) / entry
    return float((gross - cost_bp * 1e-4) * lev / RISK)


def main() -> None:
    df, _ = wf.load()
    pred = pd.read_parquet(ART / "wf_predictions.parquet")
    thr = {int(k): v["threshold"]
           for k, v in json.loads((ART / "wf_thresholds.json").read_text()).items()}
    m1arrs = load_m1_arrays()

    out = {}
    for delta in (0.10, 0.25, 0.50):
        res = simulate_limit(df, pred, thr, delta, cost_bp_limit=1.25,
                             cost_bp_mkt=2.5, m1arrs=m1arrs)
        filled = res[res.filled]
        itt_limit = res.r_mult.mean()          # unfilled contribute 0
        itt_mkt = res.mkt_r.mean()
        d = dict(n_signals=len(res), fill_rate=float(res.filled.mean()),
                 itt_limit_R=float(itt_limit), itt_market_R=float(itt_mkt),
                 itt_uplift_R=float(itt_limit - itt_mkt),
                 filled_exp_R=float(filled.r_mult.mean()) if len(filled) else np.nan,
                 filled_mkt_counterfactual_R=float(filled.mkt_r.mean()) if len(filled) else np.nan,
                 unfilled_mkt_counterfactual_R=float(res[~res.filled].mkt_r.mean()) if (~res.filled).any() else np.nan,
                 sl_samebar_frac=float((filled.reason == "sl_samebar").mean()) if len(filled) else np.nan)
        # bootstrap CI on the ITT uplift (paired by signal, day blocks)
        diff = res.r_mult - res.mkt_r
        days = pd.to_datetime(res.signal_time).dt.date
        rng = np.random.default_rng(7)
        g = [v.to_numpy() for _, v in diff.groupby(days.values)]
        means = np.array([np.concatenate([g[j] for j in rng.integers(0, len(g), len(g))]).mean()
                          for _ in range(3000)])
        d["uplift_ci"] = [float(np.quantile(means, 0.025)),
                          float(np.quantile(means, 0.975))]
        out[f"delta_{delta}"] = d
        print(delta, {k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in d.items()}, flush=True)

    (RESULTS / "b1_limit_entry.json").write_text(json.dumps(out, indent=1, default=float))
    print("saved b1_limit_entry.json")


if __name__ == "__main__":
    main()
