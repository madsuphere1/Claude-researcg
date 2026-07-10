"""XAUUSD X1 — final consolidated strategy, pre-trade analysis + backtest.

STATUS (registry C3-001): WEAKENED — positive but unproven. Read this
before using the numbers:
  * Confirmation sample (odd years 2015-2025): +0.079R/trade net,
    334 trades, PF 1.15, day-block p=0.108 (misses the 0.05 gate).
  * Exploratory pilot (even years 2014-2026): +0.128R, 414 trades.
  * Costs assume 1.25bp passive fills (registry C3-002 measures the
    real number; it is the most verdict-sensitive assumption).
  * Nothing here qualifies for any production gate (FMROS Volume 9).
    Simulated results; not investment advice.

WHAT THIS FILE IS. The single runnable artifact that codifies everything
validated across 19 closed experiments (see
FMROS/appendix/experiment_registry/registry.md):

  signal   walk-forward LightGBM on 205 features, G2 triple-barrier
           labels (TP 3xATR / SL 2xATR / 64-bar horizon), expanding
           training, annual retrain, 96-bar purge     [C1-BASE, H-I1]
  regime   3-state Gaussian HMM on [return, log RV], causal forward
           filter, trade only when P(high-vol) > 0.6          [H-A1]
  entry    passive limit at close - 0.10xATR, cancel after 4 bars,
           trade-through fill, 1.25bp cost                    [H-B1]
  exits    barrier exits, pessimistic ties (SL checked before TP),
           same-bar ambiguity resolved on M1 data           [C3-004]
  risk     1% equity risk/trade, 10x leverage cap, pre-weekend
           flatten, max 5 signals/day, no Friday >=15:00 arming
                                                       [H-G1, H-B2v2]

Every constant below is FROZEN from the pre-registered cycle-3 spec.
Changing any of them creates a NEW, untested strategy and voids the
anchor verification at the bottom of this file.

Run:  python3 strategy/xauusd_x1_final.py
First run computes walk-forward predictions for 13 folds (~15-30 min)
and caches them under strategy/artifacts/.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO / "research"))
sys.path.insert(0, str(REPO / "research" / "cycle2"))

import backtest  # noqa: E402
import wf  # noqa: E402
from features import atr  # noqa: E402
from exp_b2_barriers import GEOMETRIES, barrier_labels  # noqa: E402
from exp_b1_limit_entry import load_m1_arrays, m1_order  # noqa: E402
import exp_x1_composite_pilot as x1  # noqa: E402

ART = Path(__file__).parent / "artifacts"
ART.mkdir(exist_ok=True)

# ---------------------------------------------------------------- frozen spec
GEOM = GEOMETRIES["G2"]          # tp=3.0, sl=2.0 (xATR), h=64 bars
DELTA = 0.10                     # limit offset, xATR
K_BARS = 4                       # cancel unfilled limit after
GATE_P = 0.6                     # min P(high-vol regime)
COST_LIMIT_BP = 1.25             # passive-fill cost assumption (C3-002 open)
RISK = 0.01                      # equity risked per trade
MAX_LEV = 10.0
TICK = 0.01
MAX_TRADES_DAY = backtest.MAX_TRADES_DAY   # 5
GAP_MIN = 120                    # minutes; larger = session gap
ALL_YEARS = list(range(2014, 2027))
EVEN_YEARS = [y for y in ALL_YEARS if y % 2 == 0]   # exploratory pilot
ODD_YEARS = [y for y in ALL_YEARS if y % 2 == 1]    # confirmation sample

# anchors: recorded verdict numbers this file must reproduce exactly
ANCHORS = {"even": (414, 0.12790770985595395),
           "odd": (334, 0.07883523355330557)}


# ---------------------------------------------------- pre-trade quant checks
@dataclass
class TradeDecision:
    """Outcome of the full pre-trade checklist for one bar."""
    go: bool
    side: int = 0                          # +1 long, -1 short, 0 none
    checks: dict = field(default_factory=dict)   # name -> (passed, detail)

    def report(self) -> str:
        lines = [f"DECISION: {'ARM ' + ('LONG' if self.side == 1 else 'SHORT') if self.go else 'NO TRADE'}"]
        for name, (ok, detail) in self.checks.items():
            lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        return "\n".join(lines)


def pre_trade_decision(i, *, p_long, p_short, thr, gate, atr_val, fri_pm,
                       next_gap_min, day_count, day_pnl,
                       daily_stop=None) -> TradeDecision:
    """The quant checklist evaluated before ANY order is armed.

    Order and semantics are identical to the frozen backtest loop, so a
    live decision and a backtested decision can never diverge. `thr` is
    the walk-forward-calibrated probability threshold for the current
    model year — thresholds are chosen on validation data only.
    `daily_stop` is the Volume-7 house overlay (e.g. -0.03); it is None
    in the frozen spec and in anchor verification.
    """
    d = TradeDecision(go=False)
    c = d.checks

    ok = np.isfinite(atr_val) and atr_val > 0
    c["volatility defined"] = (ok, f"ATR(14)x2 = {atr_val:.2f}" if ok else "ATR missing")
    if not ok:
        return d

    ok = next_gap_min <= GAP_MIN
    c["no session gap ahead"] = (ok, f"next bar in {next_gap_min:.0f} min (max {GAP_MIN})")
    if not ok:
        return d

    ok = not fri_pm
    c["weekend risk window"] = (ok, "not Friday >= 15:00" if ok else "Friday >= 15:00 — gap risk, no arming")
    if not ok:
        return d

    ok = bool(gate)
    c["vol regime gate"] = (ok, f"P(high-vol) > {GATE_P}: {'in regime' if ok else 'out of regime'}")
    if not ok:
        return d

    ok = day_count < MAX_TRADES_DAY
    c["daily signal capacity"] = (ok, f"{day_count}/{MAX_TRADES_DAY} used")
    if not ok:
        return d

    if daily_stop is not None:
        ok = day_pnl > daily_stop
        c["daily loss stop"] = (ok, f"day P&L {day_pnl:+.2%} vs stop {daily_stop:.0%}")
        if not ok:
            return d

    ok = thr is not None
    c["model calibrated"] = (ok, f"threshold {thr:.3f}" if ok else "no model for this year")
    if not ok:
        return d

    side = 0
    if p_long >= thr and p_long >= p_short:
        side = 1
    elif p_short >= thr:
        side = -1
    edge = (p_long if side == 1 else p_short) - thr if side else max(p_long, p_short) - thr
    c["model edge"] = (side != 0,
                       f"p_long={p_long:.3f} p_short={p_short:.3f} thr={thr:.3f} "
                       f"(margin {edge:+.3f})")
    if side == 0:
        return d

    d.go, d.side = True, side
    return d


# ------------------------------------------------------------------- engine
def build_engine():
    """Data, G2 labels, walk-forward predictions, HMM gate, M1 arrays.

    Predictions and the regime gate are computed once for ALL years
    2014-2026 and cached; each fold trains only on data strictly before
    its test year (96-bar purge), so slicing by year parity reproduces
    the original even-year and odd-year runs input-identically.
    """
    df, feats = wf.load()
    bars = pd.read_parquet(REPO / "research" / "data" / "xauusd_15m.parquet")
    bars = bars.loc[df.index]
    a14 = atr(bars, 14)
    a14s = a14 * GEOM["sl"]                       # SL distance in price units

    labs = barrier_labels(bars, a14, GEOM["tp"], GEOM["sl"], GEOM["h"])
    d2 = df.copy()
    d2["y_tp_long"], d2["y_tp_short"] = labs.y_tp_long, labs.y_tp_short

    backtest.TP_R = GEOM["tp"] / GEOM["sl"]
    backtest.SL_R, backtest.HORIZON = 1.0, GEOM["h"]
    backtest.atr14 = lambda x, _a=a14s: _a.reindex(x.index)

    pred_f, thr_f, gate_f = (ART / "predictions.parquet",
                             ART / "thresholds.json", ART / "gate.parquet")
    if pred_f.exists():
        pred = pd.read_parquet(pred_f)
        thr = {int(k): v for k, v in json.loads(thr_f.read_text()).items()}
        gate = pd.read_parquet(gate_f)["gate"]
    else:
        wf.TEST_YEARS = ALL_YEARS
        folds = backtest.walk_forward_predictions(d2, feats)
        pred = pd.concat([f.pred for f in folds])
        thr = {f.test_year: f.threshold for f in folds}
        x1.FOLD_YEARS = ALL_YEARS
        gate = x1.hmm_gate(df, bars)
        pred.to_parquet(pred_f)
        thr_f.write_text(json.dumps(thr))
        gate.rename("gate").to_frame().to_parquet(gate_f)
    return d2, bars, a14s, pred, thr, gate, load_m1_arrays()


def simulate(df, pred, thr_by_year, gate, a14s, m1arrs, daily_stop=None):
    """Event-driven backtest. Logic-identical port of the frozen
    cycle-2/3 simulator with entry eligibility routed through
    pre_trade_decision() (one code path for backtest and live)."""
    mi, mh, ml = m1arrs
    sub = df.loc[pred.index]
    a = a14s.reindex(pred.index).to_numpy()
    o, h = sub.open.to_numpy(), sub.high.to_numpy()
    l, c = sub.low.to_numpy(), sub.close.to_numpy()
    idx = sub.index
    yrs = idx.year.to_numpy()
    p_long, p_short = pred.p_long.to_numpy(), pred.p_short.to_numpy()
    g = gate.reindex(pred.index).fillna(False).to_numpy()
    gap_min = idx.to_series().diff().dt.total_seconds().to_numpy() / 60
    next_gap = np.roll(gap_min, -1)
    next_gap[-1] = 0
    tday = (idx - pd.Timedelta(hours=18)).date
    fri_pm = (idx.dayofweek == 4) & (idx.hour >= 15)
    n = len(sub)

    trades, eq_curve = [], {}
    equity, day_eq0 = 1.0, 1.0
    day, day_n = None, 0
    pending, pos = None, None

    def close_trade(i, exit_px, reason):
        nonlocal pos, equity
        gross = pos["side"] * (exit_px - pos["entry"]) / pos["entry"]
        pnl = (gross - COST_LIMIT_BP * 1e-4) * pos["lev"]
        equity *= 1 + pnl
        trades.append(dict(entry_time=pos["t_fill"], exit_time=idx[i],
                           side=pos["side"], reason=reason, pnl=pnl,
                           r_mult=pnl / RISK, year=int(yrs[i]),
                           bars=int(i - pos["i_fill"]), p=pos["p"],
                           entry=pos["entry"], tp=pos["tp"], sl=pos["sl"],
                           lev=pos["lev"]))
        pos = None

    for i in range(n):
        if tday[i] != day:
            day, day_n, day_eq0 = tday[i], 0, equity

        # ---- manage open position: pessimistic ties, gap handling
        if pos is not None:
            if next_gap[i] > GAP_MIN and i + 1 < n:      # pre-gap flatten
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
            elif gap_min[i] > GAP_MIN:                   # gapped open
                close_trade(i, o[i], "gap")
            else:
                hit_sl = (l[i] <= pos["sl"]) if pos["side"] == 1 else (h[i] >= pos["sl"])
                hit_tp = (h[i] >= pos["tp"]) if pos["side"] == 1 else (l[i] <= pos["tp"])
                if hit_sl:                               # SL before TP: pessimistic
                    close_trade(i, pos["sl"], "sl")
                elif hit_tp:
                    close_trade(i, pos["tp"], "tp")
                elif i >= pos["expiry"]:
                    close_trade(i, c[i], "timeout")

        # ---- pending limit order: fill or expire
        if pos is None and pending is not None:
            if gap_min[i] > GAP_MIN or i > pending["exp_i"] or next_gap[i] > GAP_MIN:
                pending = None
            else:
                side, lim = pending["side"], pending["limit"]
                if (l[i] <= lim - TICK) if side == 1 else (h[i] >= lim + TICK):
                    av = pending["atr"]
                    entry = lim
                    sl_px = entry - side * av             # SL = 1 unit of 2xATR
                    tp_px = entry + side * (GEOM["tp"] / GEOM["sl"]) * av
                    lev = min(RISK / (av / entry), MAX_LEV)
                    pos = dict(side=side, entry=entry, sl=sl_px, tp=tp_px,
                               lev=lev, expiry=i + GEOM["h"], t_fill=idx[i],
                               i_fill=i, p=pending["p"])
                    same_bar_sl = (l[i] <= sl_px) if side == 1 else (h[i] >= sl_px)
                    pending = None
                    if same_bar_sl:                       # M1 resolves ambiguity
                        first = m1_order(mi, mh, ml, idx[i],
                                         idx[i] + pd.Timedelta(minutes=15),
                                         lim, -side, sl_px, -side)
                        if first in ("both_same_min", "b"):
                            close_trade(i, sl_px, "sl_samebar")

        # ---- new signal: full pre-trade checklist
        if pos is None and pending is None and i + 1 < n:
            dec = pre_trade_decision(
                i, p_long=p_long[i], p_short=p_short[i],
                thr=thr_by_year.get(yrs[i]), gate=g[i], atr_val=a[i],
                fri_pm=fri_pm[i], next_gap_min=next_gap[i],
                day_count=day_n, day_pnl=equity / day_eq0 - 1,
                daily_stop=daily_stop)
            if dec.go:
                pending = dict(side=dec.side,
                               limit=c[i] - dec.side * DELTA * a[i] / GEOM["sl"],
                               atr=a[i], exp_i=i + K_BARS,
                               p=p_long[i] if dec.side == 1 else p_short[i])
                day_n += 1
        eq_curve[idx[i]] = equity

    eq = pd.Series(eq_curve)
    daily = eq.groupby(pd.Series(tday, index=eq.index)).last()
    return pd.DataFrame(trades), daily


# ------------------------------------------------------------------ metrics
def streak_stats(r: pd.Series) -> dict:
    import itertools
    win = (r > 0).astype(int).to_list()
    sw, sl_ = [], []
    for k, grp in itertools.groupby(win):
        (sw if k else sl_).append(len(list(grp)))
    return dict(max_consec_wins=max(sw, default=0),
                max_consec_losses=max(sl_, default=0),
                mean_win_streak=float(np.mean(sw)) if sw else 0.0,
                mean_loss_streak=float(np.mean(sl_)) if sl_ else 0.0)


def year_block_bootstrap(trades: pd.DataFrame, n_boot=5000, seed=7) -> dict:
    rng = np.random.default_rng(seed)
    gs = [v.r_mult.to_numpy() for _, v in trades.groupby("year")]
    means = np.array([np.concatenate([gs[j] for j in rng.integers(0, len(gs), len(gs))]).mean()
                      for _ in range(n_boot)])
    return dict(ci_lo=float(np.quantile(means, .025)),
                ci_hi=float(np.quantile(means, .975)),
                p_leq_0=float((means <= 0).mean()))


def cost_tier_table(trades: pd.DataFrame) -> dict:
    """Re-cost the recorded trade stream analytically: each trade paid
    COST_LIMIT_BP once; expectancy at tier c shifts by (c - 1.25)bp
    scaled through leverage. Fill probabilities held fixed (ITT)."""
    out = {}
    for c_bp in (1.25, 1.5, 2.5, 4.0):
        adj = trades.r_mult - (c_bp - COST_LIMIT_BP) * 1e-4 * trades.lev / RISK
        out[f"{c_bp}bp"] = dict(expectancy_R=float(adj.mean()),
                                win_rate=float((adj > 0).mean()),
                                total_R=float(adj.sum()))
    return out


def full_report(tag: str, trades: pd.DataFrame, daily: pd.Series) -> dict:
    rep = {"n_trades": int(len(trades))}
    rep["stats"] = backtest.perf_stats(trades, daily)
    rep["streaks"] = streak_stats(trades.r_mult)
    rep["bootstrap_dayblock"] = backtest.block_bootstrap_ci(trades)
    rep["bootstrap_yearblock"] = year_block_bootstrap(trades)
    rep["cost_tiers"] = cost_tier_table(trades)
    rep["by_year"] = {str(y): dict(n=int(len(v)), exp=float(v.r_mult.mean()))
                      for y, v in trades.groupby("year")}
    rep["exit_reasons"] = trades.reason.value_counts().to_dict()
    print(f"\n===== {tag} =====")
    s = rep["stats"]
    print(f"trades {rep['n_trades']}  expectancy {s['expectancy_R']:+.3f}R  "
          f"win rate {s['win_rate']:.1%}  PF {s['profit_factor']:.2f}  "
          f"Sharpe {s['sharpe']:.2f}  maxDD {s['max_dd']:.1%}")
    print(f"day-block p(<=0) {rep['bootstrap_dayblock']['p_leq_0']:.3f}  "
          f"year-block p(<=0) {rep['bootstrap_yearblock']['p_leq_0']:.3f}")
    print("cost tiers:", {k: round(v['expectancy_R'], 3)
                          for k, v in rep["cost_tiers"].items()})
    return rep


# --------------------------------------------------------------------- main
def main() -> None:
    print("building engine (first run trains 13 walk-forward folds)...")
    d2, bars, a14s, pred, thr, gate, m1arrs = build_engine()

    results = {}
    for tag, years in (("even", EVEN_YEARS), ("odd", ODD_YEARS)):
        mask = pred.index.year.isin(years)
        trades, daily = simulate(d2, pred[mask], thr, gate, a14s, m1arrs)
        n_anchor, exp_anchor = ANCHORS[tag]
        n, exp = len(trades), trades.r_mult.mean()
        ok = n == n_anchor and abs(exp - exp_anchor) < 1e-9
        print(f"anchor {tag}: n={n} (recorded {n_anchor}), "
              f"exp={exp:+.6f} (recorded {exp_anchor:+.6f}) -> "
              f"{'MATCH' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit(f"ANCHOR MISMATCH ({tag}): this file no longer "
                             "reproduces the recorded verdicts — do not use.")
        results[tag] = full_report(
            f"{tag.upper()} YEARS ({'exploratory pilot' if tag == 'even' else 'CONFIRMATION sample'})",
            trades, daily)
        trades.to_parquet(ART / f"trades_{tag}.parquet")

    # pooled view — descriptive only (pooling was never pre-registered)
    tr_all = pd.concat([pd.read_parquet(ART / f"trades_{t}.parquet")
                        for t in ("even", "odd")]).sort_values("entry_time")
    print(f"\npooled (DESCRIPTIVE ONLY): {len(tr_all)} trades, "
          f"{tr_all.r_mult.mean():+.3f}R, win rate {(tr_all.r_mult > 0).mean():.1%}")

    # live-style demo: the checklist on the most recent bar in the data
    last = pred.index[-1]
    i = pred.index.get_loc(last)
    dec = pre_trade_decision(
        i, p_long=pred.p_long.iloc[i], p_short=pred.p_short.iloc[i],
        thr=thr.get(last.year), gate=bool(gate.reindex(pred.index).iloc[i]),
        atr_val=float(a14s.reindex(pred.index).iloc[i]),
        fri_pm=bool(last.dayofweek == 4 and last.hour >= 15),
        next_gap_min=0.0, day_count=0, day_pnl=0.0, daily_stop=-0.03)
    print(f"\n--- pre-trade checklist demo @ {last} ---")
    print(dec.report())

    (ART / "final_report.json").write_text(
        json.dumps(results, indent=1, default=float))
    print(f"\nfull report -> {ART / 'final_report.json'}")


if __name__ == "__main__":
    main()
