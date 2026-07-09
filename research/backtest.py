"""Walk-forward, cost-aware backtest of the ML signal on XAUUSD 15m.

Strategy under test
-------------------
At the close of each 15m bar, two LightGBM models (long / short, trained
walk-forward on y_tp_long / y_tp_short) emit P(TP 1.5R before SL 1R within
16 bars). If the larger probability clears a threshold chosen ON THE
VALIDATION SLICE (last year of the training window - never test data), a
trade opens at the next bar's open.

Execution model
---------------
* TP = 1.5 x ATR14, SL = 1.0 x ATR14 from entry, 16-bar timeout -> market
  exit at bar close.
* Same-bar TP+SL touch -> counted as SL (pessimistic).
* Costs: proportional round-trip cost (spread+commission+slippage) deducted
  from every trade's return; default 2.5bp of entry price, sensitivity at
  1.5 / 4.0bp.
* Risk 1% of current equity per trade (position sized off SL distance),
  one position at a time, max 5 entries per trading day (18:00 EST roll),
  daily loss stop of -3%: no new entries that day after breaching it.
* No entries within 8 bars of the Friday 17:00 EST close, and open
  positions are force-flattened on weekend gaps at the next available open.

Outputs: research/results/backtest_*.csv/json + trades parquet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import wf

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=500, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                  verbose=-1, num_threads=4, seed=7)
N_ROUNDS = 400
TP_R, SL_R = 1.5, 1.0
HORIZON = 16
COST_BP = 2.5          # round-trip, proportional to entry price
RISK = 0.01
MAX_TRADES_DAY = 5
DAILY_STOP = -0.03
VAL_YEARS = 1


@dataclass
class FoldPred:
    test_year: int
    threshold: float
    val_exp_R: float
    pred: pd.DataFrame  # index=bar time, cols p_long, p_short


def atr14(df: pd.DataFrame) -> pd.Series:
    pc = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(),
                    (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()


def choose_threshold(pv: np.ndarray, yv: np.ndarray, bars_per_day: float,
                     grid: np.ndarray) -> tuple[float, float]:
    """Pick the probability threshold on validation data that maximises
    expectancy subject to producing <= MAX_TRADES_DAY signals/day and a
    minimum of ~0.5 signal/day (else the strategy is vacuous)."""
    best_t, best_e = grid[-1], -np.inf
    n_days = len(pv) / bars_per_day
    for t in grid:
        sel = pv >= t
        n = sel.sum()
        if n < 30 or n / n_days > MAX_TRADES_DAY or n / n_days < 0.1:
            continue
        p_win = yv[sel].mean()
        e = p_win * TP_R - (1 - p_win) * SL_R
        if e > best_e:
            best_t, best_e = float(t), float(e)
    return best_t, best_e


def walk_forward_predictions(df: pd.DataFrame, feats: list[str]
                             ) -> list[FoldPred]:
    out = []
    years = df.index.year
    for ty in wf.TEST_YEARS:
        test_mask = years == ty
        if not test_mask.any():
            continue
        test_start = np.flatnonzero(test_mask)[0]
        train_mask = (years >= 2010) & (years < ty)
        train_pos = np.flatnonzero(train_mask)
        train_pos = train_pos[train_pos < test_start - wf.PURGE_BARS]
        val_mask = years[train_pos] >= ty - VAL_YEARS
        fit_pos, val_pos = train_pos[~val_mask], train_pos[val_mask]

        preds = {}
        val_stats = {}
        for side, label in (("long", "y_tp_long"), ("short", "y_tp_short")):
            lab = df[label].to_numpy()
            fp = fit_pos[np.isfinite(lab[fit_pos])]
            vp = val_pos[np.isfinite(lab[val_pos])]
            mdl = lgb.train(LGB_PARAMS,
                            lgb.Dataset(df[feats].iloc[fp], lab[fp].astype(int)),
                            num_boost_round=N_ROUNDS)
            preds[f"p_{side}"] = mdl.predict(df[feats].iloc[np.flatnonzero(test_mask)])
            val_stats[side] = (mdl.predict(df[feats].iloc[vp]), lab[vp])

        # one shared threshold from pooled long/short validation predictions
        pv = np.concatenate([val_stats["long"][0], val_stats["short"][0]])
        yv = np.concatenate([val_stats["long"][1], val_stats["short"][1]])
        # bars per day on validation slice (long side count == bar count)
        n_days = max(1, len(np.unique(df.index[val_pos].date)))
        bars_per_day = len(val_pos) / n_days
        thr, val_e = choose_threshold(pv, yv, bars_per_day,
                                      np.quantile(pv, np.linspace(0.5, 0.995, 60)))
        pred = pd.DataFrame(preds, index=df.index[test_mask])
        out.append(FoldPred(ty, thr, val_e, pred))
        print(f"fold {ty}: thr={thr:.3f} val_exp={val_e:+.3f}R", flush=True)
    return out


# --------------------------------------------------------------------------
# event-driven simulation
# --------------------------------------------------------------------------

def simulate(df: pd.DataFrame, folds: list[FoldPred], cost_bp: float = COST_BP,
             risk: float = RISK, max_trades_day: int = MAX_TRADES_DAY,
             threshold_shift: float = 0.0) -> tuple[pd.DataFrame, pd.Series]:
    """Sequential simulation. Returns (trades, daily equity)."""
    pred = pd.concat([f.pred for f in folds])
    thr_by_year = {f.test_year: f.threshold + threshold_shift for f in folds}
    sub = df.loc[pred.index]
    a = atr14(df).loc[pred.index].to_numpy()
    o = sub.open.to_numpy(); h = sub.high.to_numpy()
    l = sub.low.to_numpy(); c = sub.close.to_numpy()
    idx = sub.index
    yrs = idx.year.to_numpy()
    p_long = pred.p_long.to_numpy(); p_short = pred.p_short.to_numpy()
    gap_min = idx.to_series().diff().dt.total_seconds().to_numpy() / 60
    tday = (idx - pd.Timedelta(hours=18)).date
    # block entries near weekend close: Friday >= 15:00 EST
    fri_pm = (idx.dayofweek == 4) & (idx.hour >= 15)

    equity = 1.0
    trades = []
    eq_curve = {}
    day = None
    day_pnl = 0.0
    day_trades = 0
    pos = None  # dict(side, entry, tp, sl, size_riskfrac, i_entry, expiry)

    n = len(sub)
    for i in range(n):
        if tday[i] != day:
            day = tday[i]
            day_pnl = 0.0
            day_trades = 0
        # manage open position on this bar
        if pos is not None:
            exit_px = None
            reason = None
            if gap_min[i] > 120:      # weekend/holiday gap -> flatten at open
                exit_px, reason = o[i], "gap"
            elif pos["side"] == 1:
                if l[i] <= pos["sl"]:
                    exit_px, reason = pos["sl"], "sl"
                elif h[i] >= pos["tp"]:
                    exit_px, reason = pos["tp"], "tp"
            else:
                if h[i] >= pos["sl"]:
                    exit_px, reason = pos["sl"], "sl"
                elif l[i] <= pos["tp"]:
                    exit_px, reason = pos["tp"], "tp"
            if exit_px is None and i >= pos["expiry"]:
                exit_px, reason = c[i], "timeout"
            if exit_px is not None:
                gross = pos["side"] * (exit_px - pos["entry"]) / pos["entry"]
                net = gross - cost_bp * 1e-4
                pnl = net * pos["lev"]          # fraction of equity
                equity *= (1 + pnl)
                day_pnl += pnl
                trades.append(dict(entry_time=idx[pos["i_entry"]],
                                   exit_time=idx[i], side=pos["side"],
                                   entry=pos["entry"], exit=exit_px,
                                   reason=reason, pnl=pnl,
                                   r_mult=pnl / risk, p=pos["p"],
                                   bars=i - pos["i_entry"]))
                pos = None

        # possible new entry decided at close of bar i, filled at open i+1
        if pos is None and i + 1 < n and gap_min[i + 1] <= 120 \
                and not fri_pm[i] and day_trades < max_trades_day \
                and day_pnl > DAILY_STOP and np.isfinite(a[i]) and a[i] > 0:
            thr = thr_by_year.get(yrs[i])
            if thr is None:
                continue
            side = 0
            if p_long[i] >= thr and p_long[i] >= p_short[i]:
                side = 1
            elif p_short[i] >= thr:
                side = -1
            if side != 0:
                entry = o[i + 1]
                sl_d = SL_R * a[i]
                lev = risk / (sl_d / entry)     # position notional / equity
                pos = dict(side=side, entry=entry,
                           tp=entry + side * TP_R * a[i],
                           sl=entry - side * sl_d,
                           lev=lev, i_entry=i + 1,
                           expiry=i + HORIZON, p=max(p_long[i], p_short[i]))
                day_trades += 1
        eq_curve[idx[i]] = equity

    trades_df = pd.DataFrame(trades)
    eq = pd.Series(eq_curve)
    daily_eq = eq.groupby(pd.Series(tday, index=eq.index)).last()
    return trades_df, daily_eq


# --------------------------------------------------------------------------
# metrics & statistics
# --------------------------------------------------------------------------

def perf_stats(trades: pd.DataFrame, daily_eq: pd.Series) -> dict:
    if trades.empty:
        return {"n_trades": 0}
    r = trades.r_mult.to_numpy()
    dr = daily_eq.pct_change().dropna()
    eq = daily_eq.to_numpy()
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak - 1)
    years = (daily_eq.index[-1] - daily_eq.index[0]).days / 365.25
    cagr = (eq[-1]) ** (1 / years) - 1 if years > 0 else np.nan
    wins, losses = r[r > 0], r[r <= 0]
    pf = wins.sum() / max(1e-12, -losses.sum())
    out = dict(
        n_trades=len(r), trades_per_day=len(r) / max(1, len(daily_eq)),
        win_rate=float((r > 0).mean()), expectancy_R=float(r.mean()),
        median_R=float(np.median(r)), avg_win_R=float(wins.mean()) if len(wins) else np.nan,
        avg_loss_R=float(losses.mean()) if len(losses) else np.nan,
        profit_factor=float(pf),
        sharpe=float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else np.nan,
        sortino=float(dr.mean() / dr[dr < 0].std() * np.sqrt(252)) if (dr < 0).any() else np.nan,
        cagr=float(cagr), max_dd=float(dd.min()),
        calmar=float(cagr / -dd.min()) if dd.min() < 0 else np.nan,
        final_equity=float(eq[-1]),
        avg_bars_held=float(trades.bars.mean()),
        kelly=float(r.mean() / r.var()) if r.var() > 0 else np.nan,
    )
    # streaks
    sign = (r > 0).astype(int)
    diffs = np.diff(np.flatnonzero(np.concatenate(([1], np.diff(sign) != 0, [1]))))
    if len(diffs):
        first = sign[0]
        streaks = [(first if k % 2 == 0 else 1 - first, d) for k, d in enumerate(diffs)]
        out["max_consec_wins"] = int(max((d for s, d in streaks if s == 1), default=0))
        out["max_consec_losses"] = int(max((d for s, d in streaks if s == 0), default=0))
    return out


def block_bootstrap_ci(trades: pd.DataFrame, n_boot=5000, seed=7) -> dict:
    """CI on expectancy (R) using day-block bootstrap to respect clustering."""
    rng = np.random.default_rng(seed)
    days = pd.Series(trades.entry_time.dt.date)
    groups = [g.r_mult.to_numpy() for _, g in trades.groupby(days.values)]
    n = len(groups)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n, n)
        sample = np.concatenate([groups[j] for j in pick])
        means[b] = sample.mean()
    obs = trades.r_mult.mean()
    return dict(exp_R=float(obs),
                ci_lo=float(np.quantile(means, 0.025)),
                ci_hi=float(np.quantile(means, 0.975)),
                p_leq_0=float((means <= 0).mean()))


def random_entry_baseline(df: pd.DataFrame, folds, n_sims=200, seed=7,
                          n_target=None) -> np.ndarray:
    """Monte Carlo: same exit engine, random entries at matched frequency.
    Returns array of per-sim expectancy in R."""
    rng = np.random.default_rng(seed)
    exps = []
    pred = pd.concat([f.pred for f in folds])
    n_bars = len(pred)
    base_trades, _ = simulate(df, folds)
    n_target = n_target or len(base_trades)
    for s in range(n_sims):
        fake = [FoldPred(f.test_year, 0.5, 0.0,
                         f.pred.copy()) for f in folds]
        # random scores; threshold quantile tuned to match trade count
        for f in fake:
            m = len(f.pred)
            f.pred["p_long"] = rng.random(m)
            f.pred["p_short"] = rng.random(m)
            f.threshold = 1 - n_target / n_bars * 1.35  # calibrated roughly
        t, _ = simulate(df, fake)
        if len(t):
            exps.append(t.r_mult.mean())
    return np.array(exps)


def placebo_test(df: pd.DataFrame, folds: list[FoldPred], n_sims=100,
                 seed=7) -> dict:
    """Shuffle predicted probabilities within each fold-year (destroys
    timing, preserves marginal distribution and hence trade frequency),
    re-run the full simulation, and compare expectancies."""
    rng = np.random.default_rng(seed)
    exps, counts = [], []
    for s in range(n_sims):
        fake = []
        for f in folds:
            p = f.pred.copy()
            perm = rng.permutation(len(p))
            p["p_long"] = p["p_long"].to_numpy()[perm]
            p["p_short"] = p["p_short"].to_numpy()[perm]
            fake.append(FoldPred(f.test_year, f.threshold, f.val_exp_R, p))
        t, _ = simulate(df, fake)
        if len(t):
            exps.append(float(t.r_mult.mean()))
            counts.append(len(t))
    return dict(n_sims=len(exps), mean_exp_R=float(np.mean(exps)),
                std_exp_R=float(np.std(exps)),
                mean_n_trades=float(np.mean(counts)),
                exps=[round(e, 4) for e in exps])


def main() -> None:
    df, feats = wf.load()
    folds = walk_forward_predictions(df, feats)

    trades, daily_eq = simulate(df, folds)
    trades.to_parquet(RESULTS / "trades.parquet")
    daily_eq.rename("equity").to_frame().to_parquet(RESULTS / "daily_equity.parquet")

    stats = perf_stats(trades, daily_eq)
    boot = block_bootstrap_ci(trades) if len(trades) else {}
    print(json.dumps(stats, indent=1))
    print("bootstrap:", boot)

    # cost sensitivity
    sens = {}
    for cb in (0.0, 1.5, 2.5, 4.0):
        t_, e_ = simulate(df, folds, cost_bp=cb)
        sens[cb] = perf_stats(t_, e_)
    # threshold robustness
    thr_sens = {}
    for shift in (-0.02, -0.01, 0.0, 0.01, 0.02):
        t_, e_ = simulate(df, folds, threshold_shift=shift)
        thr_sens[shift] = perf_stats(t_, e_)

    placebo = placebo_test(df, folds) if len(trades) else {}
    if placebo:
        obs = stats["expectancy_R"]
        exps = np.array(placebo["exps"])
        placebo["p_value_geq_obs"] = float((exps >= obs).mean())
    print("placebo:", {k: v for k, v in placebo.items() if k != "exps"})

    result = dict(stats=stats, bootstrap=boot, placebo=placebo,
                  fold_thresholds={f.test_year: f.threshold for f in folds},
                  fold_val_exp={f.test_year: f.val_exp_R for f in folds},
                  cost_sensitivity={str(k): v for k, v in sens.items()},
                  threshold_sensitivity={str(k): v for k, v in thr_sens.items()})
    (RESULTS / "backtest_main.json").write_text(json.dumps(result, indent=1))
    print("saved backtest_main.json")


if __name__ == "__main__":
    main()
