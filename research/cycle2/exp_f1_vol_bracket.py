"""H-F1: monetizing the volatility forecast with direction-neutral brackets.

Cycle-1's most predictable target was magnitude (MFE/MAE AUC~0.55), not
direction. A symmetric OCO stop-bracket is the simplest direction-neutral
expression: buy-stop above / sell-stop below; the breakout side becomes the
position. If expansions are predictable, entering brackets only before
predicted expansions should be profitable despite double-leg costs.

Declared spec (REGISTRY.md):
* Label for gating model: expansion = max(MFE16, MAE16) >= 2.0 (ATR units,
  from cycle-1 labels). LightGBM walk-forward, test years 2014,16,...,26.
* Gate threshold chosen on the last training year (validation) to yield
  <= 1 bracket/day and maximise validated bracket expectancy.
* Bracket: stops at signal close +/- 0.5*ATR, armed for 8 bars; triggered
  side gets TP +2.0*ATR / SL -1.0*ATR from trigger, 32-bar horizon,
  same-bar TP+SL tie -> SL; untriggered bracket cancels at 0 cost.
* Cost 2.5bp on the executed leg only (stop entry crosses the spread).
  One bracket at a time; no Friday-PM arming; flatten on weekend gaps.

Success: walk-forward net expectancy CI > 0 (day blocks) and >=0.05
brackets/day. Sanity control: same engine on ALL bars (no gate) and on
random gates at matched frequency.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
import backtest  # noqa: E402
import wf  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "results"
FOLD_YEARS = list(range(2014, 2027, 2))
ARM, TRIG, TP_A, SL_A, HOLD = 8, 0.5, 2.0, 1.0, 32
COST_BP = 2.5
MAX_PER_DAY = 5

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=500, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              verbose=-1, num_threads=4, seed=7)


def simulate_brackets(df, gate: pd.Series, a14: pd.Series) -> pd.DataFrame:
    sub = df
    o = sub.open.to_numpy(); h = sub.high.to_numpy()
    l = sub.low.to_numpy(); c = sub.close.to_numpy()
    idx = sub.index
    a = a14.to_numpy()
    gap_min = idx.to_series().diff().dt.total_seconds().to_numpy() / 60
    tday = (idx - pd.Timedelta(hours=18)).date
    fri_pm = (idx.dayofweek == 4) & (idx.hour >= 15)
    g = gate.reindex(idx).fillna(False).to_numpy()
    n = len(sub)

    trades = []
    day = None; day_n = 0
    arm = None   # dict(up, dn, atr, expires)
    pos = None

    def record(i, exit_px, reason):
        nonlocal pos
        gross = pos["side"] * (exit_px - pos["entry"]) / pos["entry"]
        net = gross - COST_BP * 1e-4
        r = net / (SL_A * pos["atr"] / pos["entry"])
        trades.append(dict(time=str(idx[i]), side=pos["side"], reason=reason,
                           r_mult=float(r), day=str(tday[i])))
        pos = None

    for i in range(n):
        if tday[i] != day:
            day, day_n = tday[i], 0
        if pos is not None:
            if gap_min[i] > 120:
                record(i, o[i], "gap")
            else:
                hit_sl = (l[i] <= pos["sl"]) if pos["side"] == 1 else (h[i] >= pos["sl"])
                hit_tp = (h[i] >= pos["tp"]) if pos["side"] == 1 else (l[i] <= pos["tp"])
                if hit_sl:
                    record(i, pos["sl"], "sl")
                elif hit_tp:
                    record(i, pos["tp"], "tp")
                elif i >= pos["expiry"]:
                    record(i, c[i], "timeout")
        if pos is None and arm is not None:
            if gap_min[i] > 120 or i > arm["expires"]:
                arm = None
            else:
                up_hit = h[i] >= arm["up"]; dn_hit = l[i] <= arm["dn"]
                if up_hit and dn_hit:
                    # both stops in one bar: whipsaw fill both ways ->
                    # pessimistic: enter first side by proximity to open,
                    # exit stopped at the other stop level in same bar
                    side = 1 if (arm["up"] - o[i]) < (o[i] - arm["dn"]) else -1
                    entry = arm["up"] if side == 1 else arm["dn"]
                    stop = arm["dn"] if side == 1 else arm["up"]
                    gross = side * (stop - entry) / entry
                    r = (gross - COST_BP * 1e-4) / (SL_A * arm["atr"] / entry)
                    trades.append(dict(time=str(idx[i]), side=side,
                                       reason="whipsaw", r_mult=float(r),
                                       day=str(tday[i])))
                    arm = None
                elif up_hit or dn_hit:
                    side = 1 if up_hit else -1
                    entry = arm["up"] if up_hit else arm["dn"]
                    pos = dict(side=side, entry=entry, atr=arm["atr"],
                               sl=entry - side * SL_A * arm["atr"],
                               tp=entry + side * TP_A * arm["atr"],
                               expiry=i + HOLD)
                    arm = None
        if pos is None and arm is None and g[i] and i + 1 < n \
                and gap_min[i + 1] <= 120 and not fri_pm[i] and day_n < MAX_PER_DAY \
                and np.isfinite(a[i]) and a[i] > 0:
            arm = dict(up=c[i] + TRIG * a[i], dn=c[i] - TRIG * a[i],
                       atr=a[i], expires=i + ARM)
            day_n += 1
    return pd.DataFrame(trades)


def summarize(tr: pd.DataFrame, n_days: float) -> dict:
    if not len(tr):
        return {"n": 0}
    r = tr.r_mult
    rng = np.random.default_rng(7)
    g = [v.to_numpy() for _, v in r.groupby(tr.day.values)]
    means = np.array([np.concatenate([g[j] for j in rng.integers(0, len(g), len(g))]).mean()
                      for _ in range(3000)])
    return dict(n=len(tr), per_day=float(len(tr) / n_days),
                exp_R=float(r.mean()),
                ci=[float(np.quantile(means, .025)), float(np.quantile(means, .975))],
                win=float((r > 0).mean()),
                reasons=tr.reason.value_counts().to_dict())


def main() -> None:
    df, feats = wf.load()
    exp_label = ((df.mfe_16.clip(lower=0).combine(df.mae_16.clip(lower=0), max))
                 >= 2.0).astype(int)
    exp_label[df.mfe_16.isna() | df.mae_16.isna()] = np.nan
    a14 = backtest.atr14(df)
    yrs = df.index.year

    gate = pd.Series(False, index=df.index)
    out = {"base_rate": float(exp_label.mean())}
    fold_meta = {}
    for ty in FOLD_YEARS:
        te = np.flatnonzero(yrs == ty)
        if not len(te):
            continue
        tr_pos = np.flatnonzero(yrs < ty)
        tr_pos = tr_pos[tr_pos < te[0] - wf.PURGE_BARS]
        val_mask = yrs[tr_pos] >= ty - 1
        fit, val = tr_pos[~val_mask], tr_pos[val_mask]
        lab = exp_label.to_numpy()
        fitm = fit[np.isfinite(lab[fit])]
        mdl = lgb.train(PARAMS, lgb.Dataset(df[feats].iloc[fitm],
                                            lab[fitm].astype(int)),
                        num_boost_round=300)
        pv = mdl.predict(df[feats].iloc[val])
        # threshold: on validation, run tiny bracket sim per candidate
        best_t, best_e = None, -np.inf
        vdays = max(1, len(np.unique(df.index[val].date)))
        for q in (0.90, 0.95, 0.98):
            t = float(np.quantile(pv, q))
            gv = pd.Series(False, index=df.index)
            gv.iloc[val] = pv >= t
            trv = simulate_brackets(df.iloc[val[0]:val[-1] + 1], gv, a14)
            if len(trv) < 20 or len(trv) / vdays > 1.0:
                continue
            e = trv.r_mult.mean()
            if e > best_e:
                best_t, best_e = t, float(e)
        if best_t is None:
            best_t = float(np.quantile(pv, 0.98)); best_e = np.nan
        pt = mdl.predict(df[feats].iloc[te])
        gate.iloc[te] = pt >= best_t
        fold_meta[ty] = dict(threshold=best_t, val_exp=best_e,
                             gate_frac=float((pt >= best_t).mean()))
        print(ty, fold_meta[ty], flush=True)

    test_mask = df.index.year.isin(FOLD_YEARS)
    dtest = df[test_mask]
    n_days = len(np.unique(dtest.index.date))
    trades = simulate_brackets(dtest, gate, a14)
    out["folds"] = fold_meta
    out["gated"] = summarize(trades, n_days)

    # controls: all-bars gate and random gate at matched frequency
    all_gate = pd.Series(test_mask, index=df.index)
    out["ungated_control"] = summarize(simulate_brackets(dtest, all_gate, a14), n_days)
    rng = np.random.default_rng(7)
    match = gate[test_mask].mean()
    rnd_exps = []
    for _ in range(30):
        rg = pd.Series(False, index=df.index)
        rg[test_mask] = rng.random(test_mask.sum()) < match
        t_ = simulate_brackets(dtest, rg, a14)
        if len(t_):
            rnd_exps.append(float(t_.r_mult.mean()))
    out["random_gate_exp_mean"] = float(np.mean(rnd_exps))
    out["random_gate_exp_sd"] = float(np.std(rnd_exps))

    (RESULTS / "f1_vol_bracket.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "folds"}, indent=1, default=float))


if __name__ == "__main__":
    main()
