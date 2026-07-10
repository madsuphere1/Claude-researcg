"""H-C1: London-open reversion of the Asia-session move.

Hypothesis (declared in REGISTRY.md): the drift accumulated during the
Asian session (19:00 EST -> London open) partially reverts after London
open; fading it is profitable net of costs.

Design
------
Event: each trading day at the first bar at/after 03:00 EST (~08:00 London,
DST handled via tz-conversion). Signal: sign of the Asia move
(19:00 open -> event bar open). Trade: SHORT if Asia move up, LONG if down,
i.e. fade; exit after H bars (H in {8, 16, 32} declared upfront; primary
H=16). Also measured: raw event-study path of forward returns conditioned
on Asia-move quintile.

Validation: effect must exist on dev window 2010-2013 (in-sample discovery)
AND hold on 2014-2026 (out-of-sample confirmation) with day-block bootstrap
CI. Costs: 2.5bp round trip. Position sizing irrelevant for expectancy in
return units; we report bp per trade and R-equivalents using ATR sizing.

This experiment uses no ML model - it is a pure structural hypothesis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
COST_BP = 2.5


def block_ci(x: pd.Series, groups: pd.Series, n_boot=5000, seed=7):
    rng = np.random.default_rng(seed)
    g = [v.to_numpy() for _, v in x.groupby(groups.values)]
    n = len(g)
    means = np.array([np.concatenate([g[j] for j in rng.integers(0, n, n)]).mean()
                      for _ in range(n_boot)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    bars = pd.read_parquet(HERE.parents[1] / "research" / "data" / "xauusd_15m.parquet")
    bars = bars[bars.index >= "2010-01-01"]
    utc = bars.utc.dt.tz_localize("UTC")
    ldn_hour = utc.dt.tz_convert("Europe/London")
    ldn_h = ldn_hour.dt.hour + ldn_hour.dt.minute / 60
    est_h = bars.index.hour + bars.index.minute / 60

    # trading day key (rolls 18:00 EST)
    tday = pd.Series((bars.index - pd.Timedelta(hours=18)).date, index=bars.index)

    # event bar: first bar of each trading day with London hour >= 8
    is_ldn_open = (ldn_h >= 8).to_numpy()
    ev = (pd.Series(is_ldn_open, index=bars.index)
          .groupby(tday.values).cumsum() == 1) & is_ldn_open
    ev &= (est_h >= 1) & (est_h <= 5)  # sanity: open must be in the night EST window

    # Asia move: day's first open (>=19:00 prior evening) to event bar open
    day_first_open = bars.open.groupby(tday.values).transform("first")
    asia_ret = np.log(bars.open / day_first_open)

    atr = (bars.high - bars.low).ewm(alpha=1 / 14, adjust=False).mean()

    events = bars.index[ev]
    E = pd.DataFrame(index=events)
    E["asia_ret"] = asia_ret.loc[events]
    E["atr_frac"] = (atr / bars.close).loc[events]
    pos = bars.index.get_indexer(events)

    close = bars.close.to_numpy()
    open_ = bars.open.to_numpy()
    for H in (8, 16, 32):
        ok = pos + H + 1 < len(bars)
        fwd = np.full(len(pos), np.nan)
        fwd[ok] = np.log(close[pos[ok] + H] / open_[pos[ok] + 1])
        E[f"fwd_{H}"] = fwd

    E = E.dropna()
    E["dev"] = E.index < "2014-01-01"
    E["fade_dir"] = -np.sign(E.asia_ret)

    out = {}
    for win, name in ((E[E.dev], "dev_2010_13"), (E[~E.dev], "oos_2014_26")):
        rows = {}
        for H in (8, 16, 32):
            pnl_bp = (win["fade_dir"] * win[f"fwd_{H}"]) * 1e4 - COST_BP
            lo, hi = block_ci(pnl_bp, pd.Series(win.index.date, index=win.index))
            rows[f"H{H}"] = dict(n=len(win), mean_bp=float(pnl_bp.mean()),
                                 ci_lo=lo, ci_hi=hi,
                                 mean_R=float((pnl_bp / (win.atr_frac * 1e4)).mean()))
        # magnitude-conditioned: top quintile |asia_ret| only (declared: the
        # effect should be strongest after large moves if inventory-driven)
        big = win[win.asia_ret.abs() >= win.asia_ret.abs().quantile(0.8)]
        pnl_bp = (big["fade_dir"] * big["fwd_16"]) * 1e4 - COST_BP
        lo, hi = block_ci(pnl_bp, pd.Series(big.index.date, index=big.index))
        rows["H16_bigmove"] = dict(n=len(big), mean_bp=float(pnl_bp.mean()),
                                   ci_lo=lo, ci_hi=hi,
                                   mean_R=float((pnl_bp / (big.atr_frac * 1e4)).mean()))
        out[name] = rows

    # continuation check (is it momentum instead?)
    for name, win in (("dev_2010_13", E[E.dev]), ("oos_2014_26", E[~E.dev])):
        cont = (np.sign(win.asia_ret) * win["fwd_16"]) * 1e4 - COST_BP
        out[name]["H16_follow"] = dict(n=len(win), mean_bp=float(cont.mean()))

    import json
    (RESULTS / "c1_session_reversion.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
