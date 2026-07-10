"""Frozen 28-metric quant suite for C5-001. One function: given a trade
R-multiple series (and optional per-trade equity for path metrics),
return the full metric dict. Deterministic, no randomness except the
declared bootstrap helper.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

TRADES_PER_YEAR_DENOM = None   # set by caller from actual span if desired


def _streaks(win: np.ndarray) -> tuple[int, int]:
    w = l = 0
    for k, g in itertools.groupby(win):
        n = len(list(g))
        if k:
            w = max(w, n)
        else:
            l = max(l, n)
    return w, l


def _max_dd_R(r: np.ndarray) -> float:
    """Max drawdown of the cumulative-R equity curve, in R units."""
    cum = np.cumsum(r)
    peak = np.maximum.accumulate(cum)
    return float((cum - peak).min())     # <= 0


def _ulcer(r: np.ndarray) -> float:
    cum = np.cumsum(r)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(np.sqrt(np.mean(dd ** 2)))


def compute(r: pd.Series | np.ndarray, years_span: float | None = None) -> dict:
    r = np.asarray(r, float)
    n = len(r)
    if n == 0:
        return {"n_trades": 0}
    wins, losses = r[r > 0], r[r <= 0]
    win = (r > 0).astype(int)
    std = float(r.std(ddof=1)) if n > 1 else 0.0
    downside = r[r < 0]
    dd_dev = float(np.sqrt(np.mean(np.minimum(r, 0) ** 2)))
    mdd = _max_dd_R(r)
    total = float(r.sum())
    mcw, mcl = _streaks(win)
    p5, p95 = np.percentile(r, 5), np.percentile(r, 95)
    cvar = float(r[r <= p5].mean()) if (r <= p5).any() else float(p5)
    gains, absloss = wins.sum(), -losses.sum()
    tpr = (n / years_span) if years_span else float("nan")
    return {
        "n_trades": int(n),
        "trades_per_year": round(tpr, 2) if years_span else None,
        "expectancy_R": round(float(r.mean()), 4),
        "median_R": round(float(np.median(r)), 4),
        "total_R": round(total, 3),
        "win_rate": round(float(win.mean()), 4),
        "avg_win_R": round(float(wins.mean()), 4) if len(wins) else 0.0,
        "avg_loss_R": round(float(losses.mean()), 4) if len(losses) else 0.0,
        "payoff_ratio": round(float(wins.mean() / -losses.mean()), 3) if len(wins) and len(losses) and losses.mean() != 0 else float("nan"),
        "profit_factor": round(float(gains / absloss), 3) if absloss > 0 else float("inf"),
        "std_R": round(std, 4),
        "sharpe": round(float(r.mean() / std), 3) if std > 0 else float("nan"),
        "sortino": round(float(r.mean() / dd_dev), 3) if dd_dev > 0 else float("nan"),
        "downside_dev": round(dd_dev, 4),
        "max_dd_R": round(mdd, 3),
        "calmar": round(float(total / -mdd), 3) if mdd < 0 else float("nan"),
        "recovery_factor": round(float(total / -mdd), 3) if mdd < 0 else float("nan"),
        "ulcer_index": round(_ulcer(r), 4),
        "skew": round(float(pd.Series(r).skew()), 3) if n > 2 else float("nan"),
        "kurtosis": round(float(pd.Series(r).kurt()), 3) if n > 3 else float("nan"),
        "var95": round(float(p5), 4),
        "cvar95": round(cvar, 4),
        "tail_ratio": round(float(p95 / -p5), 3) if p5 < 0 else float("nan"),
        "omega": round(float(gains / absloss), 3) if absloss > 0 else float("inf"),
        "max_consec_wins": mcw,
        "max_consec_losses": mcl,
        "best_R": round(float(r.max()), 3),
        "worst_R": round(float(r.min()), 3),
    }


METRIC_KEYS = [
    "n_trades", "trades_per_year", "expectancy_R", "median_R", "total_R",
    "win_rate", "avg_win_R", "avg_loss_R", "payoff_ratio", "profit_factor",
    "std_R", "sharpe", "sortino", "downside_dev", "max_dd_R", "calmar",
    "recovery_factor", "ulcer_index", "skew", "kurtosis", "var95", "cvar95",
    "tail_ratio", "omega", "max_consec_wins", "max_consec_losses", "best_R",
    "worst_R",
]
