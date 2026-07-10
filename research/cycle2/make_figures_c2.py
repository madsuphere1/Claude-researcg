"""Cycle-2 report figures."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
FIGS = HERE.parents[1] / "report" / "figures"
plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3})
BLUE, RED, GREEN, GREY = "#2A7DE1", "#D64545", "#0F9D58", "#6B7280"


def fig_levers():
    labels = ["Cycle-1 baseline\n(market, G0, ungated)",
              "G1 geometry", "G2 geometry\n(gap-managed)",
              "Vol-gate only", "Limit entry only\n(ITT)",
              "X1 composite\n(exploratory)"]
    vals = [-0.105, -0.055, -0.027, 0.053, -0.021, 0.128]
    los = [-0.150, -0.106, -0.090, -0.032, np.nan, -0.084]
    his = [-0.061, -0.006, 0.035, 0.136, np.nan, 0.258]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [RED if v < 0 else GREEN for v in vals]
    err = [[abs(v - lo) if np.isfinite(lo) else 0 for v, lo in zip(vals, los)],
           [abs(hi - v) if np.isfinite(hi) else 0 for v, hi in zip(vals, his)]]
    ax.bar(labels, vals, yerr=err, color=colors, capsize=3)
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel("Net expectancy (R/trade) at stated costs")
    ax.set_title("Cycle-2 levers: net expectancy at 2.5bp (X1 at 1.25bp limit-fill)\n"
                 "error bars: bootstrap 95% CI (X1: year-block)")
    plt.setp(ax.get_xticklabels(), fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIGS / "c2_levers.png")
    plt.close(fig)


def fig_x1():
    eq = pd.read_parquet(HERE / "results" / "x1_composite_equity.parquet")["equity"]
    eq.index = pd.to_datetime(eq.index)
    t = pd.read_parquet(HERE / "results" / "x1_composite_trades.parquet")
    fig, ax = plt.subplots(2, 1, figsize=(9, 6),
                           gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(eq.index, eq.values, lw=0.9, color=BLUE)
    ax[0].set_title("X1 composite pilot equity (even test years 2014-2026; "
                    "flat periods = odd years / gated-off regimes)")
    yr = t.groupby("year").r_mult.sum()
    ax[1].bar(yr.index.astype(str), yr.values,
              color=[GREEN if v > 0 else RED for v in yr.values])
    n = t.groupby("year").size()
    for i, (y, v) in enumerate(yr.items()):
        ax[1].text(i, v, f"n={n[y]}", ha="center",
                   va="bottom" if v > 0 else "top", fontsize=7)
    ax[1].set_title("Sum of R by test year - profits concentrate in 2016/2020 vol regimes")
    fig.tight_layout()
    fig.savefig(FIGS / "c2_x1_composite.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_levers()
    fig_x1()
    print("figures written")
