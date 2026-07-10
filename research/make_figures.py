"""Generate report figures from results/ artifacts into report/figures/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULTS = HERE / "results"
FIGS = HERE.parent / "report" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})
C = dict(pos="#2A7DE1", neg="#D64545", neut="#6B7280", acc="#0F9D58")


def fig_price_overview():
    bars = pd.read_parquet(HERE / "data" / "xauusd_15m.parquet")
    daily = bars.close.resample("1D").last().dropna()
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(daily.index, daily.values, lw=0.7, color=C["pos"])
    ax[0].set_yscale("log")
    ax[0].set_title("XAUUSD close (daily from 15m bars, log scale)")
    rv = np.log(daily).diff().rolling(21).std() * np.sqrt(252) * 100
    ax[1].plot(rv.index, rv.values, lw=0.7, color=C["neg"])
    ax[1].set_title("21-day realized volatility (annualised, %)")
    fig.tight_layout()
    fig.savefig(FIGS / "price_overview.png")
    plt.close(fig)


def fig_feature_importance():
    gain = pd.read_csv(RESULTS / "lgbm_gain_rank.csv", index_col=0)
    perm = pd.read_csv(RESULTS / "perm_importance.csv", index_col=0)
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    top = gain.head(25).iloc[::-1]
    ax[0].barh(top.index, top.mean_gain_share, color=C["pos"])
    ax[0].set_title("LightGBM mean gain share (walk-forward)")
    topp = perm.head(25).iloc[::-1]
    ax[1].barh(topp.index, topp.mean_auc_drop, color=C["acc"])
    ax[1].set_title("Permutation importance (AUC drop, 2024-26 folds)")
    fig.tight_layout()
    fig.savefig(FIGS / "feature_importance.png")
    plt.close(fig)


def fig_wf_auc():
    stats = json.loads((RESULTS / "wf_auc_y_tp_long.json").read_text())
    years = [s["test_year"] for s in stats]
    aucs = [s["auc"] for s in stats]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(years, aucs, color=[C["pos"] if a > 0.5 else C["neg"] for a in aucs])
    ax.axhline(0.5, color="k", lw=1)
    ax.set_ylim(0.45, max(aucs) + 0.02)
    ax.set_title("Walk-forward AUC by test year (y_tp_long, LightGBM)")
    fig.tight_layout()
    fig.savefig(FIGS / "wf_auc.png")
    plt.close(fig)


def fig_model_comparison():
    res = pd.read_csv(RESULTS / "model_comparison.csv")
    summary = res.groupby("model")["auc"].agg(["mean", "std"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(summary.index, summary["mean"] - 0.5, left=0.5,
            xerr=summary["std"], color=C["pos"])
    ax.axvline(0.5, color="k", lw=1)
    ax.set_title("Mean walk-forward AUC by model (error bars: std across folds)")
    ax.set_xlabel("AUC")
    fig.tight_layout()
    fig.savefig(FIGS / "model_comparison.png")
    plt.close(fig)


def fig_equity():
    eq = pd.read_parquet(RESULTS / "daily_equity.parquet")["equity"]
    eq.index = pd.to_datetime(eq.index)
    trades = pd.read_parquet(RESULTS / "trades.parquet")
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(eq.index, eq.values, lw=0.9, color=C["pos"])
    ax[0].set_yscale("log")
    ax[0].set_title("Equity curve (1% risk per trade, 2.5bp round-trip costs)")
    dd = eq / eq.cummax() - 1
    ax[1].fill_between(dd.index, dd.values * 100, 0, color=C["neg"], alpha=0.6)
    ax[1].set_title("Drawdown (%)")
    fig.tight_layout()
    fig.savefig(FIGS / "equity_curve.png")
    plt.close(fig)

    # trade R distribution
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].hist(trades.r_mult, bins=60, color=C["pos"])
    ax[0].axvline(0, color="k", lw=1)
    ax[0].set_title(f"Trade R distribution (n={len(trades):,})")
    monthly = trades.set_index("entry_time").r_mult.resample("ME").sum()
    ax[1].bar(monthly.index, monthly.values, width=20,
              color=[C["pos"] if v > 0 else C["neg"] for v in monthly.values])
    ax[1].set_title("Monthly sum of R")
    fig.tight_layout()
    fig.savefig(FIGS / "trade_distribution.png")
    plt.close(fig)


def fig_conditional():
    """Session / regime breakdown of realised trade expectancy."""
    trades = pd.read_parquet(RESULTS / "trades.parquet")
    feats = pd.read_parquet(HERE / "data" / "features_15m.parquet",
                            columns=["adx14", "atr14_rank_1y", "sess_london",
                                     "sess_ny", "sess_tokyo", "is_fomc_day",
                                     "is_nfp_day", "hour_est"])
    t = trades.copy()
    t["sig_time"] = t.entry_time - pd.Timedelta(minutes=15)
    t = t.join(feats, on="sig_time")
    groups = {
        "London": t[t.sess_london == 1], "NY": t[t.sess_ny == 1],
        "Tokyo": t[t.sess_tokyo == 1],
        "Trend (ADX>25)": t[t.adx14 > 25], "Range (ADX<20)": t[t.adx14 < 20],
        "HighVol (top25%)": t[t.atr14_rank_1y > 0.75],
        "LowVol (bottom25%)": t[t.atr14_rank_1y < 0.25],
        "FOMC day": t[t.is_fomc_day == 1], "NFP day": t[t.is_nfp_day == 1],
        "All": t,
    }
    names, means, errs, ns = [], [], [], []
    for k, g in groups.items():
        if len(g) < 10:
            continue
        names.append(f"{k} (n={len(g)})")
        means.append(g.r_mult.mean())
        errs.append(1.96 * g.r_mult.std() / np.sqrt(len(g)))
        ns.append(len(g))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = [C["pos"] if m > 0 else C["neg"] for m in means]
    ax.barh(names, means, xerr=errs, color=colors)
    ax.axvline(0, color="k", lw=1)
    ax.set_title("Expectancy (R) by condition, 95% CI")
    fig.tight_layout()
    fig.savefig(FIGS / "conditional_expectancy.png")
    plt.close(fig)


def main() -> None:
    fig_price_overview()
    for fn in (fig_feature_importance, fig_wf_auc, fig_model_comparison,
               fig_equity, fig_conditional):
        try:
            fn()
            print("ok", fn.__name__)
        except FileNotFoundError as exc:
            print("skip", fn.__name__, exc)


if __name__ == "__main__":
    main()
