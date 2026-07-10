"""C5-001 step 2: the analysis. Reads the five trade streams and produces
(1) full-history metric suite per training window, (2) sliding-interval
metric grids, (3) the informativeness analysis — which metrics predict
forward performance, which are stable, which discriminate good policies
from dead ones. See registry C5-001 for the frozen plan.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from metrics import METRIC_KEYS, compute  # noqa: E402

REPO = HERE.parents[1]
ART = REPO / "strategy" / "artifacts"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

POLICIES = ["expanding", "rolling5", "rolling4", "rolling3", "rolling2"]
YEARS = list(range(2014, 2027))
SPANS = [2, 3, 5]
# metrics that are pure scale/count, excluded from informativeness ranking
NON_PREDICTORS = {"n_trades", "trades_per_year", "total_R", "best_R", "worst_R"}


def load_streams() -> dict[str, pd.DataFrame]:
    out = {}
    for p in POLICIES:
        f = ART / f"trades_{p}_allyears.parquet"
        if f.exists():
            t = t = pd.read_parquet(f)
            t["entry_time"] = pd.to_datetime(t["entry_time"])
            out[p] = t
    return out


def interval_metrics(tr: pd.DataFrame, a: int, b: int) -> dict:
    """Metrics over trades with year in [a, b] inclusive."""
    w = tr[(tr.year >= a) & (tr.year <= b)]
    return compute(w.r_mult.to_numpy(), years_span=(b - a + 1))


def spearman(x: pd.Series, y: pd.Series) -> float:
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 5 or d.iloc[:, 0].nunique() < 3 or d.iloc[:, 1].nunique() < 3:
        return float("nan")
    return float(d.corr(method="spearman").iloc[0, 1])


def main() -> None:
    streams = load_streams()
    print("streams:", {k: len(v) for k, v in streams.items()}, flush=True)

    # ---- (1) full-history metric suite per policy ----
    full = {p: compute(tr.r_mult.to_numpy(), years_span=13) for p, tr in streams.items()}
    (RESULTS / "c5_full_metrics.json").write_text(json.dumps(full, indent=1, default=float))

    print("\n" + "=" * 78)
    print("FULL-HISTORY METRIC SUITE (2014-2026) BY TRAINING WINDOW")
    print("=" * 78)
    hdr = f"{'metric':>17} " + " ".join(f"{p:>10}" for p in streams)
    print(hdr)
    for k in METRIC_KEYS:
        cells = []
        for p in streams:
            v = full[p].get(k)
            cells.append(f"{v:>10.3f}" if isinstance(v, (int, float)) and v is not None and np.isfinite(v) else f"{'—':>10}")
        print(f"{k:>17} " + " ".join(cells))

    # ---- (2) sliding-interval expectancy grids ----
    intervals = {}
    for p, tr in streams.items():
        intervals[p] = {}
        for s in SPANS:
            grid = {}
            for a in range(YEARS[0], YEARS[-1] - s + 2):
                b = a + s - 1
                grid[f"{a}-{b}"] = interval_metrics(tr, a, b)
            intervals[p][f"span{s}"] = grid
    (RESULTS / "c5_intervals.json").write_text(json.dumps(intervals, indent=1, default=float))

    for s in SPANS:
        print("\n" + "-" * 78)
        print(f"SLIDING {s}-YEAR INTERVAL — expectancy_R (n) by training window")
        print("-" * 78)
        labels = [f"{a}-{a+s-1}" for a in range(YEARS[0], YEARS[-1] - s + 2)]
        print(f"{'interval':>10} " + " ".join(f"{p:>12}" for p in streams))
        for lab in labels:
            cells = []
            for p in streams:
                m = intervals[p][f"span{s}"][lab]
                e, n = m.get("expectancy_R"), m.get("n_trades", 0)
                cells.append(f"{e:>+7.3f}({n:>3})" if e is not None and n else f"{'—':>12}")
            print(f"{lab:>10} " + " ".join(f"{c:>12}" for c in cells))

    # ---- (3) informativeness: does a trailing metric predict next-year P&L? ----
    # Build pooled (trailing-2y metric -> next-year expectancy) panel across policies.
    rows = []
    for p, tr in streams.items():
        for t in range(YEARS[0] + 1, YEARS[-1]):        # trailing [t-1,t], forward t+1
            trail = interval_metrics(tr, t - 1, t)
            fwd = tr[tr.year == t + 1].r_mult
            if trail.get("n_trades", 0) < 20 or len(fwd) < 10:
                continue
            row = {k: trail.get(k) for k in METRIC_KEYS}
            row["_fwd_exp"] = float(fwd.mean())
            row["_policy"] = p
            rows.append(row)
    panel = pd.DataFrame(rows)

    # persistence: lag-1 autocorr of each metric across a policy's span-2 grid, averaged
    def persistence(metric: str) -> float:
        acs = []
        for p in streams:
            series = pd.Series({lab: intervals[p]["span2"][lab].get(metric)
                                for lab in intervals[p]["span2"]})
            series = pd.to_numeric(series, errors="coerce").dropna()
            if len(series) >= 5 and series.nunique() >= 3:
                acs.append(series.autocorr(lag=1))
        return float(np.nanmean(acs)) if acs else float("nan")

    # discrimination: cross-policy Spearman(full-history metric, full-history expectancy)
    pol_exp = pd.Series({p: full[p]["expectancy_R"] for p in streams})

    def discrimination(metric: str) -> float:
        mv = pd.Series({p: full[p].get(metric) for p in streams})
        return spearman(mv, pol_exp)

    info = {}
    for k in METRIC_KEYS:
        if k in NON_PREDICTORS:
            continue
        fwd_rho = spearman(panel[k], panel["_fwd_exp"]) if k in panel else float("nan")
        per = persistence(k)
        dis = discrimination(k)
        cls = ("HELPS" if abs(fwd_rho) > 0.30 and abs(per) > 0.30
               else "DECORATIVE" if (np.isnan(fwd_rho) or abs(fwd_rho) < 0.10)
               else "WEAK")
        info[k] = dict(forward_rho=round(fwd_rho, 3) if np.isfinite(fwd_rho) else None,
                       persistence=round(per, 3) if np.isfinite(per) else None,
                       discrimination=round(dis, 3) if np.isfinite(dis) else None,
                       classification=cls)
    (RESULTS / "c5_informativeness.json").write_text(json.dumps(info, indent=1, default=float))

    print("\n" + "=" * 78)
    print(f"METRIC INFORMATIVENESS  (panel: {len(panel)} trailing->forward pairs pooled)")
    print("forward_rho = predicts next-year expectancy | persistence = lag-1 autocorr")
    print("discrimination = separates profitable vs dead policy")
    print("=" * 78)
    print(f"{'metric':>17} {'fwd_rho':>8} {'persist':>8} {'discrim':>8}  class")
    order = sorted(info, key=lambda k: -(abs(info[k]["forward_rho"]) if info[k]["forward_rho"] else 0))
    for k in order:
        r = info[k]
        fr = f"{r['forward_rho']:>+8.3f}" if r["forward_rho"] is not None else f"{'—':>8}"
        pe = f"{r['persistence']:>+8.3f}" if r["persistence"] is not None else f"{'—':>8}"
        di = f"{r['discrimination']:>+8.3f}" if r["discrimination"] is not None else f"{'—':>8}"
        print(f"{k:>17} {fr} {pe} {di}  {r['classification']}")

    helps = [k for k in info if info[k]["classification"] == "HELPS"]
    deco = [k for k in info if info[k]["classification"] == "DECORATIVE"]
    print(f"\nHELPS ({len(helps)}): {helps}")
    print(f"DECORATIVE ({len(deco)}): {deco}")
    print(f"\nsaved -> {RESULTS}")


if __name__ == "__main__":
    main()
