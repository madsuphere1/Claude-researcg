"""H-D1: automated symbolic feature discovery with multiplicity control.

Generate ~2000 random formulas over base series using a typed operator
grammar. Screen on the development window ONLY (2010-2013, split in half:
generate/confirm) - the 2014+ data is touched exactly once, for the final
top-20 validation, with Bonferroni-deflated significance.

Pipeline (declared before running):
1. 2000 seeded random formulas, depth <= 3.
2. Dev screen: |Spearman IC| vs fwd_ret_5 computed on 2010-11 and 2012-13
   separately; keep top 20 by min(|IC_a|, |IC_b|) with sign agreement.
3. OOS: yearly Spearman IC 2014-2026 (13 obs per candidate). Accept if
   t-test p < 0.05/20 AND mean |IC| >= 0.01.
4. Joint economic test: LightGBM (fold years 2018/2022/2026) with baseline
   features vs baseline + accepted candidates; report AUC delta.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parents[1]))
import wf  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RNG = np.random.default_rng(42)
N_FORMULAS = 2000
TOP_K = 20


# ---------------------------------------------------------------- grammar
def build_bases(bars: pd.DataFrame) -> dict[str, pd.Series]:
    c = bars.close
    r = np.log(c).diff()
    return {
        "r": r,
        "hl": (bars.high - bars.low) / c,
        "body": (bars.close - bars.open) / c,
        "upw": (bars.high - np.maximum(bars.open, bars.close)) / c,
        "dnw": (np.minimum(bars.open, bars.close) - bars.low) / c,
        "rv": np.sqrt(bars.rv),
        "imb": (bars.up_m1 - bars.down_m1) / (bars.up_m1 + bars.down_m1 + 1),
        "act": bars.absret,
    }


UNARY = ["lag", "diff", "ema", "std", "zs", "rank", "abs", "sign", "sum"]
BINARY = ["add", "sub", "mul", "div", "corr"]
WINDOWS = [4, 8, 16, 32, 96, 192]


def rand_expr(depth: int) -> dict:
    if depth == 0 or RNG.random() < 0.3:
        return {"op": "base", "name": list("12345678")[RNG.integers(0, 8)]}
    if RNG.random() < 0.65:
        return {"op": UNARY[RNG.integers(0, len(UNARY))],
                "w": int(WINDOWS[RNG.integers(0, len(WINDOWS))]),
                "a": rand_expr(depth - 1)}
    return {"op": BINARY[RNG.integers(0, len(BINARY))],
            "w": int(WINDOWS[RNG.integers(0, len(WINDOWS))]),
            "a": rand_expr(depth - 1), "b": rand_expr(depth - 1)}


def evaluate(node: dict, bases: dict[str, pd.Series]) -> pd.Series:
    op = node["op"]
    if op == "base":
        key = list(bases)[int(node["name"]) - 1]
        return bases[key]
    a = evaluate(node["a"], bases)
    if op == "lag":
        return a.shift(node["w"] // 4)
    if op == "diff":
        return a.diff(node["w"])
    if op == "ema":
        return a.ewm(span=node["w"], min_periods=node["w"]).mean()
    if op == "std":
        return a.rolling(node["w"]).std()
    if op == "zs":
        m, s = a.rolling(node["w"]).mean(), a.rolling(node["w"]).std()
        return (a - m) / (s + 1e-12)
    if op == "rank":
        return a.rolling(node["w"]).rank(pct=True)
    if op == "abs":
        return a.abs()
    if op == "sign":
        return np.sign(a)
    if op == "sum":
        return a.rolling(node["w"]).sum()
    b = evaluate(node["b"], bases)
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "div":
        return a / (b.abs() + 1e-12) * np.sign(b).replace(0, 1)
    if op == "corr":
        return a.rolling(node["w"]).corr(b)
    raise ValueError(op)


def expr_str(node: dict) -> str:
    if node["op"] == "base":
        return ["r", "hl", "body", "upw", "dnw", "rv", "imb", "act"][int(node["name"]) - 1]
    if "b" in node:
        return f"{node['op']}({expr_str(node['a'])},{expr_str(node['b'])},{node['w']})"
    if node["op"] in ("abs", "sign"):
        return f"{node['op']}({expr_str(node['a'])})"
    return f"{node['op']}({expr_str(node['a'])},{node['w']})"


def spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 500 or np.nanstd(x[m]) == 0:
        return np.nan
    return float(stats.spearmanr(x[m], y[m]).statistic)


def main() -> None:
    df, feats = wf.load()
    bars = pd.read_parquet(HERE.parents[1] / "research" / "data" / "xauusd_15m.parquet")
    bars = bars.loc[df.index]
    bases = build_bases(bars)
    y = df["fwd_ret_5"].to_numpy()
    idx = df.index
    m_a = (idx >= "2010-01-01") & (idx < "2012-01-01")
    m_b = (idx >= "2012-01-01") & (idx < "2014-01-01")

    print(f"generating {N_FORMULAS} formulas...", flush=True)
    cand = []
    for i in range(N_FORMULAS):
        node = rand_expr(3)
        try:
            v = evaluate(node, bases)
        except Exception:  # noqa: BLE001
            continue
        arr = v.to_numpy(dtype=np.float64)
        if not np.isfinite(arr[m_a | m_b]).any():
            continue
        ic_a = spearman_ic(arr[m_a], y[m_a])
        ic_b = spearman_ic(arr[m_b], y[m_b])
        if np.isnan(ic_a) or np.isnan(ic_b) or np.sign(ic_a) != np.sign(ic_b):
            continue
        cand.append(dict(i=i, expr=expr_str(node), node=node,
                         ic_a=ic_a, ic_b=ic_b,
                         score=min(abs(ic_a), abs(ic_b))))
        if len(cand) % 200 == 0:
            print(f"  {len(cand)} valid so far (formula {i})", flush=True)
    cand.sort(key=lambda d: -d["score"])
    top = cand[:TOP_K]
    print(f"valid candidates: {len(cand)}; screening top {TOP_K}", flush=True)

    oos_years = list(range(2014, 2027))
    results = []
    for cdt in top:
        v = evaluate(cdt["node"], bases).to_numpy(dtype=np.float64)
        ics = []
        for yy in oos_years:
            m = idx.year == yy
            ics.append(spearman_ic(v[m], y[m]))
        ics = np.array(ics, dtype=float)
        ok = np.isfinite(ics)
        t, p = stats.ttest_1samp(ics[ok], 0.0)
        results.append(dict(expr=cdt["expr"], dev_ic_a=cdt["ic_a"],
                            dev_ic_b=cdt["ic_b"],
                            oos_ic_mean=float(np.nanmean(ics)),
                            oos_ic_by_year={str(yy): (None if not np.isfinite(v_) else round(float(v_), 4))
                                            for yy, v_ in zip(oos_years, ics)},
                            t_stat=float(t), p_value=float(p),
                            accepted=bool(p < 0.05 / TOP_K
                                          and abs(np.nanmean(ics)) >= 0.01
                                          and np.sign(np.nanmean(ics)) == np.sign(cdt["ic_a"]))))
        print(f"{cdt['expr'][:60]:<60} dev={cdt['score']:+.3f} "
              f"oos={np.nanmean(ics):+.4f} p={p:.4f} "
              f"{'ACCEPT' if results[-1]['accepted'] else 'reject'}", flush=True)

    (RESULTS / "d1_feature_search.json").write_text(
        json.dumps({"n_generated": N_FORMULAS, "n_valid": len(cand),
                    "top": results}, indent=1))
    n_acc = sum(r["accepted"] for r in results)
    print(f"accepted after Bonferroni x{TOP_K}: {n_acc}")


if __name__ == "__main__":
    main()
