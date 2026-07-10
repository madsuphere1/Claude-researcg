"""H-A1: HMM regime discovery + confirmatory volatility-gate test.

Part 1 (discovery): per walk-forward fold, fit a 3-state Gaussian HMM on
[15m log-return, log realized-variance] using TRAINING data only. Measure
state persistence (dwell times), transition matrix, and state separation.

Part 2 (confirmation, declared ex-ante in REGISTRY.md): gate the cycle-1
signal to bars whose FILTERED (forward-only, no lookahead) probability of
the highest-vol state exceeds 0.6. Success: net expectancy at 2.5bp better
than baseline -0.105R AND >= 0.

Causality note: hmmlearn's decoders are smoothed (forward-backward), which
peeks at the future. Test-period state probabilities here are computed with
a manual forward filter using train-fitted parameters only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import multivariate_normal

sys.path.insert(0, str(Path(__file__).parents[1]))
import backtest  # noqa: E402
import wf  # noqa: E402
from backtest import FoldPred  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "results"
ART = HERE / "artifacts"
GATE_P = 0.6


def forward_filter(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """P(state_t | obs_1..t) - strictly causal."""
    logB = np.stack([multivariate_normal.logpdf(X, model.means_[k],
                                                np.diag(model.covars_[k].diagonal())
                                                if model.covariance_type == "diag"
                                                else model.covars_[k])
                     for k in range(model.n_components)], axis=1)
    A = model.transmat_
    alpha = np.zeros_like(logB)
    a = np.log(model.startprob_ + 1e-300) + logB[0]
    a -= a.max(); p = np.exp(a); p /= p.sum()
    alpha[0] = p
    for t in range(1, len(X)):
        pred = p @ A
        a = np.log(pred + 1e-300) + logB[t]
        a -= a.max(); p = np.exp(a); p /= p.sum()
        alpha[t] = p
    return alpha


def dwell_stats(states: np.ndarray) -> dict:
    runs = []
    cur, n = states[0], 1
    for s in states[1:]:
        if s == cur:
            n += 1
        else:
            runs.append((cur, n)); cur, n = s, 1
    runs.append((cur, n))
    df = pd.DataFrame(runs, columns=["state", "len"])
    return {int(k): float(v) for k, v in df.groupby("state")["len"].median().items()}


def main() -> None:
    df, feats = wf.load()
    bars = pd.read_parquet(HERE.parents[1] / "research" / "data" / "xauusd_15m.parquet")
    bars = bars.loc[df.index]
    r = np.log(bars.close).diff().fillna(0.0).to_numpy()
    lrv = np.log(bars.rv.replace(0, np.nan)).ffill().fillna(-20).to_numpy()
    X = np.column_stack([r * 1e3, lrv])

    pred = pd.read_parquet(ART / "wf_predictions.parquet")
    thr = json.loads((ART / "wf_thresholds.json").read_text())

    years = df.index.year
    gate = pd.Series(False, index=df.index)
    p_hi_all = pd.Series(np.nan, index=df.index)
    fold_info = {}
    for ty in wf.TEST_YEARS:
        te = np.flatnonzero(years == ty)
        if not len(te):
            continue
        tr = np.flatnonzero(years < ty)
        tr = tr[tr < te[0] - wf.PURGE_BARS]
        m = GaussianHMM(3, covariance_type="diag", n_iter=60, random_state=7)
        m.fit(X[tr])
        vol_order = np.argsort(m.means_[:, 1])          # by mean log-RV
        hi = int(vol_order[-1])
        # persistence on train (Viterbi ok in-sample - diagnostics only)
        st_tr = m.predict(X[tr])
        dw = dwell_stats(st_tr)
        occ = {int(k): float((st_tr == k).mean()) for k in range(3)}
        # causal filtered probs on [train tail + test] to warm the filter
        warm = 500
        seq = np.concatenate([tr[-warm:], te])
        alpha = forward_filter(m, X[seq])[warm:]
        p_hi = alpha[:, hi]
        gate.iloc[te] = p_hi > GATE_P
        p_hi_all.iloc[te] = p_hi
        fold_info[ty] = dict(
            hi_state=hi, dwell_median_bars=dw, occupancy=occ,
            transmat_diag=[float(m.transmat_[k, k]) for k in range(3)],
            mean_ret_1e3=[float(v) for v in m.means_[:, 0]],
            mean_logrv=[float(v) for v in m.means_[:, 1]],
            gate_frac=float((p_hi > GATE_P).mean()))
        print(f"{ty}: gate={fold_info[ty]['gate_frac']:.2%} "
              f"diag={np.round(fold_info[ty]['transmat_diag'],3)} "
              f"dwell={dw}", flush=True)

    # Part 2: gated simulation vs baseline, same engine, same thresholds
    folds, folds_gated = [], []
    for ty in wf.TEST_YEARS:
        mask = pred.index.year == ty
        if not mask.any() or str(ty) not in thr:
            continue
        p = pred[mask]
        folds.append(FoldPred(ty, thr[str(ty)]["threshold"], 0.0, p))
        pg = p.copy()
        g = gate.reindex(pg.index).fillna(False).to_numpy()
        pg.loc[~g, ["p_long", "p_short"]] = 0.0
        folds_gated.append(FoldPred(ty, thr[str(ty)]["threshold"], 0.0, pg))

    out = {"folds": fold_info, "gate_p": GATE_P}
    for name, fl in (("baseline", folds), ("vol_gated", folds_gated)):
        trades, eq = backtest.simulate(df, fl, cost_bp=2.5)
        s = backtest.perf_stats(trades, eq)
        out[name] = s
        if len(trades) > 30:
            out[name + "_bootstrap"] = backtest.block_bootstrap_ci(trades)
        print(name, {k: round(v, 4) for k, v in s.items()
                     if k in ("n_trades", "expectancy_R", "win_rate", "sharpe",
                              "max_dd", "final_equity")}, flush=True)
    # also gross (0bp) comparison
    for name, fl in (("baseline_0bp", folds), ("vol_gated_0bp", folds_gated)):
        trades, eq = backtest.simulate(df, fl, cost_bp=0.0)
        out[name] = backtest.perf_stats(trades, eq)

    (RESULTS / "a1_hmm_regimes.json").write_text(json.dumps(out, indent=1, default=float))
    print("saved a1_hmm_regimes.json")


if __name__ == "__main__":
    main()
