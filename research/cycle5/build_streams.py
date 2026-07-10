"""C5-001 step 1: ensure a trade stream exists for every training window
in the ladder {expanding, 5, 4, 3, 2}. Reuses the engine, rolling
predictor, and simulator already built for the window comparison. Only
rolling-4 and rolling-3 need retraining; the rest are cached.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO / "strategy"))
sys.path.insert(0, str(REPO / "research"))
sys.path.insert(0, str(REPO / "research" / "cycle2"))
sys.path.insert(0, str(REPO / "research" / "cycle3"))

import wf  # noqa: E402
from xauusd_x1_final import ART, build_engine, simulate  # noqa: E402
from window_comparison import get_rolling  # noqa: E402

LADDER = [5, 4, 3, 2]


def main() -> None:
    print("building engine...", flush=True)
    d2, bars, a14s, pred_exp, thr_exp, gate, m1arrs = build_engine()
    _, feats = wf.load()

    exp_stream = ART / "trades_expanding_allyears.parquet"
    if not exp_stream.exists():
        tr, _ = simulate(d2, pred_exp, thr_exp, gate, a14s, m1arrs)
        tr.to_parquet(exp_stream)
        print(f"expanding: {len(tr)} trades", flush=True)
    else:
        print("expanding: cached", flush=True)

    for W in LADDER:
        out = ART / f"trades_rolling{W}_allyears.parquet"
        if out.exists():
            print(f"rolling-{W}: cached", flush=True)
            continue
        print(f"rolling-{W}: generating predictions + simulating...", flush=True)
        pred, thr = get_rolling(d2, feats, W)
        tr, _ = simulate(d2, pred, thr, gate, a14s, m1arrs)
        tr.to_parquet(out)
        print(f"rolling-{W}: {len(tr)} trades  exp={tr.r_mult.mean():+.4f}R", flush=True)
    print("STREAMS_DONE", flush=True)


if __name__ == "__main__":
    main()
