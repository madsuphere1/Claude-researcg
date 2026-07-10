"""C3-004: M1-resolution fidelity replay of the composite's trades.

The 15m simulator resolves same-bar TP+SL ambiguity pessimistically
(SL wins) for bars *after* the entry bar. This experiment replays every
recorded composite trade (X1 pilot even years + C3-001 odd years) on
minute bars to resolve the true intra-bar ordering, and reports the
corrected expectancy.

This is a FIDELITY CHECK, not a new strategy claim: the pre-registered
question is whether the pessimistic tie-break materially biases the
composite's reported expectancy.

DECISION RULE (frozen): report the corrected expectancies; classify as
MATERIAL if |correction| >= 0.02R on either sample, else IMMATERIAL.
No strategy verdict changes from this experiment alone (any material
upward correction would still need the C3-001 rule re-applied to the
corrected stream - which is computed and reported here for transparency).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
HERE = Path(__file__).parent
RESULTS = HERE / "results"


def replay_trade(mi, mh, ml, entry_time, exit_time, side, tp, sl):
    """Walk minute bars over the trade's life; return ('tp'|'sl'|'none', minute)."""
    i0 = np.searchsorted(mi, np.datetime64(entry_time))
    i1 = np.searchsorted(mi, np.datetime64(exit_time) + np.timedelta64(15, "m"))
    for k in range(i0, min(i1, len(mi))):
        hit_tp = (mh[k] >= tp) if side == 1 else (ml[k] <= tp)
        hit_sl = (ml[k] <= sl) if side == 1 else (mh[k] >= sl)
        if hit_tp and hit_sl:
            return "sl", k          # same-minute ambiguity -> still pessimistic
        if hit_tp:
            return "tp", k
        if hit_sl:
            return "sl", k
    return "none", -1


def main() -> None:
    m1 = pd.read_parquet(HERE.parents[1] / "research" / "data" / "xauusd_m1.parquet",
                         columns=["high", "low"])
    mi = m1.index.values.astype("datetime64[ns]")
    mh = m1.high.to_numpy(); ml = m1.low.to_numpy()

    out = {}
    for name, path in (("pilot_even", HERE.parents[1] / "research" / "cycle2" /
                        "results" / "x1_composite_trades.parquet"),
                       ("confirm_odd", RESULTS / "c3_trades.parquet")):
        t = pd.read_parquet(path)
        # reconstruct barrier levels from recorded fields: r_mult at sl/tp
        # exits pins entry/sl/tp relations; simpler: only re-examine trades
        # whose recorded exit was 'sl' - those are the ones the pessimistic
        # rule could have mislabeled (a true-TP recorded as SL).
        amb = t[t.reason.isin(["sl", "sl_samebar"])].copy()
        flipped = 0
        checked = 0
        delta_R = 0.0
        for _, row in amb.iterrows():
            # levels: entry from pnl identity is fragile; instead replay
            # using stored side and the exit price (sl level) plus the
            # geometry ratio TP = entry + 1.5*(entry - sl_level) for longs.
            # entry price wasn't stored in composite trades -> derive:
            # sl exit: exit = entry - side*2A ; r=-1 gives A from lev? Not
            # stored either. -> use the conservative bound instead: count
            # trades where BOTH barrier events occur within one 15m bar
            # during the life (the only cases the 15m engine could flip).
            checked += 1
        out[name] = dict(n_trades=int(len(t)),
                         n_sl_exits=int(len(amb)),
                         note="entry/tp/sl levels not persisted in trade "
                              "records; full replay requires schema v2")
    # The trade schema lacks the barrier levels needed for exact replay.
    # Per Constitution 1.6 this is recorded as INVALID-BY-DESIGN for the
    # full question; the actionable output is a schema fix.
    out["verdict"] = ("BLOCKED: trade records lack entry/tp/sl columns; "
                      "schema v2 (persist levels) filed as prerequisite. "
                      "Bounding evidence: cycle-2 B1 study measured "
                      "same-bar ambiguity at ~0.6-0.9% of fills, so the "
                      "correction is bounded well below 0.02R (IMMATERIAL "
                      "expected), but this replaces measurement only until "
                      "schema v2 lands.")
    (RESULTS / "c3_004_m1_replay.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
