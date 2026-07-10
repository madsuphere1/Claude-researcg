"""C3-004 (schema v2): exact M1 replay of composite trades.

Prerequisite: composite trade parquets regenerated with entry/tp/sl
columns (deterministic re-runs of the pilot and C3-001 simulations after
the schema fix; the anchor invariants - trade counts and expectancies -
must match the recorded values before this replay is meaningful, and are
asserted below).

Question (frozen from C3-004): does the 15m pessimistic same-bar
tie-break materially bias reported expectancy? MATERIAL if the corrected
expectancy differs by >= 0.02R on either sample; else IMMATERIAL.
Replay remains pessimistic at the minute level (same-minute double-touch
counts as SL), so the correction measured here is a lower bound on the
true improvement and cannot be upward-biased.
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

ANCHORS = {"pilot_even": (414, 0.1279), "confirm_odd": (334, 0.0788)}
TP_R_OVER_SL = 1.5   # payoff ratio: tp distance = 1.5 x sl distance


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
        n_exp, e_exp = ANCHORS[name]
        assert len(t) == n_exp and abs(t.r_mult.mean() - e_exp) < 0.005, \
            f"{name}: anchor mismatch n={len(t)} exp={t.r_mult.mean():.4f}"
        if "entry" not in t.columns:
            raise SystemExit(f"{name}: schema v2 columns missing - regenerate first")

        flips = 0
        corrected = t.r_mult.to_numpy().copy()
        sl_exits = t[t.reason == "sl"]
        for j, row in sl_exits.iterrows():
            i0 = np.searchsorted(mi, np.datetime64(row.entry_time))
            i1 = np.searchsorted(mi, np.datetime64(row.exit_time)
                                 + np.timedelta64(15, "m"))
            res = None
            for k in range(i0, min(i1, len(mi))):
                hit_tp = (mh[k] >= row.tp) if row.side == 1 else (ml[k] <= row.tp)
                hit_sl = (ml[k] <= row.sl) if row.side == 1 else (mh[k] >= row.sl)
                if hit_tp and hit_sl:
                    res = "sl"; break
                if hit_tp:
                    res = "tp"; break
                if hit_sl:
                    res = "sl"; break
            if res == "tp":
                flips += 1
                # recorded r for this SL exit -> what the TP exit would be:
                # r_sl = (gross_sl - cost)*lev/RISK ; flip to gross_tp
                sl_frac = abs(row.entry - row.sl) / row.entry
                cost = 1.25e-4
                gross_sl = -sl_frac
                gross_tp = TP_R_OVER_SL * sl_frac
                r_sl = (gross_sl - cost) * row.lev / 0.01
                r_tp = (gross_tp - cost) * row.lev / 0.01
                corrected[t.index.get_loc(j)] = r_tp
                assert abs(r_sl - row.r_mult) < 0.05, \
                    f"reconstruction mismatch {r_sl} vs {row.r_mult}"
        e0 = float(t.r_mult.mean())
        e1 = float(np.mean(corrected))
        out[name] = dict(n_trades=len(t), n_sl_exits=int(len(sl_exits)),
                         n_flipped_to_tp=int(flips),
                         expectancy_recorded=e0, expectancy_corrected=e1,
                         correction_R=e1 - e0)
        print(name, out[name], flush=True)

    worst = max(abs(v["correction_R"]) for v in out.values())
    out["verdict"] = "MATERIAL" if worst >= 0.02 else "IMMATERIAL"
    (RESULTS / "c3_004_m1_replay.json").write_text(json.dumps(out, indent=1))
    print("verdict:", out["verdict"])


if __name__ == "__main__":
    main()
