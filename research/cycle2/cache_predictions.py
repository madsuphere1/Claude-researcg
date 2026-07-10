"""Recompute and cache cycle-1 walk-forward predictions + thresholds."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
import wf                      # noqa: E402
from backtest import walk_forward_predictions  # noqa: E402

OUT = Path(__file__).parent / "artifacts"
OUT.mkdir(exist_ok=True)

df, feats = wf.load()
folds = walk_forward_predictions(df, feats)
pred = pd.concat([f.pred for f in folds])
pred.to_parquet(OUT / "wf_predictions.parquet")
meta = {str(f.test_year): {"threshold": f.threshold, "val_exp_R": f.val_exp_R}
        for f in folds}
(OUT / "wf_thresholds.json").write_text(json.dumps(meta, indent=1))
print("cached", pred.shape)
