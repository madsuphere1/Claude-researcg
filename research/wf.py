"""Shared walk-forward utilities.

Folds are calendar-year based: train on everything from 2010 up to the test
year (expanding window), purge PURGE_BARS bars before the test start so no
training label's forward window overlaps test data, then test on one year.
2009 is discarded as indicator warm-up. The final fold tests the partial
2026 year.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
PURGE_BARS = 96          # 1 trading day >> 16-bar label horizon
TEST_YEARS = list(range(2014, 2027))

LABEL_COLS = ["y_dir_1", "y_dir_5", "y_dir_10", "fwd_ret_1", "fwd_ret_5",
              "fwd_ret_10", "y_tp_long", "y_tp_short", "mfe_16", "mae_16",
              "time_to_hit"]
PRICE_COLS = ["open", "high", "low", "close"]


@dataclass
class Fold:
    test_year: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def load() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(HERE / "data" / "features_15m.parquet")
    df = df[df.index >= "2010-01-01"]
    feature_cols = [c for c in df.columns if c not in LABEL_COLS + PRICE_COLS]
    return df, feature_cols


def year_folds(index: pd.DatetimeIndex,
               test_years: list[int] | None = None) -> list[Fold]:
    years = index.year
    folds = []
    pos = np.arange(len(index))
    for ty in (test_years or TEST_YEARS):
        test_mask = years == ty
        if not test_mask.any():
            continue
        test_start = pos[test_mask][0]
        train_end = max(0, test_start - PURGE_BARS)
        folds.append(Fold(ty, pos[:train_end], pos[test_mask]))
    return folds
