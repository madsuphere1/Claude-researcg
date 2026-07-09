"""Model comparison under identical walk-forward conditions.

Design
------
* Folds: test years 2018, 2020, 2022, 2024, 2026 (every second year keeps
  CPU budget sane while spanning regimes: pre-covid range, covid shock,
  2022 rate-hike bear, 2024-26 bull).
* Train window: last 5 years before the test year (identical rows for every
  model), purged by 96 bars.
* Target: y_tp_long (P(TP 1.5R before SL 1R within 16 bars)); timeout rows
  dropped for train and classification metrics.
* Tabular models see all engineered features (median-imputed, scaled where
  the model needs it). Sequence models see raw per-bar dynamics of the last
  32 bars for 18 dynamic channels - a fair test of whether sequence learning
  extracts more than hand engineering.
* Metrics: AUC, log-loss, Brier, plus an economic proxy: mean R-payoff of
  the top-5% highest-probability bars (+1.5R win / -1R loss).

TFT / N-BEATS / TabNet are represented by the Transformer / TCN / MLP
respectively; full versions are impractical on this 4-core CPU box and are
listed as future work.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import wf

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
torch.manual_seed(7)
np.random.seed(7)

COMPARE_YEARS = [2018, 2020, 2022, 2024, 2026]
TRAIN_YEARS = 5
SEQ_LEN = 32
SEQ_FEATURES = ["ret_1", "body_atr", "close_pos", "range_atr", "upwick_frac",
                "dnwick_frac", "m1_imb", "absret_z", "rv_1", "vol_expansion",
                "bb_pos", "rsi14", "dist_vwap_day", "hour_sin", "hour_cos",
                "adx14", "atr_ratio", "efficiency_1"]
LABEL = "y_tp_long"
WIN_R, LOSS_R = 1.5, -1.0


# --------------------------------------------------------------------------
# torch models
# --------------------------------------------------------------------------

class RNNClf(nn.Module):
    def __init__(self, n_in: int, kind: str = "lstm", hidden: int = 48):
        super().__init__()
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.rnn = rnn(n_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


class TCNClf(nn.Module):
    """Small causal temporal CNN."""

    def __init__(self, n_in: int, ch: int = 48):
        super().__init__()
        layers = []
        c = n_in
        for d in (1, 2, 4, 8):
            layers += [nn.Conv1d(c, ch, 3, padding=2 * d, dilation=d),
                       nn.ReLU()]
            c = ch
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(ch, 1)

    def forward(self, x):                      # x: (B, T, F)
        h = self.net(x.transpose(1, 2))        # causal: crop right side
        return self.head(h[:, :, x.shape[1] - 1]).squeeze(-1)


class TransformerClf(nn.Module):
    def __init__(self, n_in: int, d: int = 48, heads: int = 4, layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(n_in, d)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True,
                                         dropout=0.1)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        h = self.enc(self.proj(x) + self.pos)
        return self.head(h[:, -1]).squeeze(-1)


def train_torch(model: nn.Module, Xtr, ytr, Xte, epochs=3, bs=1024, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    n = len(Xtr_t)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
    model.eval()
    preds = []
    with torch.no_grad():
        Xte_t = torch.tensor(Xte, dtype=torch.float32)
        for i in range(0, len(Xte_t), 4096):
            preds.append(torch.sigmoid(model(Xte_t[i:i + 4096])).numpy())
    return np.concatenate(preds)


def build_sequences(df: pd.DataFrame, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, SEQ_LEN, F) windows ending at each masked row."""
    arr = df[SEQ_FEATURES].to_numpy(dtype=np.float32)
    med = np.nanmedian(arr, axis=0)
    arr = np.where(np.isfinite(arr), arr, med)
    sd = arr.std(axis=0) + 1e-9
    arr = np.clip((arr - arr.mean(axis=0)) / sd, -8, 8)
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(arr, SEQ_LEN, axis=0)  # (n-L+1, F, L)
    win = win.transpose(0, 2, 1)
    pos = np.flatnonzero(mask)
    pos = pos[pos >= SEQ_LEN - 1]
    return win[pos - SEQ_LEN + 1], pos


# --------------------------------------------------------------------------

def metrics(y: np.ndarray, p: np.ndarray, payoff_mask_frac=0.05) -> dict:
    out = {
        "auc": float(roc_auc_score(y, p)),
        "logloss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
        "base_rate": float(y.mean()),
    }
    k = max(1, int(len(p) * payoff_mask_frac))
    top = np.argsort(p)[-k:]
    wins = y[top]
    out["top5_exp_R"] = float(wins.mean() * WIN_R + (1 - wins.mean()) * LOSS_R)
    out["top5_n"] = int(k)
    return out


def run() -> None:
    df, feats = wf.load()
    rows = []
    for ty in COMPARE_YEARS:
        yr = df.index.year
        test_mask = (yr == ty)
        tr_lo, tr_hi = ty - TRAIN_YEARS, ty
        train_mask = (yr >= tr_lo) & (yr < tr_hi)
        test_start = np.flatnonzero(test_mask)[0]
        train_pos = np.flatnonzero(train_mask)
        train_pos = train_pos[train_pos < test_start - wf.PURGE_BARS]
        test_pos = np.flatnonzero(test_mask)

        lab_ok = df[LABEL].notna().to_numpy()
        tr_pos = train_pos[lab_ok[train_pos]]
        te_pos = test_pos[lab_ok[test_pos]]
        ytr = df[LABEL].to_numpy()[tr_pos].astype(int)
        yte = df[LABEL].to_numpy()[te_pos].astype(int)

        Xtr_raw = df[feats].iloc[tr_pos]
        Xte_raw = df[feats].iloc[te_pos]
        med = Xtr_raw.median()
        Xtr_i = Xtr_raw.fillna(med)
        Xte_i = Xte_raw.fillna(med)
        scaler = StandardScaler().fit(Xtr_i)
        Xtr_s = np.clip(scaler.transform(Xtr_i), -8, 8)
        Xte_s = np.clip(scaler.transform(Xte_i), -8, 8)

        print(f"== fold {ty}: train {len(tr_pos):,} test {len(te_pos):,}", flush=True)

        def record(name, p, secs):
            m = metrics(yte, p)
            m.update(model=name, test_year=ty, secs=round(secs, 1))
            rows.append(m)
            print(f"  {name:<14} auc={m['auc']:.4f} top5R={m['top5_exp_R']:+.3f} ({secs:.0f}s)", flush=True)

        t0 = time.time()
        mdl = lgb.train(dict(objective="binary", learning_rate=0.05,
                             num_leaves=63, min_data_in_leaf=500,
                             feature_fraction=0.7, bagging_fraction=0.8,
                             bagging_freq=1, lambda_l2=5.0, verbose=-1,
                             num_threads=4, seed=7),
                        lgb.Dataset(Xtr_raw, ytr), num_boost_round=400)
        p_lgb = mdl.predict(Xte_raw)
        record("lightgbm", p_lgb, time.time() - t0)

        t0 = time.time()
        xm = xgb.XGBClassifier(n_estimators=400, learning_rate=0.05,
                               max_depth=6, subsample=0.8, colsample_bytree=0.7,
                               reg_lambda=5.0, tree_method="hist", n_jobs=4,
                               min_child_weight=50, eval_metric="logloss",
                               random_state=7)
        xm.fit(Xtr_raw, ytr)
        p_xgb = xm.predict_proba(Xte_raw)[:, 1]
        record("xgboost", p_xgb, time.time() - t0)

        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=300, max_depth=12,
                                    min_samples_leaf=200, n_jobs=4,
                                    random_state=7)
        rf.fit(Xtr_i, ytr)
        record("random_forest", rf.predict_proba(Xte_i)[:, 1], time.time() - t0)

        t0 = time.time()
        lr_ = LogisticRegression(C=0.1, max_iter=2000, n_jobs=4)
        lr_.fit(Xtr_s, ytr)
        p_lr = lr_.predict_proba(Xte_s)[:, 1]
        record("logistic", p_lr, time.time() - t0)

        t0 = time.time()
        svm = SGDClassifier(loss="modified_huber", alpha=1e-5, max_iter=30,
                            random_state=7)
        svm.fit(Xtr_s, ytr)
        record("linear_svm", svm.predict_proba(Xte_s)[:, 1], time.time() - t0)

        t0 = time.time()
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                            n_iter_no_change=3, max_iter=60, random_state=7)
        mlp.fit(Xtr_s, ytr)
        record("mlp", mlp.predict_proba(Xte_s)[:, 1], time.time() - t0)

        # sequence models
        lab_tr_mask = np.zeros(len(df), bool)
        lab_tr_mask[tr_pos] = True
        lab_te_mask = np.zeros(len(df), bool)
        lab_te_mask[te_pos] = True
        Str, ptr = build_sequences(df, lab_tr_mask)
        Ste, pte = build_sequences(df, lab_te_mask)
        ytr_s = df[LABEL].to_numpy()[ptr].astype(int)
        yte_s = df[LABEL].to_numpy()[pte].astype(int)

        for name, ctor in (("lstm", lambda: RNNClf(len(SEQ_FEATURES), "lstm")),
                           ("gru", lambda: RNNClf(len(SEQ_FEATURES), "gru")),
                           ("tcn", lambda: TCNClf(len(SEQ_FEATURES))),
                           ("transformer", lambda: TransformerClf(len(SEQ_FEATURES)))):
            t0 = time.time()
            p = train_torch(ctor(), Str, ytr_s, Ste)
            m = metrics(yte_s, p)
            m.update(model=name, test_year=ty, secs=round(time.time() - t0, 1))
            rows.append(m)
            print(f"  {name:<14} auc={m['auc']:.4f} top5R={m['top5_exp_R']:+.3f} ({m['secs']}s)", flush=True)

        # simple ensemble of the three strongest families
        p_ens = (pd.Series(p_lgb).rank(pct=True).to_numpy()
                 + pd.Series(p_xgb).rank(pct=True).to_numpy()
                 + pd.Series(p_lr).rank(pct=True).to_numpy()) / 3
        record("ensemble_rank", p_ens, 0.0)

        pd.DataFrame(rows).to_csv(RESULTS / "model_comparison.csv", index=False)

    res = pd.DataFrame(rows)
    summary = res.groupby("model")[["auc", "logloss", "brier", "top5_exp_R"]].mean()
    summary = summary.sort_values("auc", ascending=False)
    summary.to_csv(RESULTS / "model_comparison_summary.csv")
    print("\n=== mean across folds ===\n", summary.round(4).to_string())


if __name__ == "__main__":
    run()
