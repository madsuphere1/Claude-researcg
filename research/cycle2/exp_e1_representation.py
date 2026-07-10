"""H-E1: self-supervised representation probe.

A small masked-reconstruction encoder (GRU) is trained on the DEV window
(2010-2013) only: 32-bar windows of 6 normalised channels, 25% of timesteps
masked, MSE reconstruction. The final hidden state (32-d) is the embedding.

Test: LightGBM AUC on y_tp_long for folds 2018 / 2022 / 2026 with
(a) baseline 205 features, (b) baseline + 32 embedding dims,
(c) embeddings alone. Success (registry): uplift (b)-(a) >= +0.002 mean AUC.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parents[1]))
import wf  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "results"
SEQ = 32
EMB = 32
CH = 6
FOLDS = [2018, 2022, 2026]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=500, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              verbose=-1, num_threads=4, seed=7)
torch.manual_seed(7)


class MaskedGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.GRU(CH, EMB, batch_first=True)
        self.dec = nn.Linear(EMB, CH)

    def forward(self, x):
        h, hn = self.enc(x)
        return self.dec(h), hn[-1]


def build_windows(bars: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex]:
    c = bars.close
    chans = pd.DataFrame({
        "r": np.log(c).diff(),
        "hl": (bars.high - bars.low) / c,
        "body": (bars.close - bars.open) / c,
        "rv": np.sqrt(bars.rv),
        "imb": (bars.up_m1 - bars.down_m1) / (bars.up_m1 + bars.down_m1 + 1),
        "act": bars.absret,
    })
    dev = chans[chans.index < "2014-01-01"]
    mu, sd = dev.mean(), dev.std() + 1e-12
    z = ((chans - mu) / sd).clip(-8, 8).fillna(0).to_numpy(dtype=np.float32)
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(z, SEQ, axis=0).transpose(0, 2, 1)
    return win, chans.index[SEQ - 1:]


def main() -> None:
    df, feats = wf.load()
    bars = pd.read_parquet(HERE.parents[1] / "research" / "data" / "xauusd_15m.parquet")
    bars = bars.loc[df.index]
    win, widx = build_windows(bars)

    dev_mask = widx < "2014-01-01"
    Xdev = torch.tensor(win[dev_mask])
    model = MaskedGRU()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    n = len(Xdev)
    print(f"training masked GRU on {n:,} dev windows", flush=True)
    for ep in range(3):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 512):
            xb = Xdev[perm[i:i + 512]]
            mask = torch.rand(xb.shape[0], SEQ, 1) < 0.25
            xin = xb.masked_fill(mask, 0.0)
            opt.zero_grad()
            rec, _ = model(xin)
            loss = lossf(rec[mask.expand_as(rec)], xb[mask.expand_as(xb)])
            loss.backward()
            opt.step()
            tot += float(loss) * len(xb)
        print(f"epoch {ep}: masked MSE {tot / n:.4f}", flush=True)

    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(win), 8192):
            _, hn = model(torch.tensor(win[i:i + 8192]))
            embs.append(hn.numpy())
    E = pd.DataFrame(np.concatenate(embs), index=widx,
                     columns=[f"emb_{k}" for k in range(EMB)]).reindex(df.index)

    lab = df["y_tp_long"].to_numpy()
    yrs = df.index.year
    dfe = pd.concat([df[feats], E], axis=1)
    out = {}
    for ty in FOLDS:
        te = np.flatnonzero(yrs == ty)
        tr = np.flatnonzero(yrs < ty)
        tr = tr[tr < te[0] - wf.PURGE_BARS]
        tr = tr[np.isfinite(lab[tr])]
        tev = te[np.isfinite(lab[te])]
        r = {}
        for name, cols in (("baseline", feats),
                           ("baseline+emb", feats + list(E.columns)),
                           ("emb_only", list(E.columns))):
            mdl = lgb.train(PARAMS, lgb.Dataset(dfe[cols].iloc[tr],
                                                lab[tr].astype(int)),
                            num_boost_round=400)
            r[name] = float(roc_auc_score(lab[tev].astype(int),
                                          mdl.predict(dfe[cols].iloc[tev])))
        out[str(ty)] = r
        print(ty, {k: round(v, 4) for k, v in r.items()}, flush=True)

    mean_uplift = float(np.mean([out[str(t)]["baseline+emb"] - out[str(t)]["baseline"]
                                 for t in FOLDS]))
    out["mean_uplift"] = mean_uplift
    (RESULTS / "e1_representation.json").write_text(json.dumps(out, indent=1))
    print("mean uplift:", round(mean_uplift, 4),
          "ACCEPT" if mean_uplift >= 0.002 else "REJECT")


if __name__ == "__main__":
    main()
