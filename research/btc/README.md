# BTCUSD Research Program

A second, independent research program under FMROS, parallel to the
XAUUSD program. Same constitution: near-efficiency null, mechanism before
mining, walk-forward out-of-sample, cost-in-R honesty, negative results
reported at full prominence. **Not** a copy of the gold work — Bitcoin
differs structurally, and those differences are the research.

## How BTC differs from XAUUSD (and why it matters)

| | XAUUSD (gold) | BTCUSDT |
|---|---|---|
| Sessions | London/NY, weekend gaps | **24/7, no gaps** |
| Data | HistData, no true volume | **real volume, trades, taker-buy flow** |
| Volatility | ~1% daily | ~3-5% daily (regimes extreme) |
| Retail cost | ~2.5 bp spread | **~10 bp taker fee** (much higher) |
| History | 2010-2026 | 2017-08 → 2026 (shorter) |

Two implications set the research agenda:
1. **Real order-flow exists.** Taker-buy imbalance, trade count, and
   volume are genuine microstructure signals gold never had — the first
   thing worth testing (P7 Microstructure gets real teeth here).
2. **Costs are ~4× gold's.** The breakeven bar is much higher, but BTC's
   gross moves are also much larger. Whether net edge survives is the
   open question, exactly as in gold — decided by cost-in-R, not by
   whether a signal exists.

## Data
`data/btcusdt_15m.parquet` — Binance public dumps (data.binance.vision),
15m klines, OHLCV + quote_volume + trades + taker_buy_base, UTC.
Acquisition: `fetch_binance.py`.

## Cycle plan (mirrors the gold program's spine)
* **B1-BASE** — does a predictive signal exist? Walk-forward AUC on
  triple-barrier labels, per-year sign test. (`baseline.py`)
* **B1-FLOW** — do the crypto-only order-flow features (taker imbalance,
  volume, trades) add OOS AUC beyond price alone? The genuinely new
  question.
* **B1-ECON** — does any signal survive ~10bp crypto cost? Cost-tier
  table, breakeven bp.
* Later: regime (bull/bear/chop like gold), execution, the trend-ride
  vs mean-revert question (BTC is famously trend-persistent).

All results in `results/`; verdicts in the shared registry with a `B`
prefix. Simulated throughout; not investment advice.
