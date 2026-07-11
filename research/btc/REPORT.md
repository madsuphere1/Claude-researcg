# BTCUSD — Cycle 1 (founding): signal exists, order-flow helps, costs bite harder

First cycle of the Bitcoin research program. Real Binance data,
310,462 fifteen-minute bars, 2017-08 → 2026-06, with true volume, trade
counts, and taker-buy flow. Walk-forward, out-of-sample, 8 test years
(2019–2026), triple-barrier labels (TP 1.5×ATR / SL 1×ATR / 16-bar).
Simulated; not investment advice.

## B1-BASE — does a predictive signal exist? **CONFIRMED**
Price-only walk-forward AUC = **0.5301**, positive in **8/8** test years,
sign-test **p = 0.0039**. A real, statistically significant directional
signal exists on BTC 15m — marginally stronger and more consistent than
gold's (0.529, 13/13). Per-year AUC declines late (0.544 in 2020 →
0.512 in 2025–26), the same alpha-decay pattern seen in gold.

## B1-FLOW — do crypto-only order-flow features add value? **CONFIRMED**
Adding five order-flow features gold never had — taker-buy imbalance,
volume z-score, trade-count z-score, average trade size, smoothed
imbalance — raised walk-forward AUC from 0.5301 to **0.5347, uplift
+0.0046**. This clears the +0.003 bar the gold program used, and it is
the first genuinely *new* result of the crypto program: **aggressive-buy
order flow carries incremental predictive information** that price alone
lacks. Gold's data could never test this; Bitcoin's can, and it's real.

## B1-ECON — does it survive crypto costs? **REFUTED**
Top-decile-conviction long trades: gross **+0.0162R** (win rate 40.6%,
barely above the 40% breakeven for 1.5:1). Net expectancy after cost:

| Cost | Net R | 
|---|---|
| 2 bp | −0.048 |
| 5 bp | −0.145 |
| **10 bp (retail crypto taker)** | **−0.307** |

Negative at every tier. The mechanism is the same cost-in-R identity as
gold, but harsher: a 15m bar's 1×ATR stop is only ~0.3% of price, so even
2 bp is ~0.065R of drag, and crypto's realistic **10 bp taker fee is
~0.32R** — four times gold's 2.5 bp spread. **Bitcoin's signal is
slightly richer than gold's, but its cost wall is far higher**, so the
naive 15m strategy is *more* cost-dead, not less.

## Verdict and agenda
Same shape as gold's Cycle 1 — real signal, killed by cost — with two
BTC-specific twists: (1) order flow is a genuine, testable edge source
here; (2) the 10 bp fee makes short-horizon trading brutal, which pushes
the viable design toward **fewer, larger moves** (wider barriers / longer
horizons / trend-riding) where cost-in-R shrinks — exactly the lever that
rescued the gold EA. Next: B2 wide-barrier / trend-hold economics, and a
B-FLOW deep dive (does taker imbalance time entries well enough to beat
10 bp?). No production gate touched.
