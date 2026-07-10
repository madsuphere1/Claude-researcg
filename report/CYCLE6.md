# Cycle 6 — Metric structure: hierarchy, redundancy, and why Cycle 5 was half-wrong

The operator rejected Cycle 5 on a correct methodological ground: it
ranked 28 metrics as flat, independent columns and read a "pattern" off
them. Metrics are not independent columns — they are a coupled algebraic
system (the operator's analogy: PV=nRT, where you cannot move one
variable holding the others fixed because they are bound by an equation
of state). This cycle does it properly: prove the redundancy, measure
the true dimensionality, build the hierarchy, and test **combinations**
of the real primitives out-of-sample. The result corrects a false claim
from Cycle 5.

## 1. The redundancy is exact, not incidental

Three identities, verified on the data:

* `expectancy_R = p·avg_win + (1−p)·avg_loss` — holds to 9×10⁻⁵.
* `omega ≡ profit_factor` — **exactly equal**; they were two identical
  columns in Cycle 5.
* `calmar ≡ recovery_factor` — **exactly equal**; likewise.

Cycle 5 reported `omega`, `profit_factor`, `calmar`, `recovery_factor`,
`sharpe`, `sortino`, `expectancy` as seven metrics "agreeing" that
forward-ρ ≈ 0.14. They were not agreeing — they are one quantity
(mean return, variously normalised) printed seven times. Counting it
seven times is the fallacy.

## 2. 28 metrics collapse to ~6 independent axes

Correlation clustering (|ρ|≥0.90) and PCA on 155 interval cells:

* **The first principal component alone explains 50.8% of all metric
  variance.** Half of everything the 28 metrics "say" is a single
  underlying dimension.
* 90% of variance is captured by **6 components**; 95% by 8.
* The dominant cluster is **eleven metrics moving as one**:
  `{expectancy, sharpe, sortino, profit_factor, omega, calmar,
  recovery_factor, win_rate, total_R, downside_dev, skew}`. This is the
  **return/profitability axis**. Ranking its members separately (as
  Cycle 5 did) triple-counts nothing but noise.

The genuinely independent axes (one representative each):

| Axis | Cluster members | Nature |
|---|---|---|
| Return / profitability | the 11-metric cluster above | the edge itself |
| Drawdown | max_dd_R, ulcer_index | path risk |
| Tail | var95, cvar95, tail_ratio | **barrier-pinned (near-constant)** |
| Dispersion | std_R | volatility of R |
| Trade geometry | avg_win, avg_loss, payoff_ratio, median_R | **barrier-pinned** |
| Higher moment / streak | kurtosis, max_consec_losses (and max_consec_wins) | shape / clustering |
| Scale | n_trades, trades_per_year | count |

Two of these axes (tail, trade geometry) are **pinned by the TP/SL
barrier geometry** and barely move across windows — structurally blind,
as flagged in Cycle 5. So the *effective* number of informative,
model-sensitive axes is closer to **four**: return, drawdown,
dispersion, and streak/shape.

## 3. The hierarchy (how the metrics are built)

```
PRIMITIVES (measured directly from the trade stream)
  p   = win_rate
  W   = avg_win_R          L = avg_loss_R
  σ   = std_R              σ_d = downside_dev
  MDD = max_dd_R           CL = max_consec_losses
  skew, kurtosis           N = n_trades
        │
        ▼   DERIVED (deterministic functions — carry no new information)
  expectancy = p·W + (1−p)·L
  payoff     = W / |L|
  profit_factor = p·W / ((1−p)|L|)  ≡ omega
  sharpe     = expectancy / σ
  sortino    = expectancy / σ_d
  calmar     = total / |MDD|        ≡ recovery_factor
  total      = N · expectancy
```

A derived metric can never carry information its primitives lack; it can
only re-weight them. Evaluate at the primitive layer, not the derived
layer.

## 4. Combinations, tested out-of-sample — and nothing survives

The operator's core ask: search **combinations** of the true primitives
for a forward-predictive pattern no single metric shows. Done rigorously —
9 primitives, all singles and all pairs, leave-one-out cross-validated
OOS R², with a 500-draw permutation null that re-runs the *entire*
best-of search on shuffled targets (so the search itself cannot
manufacture a hit):

| | Best combination | OOS R² |
|---|---|---|
| Best single primitive | max_consec_losses | **+0.006** |
| Best pair | std_R + max_consec_losses | **+0.002** |
| Permutation null (best-of-search) | — | mean −0.013, 95th pct +0.098 |
| **Permutation p(best ≥ observed)** | — | **0.284** |

The best combination explains essentially **zero** out-of-sample
variance, the best pair does **not** beat the best single, and the whole
result sits **well inside the permutation null (p = 0.284)** — i.e. it is
indistinguishable from what random shuffling produces.

**This corrects Cycle 5.** C5 reported max_dd, max_consec_losses, and
tail_ratio as metrics that "HELP" (forward Spearman ρ ≈ 0.31–0.35). That
ρ was **in-sample**. Under out-of-sample cross-validation with a
permutation null, the predictive power evaporates (OOS R² ≈ 0, p = 0.28).
The Cycle 5 "HELPS" finding is **withdrawn** — it was an artifact of
in-sample correlation on a 42-point panel against 23 metrics, exactly the
overfitting the operator suspected. Per the Constitution, this is
recorded as a correction, not silently edited: C5's informativeness
ranking is superseded by this OOS test.

## 5. What this establishes

* **Metric selection**: work with the ~4 informative independent axes
  (return, drawdown, dispersion, streak/shape) — one representative each,
  chosen at the primitive layer. Never rank collinear derived metrics;
  never treat barrier-pinned metrics as signal.
* **Metric evaluation**: forward-predictive claims must be out-of-sample
  (cross-validated) with a permutation null. In-sample Spearman ρ across
  many metrics is a fallacy generator — it produced Cycle 5's false
  positive.
* **The pattern the operator hoped combinations would reveal is not
  present** in linear pairs of these primitives on this data. That is a
  genuine negative result — and it is consistent with the whole
  program's spine: on this instrument, past performance (in any metric or
  combination tested) does not forecast forward performance. The edge is
  regime-driven, not metric-schedulable.

### Honest limits (what this does NOT prove)
n = 42 trailing→forward pairs is a low-power test; "indistinguishable
from noise" means *no pattern is demonstrable here*, not that none exists
— absence of evidence. Only linear combinations up to pairs were tested;
higher-order or nonlinear interactions are **not honestly testable at
this sample size** (they would overfit), so they remain open, not
excluded. A larger sample (the 2027+ accrual, C3-008) is the only way to
raise the power. The four independent axes and the OOS+permutation
protocol are now the frozen method for that future test.

Registry: C6-001 closed; supersedes C5-001's informativeness ranking.
No production gate touched. Simulated; not investment advice.
