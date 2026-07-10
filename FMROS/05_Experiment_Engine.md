# Volume 5 — Experiment Engine

The machinery that turns the registry into an autonomous scientist:
what runs next, what counts, what dies.

## 5.1 The experiment registry

Location: `FMROS/appendix/experiment_registry/registry.md` (append-only
table) plus one file per experiment
(`FMROS/appendix/experiment_registry/entries/<ID>.md`, template in
`appendix/templates/experiment.md`).

Entry states: `PROPOSED → DESIGNED → SCREENED → PRE-REGISTERED → RUN →
DIAGNOSED → {CONFIRMED | WEAKENED | REFUTED | INVALID}`.

Mandatory fields (see template): ID; program; hypothesis + mechanism;
economic term attacked; multiplicity universe; dev-screen design and
result; frozen confirmatory design (data slice, folds, costs, controls,
grids with primary values); **decision rule verbatim**; results file
path; git SHA; verdict; lessons; superseded-by links.

Two ID conventions are already in force and are grandfathered: cycle-2
IDs (`H-A1`…`H-I1`, `X1`) and cycle-3+ IDs (`C3-…`, sequential).

## 5.2 Priority scoring

When multiple PROPOSED/DESIGNED entries compete for compute, the Chief
Scientist scores each 1–5 on:

* **I — Information value.** How much does either verdict change the
  program's beliefs or unblock other work? (The cycle-3 X1 confirmation
  is a 5: it gates the entire production question.)
* **P — Prior plausibility × effect size.** Mechanism strength times the
  economic magnitude if true. (A 0.001-AUC curiosity scores 1 even if
  certain.)
* **C — Cost.** Inverse of compute + implementation time.
* **D — Data readiness.** 0 if the data doesn't exist yet (converts the
  entry into a data-acquisition task for A5).

Priority = I × P × C, gated by D. Ties break toward the experiment that
closes a program question permanently (dead ends are valuable —
Volume 3 exists largely to stop re-litigating them).

Estimated information gain must be written down before the run; the
Auditor compares estimates to realized surprise over time to calibrate
the Chief Scientist's scoring.

## 5.3 Acceptance criteria (house defaults)

Any experiment may declare stricter rules; none may declare weaker
without a Statistician-signed justification in the entry.

| Claim type | Default acceptance |
|---|---|
| Feature/factor validity | OOS IC or importance stable in sign across ≥ 70% of yearly folds AND deflated p < 0.05 within its declared universe |
| Model improvement | Mean walk-forward AUC uplift ≥ +0.003 on ≥ 3 folds spanning ≥ 6 years, never negative in more than 1 fold |
| Expectancy/strategy | Net > 0 at 2.5bp; day-block p < 0.05 AND coarsest-block p < 0.05; n ≥ 300 trades across ≥ 3 regimes; survives declared sensitivity grid |
| Execution/cost effect | ITT uplift CI excluding 0 at day blocks; effect direction stable across the declared parameter grid |
| Risk rule | Improves MAR or tail metric on positive stream without creating profit on the negative-stream sanity check |

## 5.4 Rejection criteria

* Falsification criterion met → REFUTED (final for that mechanism+data).
* Point estimate wrong-signed at the primary spec → REFUTED even if some
  grid corner looks good (corners are hypothesis-generating only).
* Right sign, missed significance → WEAKENED; follow-up rules per §1.6.
* Any post-hoc modification of spec or rule → the run is void; re-file.

## 5.5 Statistical thresholds

Centralized so no experiment invents its own: α = 0.05 at the gating
blocking level; Bonferroni within declared universes; bootstrap B ≥ 3000
(5000 for verdict-gating numbers); placebo N ≥ 50 (200 where cheap);
purge ≥ 96 bars (≥ 6× label horizon); AUC uplifts matter at ≥ 0.003;
expectancy reported to 3 decimals in R with CI.

## 5.6 Experiment lifecycle mechanics

1. **PROPOSED**: hypothesis template filled; enters priority queue.
2. **DESIGNED**: script skeleton + docstring decision rule written;
   Statistician sign-off recorded.
3. **SCREENED** (optional): dev-window result; a failed screen closes the
   entry as REFUTED-AT-SCREEN (cheap, honorable death — H-C1 pattern:
   its dev window already showed nothing).
4. **PRE-REGISTERED**: confirmatory spec frozen (grids, costs, folds,
   rule); git commit of the script *before* the run is the timestamp.
5. **RUN**: executed by script; results file committed.
6. **DIAGNOSED**: dissection (by year/regime/exit-reason/tail), Reviewer
   pass, Risk pass. New anomalies become new PROPOSED entries, not
   silent spec edits.
7. **Verdict** recorded with lessons. CONFIRMED strategies move to the
   Volume 9 pipeline; everything else feeds Volume 3 agendas.

## 5.7 Automatic rollback

Applies to the code/artifact side of experiments:

* Simulator or feature-library changes trigger re-runs of the **anchor
  set** (a registry-tagged subset: currently cycle-1 baseline backtest,
  B2v2, A1 gate, B1 δ=0.1, X1 pilot). Any unexplained delta > tolerance
  (0.005R expectancy or 0.002 AUC) blocks the change — revert or explain.
* Data revisions (A5 changelog events) invalidate downstream caches
  automatically: cached predictions and registry verdicts that consumed
  the revised data get flagged STALE until re-run.
* A verdict found to rest on an INVALID result cascades: dependent
  entries revert to their prior state and the report that cited them
  gets an erratum note (never silent edits).

## 5.8 The autonomous loop (Track J made operational)

One cycle of the discovery engine:

```
1. Chief Scientist: refresh priority queue from program agendas + last diagnoses
2. Take top entry → complete lifecycle §5.6
3. Every verdict → update program agenda (Vol. 3) + lessons
4. Every N=5 experiments or at cycle end → Auditor pass + cycle report
5. If a CONFIRMED strategy exists → hand to Volume 9 pipeline (parallel)
6. GOTO 1
```

Stopping condition: the program never "finishes"; it pauses when the
priority queue's top information value falls below the cost of running
it — which is itself reported to the operator as the honest
"diminishing returns" signal.
