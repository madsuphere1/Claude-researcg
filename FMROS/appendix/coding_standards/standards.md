# Coding Standards

Extends Constitution §1.10. Language: Python 3.11+, pandas/numpy/
scikit-learn/LightGBM stack as pinned by the repo.

## Structure
* One experiment = one script under `research/cycle<N>/exp_<id>_<slug>.py`,
  runnable from its directory, writing only under its `results/`.
* Shared logic lives in `research/` modules (`wf.py`, `features.py`,
  `backtest.py`); experiments import, never copy-paste, simulator code.
* Docstring = hypothesis + design + decision rule (frozen pre-run).

## Causality rules (enforced in review)
* Features: information ≤ bar close t. Confirmation-lagged structures
  (swings k-bar-confirmed) place flags at confirmation time.
* Any filter/decoder/embedding applied out-of-sample must be forward-only
  (no Viterbi/`predict_proba`/centered windows on test spans).
* Labels: strictly t+1 onward; entry at next open; no trade spans a >2h
  gap.
* Fold choices (thresholds, gates): functions of train+validation only.

## Simulation rules
* All P&L via the event engine: one position, trade caps, daily stop,
  pessimistic same-bar ties (M1 escalation where it matters), pre-gap
  flatten, leverage cap, costs per leg inside the loop.
* Never compute strategy returns by label arithmetic.

## Determinism & artifacts
* `np.random.default_rng(seed)` / fixed seeds everywhere; torch seeded.
* Results to JSON/parquet with every number the report will cite.
* Print the verdict against the decision rule at the end of main().

## Pitfall list (each cost us once — check explicitly)
* `pgrep -f` matching its own command line in orchestration scripts.
* LightGBM: no JSON-special characters in feature names.
* hmmlearn decoders are smoothed → forward filter by hand.
* `Timestamp.utcnow()` deprecation; tz-handling: dataset clock is fixed
  EST (UTC-5); sessions computed via real tz conversion.
* Stats helpers assume column names (`bars`, `p`); simulator variants
  must emit the full trade schema.
* Weekend-gap × inverse-ATR sizing tail (design-level, §7.4).

## Review requirements
Registry-bound experiments get an adversarial second pass (other agent,
or same agent with fresh context) covering: data path & purge, cost
application point, cohort/selection logic, decision-rule conformance.
