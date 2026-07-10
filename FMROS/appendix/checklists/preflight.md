# Pre-flight Checklist (before any confirmatory run)

- [ ] Registry entry exists and is in PRE-REGISTERED state
- [ ] Decision rule in script docstring matches registry verbatim
- [ ] Script committed (SHA recorded) BEFORE the run
- [ ] Data slices to be consumed are unburned (check registry burn list)
- [ ] Purge ≥ 96 bars verified at every train/test boundary
- [ ] All fold-level choices are functions of train+validation only
- [ ] Costs applied inside the simulator, correct tier(s)
- [ ] Controls implemented (placebo / random gate / baseline as declared)
- [ ] Seeds fixed; run is deterministic
- [ ] Leakage screens passed for any NEW features (corr, shift, smoothed-state, boundary)
- [ ] Tail mechanisms considered: max holding period × gap calendar; leverage at low-ATR entries
- [ ] Output paths under results/; script prints its own verdict vs the rule
