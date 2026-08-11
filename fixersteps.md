# Fix Plan — post-Brownian follow-ups

Six ordered steps closing the gaps found after implementing [brownianPlan.md](brownianPlan.md)
on `feature/brownian-motion`. Ordered by dependency: each step is safe to stop after.

**Baseline for rollback:** tag `pre-brownian-baseline` (`3d078a3`), branch
`backup/pre-brownian-main`. Current head: `590292a`.

**Status quo.** 151 tests pass; ruff / black / mypy clean; coverage 91%; legacy dynamics
bit-identical to the baseline on 10/10 probes; `run_experiments.py --quick` completes on the
9 classic regimes. Everything below is a gap, not a breakage.

| # | Step | Effort | Risk | Blocks |
|---|------|--------|------|--------|
| 1 | Coerce config types at construction | 0.5 h | low | — |
| 2 | Stop smoke runs overwriting published results | 0.5 h | low | 4, 6 |
| 3 | Wire the Brownian features into the CLI | 2 h | low | 4, 5 |
| 4 | Run the Almgren-Chriss optimality validation | 3 h + compute | medium | 5, 6 |
| 5 | Teach the report generator about the price process | 1.5 h | low | 6 |
| 6 | Decide and execute the default flip | 1 d + compute | **high** | — |

---

## Step 1 — Coerce config types at construction

**Problem.** `MarketConfig` accepts whatever `from_dict` hands it, with no type check. A YAML
typo produces a *string* that is truthy and passes validation:

```python
MarketConfig.from_dict({"imbalance_ou_scaling": "false# oops"})
# -> 'false# oops', bool() is True, RunConfig(...).validate() PASSES
```

This is not hypothetical — it happened while writing `config/default.yaml` (a missing space
before a `#` comment). It would have silently switched on the stationary imbalance scaling and
invalidated a whole run with no error anywhere.

**Fix.** Give `MarketConfig` a `__post_init__` that coerces and rejects, mirroring the pattern
already used by `ExecutionConfig.__post_init__` ([config.py:143](rl_execution/config.py:143)):

- floats/ints → `float()` / `int()`, raising `ValueError` on junk rather than propagating a string;
- `bool` fields → accept real bools and the exact strings `true`/`false`/`yes`/`no`/`1`/`0`
  (case-insensitive), reject anything else;
- enum-like strings (`price_process`, `rng_mode`, `normal_method`, `vol_process`,
  `latency_model`) → `str().strip().lower()`, so trailing whitespace is not a new failure mode.

Keep the membership checks where they are in `_validate_manual`; this step is about *types*
reaching validation intact.

**Files.** `rl_execution/config.py`, `tests/test_brownian.py` (or a new `tests/test_config.py`).

**Tests.**
- `test_bool_fields_reject_non_boolean_strings` — the case above now raises.
- `test_bool_fields_accept_yaml_booleans` — `true` / `False` / `1` all coerce.
- `test_numeric_fields_coerce_from_strings` — `dt: "0.5"` becomes `0.5`.
- `test_enum_fields_are_case_and_whitespace_tolerant` — `" GBM "` → `"gbm"`.

**Done when.** The snippet above raises `ValueError`, and the full suite plus
`load_config("config/default.yaml")` still pass.

---

## Step 2 — Stop smoke runs overwriting published results

**Problem.** `scripts/run_experiments.py --quick` writes to `RESULTS_DIR` and `MODELS_DIR`
unconditionally. A single smoke run overwrote the committed 60k-step
`results/{regime_results,paired_vs_twap,reward_logs,summary}.{csv,json}` and all five
`models/*.json` sidecars with tiny-budget numbers. They were recoverable only because they
happen to be tracked; `results/*.pkl` and `*.log` are gitignored and would have been lost.

Steps 4 and 6 both involve running the pipeline repeatedly, so fix this first.

**Fix.**
- Add `--out-dir` (default `RESULTS_DIR`) and `--models-dir` (default `MODELS_DIR`) to
  `parse_args` in [cli/experiments.py](rl_execution/cli/experiments.py:53) and thread them
  through instead of importing the constants at each write site.
- Make `--quick` default `--out-dir` to `results/quick/` and `--models-dir` to `models/quick/`
  unless the user passes them explicitly, so a smoke run can never touch published artefacts.
- Add `results/quick/` and `models/quick/` to `.gitignore`.

**Files.** `rl_execution/cli/experiments.py`, `rl_execution/cli/multiseed.py` (same pattern),
`.gitignore`.

**Tests.** Extend `tests/test_cli.py`: `test_quick_run_writes_to_a_sandbox_dir` — run the quick
pipeline against `tmp_path` and assert the real `RESULTS_DIR` files are untouched (compare
mtimes or hashes before/after).

**Done when.** `python scripts/run_experiments.py --quick` leaves `git status` clean.

---

## Step 3 — Wire the Brownian features into the CLI

**Problem.** Three shipped features are library-only and unreachable from any entry point:
`evaluate(antithetic=True)`, `randomized_market_config(brownian=True)` and
`schedule_gap_table`. As delivered, the pipeline cannot use any of them.

**Fix.**

*`rlx-experiments`* ([cli/experiments.py](rl_execution/cli/experiments.py:53)):
- `--antithetic` — passes through `evaluate_across_regimes` to `evaluate`. Requires a
  non-legacy `price_process`; error out early with a clear message if the selected regimes are
  all legacy, rather than silently doing nothing.
- `--include-brownian` — `regimes = list_regimes(include_brownian=True)`. Keep the classic nine
  as the default (see the regression test added in `590292a`); when this flag is on, write the
  Brownian grid to a **separate** CSV so the two price models are never Holm-corrected as one
  family.
- `--schedule-reference AlmgrenChriss` — emit `schedule_gap_table` to
  `results/schedule_gap.csv`.

*`rlx-train`* ([cli/train.py](rl_execution/cli/train.py)): `--brownian-randomization`, routed to
`randomized_market_config(rng, brownian=True)` via `RandomizedRegimeEnv`
([runner.py:36](rl_execution/experiments/runner.py:36)).

*`evaluate_across_regimes`* ([runner.py:57](rl_execution/experiments/runner.py:57)): add an
`antithetic: bool = False` parameter and pass it down.

Then propagate `block=2` into `paired_is_table` wherever `--antithetic` is set — antithetic
episodes are dependent within a pair, and the un-blocked bootstrap is anti-conservative.

**Files.** `rl_execution/cli/experiments.py`, `rl_execution/cli/train.py`,
`rl_execution/experiments/runner.py`, `tests/test_cli.py`, README quickstart.

**Tests.**
- `test_experiments_cli_accepts_antithetic` — smoke run with the flag completes and produces an
  even episode count.
- `test_antithetic_flag_forces_block_two` — the paired table is built with `block=2`.
- `test_include_brownian_writes_a_separate_grid` — two CSVs, disjoint regime sets.

**Done when.** `rlx-experiments --quick --include-brownian --antithetic` runs and writes both
grids plus `schedule_gap.csv`.

---

## Step 4 — Run the Almgren-Chriss optimality validation

**Problem.** This is the headline claim of the whole Brownian effort and it is currently
*machinery only*. `abm_ac_optimal` and `schedule_gap_table` are unit-tested, but no agent has
ever been trained in that regime, so "the learned policy recovers the closed-form optimum" is
unevidenced.

**Why it matters.** Under `abm_ac_optimal` the price is arithmetic Brownian motion with zero
drift, zero alpha and constant volatility, and the book is deep enough that the walk cost stays
close to linear — the exact conditions under which the Almgren-Chriss trajectory is provably
optimal. A small schedule distance is direct evidence about the *learner*, not another cost
comparison that a lucky price path could explain.

**Procedure.**
1. Train each agent on `abm_ac_optimal` (not domain-randomised — this is a single-regime probe)
   at the production budget, ≥60k steps; per the project notes DQN/A2C collapse to
   "wait then force-liquidate" below ~20k.
2. Evaluate against `AlmgrenChriss(risk_aversion=...)` matched to the env's
   `RewardConfig.risk_aversion`, plus TWAP as the null, over ≥400 paired episodes with
   `--antithetic`.
3. Emit `schedule_gap_table(results, reference="AlmgrenChriss")`.
4. Sweep `imbalance_alpha` from 0 → 0.05 in the same regime and plot the schedule distance
   against it — this is the falsification test recorded in the project notes: at α = 0 the
   learned schedule **must** collapse onto the AC sinh curve, and a non-zero intercept means a
   simulator exploit rather than skill.

**Interpretation, fixed in advance** (so the result cannot be rationalised after the fact):

| `schedule_L1` at α = 0 | Reading |
|---|---|
| < 0.10 | Learner recovers the optimum; the framework is validated |
| 0.10 – 0.30 | Directionally right, under-trained or under-tuned — report honestly |
| > 0.30 | Either the agent has not solved it or the regime is not as linear as assumed; investigate the book-walk convexity before claiming anything |

**Files.** New `rl_execution/experiments/ac_validation.py` (or a script under `scripts/`),
`results/schedule_gap.csv`, a figure via `plot_execution_schedule`.

**Done when.** A committed table and figure show the RL schedule against the AC curve, with the
α-sweep, and the finding — whatever it is — is written down.

---

## Step 5 — Teach the report generator about the price process

**Problem.** [cli/report.py](rl_execution/cli/report.py:94) §2 "Market & limit-order-book model"
is hardcoded prose describing only the legacy multiplicative walk. A regenerated
`reports/REPORT.md` therefore understates the framework and would misdescribe any run that used
a Brownian process.

**Fix.**
- Make §2 read `MarketConfig` from the run summary and describe the *actual* process: the SDE,
  its exact discretisation, `dt`, the volatility process and the leverage correlation.
- Add a short §2.1 on the RNG design: independent substreams, the exogenous market drawn before
  the agent acts, and what that buys — under legacy, a strategy's own trades consumed the normal
  draws that would have driven the price, so two strategies on one seed diverged by ~14 price
  units over 12 steps with impact switched off; under a Brownian process they face bit-identical
  markets, which is what makes the paired comparison a real variance reduction.
- If `results/schedule_gap.csv` exists (Step 4), add a §5.1 "Optimality check" rendering it.
- Note the antithetic setting and the bootstrap `block` in the statistical-methods section, so
  the CI construction is reproducible from the report alone.

**Files.** `rl_execution/cli/report.py`, `tests/test_cli.py`.

**Tests.** `test_report_describes_the_configured_price_process` — generate a report from a
summary carrying `price_process="gbm"` and assert the text says so and does not claim the legacy
model.

**Done when.** `rlx-report` produces a REPORT.md whose §2 matches the config that produced the
results.

---

## Step 6 — Decide and execute the default flip

**This is the only step that changes published numbers. Do not start it until 1–5 are done.**

Three decisions, left open deliberately in `brownianPlan.md` §7:

**6a. Does `price_process` default to `gbm`?**
Recommendation: yes, at a tagged release. It fixes the CRN leak for every future result and adds
the Itô correction. Cost: `results/`, `reports/REPORT.md` and the deck's numbers all move.

**6b. Does `imbalance_ou_scaling` default to `True`?**
Recommendation: yes, **but only together with a recalibration of `imbalance_alpha`.** The
stationary scaling raises the realised imbalance sd from 0.285 to ~0.72 (not 1.0 — clipping to
[-1, 1] saturates a unit-variance Gaussian about 32% of the time). That is ~2.5× stronger, and
the project notes already judge the current alpha ~4–7× too predictive versus real order-book
signals (per-step R² = 0.336, IC = 0.58 at α = 0.05, against a realistic IC of 0.05–0.15).
Flipping the scaling without lowering α makes an already-overstated edge worse. Solve for the α
that holds R² fixed, and report improvement as a *function* of α rather than a single number.

**6c. exp-OU or full Heston for stochastic volatility?**
Recommendation: stay with exp-OU — positive by construction, exactly discretisable, one fewer
scheme to defend. Revisit only if a reviewer asks.

**Execution, as one commit:**
1. Change the defaults in `MarketConfig` and `config/default.yaml`.
2. Retune `imbalance_alpha` (6b) and record the calculation.
3. Regenerate every artefact: `results/*`, `reports/REPORT.md`, figures.
4. Update the README "Interpreting the numbers" and the proposal deck's figures.
5. Add a `CHANGELOG.md` migration note stating that `price_process: legacy` reproduces the old
   results and that `config_hash` changes for every run.
6. Keep `test_legacy_mode_is_bit_identical` green — it pins the *legacy path*, which must
   survive the default change untouched.

**Rollback.** `git reset --hard pre-brownian-baseline`, or set `price_process: legacy` and
`imbalance_ou_scaling: false` in `config/default.yaml` to reproduce the old numbers on the new
code.

---

## Verification to run after every step

```bash
python -m pytest tests/ -q && python -m ruff check . && python -m black --check . && python -m mypy rl_execution
```

Legacy must stay bit-identical throughout Steps 1–5 — `test_legacy_mode_is_bit_identical` and
`test_legacy_env_episode_is_bit_identical` are the guards. Step 6 is the only step permitted to
change published numbers, and it changes them by changing *defaults*, never the legacy path
itself.
