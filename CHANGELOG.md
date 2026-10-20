# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed: the problem formulation (roadmap Phase 1)
- **Calibrated market defaults** so impact and price risk trade off: per-step volatility
  0.004 (was 0.02), temporary impact 1.5e-4 (was 1e-5), permanent impact 2e-5 (was 5e-6),
  stationary imbalance scaling on, and `imbalance_alpha` 4.5e-4 for an observed information
  coefficient of ~0.075 (v0.1's signal was ~0.43). Dumping at arrival now costs ~158 bps and
  TWAP ~19 bps with ~104 bps of price risk. `rl_execution.experiments.calibration` and
  `scripts/calibrate.py` measure all of this and solve for the signal strength and risk
  aversion; `tests/test_calibration.py` pins the bands. All regimes and the domain
  randomiser were rebuilt around the new defaults; the randomiser's ranges stop short of the
  `low_vol` / `high_vol` / `thin` / `deep` presets so those can be held out.
- **Reward = the Almgren-Chriss objective, exactly.** Mark-to-market rewards (fill cost
  against the decision-time mid, plus the P&L of the inventory still held, minus
  `lambda * (sigma_t x)^2`) sum to exactly `-(IS + lambda * sum sigma^2 x^2)`, tested to
  1e-8 across price processes, sides and latency. The risk term uses the model volatility
  for the step (the old realised-volatility estimate was zero at reset). Removed the
  double-counted `temp_impact_penalty` and the `unexecuted_penalty` / `force_liquidation`
  flags: the last step always liquidates. `risk_aversion` is now per bps^2 (1.4e-3, giving
  an Almgren-Chriss urgency kappa*T ~2.5) and `reward_scale` defaults to 1 (rewards in bps).
- **Almgren-Chriss uses the reward's own risk aversion** (`ac_risk_aversion`), so in a
  signal-free arithmetic-Brownian market it is the optimum of the agents' objective.
- **TWAP-relative action**: `a in [-1, 1]`, -1 waits, 0 trades at the TWAP pace, +1
  liquidates now; the discrete space is a grid of TWAP multipliers. An untrained PPO policy
  now starts as TWAP instead of "wait, then dump". Baselines go through the same interface
  via `ExecutionEnv.fraction_to_action`.
- **Observation**: 7 O(1) features (inventory, time, price move in sigma units, imbalance,
  volatility ratio, spread, depth against a fixed reference); price move and imbalance are
  signed by side; `prev_action` dropped; the depth feature now distinguishes thin and deep
  markets.
- **Report**: ranks on the objective `J`, compares against Almgren-Chriss by default
  (`--benchmark`, `--metric`), and adds IS tail, effective steps traded and the fitted
  SignalRule coefficient.

### Added
- Null baselines `DumpAtStart`, `WaitThenDump` and `SignalRule` (AC pace tilted by the book
  signal, coefficient fitted per regime on validation seeds via the new `BaseStrategy.fit`).
- `effective_steps` and `objective_bps` episode metrics.

### Fixed
- `_stack_mean` and `schedule_gap` truncated schedules to the shortest episode, so a
  front-loaded policy looked half as far from TWAP as it is; both now zero-pad.

### Changed: lean core

A lean core: fewer moving parts, each one trained and tested properly.

### Removed
- The non-Brownian `legacy` price walk, together with `rng_mode` (independent substreams are
  now the only mode), the always-adverse latency model and the bit-identity tests that pinned
  them. The legacy walk let a strategy's own trades consume the random draws that drove the
  price, so paired comparisons were not true common random numbers.
- A2C, SAC and SB3-DQN; the agents are now PPO and the from-scratch DQN / Double DQN.
- The POV and Random baselines.
- Real-LOB replay (`rl_execution.data`), the multi-agent simulator and the Streamlit dashboard.
- MLflow / W&B tracking, the model registry, Optuna HPO, the multi-seed harness and the
  deflated / probabilistic Sharpe helpers.
- Docker files, issue / PR templates, `CONTRIBUTING.md` and internal planning documents.
- The v0.1 `results/`, `models/` sidecars and `reports/REPORT.md`. They could not be
  regenerated from the repository, and the published PPO row matched a do-nothing policy.

### Changed
- Every price process is Brownian; the default is arithmetic BM (`price_process: abm`), and
  the `sideways` regime and mean-reverting randomised draws use `ou`.
- One command line, `rlx train | evaluate | report`, replaces the five `rlx-*` entry points and
  the wrapper scripts. `train --seeds 0-9` trains several seeds and skips completed ones.
- Runs live in `runs/<experiment>/<Agent>/seed_<k>/` with `config.yaml`, weights,
  `train_log.csv` and `provenance.json`, which now records the weights' SHA-256; `load_run`
  refuses weights that do not match.
- `rlx evaluate` uses a dedicated block of test seeds and supports `--antithetic`; `rlx report`
  takes the training seed as the unit of analysis for confidence intervals and Holm-corrects
  across every comparison.
- Python 3.10+; numpy 2 is allowed; CI runs a single lint, type-check and test job.
- The Windows OpenMP workaround (`KMP_DUPLICATE_LIB_OK`) is only applied on Windows.

### Fixed
- The agent tests now train past `learning_starts`, so the DQN tests run gradient updates.

## [0.1.1]

### Added
- Sandboxed smoke runs: `rlx-experiments` and `rlx-multiseed` gain `--out-dir` / `--models-dir`,
  and `--quick` defaults them to `results/quick/` and `models/quick/` (figures follow the results
  directory), so a tiny-budget smoke run can no longer overwrite published results, models or
  figures. Both directories are gitignored.
- Brownian-motion price dynamics:
  - `rl_execution.envs.brownian`: named RNG **substreams** (`Stream` / `NoiseStreams`, spawned
    from one seed via `SeedSequence`), Wiener increments (`BrownianEngine`), the **Brownian
    bridge** (`bridge_value`, `BrownianEngine.bridge` / `.refine`), Cholesky-correlated normals
    and a choice of ziggurat or **inverse-CDF** normal generation (monotone in the uniform, so
    antithetic pairing is exact and QMC constructions are possible).
  - `rl_execution.envs.processes`: explicit SDEs with **exact** discretisations — arithmetic
    Brownian motion (`abm`), geometric Brownian motion (`gbm`, with the Itô correction the old
    `S*(1+r)` update omitted), Ornstein-Uhlenbeck (`ou`), plus an **exp-OU stochastic
    volatility** producing genuine volatility clustering and a **leverage** correlation
    between the price and volatility Brownians.
  - The whole exogenous market (price, volatility, imbalance, spread, volume, per-level depth)
    is now drawn at `reset()` under a Brownian process, with permanent impact applied as an
    additive overlay. Agent actions can no longer consume randomness that perturbs the market,
    so `compare_strategies` / `paired_is_table` are genuine common random numbers.
  - `latency_model="bridge"` prices a delayed fill on the Brownian bridge between the two
    decision steps (unbiased) instead of the always-adverse folded normal; the deviation is
    now transient in both the synthetic and historical simulators.
  - `evaluate(antithetic=True)` pairs each seed with its `Z -> -Z` twin; `paired_bootstrap_ci`
    and `paired_is_table` gain `block=` so dependent pairs are resampled as blocks.
  - New regimes `abm_ac_optimal`, `gbm_medium`, `clustered_vol`, `leverage`, and
    `schedule_gap` / `schedule_gap_table` to measure how far a learned policy's schedule sits
    from the closed-form Almgren-Chriss trajectory in the regime where it is provably optimal.
  - `imbalance_ou_scaling=True` switches the latent imbalance to the stationary
    `sqrt(1 - phi^2)` innovation; the historical `1 - phi` scaling made the realised factor
    about 2.5x weaker than `imbalance_strength` / `imbalance_alpha` implied.
  - `scripts/brownian_demo.py` plus `plot_brownian_paths`, `plot_return_qq` and `plot_vol_acf`.
  - `MarketConfig` gains `price_process`, `dt`, `rng_mode`, `normal_method`, `vol_process`,
    `vol_persistence`, `leverage_corr`, `imbalance_ou_scaling` and `latency_model`, all
    range-validated. **All default to the legacy behaviour**, which is covered by a
    bit-identity regression test, so existing results and configs are unaffected.
- Statistical & research rigor:
  - `rl_execution.metrics.stats`: percentile **bootstrap CIs**, a **paired** bootstrap that
    preserves common-random-number pairing, Holm / Benjamini-Hochberg multiple-testing
    correction (uses `statsmodels` when present, exact NumPy fallback otherwise) and the
    **deflated / probabilistic Sharpe ratio** (Bailey-López de Prado).
  - `paired_is_table` now also reports the paired-bootstrap CI and a two-sided p-value of
    `vs_<bench>`; new `corrected_significance` Holm-corrects the paired tests across the whole
    regime × strategy grid so a significant regime is not cherry-picked from many tests.
  - `rl_execution.experiments.run_multiseed`: multi-seed train/eval with the **training seed**
    as the unit of analysis — an across-seed bootstrap CI (the headline) plus a within-run
    episode CI — reusing already-trained `{config_hash, seed}` models from the registry instead
    of retraining (optional `joblib` parallelism over seeds).
  - Optuna **hyperparameter optimization** (`optimize`, `make_study`, `search_space`):
    resumable SQLite-backed studies and per-algorithm search spaces, selecting on a validation
    split of seeds/regimes disjoint from the final evaluation.
  - `rlx-experiments` now also writes `results/corrected_significance.csv`; a new `rlx-multiseed`
    CLI produces the across-seed CIs, and `rlx-report` renders the corrected-significance,
    across-seed and statistical-methods sections.
  - `tracking` extra gains `statsmodels`, `optuna` and `joblib`.
- Reproducibility & experiment tracking:
  - A swappable experiment-tracking abstraction (`rl_execution.tracking`) with MLflow-local
    (default, fully offline), Weights & Biases (opt-in, lazy) and Null (CI/tests) backends,
    selected via `get_tracker`.
  - A versioned **model registry** (MLflow local file store) with run lineage — models are
    registered as numbered versions instead of overwriting `models/<agent>.*`.
  - Deterministic global seeding (`set_global_seeds`: python/numpy/torch/cuDNN/SB3), wired
    into training so a single seed reproduces a run.
  - Provenance stamping: every saved model's JSON sidecar now records
    `{git_sha, git_dirty, config_hash, data_hash, lib_versions, seed, python, platform, created_at}`.
  - `RunConfig` — a single serialisable object that fully describes a run, round-tripping to
    YAML (`rlx-train --config run.yaml --seed 3`) with pydantic range validation.
  - `rlx-train` / `rlx-experiments` now seed, log params/metrics/artifacts to the tracker and
    register a model version; new flags `--config`, `--tracker`, `--experiment`,
    `--no-register`, `--save-config`.
  - Optional `tracking` extra: `pip install -e ".[tracking]"` (mlflow, wandb, omegaconf, pydantic).
- Continuous-integration workflow (lint, format check, type check, test matrix, coverage).
- `ruff`, `black`, `mypy` and `pre-commit` configuration with enforced quality gates.
- Console entry points (`rlx-train`, `rlx-evaluate`, `rlx-experiments`, `rlx-report`) backed by
  a new `rl_execution.cli` package; the files under `scripts/` are now thin wrappers with no
  `sys.path` manipulation.
- Containerised workflow via `Dockerfile` and `docker-compose.yml` (CPU).
- Repository hygiene: `LICENSE`, `CONTRIBUTING.md`, `.gitattributes`, `.dockerignore`,
  issue/PR templates.

### Changed
- `MarketConfig` now coerces every field to its declared type at construction and raises
  `ValueError` on junk. A YAML typo such as `imbalance_ou_scaling: false# oops` previously
  reached validation as a truthy *string*, silently switching a setting on for a whole run.
- Replaced the blanket `warnings.filterwarnings("ignore")` in the command-line tools with a
  scoped helper that silences only known third-party warning noise.

## [0.1.0]

### Added
- Initial RL optimal-execution framework: limit-order-book simulator, Gymnasium environment,
  classical baselines (TWAP, VWAP, POV, Random, Almgren-Chriss), from-scratch DQN/DoubleDQN and
  Stable-Baselines3 (PPO/A2C/SAC) agents, a uniform backtesting engine, cross-regime experiment
  runners, execution-quality metrics, visualisation, a Streamlit dashboard and an auto-generated
  research report.
