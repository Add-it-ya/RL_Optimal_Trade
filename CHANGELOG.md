# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
- Replaced the blanket `warnings.filterwarnings("ignore")` in the command-line tools with a
  scoped helper that silences only known third-party warning noise.

## [0.1.0]

### Added
- Initial RL optimal-execution framework: limit-order-book simulator, Gymnasium environment,
  classical baselines (TWAP, VWAP, POV, Random, Almgren-Chriss), from-scratch DQN/DoubleDQN and
  Stable-Baselines3 (PPO/A2C/SAC) agents, a uniform backtesting engine, cross-regime experiment
  runners, execution-quality metrics, visualisation, a Streamlit dashboard and an auto-generated
  research report.
