# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
