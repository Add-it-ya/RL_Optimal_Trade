# Contributing

Thanks for your interest in improving this project. This guide covers local setup and the
quality gates that run in CI.

## Development setup

```bash
# 1. Create and activate a virtual environment (any tool works).
python -m venv .venv && . .venv/Scripts/activate     # Windows
# python -m venv .venv && source .venv/bin/activate  # Linux/macOS

# 2. Install CPU-only PyTorch (skip if you want a CUDA build — see the README).
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install the package with all extras and the dev tooling.
pip install -e ".[rl,dashboard,dev]"

# 4. Install the pre-commit hooks.
pre-commit install
```

## Quality gates

The same checks run locally (via pre-commit) and in CI:

```bash
ruff check .          # lint (import order, pyflakes, pycodestyle)
black --check .       # formatting
mypy rl_execution     # static type checking
pytest                # test suite
```

Run `ruff check --fix .` and `black .` to auto-fix lint and formatting. CI enforces a test
coverage floor (see `.github/workflows/ci.yml`).

## Running the pipeline

```bash
rlx-experiments --quick      # fast smoke run
rlx-report                   # build reports/REPORT.md
streamlit run dashboard/app.py
```

## Conventions

- Keep public functions and classes type-annotated and documented.
- Add or update tests for any behaviour change; keep the suite green.
- Update `CHANGELOG.md` under `[Unreleased]` for user-visible changes.
