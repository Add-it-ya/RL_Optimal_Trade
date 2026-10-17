# RL for Optimal Trade Execution

A compact **reinforcement-learning framework for optimal trade execution**. An agent works a
large parent order over a fixed horizon, choosing at each step how much of the remaining
inventory to sell (or buy) into a simulated limit order book, so as to minimise implementation
shortfall, market impact and inventory risk. It is benchmarked against the classical
strategies used on trading desks, including the closed-form **Almgren-Chriss** optimum.

> **Status (v0.2).** The codebase was cut down to a lean core: Brownian-only market dynamics,
> two agents, three commands. The v0.1 headline results were withdrawn after an audit showed
> the reward favoured a do-nothing, dump-at-the-end policy and the published weights could no
> longer be reproduced. New results will be published from the recalibrated simulator, with
> confidence intervals across training seeds.

---

## ✨ What's inside

| Area | What's included |
|------|-----------------|
| **Environment** | A **Gymnasium** env over a simulated **limit order book**: depth ladder, spread and imbalance dynamics, convex book-walk slippage, temporary + permanent impact, commissions, optional latency. |
| **Price dynamics** | Explicit **Brownian-motion** processes with exact discretisations: **arithmetic** BM (the Almgren-Chriss world), **geometric** BM, **Ornstein-Uhlenbeck**, plus **exp-OU stochastic volatility** with clustering and a **leverage** correlation. Named RNG substreams, **Brownian-bridge** latency fills and **antithetic** variates. |
| **Actions** | Discrete grid *and* continuous Box: the fraction of remaining inventory to trade. |
| **Baselines** | **TWAP**, **VWAP** and **Almgren-Chriss** (closed-form risk-averse optimum). |
| **RL agents** | **PPO** (Stable-Baselines3) and **DQN / Double DQN** written from scratch in PyTorch. |
| **Evaluation** | Every strategy runs on the same pre-drawn market paths (**common random numbers**), so comparisons are paired. Bootstrap CIs with the **training seed as the unit of analysis**, and **Holm** correction across all comparisons. |
| **Reproducibility** | Each run folder records its config, git SHA, library versions and the **SHA-256 of its weights**; loading refuses weights that don't match. |

---

## Quickstart

```bash
pip install -e ".[rl,dev]"
```

Train, evaluate on held-out test seeds, and write a report:

```bash
rlx train --agent ppo --timesteps 200000 --seeds 0-4 --regime abm_ac_optimal --experiment e1
```
```bash
rlx evaluate --experiment e1 --episodes 2000 --antithetic
```
```bash
rlx report --experiment e1
```

- `rlx train` writes one folder per seed to `runs/<experiment>/<Agent>/seed_<k>/`. Seeds that
  are already complete are skipped, so an interrupted job can simply be re-launched.
- `rlx evaluate` scores every run of the experiment plus the baselines on a block of test
  seeds that training never sees, and writes `results/<experiment>/` (per-episode CSV, summary,
  figures).
- `rlx report` turns that into `results/<experiment>/REPORT.md`: IS level and spread per
  strategy, the paired gap to TWAP with a 95% CI, and Holm-adjusted p-values.

`rlx train --config run.yaml` accepts a full `RunConfig` in YAML (see `config/default.yaml` for
the market and execution sections); explicit flags override it.

Use the library directly:

```python
from rl_execution.config import ExecutionConfig
from rl_execution.envs import ExecutionEnv
from rl_execution.baselines import TWAP, AlmgrenChriss
from rl_execution.backtest import compare_strategies, paired_is_table
from rl_execution.experiments import get_regime

factory = lambda: ExecutionEnv(get_regime("abm_ac_optimal"), ExecutionConfig(side="sell"))
results = compare_strategies(factory, {"TWAP": TWAP(), "AC": AlmgrenChriss(1e-6)},
                             n_episodes=400, progress=False, antithetic=True)
print(paired_is_table(results, benchmark="TWAP", block=2))
```

---

## 🧱 Project structure

```
rl_execution/
├── config.py             # MarketConfig / ExecutionConfig / RewardConfig / RunConfig
├── envs/
│   ├── brownian.py       # RNG substreams, Wiener increments, Brownian bridge, antithetics
│   ├── processes.py      # ABM / GBM / OU price processes + exp-OU stochastic volatility
│   ├── lob.py            # order-book snapshot + market-order book walking
│   ├── market.py         # market simulator: pre-drawn exogenous paths + impact overlay
│   └── execution_env.py  # the Gymnasium environment
├── baselines/            # TWAP, VWAP, Almgren-Chriss
├── agents/               # PPO (SB3 wrapper) and DQN / Double DQN (PyTorch)
├── backtest/             # uniform backtesting engine, paired comparisons
├── metrics/              # execution-quality metrics, bootstrap CIs, multiple testing
├── experiments/          # regime presets, domain randomisation, cross-regime runner
├── viz/                  # plotting
├── training.py           # training + run folders (config, weights, provenance)
└── cli.py                # rlx train | evaluate | report
scripts/    brownian_demo.py
config/     default.yaml
tests/      pytest suite
```

---

## 🌊 Brownian price dynamics

`MarketConfig.price_process` selects how the mid-price moves. Each option is an explicit SDE
with an **exact** discretisation:

| Process | SDE | Exact step |
|---|---|---|
| `abm` | `dS = mu S0 dt + sigma S0 dW` | `S + mu S0 dt + sigma S0 sqrt(dt) Z` |
| `gbm` | `dS = mu S dt + sigma S dW` | `S exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z)` |
| `ou`  | `dS = kappa (S0 - S) dt + sigma S0 dW` | mean-reverting, stationary variance preserved |

Layered on top: `vol_process="expou"` gives an exponential-OU log-volatility with genuine
**clustering** (so the env's `volatility` feature carries forecast value), `leverage_corr < 0`
makes volatility rise after the price falls, and `latency_steps > 0` prices a delayed fill on
the **Brownian bridge** between the two decision steps.

Two properties matter beyond realism:

* **Common random numbers hold.** The whole exogenous market (price, volatility, imbalance,
  spread, volume, per-level depth) is drawn at `reset()`, and market impact is an additive
  overlay. Nothing an agent does can consume randomness that would move the market, so two
  strategies on one seed face bit-identical markets and paired differences cancel the shared
  price risk.
* **Ground truth for the learner.** Under `abm` with no alpha and linear impact, the
  closed-form **Almgren-Chriss** schedule is provably optimal. The `abm_ac_optimal` regime
  turns "did the agent solve the problem?" into a measurable schedule distance
  (`schedule_gap_table`) rather than another cost comparison.

```bash
python scripts/brownian_demo.py          # paths, QQ-plot, vol clustering, quadratic variation
```

---

## ⚡ Training on a GPU machine

The policies are small MLPs and the environment step runs on the CPU, so for this code a
machine's **CPU cores** matter more than its GPU: run several seeds in parallel, one process
each, with `OMP_NUM_THREADS=1` so they don't fight over threads. A CUDA build of PyTorch is
only needed for GPU work:

```bash
nvidia-smi
```
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Pick the `cuXXX` wheel that matches the CUDA version `nvidia-smi` reports.

## 🛠 Troubleshooting

- **`OMP: Error #15: ... libiomp5md.dll already initialized`** (Windows + Anaconda + PyTorch).
  The package sets `KMP_DUPLICATE_LIB_OK=TRUE` on Windows before torch is imported. If you
  import torch *before* `rl_execution`, set that environment variable yourself.

## ⚠️ Interpreting the numbers

The regime presets embed an **intentionally amplified** order-book-imbalance signal
(`imbalance_alpha`), far more predictive than real order-book signals, so that a
state-reactive policy has something to learn. Basis-point improvements obtained with it are
**not** realistic P&L. Report any improvement together with the signal strength that produced
it, and use `abm_ac_optimal` (no signal) to check that a learner recovers the known optimum.

## ✅ Testing

```bash
python -m pytest -q
```

## 📦 Requirements

Python ≥ 3.10; numpy / pandas / scipy / gymnasium / matplotlib / seaborn, plus `torch` and
`stable-baselines3` for the agents (the `rl` extra). See `requirements.txt`.

## License

MIT.
