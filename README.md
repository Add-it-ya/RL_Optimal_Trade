# RL for Optimal Trade Execution

A production-quality, modular **Reinforcement Learning framework for optimal trade
execution**. It trains agents to work a large parent order over a fixed horizon so as to
minimise execution cost, market impact, slippage and inventory risk — and benchmarks them
against the classical execution strategies used on real trading desks.

> **Goal.** Learn a policy that decides *how much of the remaining inventory to execute at
> each step* and consistently beats TWAP, VWAP and Random execution on implementation
> shortfall and execution cost, while staying robust across market regimes.

---

## ✨ Features

| Area | What's included |
|------|-----------------|
| **Environment** | Custom **Gymnasium** env over a simulated **limit order book** (depth ladder, spread & imbalance dynamics, convex book-walk slippage). State = inventory, time, mid-return, spread, volatility, imbalance, depth, previous action. |
| **Actions** | **Discrete** grid *and* **continuous** Box — fraction of remaining inventory to trade. |
| **Reward** | Negative implementation-shortfall contribution (slippage + spread + **temporary impact**), **permanent impact**, transaction costs, **inventory-risk** penalty, unexecuted-inventory penalty. |
| **Baselines** | **TWAP, VWAP, POV, Random, Almgren-Chriss** (closed-form risk-averse optimal). |
| **RL agents** | **DQN** & **Double DQN** (from scratch, PyTorch) + **PPO, A2C, SAC** (Stable-Baselines3). |
| **Metrics** | Implementation shortfall, execution cost, market-impact cost, Sharpe, average fill price, inventory trajectory. |
| **Experiments** | Volatility (low/med/high), liquidity (thin/normal/deep), trend (bull/bear/sideways) + **domain-randomised** training. |
| **Advanced** | **Real LOB data** replay (LOBSTER/CSV), **multi-agent** simulation, latency, transaction costs, risk-adjusted rewards. |
| **Deliverables** | Backtesting engine, trained models, auto-generated **research report**, **Streamlit dashboard**, test suite. |

---

## Quickstart

```bash
# 1. install
pip install -r requirements.txt          # or: pip install -e ".[rl,dashboard,dev]"

# 2. fast end-to-end smoke run (trains tiny agents, evaluates, makes figures)
python scripts/run_experiments.py --quick

# 3. full run (trains all 5 agents with domain randomisation, evaluates across regimes)
python scripts/run_experiments.py --timesteps 120000 --episodes 200

# 4. build the research report and launch the dashboard
python scripts/make_report.py
streamlit run dashboard/app.py
```

Train / evaluate a single agent:

```bash
python scripts/train.py --agent ppo --timesteps 100000 --randomized
python scripts/evaluate.py --regime bear --episodes 200
```

Use the library directly:

```python
from rl_execution.config import ExecutionConfig, MarketConfig
from rl_execution.envs import ExecutionEnv
from rl_execution.baselines import TWAP, AlmgrenChriss
from rl_execution.backtest import compare_strategies, results_table

factory = lambda: ExecutionEnv(MarketConfig(drift=-0.0008), ExecutionConfig(side="sell"))
results = compare_strategies(factory, {"TWAP": TWAP(), "AC": AlmgrenChriss(1e-7)},
                             n_episodes=200, progress=False)
print(results_table(results))
```

---

## 🧱 Project structure

```
rl_execution/
├── config.py             # MarketConfig / ExecutionConfig / RewardConfig
├── envs/
│   ├── lob.py            # order-book snapshot + market-order book-walking
│   ├── market.py         # stochastic market simulator + impact model
│   ├── execution_env.py  # the Gymnasium environment
│   └── multi_agent.py    # multi-agent (shared-impact) simulation
├── baselines/            # TWAP, VWAP, POV, Random, Almgren-Chriss
├── agents/               # DQN & DoubleDQN (PyTorch) + SB3 PPO/A2C/SAC wrappers
├── metrics/              # execution-quality metrics
├── backtest/             # uniform backtesting engine
├── experiments/          # regime presets + cross-regime runners + domain randomisation
├── data/                 # real/historical LOB loaders + replay simulator
├── viz/                  # plotting (inventory decay, schedules, costs, heatmaps, …)
├── utils/                # paths, IO, YAML config
└── training.py           # train / save / load helpers
scripts/    train.py · evaluate.py · run_experiments.py · make_report.py
dashboard/  app.py        # Streamlit performance dashboard
config/     default.yaml
tests/      pytest suite
```

---

## 🧠 How the agent can beat the baselines

The simulator embeds a small, persistent **order-book-imbalance alpha**: imbalance weakly
predicts the next mid move. Static schedules (TWAP/VWAP/POV) ignore the state entirely, and
Random just front-loads on average. A **state-reactive RL policy** learns to (a) trade more
when the book signals favourable moves and hold back otherwise, (b) trade off market impact
against price/inventory risk (à la Almgren-Chriss), and (c) adapt to drift and liquidity.
Drift magnitudes are deliberately kept small relative to per-step volatility so that
*timing skill*, not trivial front-loading, is what differentiates strategies.

Implementation shortfall and execution cost are reported in **basis points of arrival
notional**, signed so **lower = better** for both buy and sell orders.

---

## 📊 Outputs

Running the pipeline writes:

- `results/regime_results.csv` — tidy metrics for every (strategy × regime).
- `results/results.pkl`, `results/summary.json`, `results/reward_logs.json`.
- `models/<agent>.{pt,zip}` + `<agent>.json` sidecars (trained models).
- `reports/figures/*.png` and `reports/REPORT.md` — the research report.

---

## 🔬 Using real LOB data

```python
from rl_execution.data import load_lobster, HistoricalMarketSource
from rl_execution.envs import ExecutionEnv
from rl_execution.config import MarketConfig, ExecutionConfig

df = load_lobster("AAPL_orderbook.csv", levels=5)          # or load_lob_csv(...)
src = HistoricalMarketSource(df, MarketConfig())
env = ExecutionEnv(MarketConfig(), ExecutionConfig(), market_source=src)
```

The replay simulator overlays the configured market-impact model on top of the recorded
prices, so the same agents and baselines run unchanged on historical data.

---

## ⚡ GPU / CUDA

All agents accept a `device` argument (`"cpu"` default, `"cuda"`, or `"auto"`), exposed on
the scripts via `--device`:

```bash
python scripts/train.py --agent sac --timesteps 100000 --device cuda
python scripts/run_experiments.py --device cuda
```

**Recommendation: keep `--device cpu`.** The policies here are tiny 64×64 MLPs and the
throughput bottleneck is the environment step (CPU), not matrix multiplies. Moving tiny
tensors to the GPU every step usually makes training *slower* due to transfer latency —
this is the standard guidance for MLP-based RL (Stable-Baselines3 warns about it too). A
GPU helps when networks are large or observations are images, neither of which applies here.

The bundled `requirements.txt` / the wheel installed in this repo is **CPU-only PyTorch**.
To actually use CUDA you must install a matching CUDA build first, e.g.:

```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your CUDA
python -c "import torch; print(torch.cuda.is_available())"             # expect: True
```

## 🛠 Troubleshooting

- **`OMP: Error #15: ... libiomp5md.dll already initialized`** (common on Windows +
  Anaconda + PyTorch). The package sets `KMP_DUPLICATE_LIB_OK=TRUE` in
  `rl_execution/__init__.py` before torch is imported, which resolves it. If you import
  torch *before* `rl_execution`, set that env var yourself first.
- **`torch.cuda.is_available()` is False** — expected; the bundled wheel is CPU-only (see
  the GPU section above to install a CUDA build).
- **RL results look poor / agents "do nothing then dump"** — undertrained. Use ≥60k steps
  for the value-based agents (`--timesteps 80000`).

## ⚠️ Interpreting the numbers (important)

The simulator embeds an **intentionally amplified** order-book-imbalance alpha so the RL
learning signal is unambiguous. The absolute basis-point improvements (e.g. RL beating
TWAP by hundreds of bps) are therefore **optimistic and not realistic P&L** — real signals
are far weaker. The meaningful result is the **relative ordering and robustness** of
strategies, and the methodology (paired evaluation, regime sweeps). See
`reports/REPORT.md` §6 for the full findings, discussion and limitations.

## ✅ Testing

```bash
python -m pytest -q
```

## 📦 Requirements

Python ≥ 3.9; core stack (numpy/pandas/scipy/gymnasium/matplotlib/seaborn) plus
`torch` and `stable-baselines3` for the RL agents and `streamlit` for the dashboard.
See `requirements.txt`.

## License

MIT.
