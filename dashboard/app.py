"""Streamlit performance dashboard for the execution framework.

Run with::

    streamlit run dashboard/app.py

If experiment results exist (``results/results.pkl`` from ``run_experiments.py``) the
dashboard renders the saved comparison; otherwise it can run a quick *live* baseline
comparison on a chosen regime so the dashboard is useful even before training.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import streamlit as st

from rl_execution import viz
from rl_execution.backtest import compare_strategies, paired_is_table, results_table
from rl_execution.config import ActionType, ExecutionConfig, Side
from rl_execution.envs import ExecutionEnv
from rl_execution.experiments import build_baselines, get_regime, list_regimes
from rl_execution.utils.io import RESULTS_DIR, load_json, load_pickle

st.set_page_config(page_title="RL Optimal Execution", layout="wide")
st.title("🎯 RL for Optimal Trade Execution — Performance Dashboard")

RESULTS_PKL = RESULTS_DIR / "results.pkl"
RESULTS_CSV = RESULTS_DIR / "regime_results.csv"
REWARD_LOG = RESULTS_DIR / "reward_logs.json"

mode = st.sidebar.radio(
    "Data source",
    ["Saved experiment results", "Live baseline backtest"],
    index=0 if RESULTS_PKL.exists() else 1,
)


def render_regime_block(results_by_strat, title: str):
    table = results_table(results_by_strat)
    rl = [n for n in table.index if n in {"DQN", "DoubleDQN", "PPO", "A2C", "SAC"}]
    st.subheader(title)
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("**Metrics (sorted by implementation shortfall)**")
        st.dataframe(table.round(2), use_container_width=True)
        if "TWAP" in results_by_strat:
            st.markdown("**Paired vs TWAP** (vs_TWAP<0 = better; common random numbers)")
            st.dataframe(
                paired_is_table(results_by_strat, "TWAP").round(2), use_container_width=True
            )
    with c2:
        st.pyplot(viz.plot_rl_vs_baselines(table, rl) if rl else viz.plot_cost_comparison(table))
    c3, c4 = st.columns(2)
    with c3:
        st.pyplot(viz.plot_inventory_decay(results_by_strat))
        st.pyplot(viz.plot_reward_curve(results_by_strat))
    with c4:
        st.pyplot(viz.plot_execution_schedule(results_by_strat))
        st.pyplot(viz.plot_is_distribution(results_by_strat))


# ===================================================================== saved results
if mode == "Saved experiment results":
    if not RESULTS_PKL.exists():
        st.warning(
            "No saved results found. Run `python scripts/run_experiments.py` "
            "or switch to *Live baseline backtest*."
        )
    else:
        results = load_pickle(RESULTS_PKL)
        regimes = list(results.keys())
        regime = st.sidebar.selectbox("Regime", regimes, index=min(1, len(regimes) - 1))
        if RESULTS_CSV.exists():
            df = pd.read_csv(RESULTS_CSV)
            st.markdown("### Robustness across regimes (IS, bps — lower is better)")
            st.pyplot(viz.plot_regime_heatmap(df, value="IS_bps"))
        render_regime_block(results[regime], f"Regime: `{regime}`")
        if REWARD_LOG.exists():
            st.markdown("### Training reward curves")
            st.pyplot(viz.plot_training_curve(load_json(str(REWARD_LOG))))

# ===================================================================== live backtest
else:
    st.sidebar.markdown("### Live backtest settings")
    regime = st.sidebar.selectbox("Regime", list_regimes(), index=4)
    side = st.sidebar.selectbox("Side", ["sell", "buy"])
    inventory = st.sidebar.number_input("Inventory (shares)", 1000, 200_000, 10_000, step=1000)
    horizon = st.sidebar.slider("Horizon (steps)", 5, 60, 20)
    episodes = st.sidebar.slider("Episodes", 10, 500, 100, step=10)
    run = st.sidebar.button("Run backtest", type="primary")

    st.info(
        "Compares the classical baselines on the chosen regime. "
        "Train RL agents with `scripts/run_experiments.py` to add them here."
    )
    if run:
        exec_config = ExecutionConfig(
            total_inventory=float(inventory),
            horizon=int(horizon),
            side=Side(side),
            action_type=ActionType.CONTINUOUS,
        )
        market_config = get_regime(regime)

        def factory():
            return ExecutionEnv(market_config, exec_config)

        with st.spinner(f"Running {episodes} episodes × {len(build_baselines())} strategies ..."):
            results = compare_strategies(
                factory,
                build_baselines(),
                n_episodes=int(episodes),
                base_seed=10_000,
                progress=False,
            )
        render_regime_block(results, f"Live results — regime `{regime}` ({side})")
