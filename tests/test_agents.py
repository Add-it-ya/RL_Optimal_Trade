"""Light-weight agent tests (tiny budgets so the suite stays fast)."""

import numpy as np
import pytest

from rl_execution.agents import AgentStrategy, make_agent, required_action_type
from rl_execution.backtest import run_episode
from rl_execution.config import ExecutionConfig, MarketConfig
from rl_execution.envs import ExecutionEnv


def _env(name):
    ec = ExecutionConfig(action_type=required_action_type(name))
    return ExecutionEnv(MarketConfig(), ec)


@pytest.mark.parametrize("name", ["dqn", "doubledqn", "ppo", "a2c", "sac"])
def test_agent_builds_trains_predicts(name):
    env = _env(name)
    agent = make_agent(name, env, seed=0)
    agent.train(400)
    obs, _ = env.reset(seed=0)
    action = agent.predict(obs, deterministic=True)
    assert env.action_space.contains(
        np.asarray(action).reshape(env.action_space.shape)
        if env.action_space.shape
        else int(action)
    )


@pytest.mark.parametrize("name", ["dqn", "doubledqn"])
def test_custom_dqn_save_load(tmp_path, name):
    env = _env(name)
    agent = make_agent(name, env, seed=0)
    agent.train(400)
    path = str(tmp_path / "model.pt")
    agent.save(path)
    from rl_execution.agents.dqn import DQNAgent

    loaded = DQNAgent.load(path, env=env)
    obs, _ = env.reset(seed=1)
    assert loaded.predict(obs) == agent.predict(obs)


def test_agent_strategy_runs_episode():
    env = _env("ppo")
    agent = make_agent("ppo", env, seed=0)
    agent.train(400)
    summary, hist, rewards, _ = run_episode(env, AgentStrategy(agent, "PPO"), seed=0)
    assert summary["unexecuted_shares"] == pytest.approx(0.0, abs=1e-6)
    assert np.isfinite(rewards.sum())
