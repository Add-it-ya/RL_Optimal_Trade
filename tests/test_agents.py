"""Light-weight agent tests (tiny budgets so the suite stays fast)."""

import numpy as np
import pytest

from rl_execution.agents import AgentStrategy, make_agent, required_action_type
from rl_execution.backtest import run_episode
from rl_execution.config import ExecutionConfig, MarketConfig
from rl_execution.envs import ExecutionEnv

# Small enough to be fast, large enough that the DQN agents actually take gradient steps
# (their default ``learning_starts`` of 1000 would make a 400-step test a no-op).
_FAST = {
    "dqn": dict(learning_starts=50, batch_size=32),
    "doubledqn": dict(learning_starts=50, batch_size=32),
    "ppo": dict(n_steps=64, batch_size=32, n_epochs=2),
}


def _env(name):
    ec = ExecutionConfig(action_type=required_action_type(name))
    return ExecutionEnv(MarketConfig(), ec)


@pytest.mark.parametrize("name", ["dqn", "doubledqn", "ppo"])
def test_agent_builds_trains_predicts(name):
    env = _env(name)
    agent = make_agent(name, env, seed=0, **_FAST[name])
    agent.train(400)
    obs, _ = env.reset(seed=0)
    action = agent.predict(obs, deterministic=True)
    assert env.action_space.contains(
        np.asarray(action).reshape(env.action_space.shape)
        if env.action_space.shape
        else int(action)
    )


def test_dqn_takes_gradient_steps():
    env = _env("doubledqn")
    agent = make_agent("doubledqn", env, seed=0, **_FAST["doubledqn"])
    agent.train(400)
    assert len(agent.train_log) > 0


@pytest.mark.parametrize("name", ["dqn", "doubledqn"])
def test_custom_dqn_save_load(tmp_path, name):
    env = _env(name)
    agent = make_agent(name, env, seed=0, **_FAST[name])
    agent.train(400)
    path = str(tmp_path / "model.pt")
    agent.save(path)
    from rl_execution.agents.dqn import DQNAgent

    loaded = DQNAgent.load(path, env=env)
    obs, _ = env.reset(seed=1)
    assert loaded.predict(obs) == agent.predict(obs)


def test_ppo_save_load_round_trip(tmp_path):
    env = _env("ppo")
    agent = make_agent("ppo", env, seed=0, **_FAST["ppo"])
    agent.train(128)
    path = str(tmp_path / "model.zip")
    agent.save(path)
    from rl_execution.agents.ppo import PPOAgent

    loaded = PPOAgent.load(path, env=env)
    obs, _ = env.reset(seed=1)
    assert np.allclose(loaded.predict(obs), agent.predict(obs))


def test_agent_names_normalise():
    from rl_execution.agents import normalize_agent_name

    assert normalize_agent_name("Double-DQN") == "doubledqn"
    assert normalize_agent_name("DDQN") == "doubledqn"
    with pytest.raises(KeyError):
        normalize_agent_name("sac")


def test_agent_strategy_runs_episode():
    env = _env("ppo")
    agent = make_agent("ppo", env, seed=0, **_FAST["ppo"])
    agent.train(128)
    summary, hist, rewards, _ = run_episode(env, AgentStrategy(agent, "PPO"), seed=0)
    assert summary["unexecuted_shares"] == pytest.approx(0.0, abs=1e-6)
    assert np.isfinite(rewards.sum())
