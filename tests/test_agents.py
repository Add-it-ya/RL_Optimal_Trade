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
@pytest.mark.parametrize("dueling", [False, True])
def test_custom_dqn_save_load(tmp_path, name, dueling):
    env = _env(name)
    agent = make_agent(name, env, seed=0, dueling=dueling, **_FAST[name])
    agent.train(400)
    path = str(tmp_path / "model.pt")
    agent.save(path)
    from rl_execution.agents.dqn import DQNAgent

    loaded = DQNAgent.load(path, env=env)
    assert loaded._dueling is dueling
    obs, _ = env.reset(seed=1)
    assert loaded.predict(obs) == agent.predict(obs)


def test_dueling_head_centres_the_advantages():
    """Q = V + A - mean(A): the mean Q over actions equals the state value."""
    torch = pytest.importorskip("torch")
    from rl_execution.agents.dqn import QNetwork

    net = QNetwork(obs_dim=7, n_actions=10, dueling=True)
    x = torch.randn(5, 7)
    q = net(x)
    assert q.shape == (5, 10)
    head = net.net[-1]
    value = head.value(net.net[:-1](x))
    assert torch.allclose(q.mean(dim=1, keepdim=True), value, atol=1e-6)


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


def test_dqn_target_tracks_online_network_by_polyak_averaging():
    torch = pytest.importorskip("torch")
    agent = make_agent("doubledqn", _env("doubledqn"), seed=0, **_FAST["doubledqn"], tau=0.5)
    before = [p.detach().clone() for p in agent.q_target.net.parameters()]
    agent.train(80)  # a few updates past learning_starts (one every train_freq steps)
    after = list(agent.q_target.net.parameters())
    online = list(agent.q.net.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))  # the target moved ...
    assert any(not torch.equal(a, o) for a, o in zip(after, online))  # ... but lags online


@pytest.mark.parametrize("name", ["ppo", "doubledqn"])
def test_weights_round_trip(name):
    agent = make_agent(name, _env(name), seed=0, **_FAST[name])
    snapshot = agent.get_weights()
    agent.train(200)
    obs, _ = _env(name).reset(seed=3)
    agent.set_weights(snapshot)
    fresh = make_agent(name, _env(name), seed=0, **_FAST[name])
    assert np.allclose(agent.predict(obs), fresh.predict(obs))


def test_ppo_trains_on_a_vectorised_env():
    from rl_execution.training import train_agent

    result = train_agent(
        "ppo",
        total_timesteps=256,
        regime="abm_ac_optimal",
        randomized=False,
        seed=0,
        n_envs=4,
        agent_kwargs=dict(n_steps=32, batch_size=64, n_epochs=1),
    )
    assert result.agent.model.n_envs == 4
    # 256 steps split over 4 envs: at least 64 // 20 = 3 finished episodes from each
    assert len(result.episodes) >= 4 * (256 // 4 // 20)
    steps = [row["step"] for row in result.episodes]
    assert steps == sorted(steps)


def test_checkpoint_selection_keeps_the_best_validation_score():
    from rl_execution.training import train_agent

    result = train_agent(
        "doubledqn",
        total_timesteps=300,
        regime="abm_ac_optimal",
        randomized=False,
        seed=0,
        eval_every=100,
        val_episodes=10,
        agent_kwargs=dict(_FAST["doubledqn"]),
    )
    curve = {row["step"]: row["objective_bps"] for row in result.validation}
    assert sorted(curve) == [0, 100, 200, 300]
    assert result.selected_step == min(curve, key=curve.get)

    # the returned agent carries the selected weights: re-scoring reproduces that score
    from rl_execution.agents import AgentStrategy
    from rl_execution.backtest import evaluate
    from rl_execution.experiments.runner import VALIDATION_BASE_SEED
    from rl_execution.training import make_env_factory

    factory = make_env_factory("doubledqn", regime="abm_ac_optimal")
    rescored = evaluate(
        factory, AgentStrategy(result.agent), 10, VALIDATION_BASE_SEED, antithetic=True
    )
    assert rescored.aggregate["objective_bps_mean"] == pytest.approx(curve[result.selected_step])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_untrained_ppo_starts_as_twap(seed):
    """With the TWAP-relative action an untrained policy (mean action ~0) trades at the
    TWAP pace -- not "wait, then dump", which is where v0.1's [0, 1] action started."""
    env = _env("ppo")
    agent = make_agent("ppo", env, seed=seed)
    _, hist, _, _ = run_episode(env, AgentStrategy(agent, "PPO"), seed=0)
    schedule = hist["shares"].to_numpy() / env.total_inventory
    assert len(schedule) == env.horizon
    assert np.allclose(schedule, 1.0 / env.horizon, atol=0.01)
