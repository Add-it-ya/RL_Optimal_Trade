"""Reproducibility: deterministic seeding, provenance stamping and RunConfig round-trip."""

import pytest

from rl_execution.agents import required_action_type
from rl_execution.config import ExecutionConfig, MarketConfig, RunConfig, Side
from rl_execution.envs import ExecutionEnv
from rl_execution.experiments.regimes import get_regime
from rl_execution.training import save_agent, train_agent
from rl_execution.utils.config_io import dump_run_config, load_run_config
from rl_execution.utils.io import load_json
from rl_execution.utils.provenance import capture_provenance, config_hash, git_sha

# Tiny DQN budget that still exercises gradient updates (timesteps > learning_starts).
_FAST = dict(learning_starts=20, batch_size=8, buffer_size=500, target_update_interval=50)


def _train_tiny(seed):
    return train_agent(
        "dqn",
        total_timesteps=200,
        regime="normal_liquidity",
        randomized=False,
        seed=seed,
        device="cpu",
        agent_kwargs=dict(_FAST),
    )


def _first_actions(agent, n=6, seed=7):
    env = ExecutionEnv(
        get_regime("normal_liquidity"),
        ExecutionConfig(action_type=required_action_type("dqn")),
    )
    obs, _ = env.reset(seed=seed)
    actions = []
    for _ in range(n):
        a = agent.predict(obs, deterministic=True)
        actions.append(int(a))
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    return actions


def test_seeding_is_deterministic():
    """Two runs with the same seed produce identical weights, curves and actions (CPU)."""
    torch = pytest.importorskip("torch")
    agent1, rewards1 = _train_tiny(123)
    agent2, rewards2 = _train_tiny(123)

    # identical training -> bit-identical learned weights ...
    sd1, sd2 = agent1.q.net.state_dict(), agent2.q.net.state_dict()
    for key in sd1:
        assert torch.equal(sd1[key], sd2[key]), f"weights diverged at {key}"
    # ... identical per-episode reward curve ...
    assert rewards1 == pytest.approx(rewards2)
    # ... and identical first-N actions on a fixed observation sequence.
    assert _first_actions(agent1) == _first_actions(agent2)


def test_different_seeds_diverge():
    """Sanity check: a different seed actually changes the learned policy."""
    torch = pytest.importorskip("torch")
    agent1, _ = _train_tiny(1)
    agent2, _ = _train_tiny(2)
    sd1, sd2 = agent1.q.net.state_dict(), agent2.q.net.state_dict()
    assert any(not torch.equal(sd1[k], sd2[k]) for k in sd1)


def test_capture_provenance_has_lineage_fields():
    rc = RunConfig(agent="dqn", seed=7)
    prov = capture_provenance(seed=7, config=rc)
    expected = {
        "git_sha",
        "git_dirty",
        "config_hash",
        "data_hash",
        "seed",
        "lib_versions",
        "python",
        "platform",
        "created_at",
    }
    assert expected <= set(prov)
    assert prov["seed"] == 7
    assert prov["config_hash"] == config_hash(rc)
    assert prov["data_hash"] is None  # pure-synthetic until the data pipeline (Step 3)
    assert "numpy" in prov["lib_versions"]


def test_provenance_sidecar(tmp_path, monkeypatch):
    """A saved model's JSON sidecar carries git_sha, config_hash, lib_versions and seed."""
    pytest.importorskip("torch")
    monkeypatch.setattr("rl_execution.training.MODELS_DIR", tmp_path)
    rc = RunConfig(agent="dqn", seed=5, regime="normal_liquidity", randomized=False)
    agent, _ = _train_tiny(5)
    save_agent(agent, "dqn", tag="dqn", seed=5, config=rc)

    sidecar = load_json(str(tmp_path / "dqn.json"))
    prov = sidecar["provenance"]
    assert prov["seed"] == 5
    assert prov["git_sha"]  # a SHA, or the literal "unknown" outside a checkout
    assert prov["config_hash"] == config_hash(rc)
    assert "numpy" in prov["lib_versions"]


def test_runconfig_yaml_roundtrip(tmp_path):
    rc = RunConfig(
        agent="ppo",
        seed=3,
        timesteps=1234,
        market=MarketConfig(volatility=0.03),
        execution=ExecutionConfig(horizon=15, side=Side.BUY),
    )
    path = tmp_path / "run.yaml"
    dump_run_config(rc, str(path))
    rc2 = load_run_config(str(path))

    assert rc2.to_dict() == rc.to_dict()
    assert rc2.execution.side is Side.BUY
    assert config_hash(rc2) == config_hash(rc)


def test_runconfig_validation_accepts_defaults():
    RunConfig().validate()  # should not raise


@pytest.mark.parametrize(
    "kwargs",
    [
        {"market": MarketConfig(imbalance_alpha=2.0)},
        {"execution": ExecutionConfig(horizon=0)},
        {"timesteps": 0},
        {"tracker": "bogus"},
    ],
)
def test_runconfig_validation_rejects_bad_ranges(kwargs):
    with pytest.raises(ValueError):
        RunConfig(**kwargs).validate()


def test_config_hash_is_stable_and_sensitive():
    rc = RunConfig(agent="dqn", seed=1)
    assert config_hash(rc) == config_hash(RunConfig(agent="dqn", seed=1))
    assert config_hash(rc) != config_hash(RunConfig(agent="dqn", seed=2))


def test_git_sha_returns_string():
    sha = git_sha()
    assert isinstance(sha, str) and sha  # SHA in a checkout, else "unknown"
