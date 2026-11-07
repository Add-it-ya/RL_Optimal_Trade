"""Reproducibility: deterministic seeding, provenance stamping and RunConfig round-trip."""

import pytest

from rl_execution.agents import required_action_type
from rl_execution.config import ExecutionConfig, MarketConfig, RunConfig, Side
from rl_execution.envs import ExecutionEnv
from rl_execution.experiments.regimes import get_regime
from rl_execution.training import find_runs, load_run, run_dir, save_run, train_agent
from rl_execution.utils.config_io import dump_run_config, load_run_config
from rl_execution.utils.io import load_json
from rl_execution.utils.provenance import capture_provenance, config_hash, file_sha256, git_sha

# Tiny DQN budget that still exercises gradient updates (timesteps > learning_starts).
_FAST = dict(learning_starts=20, batch_size=8, buffer_size=500)


def _train_tiny(seed, **kwargs):
    return train_agent(
        "dqn",
        total_timesteps=200,
        regime="normal_liquidity",
        randomized=False,
        seed=seed,
        device="cpu",
        agent_kwargs=dict(_FAST),
        **kwargs,
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
    run1, run2 = _train_tiny(123), _train_tiny(123)
    agent1, agent2 = run1.agent, run2.agent

    # identical training -> bit-identical learned weights ...
    sd1, sd2 = agent1.q.net.state_dict(), agent2.q.net.state_dict()
    for key in sd1:
        assert torch.equal(sd1[key], sd2[key]), f"weights diverged at {key}"
    # ... identical per-episode training log ...
    assert run1.episodes == run2.episodes and len(run1.episodes) > 0
    # ... and identical first-N actions on a fixed observation sequence.
    assert _first_actions(agent1) == _first_actions(agent2)


def test_different_seeds_diverge():
    """Sanity check: a different seed actually changes the learned policy."""
    torch = pytest.importorskip("torch")
    agent1, agent2 = _train_tiny(1).agent, _train_tiny(2).agent
    sd1, sd2 = agent1.q.net.state_dict(), agent2.q.net.state_dict()
    assert any(not torch.equal(sd1[k], sd2[k]) for k in sd1)


def test_capture_provenance_has_lineage_fields():
    rc = RunConfig(agent="dqn", seed=7)
    prov = capture_provenance(seed=7, config=rc)
    expected = {
        "git_sha",
        "git_dirty",
        "config_hash",
        "seed",
        "lib_versions",
        "python",
        "platform",
        "created_at",
    }
    assert expected <= set(prov)
    assert prov["seed"] == 7
    assert prov["config_hash"] == config_hash(rc)
    assert "numpy" in prov["lib_versions"]


def _saved_run(tmp_path, seed=5):
    rc = RunConfig(
        agent="dqn",
        seed=seed,
        regime="normal_liquidity",
        randomized=False,
        timesteps=200,
        agent_kwargs=dict(_FAST),
    )
    result = _train_tiny(seed, eval_every=100, val_episodes=4)
    out = save_run(result, rc, run_dir(tmp_path, rc.experiment, rc.agent, seed))
    return result.agent, rc, out


def test_run_folder_records_provenance(tmp_path):
    """A run folder carries its config, git SHA, config hash, versions and weight hash."""
    pytest.importorskip("torch")
    _, rc, out = _saved_run(tmp_path)

    assert out == tmp_path / "default" / "DQN" / "seed_5"
    assert {"config.yaml", "model.pt", "provenance.json", "train_log.csv", "val_curve.csv"} <= {
        p.name for p in out.iterdir()
    }
    prov = load_json(out / "provenance.json")
    assert prov["seed"] == 5
    assert prov["git_sha"]  # a SHA, or the literal "unknown" outside a checkout
    assert prov["config_hash"] == config_hash(rc)
    assert prov["weights_sha256"] == file_sha256(out / "model.pt")
    assert "numpy" in prov["lib_versions"]
    assert prov["selected_step"] in (0, 100, 200)
    assert prov["selected_val_objective_bps"] <= prov["initial_val_objective_bps"]
    assert find_runs(tmp_path) == [out]


def test_load_run_round_trips_the_policy(tmp_path):
    pytest.importorskip("torch")
    agent, rc, out = _saved_run(tmp_path)
    loaded, rc2 = load_run(out)
    assert rc2.to_dict() == rc.to_dict()
    assert _first_actions(loaded) == _first_actions(agent)


def test_load_run_rejects_swapped_weights(tmp_path):
    """Weights that do not match provenance.json are refused, not silently loaded."""
    pytest.importorskip("torch")
    _, _, out = _saved_run(tmp_path, seed=5)
    _train_tiny(6).agent.save(str(out / "model.pt"))  # a different run's weights under this folder
    with pytest.raises(ValueError, match="do not match"):
        load_run(out)


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
        {"agent": "sac"},
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
