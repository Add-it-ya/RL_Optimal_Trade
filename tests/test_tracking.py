"""Experiment-tracking abstraction, MLflow-local model registry and the W&B adapter.

The MLflow tests use a fully isolated local SQLite store under ``tmp_path`` (no network); the
W&B adapter is exercised against an injected fake ``wandb`` module so it needs no account.
"""

import sys
import types

import pytest

from rl_execution.agents import required_action_type
from rl_execution.config import ExecutionConfig
from rl_execution.envs import ExecutionEnv
from rl_execution.experiments.regimes import get_regime
from rl_execution.tracking import (
    MLflowTracker,
    NullTracker,
    WandbTracker,
    flatten_dict,
    get_tracker,
)
from rl_execution.tracking.registry import (
    load_registered_agent,
    register_model,
    resolve_version,
)


@pytest.fixture
def mlflow_store(tmp_path, monkeypatch):
    """Isolate MLflow's tracking DB + artifact root under a temp dir (offline)."""
    pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", (tmp_path / "mlruns").as_uri())
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    return tmp_path


# --------------------------------------------------------------------------- base / factory
def test_null_tracker_is_noop():
    t = NullTracker()
    with t as ctx:
        assert ctx is t
        ctx.start_run("r", {"a": 1}, {"git_sha": "x"})
        ctx.log_params({"a": 1})
        ctx.log_metrics({"m": 1.0}, step=0)
        ctx.log_artifact("nonexistent")
        assert ctx.log_model("p", "n", {}) == ""


def test_flatten_dict():
    assert flatten_dict({"a": {"b": 1}, "c": 2}) == {"a.b": 1, "c": 2}
    assert flatten_dict({}) == {}


def test_get_tracker_null():
    assert isinstance(get_tracker("null"), NullTracker)


def test_get_tracker_mlflow_missing_falls_back_to_null(monkeypatch):
    # A None entry in sys.modules makes `import mlflow` raise ImportError.
    monkeypatch.setitem(sys.modules, "mlflow", None)
    assert isinstance(get_tracker("mlflow"), NullTracker)


def test_get_tracker_auto_prefers_mlflow(mlflow_store):
    t = get_tracker("auto", experiment="t")
    try:
        assert isinstance(t, MLflowTracker)
    finally:
        t.finish()


# --------------------------------------------------------------------------- MLflow tracker
def test_mlflow_tracker_logs_params_and_metrics(mlflow_store):
    import mlflow
    from mlflow.tracking import MlflowClient

    with get_tracker("mlflow", experiment="t") as t:
        t.start_run("run1", {"market": {"volatility": 0.02}, "seed": 1}, {"git_sha": "abc"})
        t.log_metrics({"train/episode_reward": -5.0}, step=0)
        t.log_metrics({"train/episode_reward": -3.0}, step=1)

    exp = mlflow.get_experiment_by_name("t")
    runs = MlflowClient().search_runs([exp.experiment_id])
    assert len(runs) == 1
    assert runs[0].data.params["market.volatility"] == "0.02"
    assert runs[0].data.params["seed"] == "1"
    assert runs[0].data.metrics["train/episode_reward"] == -3.0
    assert runs[0].data.tags["git_sha"] == "abc"


# --------------------------------------------------------------------------- model registry
def test_register_increments_versions(mlflow_store, tmp_path):
    f = tmp_path / "m.pt"
    f.write_text("weights")
    v1 = register_model(str(f), "demo", {"seed": 1})
    v2 = register_model(str(f), "demo", {"seed": 2})
    assert (v1, v2) == ("1", "2")
    assert resolve_version("demo", "latest") == "2"


def test_registry_roundtrip(mlflow_store, tmp_path, monkeypatch):
    """register -> load_registered('dqn','latest') rebuilds a usable agent (offline)."""
    pytest.importorskip("torch")
    from rl_execution.training import model_artifact_path, save_agent, train_agent

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr("rl_execution.training.MODELS_DIR", models_dir)

    agent, _ = train_agent(
        "dqn",
        total_timesteps=120,
        regime="normal_liquidity",
        randomized=False,
        seed=1,
        agent_kwargs=dict(learning_starts=20, batch_size=8, buffer_size=300),
    )
    save_agent(agent, "dqn", tag="dqn", seed=1)
    version = register_model(str(model_artifact_path("dqn", "dqn")), "dqn", {"seed": 1})
    assert version == "1"

    env = ExecutionEnv(
        get_regime("normal_liquidity"),
        ExecutionConfig(action_type=required_action_type("dqn")),
    )
    loaded = load_registered_agent("dqn", env, "latest")
    obs, _ = env.reset(seed=2)
    assert int(loaded.predict(obs)) == int(agent.predict(obs))


def test_resolve_version_unknown_model_raises(mlflow_store):
    with pytest.raises(ValueError):
        resolve_version("does-not-exist", "latest")


# --------------------------------------------------------------------------- W&B (fake)
def _make_fake_wandb():
    """A minimal stand-in for the wandb module covering the methods WandbTracker uses."""

    class FakeArtifact:
        def __init__(self, name, type, metadata=None):
            self.name, self.type, self.metadata, self.files = name, type, metadata, []

        def add_file(self, path):
            self.files.append(path)

    class FakeConfig:
        def __init__(self):
            self.data = {}

        def update(self, d, allow_val_change=False):
            self.data.update(d)

    class FakeLogged:
        version = "v7"

        def wait(self):
            pass

    class FakeRun:
        def __init__(self, **kw):
            self.kw, self.config = kw, FakeConfig()
            # real wandb seeds run.config from the init `config=` kwarg
            self.config.data.update(kw.get("config") or {})
            self.logged, self.artifacts, self.finished = [], [], False

        def log(self, metrics, step=None):
            self.logged.append((dict(metrics), step))

        def log_artifact(self, art, aliases=None):
            self.artifacts.append((art, aliases))
            return FakeLogged()

        def finish(self, exit_code=0):
            self.finished, self.exit_code = True, exit_code

    fake = types.SimpleNamespace(run=None, Artifact=FakeArtifact)

    def _init(**kw):
        fake.run = FakeRun(**kw)
        return fake.run

    fake.init = _init
    return fake


def test_wandb_tracker_with_fake(monkeypatch, tmp_path):
    fake = _make_fake_wandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)

    model = tmp_path / "dqn.pt"
    model.write_text("w")
    (tmp_path / "dqn.json").write_text("{}")

    with WandbTracker(project="p") as t:
        t.start_run("run", {"seed": 1}, {"git_sha": "abc"})
        t.log_params({"x": 2})
        t.log_metrics({"m": 1.0}, step=0)
        t.log_artifact(str(model))
        version = t.log_model(str(model), "dqn", {"seed": 1})

    assert version == "v7"
    assert fake.run.finished
    assert fake.run.config.data["provenance"] == {"git_sha": "abc"}
    assert fake.run.logged == [({"m": 1.0}, 0)]
    # model artifact also picked up the JSON sidecar
    model_art = [a for a, _ in fake.run.artifacts if a.type == "model"][0]
    assert any(f.endswith("dqn.json") for f in model_art.files)


def test_get_tracker_auto_prefers_wandb_when_key(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", _make_fake_wandb())
    monkeypatch.setenv("WANDB_API_KEY", "k")
    assert isinstance(get_tracker("auto", experiment="p"), WandbTracker)
