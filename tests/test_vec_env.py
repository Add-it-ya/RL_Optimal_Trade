"""The batched torch environment against the numpy reference, and the PPO that trains on it.

Parity test 1 (identical dynamics on identical markets) and parity test 2 (the torch-drawn
markets have the numpy simulator's statistics) live here and run on the CPU.  Parity test 3
(the batched PPO reproduces the SB3 E1 result) is a training experiment, run by
``scripts/lab/run_phase4.sh`` on a GPU machine.
"""

import numpy as np
import pytest
import torch

from rl_execution.agents import AgentStrategy
from rl_execution.backtest import evaluate
from rl_execution.config import ActionType, ExecutionConfig, Side
from rl_execution.envs import ExecutionEnv, NoiseStreams, generate_market_paths
from rl_execution.experiments import get_regime
from rl_execution.experiments.regimes import _RANDOMIZED_MEAN_REVERSION, _RANDOMIZED_UNIFORM
from rl_execution.vec import (
    BatchedExecutionEnv,
    BatchMarket,
    generate_paths,
    reference_paths,
    sample_randomized_markets,
)
from rl_execution.vec.ppo import (
    ActorCritic,
    BatchedValidator,
    TorchPolicyAgent,
    VecPPOConfig,
    train_vec_ppo,
)

PARITY_REGIMES = ["abm_ac_optimal", "normal_liquidity", "sideways", "gbm_medium"]
PARITY_REGIMES += ["clustered_vol", "high_vol", "thin", "signal_ic100"]


def _run_both(regime, side, action_type, dtype, n=8, seed=0):
    """Step the numpy env and the batched env through the same markets and actions."""
    mc, ec = get_regime(regime), ExecutionConfig(side=side, action_type=action_type)
    H = ec.horizon
    rng = np.random.default_rng(seed)
    if action_type is ActionType.DISCRETE:
        actions = rng.integers(0, 10, size=(H, n))
    else:
        actions = rng.uniform(-1.2, 1.2, size=(H, n))  # beyond [-1, 1] exercises the clip
    configs, paths = reference_paths(
        lambda: ExecutionEnv(mc, ec), n, 7_000, antithetic=True, dtype=dtype
    )
    batch = BatchedExecutionEnv(n, configs, ec, dtype=dtype)
    b_obs, b_rew, b_fill = [batch.reset(paths).double().numpy()], [], []
    for t in range(H):
        obs, rew, _, info = batch.step(torch.as_tensor(actions[t]))
        b_obs.append(obs.double().numpy())
        b_rew.append(rew.double().numpy())
        b_fill.append(info["filled"].double().numpy())
    b_obj = batch.episode_summary()["objective_bps"].double().numpy()

    gaps = {"obs": 0.0, "reward": 0.0, "filled": 0.0, "objective": 0.0}
    for i in range(n):
        env = ExecutionEnv(mc, ec)
        obs, _ = env.reset(seed=7_000 + i // 2, options={"antithetic": bool(i % 2)})
        gaps["obs"] = max(gaps["obs"], np.abs(obs - b_obs[0][i]).max())
        for t in range(H):
            a = int(actions[t, i]) if action_type is ActionType.DISCRETE else [actions[t, i]]
            obs, rew, term, _, info = env.step(a)
            gaps["obs"] = max(gaps["obs"], np.abs(obs - b_obs[t + 1][i]).max())
            gaps["reward"] = max(gaps["reward"], abs(rew - b_rew[t][i]))
            gaps["filled"] = max(gaps["filled"], abs(info["shares"] - b_fill[t][i]))
            if term:
                # a finished episode keeps stepping in the batch, trading nothing, earning 0
                rest = [abs(b_rew[u][i]) + abs(b_fill[u][i]) for u in range(t + 1, H)]
                gaps["reward"] = max([gaps["reward"], *rest])
                break
        summary = info["episode_summary"]
        gaps["objective"] = max(gaps["objective"], abs(summary["objective_bps"] - b_obj[i]))
    return gaps


# --------------------------------------------------------------------------- parity test 1
@pytest.mark.parametrize("regime", PARITY_REGIMES)
@pytest.mark.parametrize("side", [Side.SELL, Side.BUY])
@pytest.mark.parametrize("action_type", [ActionType.CONTINUOUS, ActionType.DISCRETE])
def test_batched_env_matches_the_numpy_env_step_by_step(regime, side, action_type):
    """float64, identical markets: the same fills, rewards and observations.

    Observations differ only by the numpy env's float32 rounding (~1e-7).
    """
    gaps = _run_both(regime, side, action_type, torch.float64)
    assert gaps["reward"] < 1e-10
    assert gaps["filled"] < 1e-8
    assert gaps["objective"] < 1e-8
    assert gaps["obs"] < 1e-6


def test_float32_parity_is_within_float32_resolution():
    """In float32 (the training precision) prices near 100 resolve to ~1e-5, so rewards
    agree to ~1e-5 scaled units (~1e-3 bps) and objectives to ~1e-2 bps: four orders of
    magnitude below the per-episode noise, and reported numbers come from numpy anyway."""
    gaps = _run_both("thin", Side.SELL, ActionType.CONTINUOUS, torch.float32, n=16)
    assert gaps["reward"] < 1e-4
    assert gaps["objective"] < 1e-2
    assert gaps["obs"] < 5e-4


def test_rewards_sum_to_minus_the_scaled_objective_on_torch_drawn_markets():
    for regime in ("normal_liquidity", "gbm_medium", "sideways"):
        ec = ExecutionConfig()
        env = BatchedExecutionEnv(256, get_regime(regime), ec, dtype=torch.float64, seed=3)
        env.reset()
        actions = torch.rand(ec.horizon, 256, generator=torch.Generator().manual_seed(1)) * 2 - 1
        total = torch.zeros(256, dtype=torch.float64)
        for t in range(ec.horizon):
            total += env.step(actions[t])[1]
        objective = env.episode_summary()["objective_bps"]
        assert torch.allclose(total, -ec.reward.reward_scale * objective, atol=1e-9)


def test_every_episode_ends_by_the_horizon_and_alive_masks_the_rest():
    ec = ExecutionConfig()
    env = BatchedExecutionEnv(64, get_regime("normal_liquidity"), ec, seed=0)
    env.reset()
    dumped = None
    for t in range(ec.horizon):
        action = torch.ones(64) if t == 0 else torch.zeros(64)  # +1 = liquidate now
        _, reward, done, info = env.step(action)
        if t == 0:
            dumped = done
        else:
            assert not info["alive"].any() and (reward == 0).all()
    assert dumped.all()


# --------------------------------------------------------------------------- parity test 2
def _numpy_paths(mc, n, seed=11):
    paths = [generate_market_paths(mc, 22, NoiseStreams.from_seed(seed + i)) for i in range(n)]
    return {
        "mid": np.stack([p.mid for p in paths]),
        "sigma": np.stack([p.sigma for p in paths]),
        "imbalance": np.stack([p.imbalance for p in paths]),
        "spread": np.stack([p.spread for p in paths]),
        "depth": np.stack([p.depth_bid for p in paths]),
    }


def _torch_paths(mc, n, seed=12):
    market = BatchMarket.from_configs(mc, n=n, dtype=torch.float64)
    p = generate_paths(market, 22, torch.Generator().manual_seed(seed))
    return {
        "mid": p.mid.numpy(),
        "sigma": p.sigma.numpy(),
        "imbalance": p.imbalance.numpy(),
        "spread": p.spread.numpy(),
        "depth": p.depth_bid.numpy(),
    }


def _stats(d):
    ret = np.diff(d["mid"], axis=1) / d["mid"][:, :-1]
    ar = np.abs(ret)
    ic = np.corrcoef(d["imbalance"][:, 1:-1].ravel(), ret[:, 1:].ravel())[0, 1]
    acf = np.corrcoef(ar[:, :-1].ravel(), ar[:, 1:].ravel())[0, 1]
    return {
        "ret_mean": ret.mean(),
        "ret_std": ret.std(),
        "final_std": (d["mid"][:, -1] / d["mid"][:, 0]).std(),
        "imb_mean": d["imbalance"].mean(),
        "imb_std": d["imbalance"].std(),
        "ic": ic,
        "abs_ret_acf": acf,
        "sigma_mean": d["sigma"].mean(),
        "spread_mean": d["spread"][:, 1:].mean(),
        "depth_mean": d["depth"].mean(),
    }


@pytest.mark.parametrize(
    "regime", ["signal_ic100", "clustered_vol", "leverage", "high_vol", "sideways", "gbm_medium"]
)
def test_torch_drawn_markets_have_the_numpy_statistics(regime):
    """Increment moments, signal IC, |return| autocorrelation and the book inputs agree to
    within sampling error (4,000 episodes x 21 steps from each generator)."""
    mc = get_regime(regime)
    a, b = _stats(_numpy_paths(mc, 4_000)), _stats(_torch_paths(mc, 4_000))
    assert b["ret_std"] == pytest.approx(a["ret_std"], rel=0.03)
    assert b["final_std"] == pytest.approx(a["final_std"], rel=0.05)
    assert abs(b["ret_mean"] - a["ret_mean"]) < 0.05 * a["ret_std"]
    assert abs(b["imb_mean"] - a["imb_mean"]) < 0.03
    assert b["imb_std"] == pytest.approx(a["imb_std"], rel=0.03)
    assert abs(b["ic"] - a["ic"]) < 0.02
    assert abs(b["abs_ret_acf"] - a["abs_ret_acf"]) < 0.03
    assert b["sigma_mean"] == pytest.approx(a["sigma_mean"], rel=0.03)
    assert b["spread_mean"] == pytest.approx(a["spread_mean"], rel=0.02)
    assert b["depth_mean"] == pytest.approx(a["depth_mean"], rel=0.01)


def test_randomized_markets_stay_inside_the_training_family():
    m = sample_randomized_markets(5_000, torch.Generator().manual_seed(0), dtype=torch.float64)
    for name, (low, high, unit) in _RANDOMIZED_UNIFORM.items():
        lo, hi = sorted((low * unit, high * unit))
        values = m.p[name]
        assert values.min() >= lo - 1e-12 and values.max() <= hi + 1e-12, name
        assert values.max() - values.min() > 0.9 * (hi - lo), name  # covers the range
    reversion = set(np.round(m.p["mean_reversion"].numpy(), 12))
    assert reversion <= set(_RANDOMIZED_MEAN_REVERSION)
    assert torch.equal(m.kappa, m.p["mean_reversion"])
    share_ou = float((m.kappa > 0).double().mean())
    assert share_ou == pytest.approx(1 / 3, abs=0.03)


# --------------------------------------------------------------------------- PPO
def _validator(regime="abm_ac_optimal", n=64):
    mc, ec = get_regime(regime), ExecutionConfig()
    configs, paths = reference_paths(lambda: ExecutionEnv(mc, ec), n, 1_000_000, antithetic=True)
    return BatchedValidator(BatchedExecutionEnv(n, configs, ec), paths), mc, ec


def test_batched_validation_matches_numpy_evaluation():
    """Checkpoint selection on the GPU sees the numpy pipeline's validation episodes."""
    torch.manual_seed(0)
    validator, mc, ec = _validator()
    agent = TorchPolicyAgent(ActorCritic(log_std_init=-1.0))
    with torch.no_grad():  # a policy that is far from TWAP and state-dependent
        agent.model.actor[-1].weight.mul_(100.0)
    batched = validator.score(agent.model)["objective_bps"]
    res = evaluate(
        lambda: ExecutionEnv(mc, ec), AgentStrategy(agent), 64, 1_000_000, antithetic=True
    )
    assert batched == pytest.approx(res.aggregate["objective_bps_mean"], abs=1e-2)


def test_untrained_vecppo_starts_as_twap():
    validator, mc, ec = _validator()
    torch.manual_seed(1)
    untrained = validator.score(ActorCritic())
    twap = validator.env
    twap.reset(validator.paths)
    for _ in range(ec.horizon):
        twap.step(torch.zeros(validator.env.n))
    twap_objective = float(twap.episode_summary()["objective_bps"].mean())
    assert untrained["objective_bps"] == pytest.approx(twap_objective, abs=0.2)
    assert untrained["effective_steps"] > 19.5


def test_actor_round_trips_through_a_file(tmp_path):
    torch.manual_seed(2)
    agent = TorchPolicyAgent(
        ActorCritic(hidden=(32, 16)), {"hidden": [32, 16], "log_std_init": -1.0}
    )
    agent.save(str(tmp_path / "model.pt"))
    loaded = TorchPolicyAgent.load(str(tmp_path / "model.pt"))
    obs = np.random.default_rng(0).uniform(0, 1, size=(5, 7)).astype(np.float32)
    for o in obs:
        assert np.array_equal(agent.predict(o), loaded.predict(o))
        assert agent.predict(o).shape == (1,)


def test_train_vec_ppo_counts_real_steps_and_selects_a_checkpoint():
    torch.manual_seed(3)
    validator, mc, ec = _validator(n=32)
    env = BatchedExecutionEnv(32, mc, ec, seed=4)
    out = train_vec_ppo(env, 3_000, VecPPOConfig(n_envs=32, minibatch_size=128), validator, 1_000)
    steps = [row["step"] for row in out["validation"]]
    assert steps[0] == 0 and steps == sorted(steps) and steps[-1] >= 3_000
    assert out["selected_step"] in steps
    assert all(row["step"] <= 32 * ec.horizon * (i + 1) for i, row in enumerate(out["episodes"]))
    best = min(row["objective_bps"] for row in out["validation"])
    assert validator.score(out["agent"].model)["objective_bps"] == pytest.approx(best, abs=1e-4)
