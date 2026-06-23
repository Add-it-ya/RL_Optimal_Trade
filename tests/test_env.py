import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rl_execution.config import ActionType, ExecutionConfig, MarketConfig, Side
from rl_execution.envs import ExecutionEnv


@pytest.mark.parametrize("action_type", [ActionType.CONTINUOUS, ActionType.DISCRETE])
def test_passes_gym_checker(action_type):
    env = ExecutionEnv(MarketConfig(), ExecutionConfig(action_type=action_type))
    check_env(env, skip_render_check=True)


@pytest.mark.parametrize("action_type", [ActionType.CONTINUOUS, ActionType.DISCRETE])
def test_obs_in_space(action_type):
    env = ExecutionEnv(MarketConfig(), ExecutionConfig(action_type=action_type))
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs.shape == (8,)


def test_force_liquidation_executes_everything():
    env = ExecutionEnv(MarketConfig(), ExecutionConfig(force_liquidation=True))
    obs, _ = env.reset(seed=1)
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(
            np.array([0.0], dtype=np.float32)
        )  # never trade voluntarily
        done = term or trunc
    summ = info["episode_summary"]
    assert summ["unexecuted_shares"] == pytest.approx(0.0, abs=1e-6)
    assert summ["executed_shares"] == pytest.approx(env.total_inventory, rel=1e-6)


def test_episode_length_equals_horizon():
    env = ExecutionEnv(MarketConfig(), ExecutionConfig(horizon=15))
    env.reset(seed=2)
    steps = 0
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(np.array([0.05], dtype=np.float32))
        steps += 1
        done = term or trunc
    assert steps <= 15


def test_reward_sum_tracks_negative_is():
    # With zero risk aversion / temp penalty and no commission, cumulative reward should
    # equal -implementation_shortfall (bps) up to permanent-impact bookkeeping.
    from rl_execution.config import RewardConfig

    ec = ExecutionConfig(
        commission_bps=0.0,
        reward=RewardConfig(risk_aversion=0.0, temp_impact_penalty=0.0, reward_scale=1e4),
    )
    env = ExecutionEnv(MarketConfig(volatility=0.0, drift=0.0, imbalance_alpha=0.0), ec)
    env.reset(seed=3)
    total_r = 0.0
    done = False
    while not done:
        _, r, term, trunc, info = env.step(np.array([0.1], dtype=np.float32))
        total_r += r
        done = term or trunc
    is_bps = info["episode_summary"]["implementation_shortfall_bps"]
    assert total_r == pytest.approx(-is_bps, rel=0.05, abs=1.0)


def test_buy_side_runs():
    env = ExecutionEnv(MarketConfig(), ExecutionConfig(side=Side.BUY))
    env.reset(seed=4)
    done = False
    while not done:
        _, r, term, trunc, info = env.step(np.array([0.2], dtype=np.float32))
        done = term or trunc
    assert np.isfinite(info["episode_summary"]["implementation_shortfall_bps"])
