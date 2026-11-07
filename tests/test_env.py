import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rl_execution.config import ActionType, ExecutionConfig, MarketConfig, RewardConfig, Side
from rl_execution.envs import ExecutionEnv
from rl_execution.envs.execution_env import MULTIPLIERS
from rl_execution.experiments.regimes import get_regime


def _run(env, actions, seed=0):
    """Play ``actions`` (cycled) until the episode ends; return (rewards, summary)."""
    env.reset(seed=seed)
    rewards, i, done, info = [], 0, False, {}
    while not done:
        a = actions[i % len(actions)]
        _, r, term, trunc, info = env.step(np.array([a], dtype=np.float32))
        rewards.append(r)
        done, i = term or trunc, i + 1
    return np.asarray(rewards), info["episode_summary"]


@pytest.mark.parametrize("action_type", [ActionType.CONTINUOUS, ActionType.DISCRETE])
def test_passes_gym_checker(action_type):
    env = ExecutionEnv(MarketConfig(), ExecutionConfig(action_type=action_type))
    check_env(env, skip_render_check=True)


@pytest.mark.parametrize("action_type", [ActionType.CONTINUOUS, ActionType.DISCRETE])
def test_obs_in_space(action_type):
    env = ExecutionEnv(MarketConfig(), ExecutionConfig(action_type=action_type))
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs.shape == (len(ExecutionEnv.FEATURE_NAMES),) == (7,)


# =============================================================== reward == objective
@pytest.mark.parametrize("process", ["abm", "gbm", "ou"])
@pytest.mark.parametrize("side", ["sell", "buy"])
@pytest.mark.parametrize("latency", [0.0, 0.4])
def test_rewards_sum_exactly_to_minus_the_objective(process, side, latency):
    """Mark-to-market rewards telescope: sum r_t = -(IS + lambda * sum sigma^2 x^2)."""
    market = MarketConfig(
        price_process=process,
        mean_reversion=0.1,
        vol_process="expou",
        vol_of_vol=0.4,
        vol_persistence=0.8,
    )
    env = ExecutionEnv(market, ExecutionConfig(side=side, latency_steps=latency))
    scale = env.exec_config.reward.reward_scale
    rng = np.random.default_rng(7)
    for seed in range(5):
        rewards, summary = _run(env, list(rng.uniform(-1, 1, 20)), seed=seed)
        objective = summary["implementation_shortfall_bps"] + summary["risk_penalty_bps"]
        assert summary["objective_bps"] == pytest.approx(objective)
        assert rewards.sum() == pytest.approx(-scale * objective, abs=1e-10)


def test_without_risk_aversion_rewards_sum_to_minus_shortfall():
    ec = ExecutionConfig(reward=RewardConfig(risk_aversion=0.0, reward_scale=1.0))
    rewards, summary = _run(ExecutionEnv(MarketConfig(), ec), [0.0])
    assert summary["risk_penalty_bps"] == 0.0
    assert rewards.sum() == pytest.approx(-summary["implementation_shortfall_bps"], abs=1e-8)


def test_reward_scale_only_changes_units():
    raw = ExecutionConfig(reward=RewardConfig(reward_scale=1.0))
    base, summary = _run(ExecutionEnv(MarketConfig(), raw), [0.2, -0.3])
    scaled, scaled_summary = _run(ExecutionEnv(), [0.2, -0.3])  # default scale 0.01
    assert np.allclose(scaled, 0.01 * base)
    assert scaled_summary["objective_bps"] == pytest.approx(summary["objective_bps"])


def test_risk_penalty_is_the_ac_variance_term_for_twap():
    """Constant-vol ABM: penalty = lambda * sigma_bps^2 * sum of squared holdings."""
    market = get_regime("abm_ac_optimal")
    env = ExecutionEnv(market)
    _, summary = _run(env, [0.0])  # a = 0 is the TWAP pace
    n = env.horizon
    held = (n - 1 - np.arange(n - 1)) / n  # fraction held over each of the first n-1 steps
    sigma_bps = market.volatility * 1e4
    expected = RewardConfig().risk_aversion * sigma_bps**2 * float(np.sum(held**2))
    assert summary["risk_penalty_bps"] == pytest.approx(expected, rel=1e-9)


def test_risk_penalty_is_charged_from_the_first_step():
    """The old realised-vol estimate was zero at reset; the model sigma is not."""
    env = ExecutionEnv()
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert info["risk_bps"] > 0


# =============================================================== TWAP-relative action
def test_action_endpoints_and_twap_centre():
    env = ExecutionEnv()
    env.reset(seed=0)
    n = env.horizon
    assert env.action_to_fraction(np.array([-1.0])) == 0.0
    assert env.action_to_fraction(np.array([0.0])) == pytest.approx(1.0 / n)
    assert env.action_to_fraction(np.array([1.0])) == pytest.approx(1.0)
    grid = np.linspace(-1, 1, 41)
    fractions = [env.action_to_fraction(np.array([a])) for a in grid]
    assert np.all(np.diff(fractions) > 0)  # strictly monotone


@pytest.mark.parametrize("t", [0, 7, 18, 19])
def test_fraction_to_action_inverts_the_mapping(t):
    env = ExecutionEnv()
    env.reset(seed=0)
    env.t = t
    for fraction in [0.0, 0.01, 1.0 / env.steps_left, 0.3, 0.75, 1.0]:
        action = env.fraction_to_action(fraction)
        assert env.action_space.contains(action)
        if env.steps_left > 1 or fraction <= 1.0:
            assert env.action_to_fraction(action) == pytest.approx(fraction, abs=1e-6)


def test_discrete_actions_are_twap_multipliers():
    env = ExecutionEnv(MarketConfig(), ExecutionConfig(action_type=ActionType.DISCRETE))
    env.reset(seed=0)
    assert env.action_space.n == len(MULTIPLIERS)
    one = MULTIPLIERS.index(1.0)
    assert env.action_to_fraction(one) == pytest.approx(1.0 / env.horizon)
    assert env.fraction_to_action(1.0 / env.horizon) == one


def test_last_step_liquidates_everything_even_when_waiting():
    _, summary = _run(ExecutionEnv(), [-1.0])  # a = -1 waits every step
    assert summary["unexecuted_shares"] == pytest.approx(0.0, abs=1e-6)
    assert summary["executed_shares"] == pytest.approx(10_000.0)


def test_full_liquidation_ends_the_episode_early():
    rewards, summary = _run(ExecutionEnv(), [1.0])  # a = +1 sells everything now
    assert len(rewards) == 1
    assert summary["risk_penalty_bps"] == 0.0


def test_episode_length_equals_horizon():
    rewards, _ = _run(ExecutionEnv(MarketConfig(), ExecutionConfig(horizon=15)), [0.0])
    assert len(rewards) == 15


# =============================================================== observation
def test_observation_is_signed_by_side():
    """The same market looks mirror-imaged to a buyer and a seller."""
    sell = ExecutionEnv(MarketConfig(), ExecutionConfig(side=Side.SELL))
    buy = ExecutionEnv(MarketConfig(), ExecutionConfig(side=Side.BUY))
    sell.reset(seed=3)
    buy.reset(seed=3)
    zero = np.array([-1.0], dtype=np.float32)  # no trading, so both see the same market
    for _ in range(5):
        o_sell, *_ = sell.step(zero)
        o_buy, *_ = buy.step(zero)
    names = ExecutionEnv.FEATURE_NAMES
    for f in ("price_move", "imbalance"):
        i = names.index(f)
        assert o_sell[i] == pytest.approx(-o_buy[i], abs=1e-6)
    for f in ("inventory_remaining", "time_remaining", "volatility", "spread", "depth"):
        i = names.index(f)
        assert o_sell[i] == pytest.approx(o_buy[i])


def test_depth_feature_tells_thin_from_deep_markets():
    i = ExecutionEnv.FEATURE_NAMES.index("depth")
    thin, _ = ExecutionEnv(get_regime("thin")).reset(seed=0)
    deep, _ = ExecutionEnv(get_regime("deep")).reset(seed=0)
    assert deep[i] > 3 * thin[i]


def test_buy_side_runs():
    _, summary = _run(ExecutionEnv(MarketConfig(), ExecutionConfig(side=Side.BUY)), [0.2])
    assert np.isfinite(summary["implementation_shortfall_bps"])
