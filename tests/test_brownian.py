"""Brownian-motion layer: Wiener properties, bridges, RNG substreams and price processes.

The statistical assertions are seeded and sized so the suite is deterministic; tolerances
are set at roughly four standard errors of the quantity under test.
"""

import numpy as np
import pytest
from scipy import stats

from rl_execution.backtest import evaluate, schedule_gap, schedule_gap_table
from rl_execution.baselines import TWAP, AlmgrenChriss
from rl_execution.config import ExecutionConfig, MarketConfig, RunConfig, Side
from rl_execution.envs import ExecutionEnv, MarketSimulator
from rl_execution.envs.brownian import (
    BrownianEngine,
    NoiseStreams,
    Stream,
    _ndtri,
    bridge_value,
    brownian_path,
    correlate,
    correlated_normals,
    make_streams,
    normals,
)
from rl_execution.envs.processes import (
    ArithmeticBM,
    GeometricBM,
    OrnsteinUhlenbeck,
    generate_market_paths,
    make_price_process,
    volatility_path,
)
from rl_execution.experiments.regimes import get_regime, list_regimes
from rl_execution.metrics.stats import paired_bootstrap_ci


# =============================================================== Wiener process properties
def test_increments_are_gaussian():
    """dW / sqrt(dt) is standard normal (Kolmogorov-Smirnov)."""
    eng = BrownianEngine.draw(np.random.default_rng(0), 20_000, dt=0.25)
    z = eng.dW / np.sqrt(eng.dt)
    assert stats.kstest(z, "norm").pvalue > 0.01


def test_variance_grows_linearly_in_time():
    """Var(W_t) = t, so the variance ratio between two times is the time ratio."""
    rng = np.random.default_rng(1)
    dt, n, m = 0.5, 10, 3_000
    paths = np.stack([brownian_path(rng, n, dt) for _ in range(m)])
    for t in (2, 5, 10):
        var = float(paths[:, t].var())
        assert var == pytest.approx(t * dt, rel=0.08), f"Var(W_{t}) = {var}"


def test_increments_are_uncorrelated():
    eng = BrownianEngine.draw(np.random.default_rng(2), 50_000)
    d = eng.dW - eng.dW.mean()
    for lag in (1, 2, 3):
        acf = float((d[:-lag] * d[lag:]).mean() / d.var())
        assert abs(acf) < 0.02, f"lag-{lag} autocorrelation {acf}"


def test_quadratic_variation_converges():
    """sum (dW)^2 -> T as dt -> 0; the sampling error shrinks like sqrt(2/n)."""
    rng = np.random.default_rng(3)
    errors = {}
    for n in (50, 5_000):
        qv = [BrownianEngine.draw(rng, n, dt=1.0 / n).quadratic_variation() for _ in range(20)]
        errors[n] = float(np.mean(np.abs(np.array(qv) - 1.0)))
    assert errors[5_000] < errors[50] / 5


def test_time_rescaling_invariance():
    """Halving dt and doubling the step count leaves the terminal law unchanged."""
    rng = np.random.default_rng(4)
    coarse = [brownian_path(rng, 10, 0.1)[-1] for _ in range(2_000)]
    fine = [brownian_path(rng, 20, 0.05)[-1] for _ in range(2_000)]
    assert stats.ks_2samp(coarse, fine).pvalue > 0.01


def test_engine_rejects_non_positive_dt():
    with pytest.raises(ValueError):
        BrownianEngine(np.zeros(3), dt=0.0)


# =============================================================== Brownian bridge
def test_bridge_matches_endpoints():
    eng = BrownianEngine.draw(np.random.default_rng(5), 12)
    assert eng.bridge(3, 0.0, z=2.5) == pytest.approx(eng.path[3])
    assert eng.bridge(3, 1.0, z=2.5) == pytest.approx(eng.path[4])


def test_bridge_midpoint_moments():
    """At delta = 0.5 the bridge is centred on the chord with variance dt / 4."""
    rng = np.random.default_rng(6)
    eng = BrownianEngine.draw(rng, 1, dt=2.0)
    draws = np.array([eng.bridge(0, 0.5, z=float(zz)) for zz in rng.standard_normal(20_000)])
    chord = 0.5 * (eng.path[0] + eng.path[1])
    assert draws.mean() == pytest.approx(chord, abs=0.02)
    assert draws.var() == pytest.approx(0.25 * eng.dt, rel=0.05)


def test_bridge_value_is_unbiased_between_endpoints():
    mean = np.mean(
        [
            bridge_value(10.0, 14.0, 0.25, 4.0, z)
            for z in np.random.default_rng(7).standard_normal(5_000)
        ]
    )
    assert mean == pytest.approx(11.0, abs=0.05)  # 10 + 0.25 * (14 - 10)


def test_bridge_out_of_range_step_raises():
    eng = BrownianEngine.draw(np.random.default_rng(8), 4)
    with pytest.raises(IndexError):
        eng.bridge(4, 0.5, z=0.0)


def test_refine_returns_interior_points():
    rng = np.random.default_rng(9)
    eng = BrownianEngine.draw(rng, 3)
    pts = eng.refine(1, 4, rng)
    assert pts.shape == (4,)
    assert np.all(np.isfinite(pts))
    # the refined sub-path is centred on the chord joining the step's endpoints
    mids = np.mean([eng.refine(1, 1, rng)[0] for _ in range(3_000)])
    assert mids == pytest.approx(0.5 * (eng.path[1] + eng.path[2]), abs=0.05)


def test_refine_with_zero_points_is_empty():
    eng = BrownianEngine.draw(np.random.default_rng(10), 3)
    assert eng.refine(0, 0, np.random.default_rng(0)).size == 0


# =============================================================== random number generation
def test_inverse_cdf_matches_ziggurat_distribution():
    a = normals(np.random.default_rng(11), 20_000, method="ziggurat")
    b = normals(np.random.default_rng(12), 20_000, method="inverse_cdf")
    assert stats.ks_2samp(a, b).pvalue > 0.01


def test_inverse_cdf_is_monotone_in_the_uniform():
    """Monotonicity is what makes antithetic pairing exact and QMC constructions valid."""
    u = np.linspace(1e-6, 1 - 1e-6, 500)
    assert np.all(np.diff(_ndtri(u)) > 0)


def test_unknown_normal_method_raises():
    with pytest.raises(ValueError):
        normals(np.random.default_rng(0), 4, method="nope")


def test_correlate_produces_target_correlation():
    rng = np.random.default_rng(13)
    z1, z2 = normals(rng, 40_000), normals(rng, 40_000)
    for rho in (-0.8, -0.3, 0.0, 0.6):
        zc = correlate(z1, z2, rho)
        assert float(np.corrcoef(z1, zc)[0, 1]) == pytest.approx(rho, abs=0.02)
        assert float(zc.std()) == pytest.approx(1.0, rel=0.03)


def test_correlated_normals_hit_target_matrix():
    target = np.array([[1.0, -0.5, 0.2], [-0.5, 1.0, 0.1], [0.2, 0.1, 1.0]])
    z = correlated_normals(np.random.default_rng(14), target, 40_000)
    assert np.allclose(np.corrcoef(z), target, atol=0.02)


def test_correlated_normals_rejects_non_square():
    with pytest.raises(ValueError):
        correlated_normals(np.random.default_rng(0), np.ones((2, 3)), 10)


def test_substreams_are_independent():
    s = NoiseStreams.from_seed(15)
    a, b = s.normals(Stream.PRICE, 20_000), s.normals(Stream.VOL, 20_000)
    assert abs(float(np.corrcoef(a, b)[0, 1])) < 0.02


def test_substream_isolation_under_added_draws():
    """Adding draws to one stream must not move another -- the reproducibility guarantee."""
    baseline = NoiseStreams.from_seed(16).normals(Stream.PRICE, 5)

    perturbed = NoiseStreams.from_seed(16)
    perturbed.normals(Stream.DEPTH, 137)  # a new noise source elsewhere in the simulator
    perturbed.normals(Stream.SPREAD, 42)
    assert np.array_equal(perturbed.normals(Stream.PRICE, 5), baseline)


def test_shared_mode_reuses_one_generator():
    rng = np.random.default_rng(17)
    s = NoiseStreams.shared(rng)
    assert s.shared_generator
    assert all(s[stream] is rng for stream in Stream)


def test_antithetic_pairs_sum_to_zero():
    plain = NoiseStreams.from_seed(18, antithetic=False).normals(Stream.PRICE, 100)
    flipped = NoiseStreams.from_seed(18, antithetic=True).normals(Stream.PRICE, 100)
    assert np.allclose(plain + flipped, 0.0)


def test_make_streams_modes():
    rng = np.random.default_rng(19)
    assert make_streams(rng, "legacy").shared_generator
    assert not make_streams(rng, "streams").shared_generator
    with pytest.raises(ValueError):
        make_streams(rng, "bogus")


# =============================================================== price processes
def _paths(process, n_paths, n_steps, sigma, seed=0):
    rng = np.random.default_rng(seed)
    zero = np.zeros(n_steps)
    sig = np.full(n_steps, sigma)
    return np.stack([process.path(normals(rng, n_steps), sig, zero) for _ in range(n_paths)])


def test_gbm_is_strictly_positive():
    cfg = MarketConfig(price_process="gbm", volatility=0.2)
    paths = _paths(GeometricBM(cfg), 2_000, 40, 0.2, seed=20)
    assert np.all(paths > 0)


def test_gbm_log_returns_have_ito_corrected_mean():
    """E[d log S] = (mu - sigma^2 / 2) dt -- the correction a naive S*(1+r) update omits."""
    mu, sigma, dt = 0.01, 0.2, 0.5
    cfg = MarketConfig(price_process="gbm", drift=mu, volatility=sigma, dt=dt)
    paths = _paths(GeometricBM(cfg), 4_000, 20, sigma, seed=21)
    log_ret = np.diff(np.log(paths), axis=1).ravel()
    assert log_ret.mean() == pytest.approx((mu - 0.5 * sigma**2) * dt, abs=0.004)


def test_gbm_is_martingale_under_zero_drift():
    cfg = MarketConfig(price_process="gbm", drift=0.0, volatility=0.02)
    paths = _paths(GeometricBM(cfg), 6_000, 20, 0.02, seed=22)
    assert float(paths[:, -1].mean() / cfg.initial_price) == pytest.approx(1.0, abs=0.006)


def test_abm_matches_closed_form_moments():
    mu, sigma, n, dt = 0.001, 0.02, 20, 1.0
    cfg = MarketConfig(price_process="abm", drift=mu, volatility=sigma, dt=dt)
    s0 = cfg.initial_price
    paths = _paths(ArithmeticBM(cfg), 6_000, n, sigma, seed=23)
    terminal = paths[:, -1]
    assert terminal.mean() == pytest.approx(s0 + mu * s0 * n * dt, abs=0.15)
    assert terminal.var() == pytest.approx((sigma * s0) ** 2 * n * dt, rel=0.07)


def test_ou_reverts_to_mean_with_stationary_variance():
    kappa, sigma = 0.3, 0.02
    cfg = MarketConfig(price_process="ou", mean_reversion=kappa, volatility=sigma)
    path = _paths(OrnsteinUhlenbeck(cfg), 1, 40_000, sigma, seed=24)[0]
    tail = path[2_000:]  # discard the deterministic start at s0
    assert tail.mean() == pytest.approx(cfg.initial_price, abs=0.15)
    assert tail.var() == pytest.approx((sigma * cfg.initial_price) ** 2 / (2 * kappa), rel=0.15)


def test_ou_with_zero_kappa_falls_back_to_abm():
    cfg = MarketConfig(price_process="ou", mean_reversion=0.0, volatility=0.02)
    z, sig, alpha = normals(np.random.default_rng(25), 30), np.full(30, 0.02), np.zeros(30)
    assert np.allclose(
        OrnsteinUhlenbeck(cfg).path(z, sig, alpha), ArithmeticBM(cfg).path(z, sig, alpha)
    )


def test_make_price_process_registry():
    assert isinstance(make_price_process(MarketConfig(price_process="gbm")), GeometricBM)
    assert isinstance(make_price_process(MarketConfig(price_process="abm")), ArithmeticBM)
    with pytest.raises(ValueError):
        make_price_process(MarketConfig(price_process="legacy"))
    with pytest.raises(ValueError):
        make_price_process(MarketConfig(price_process="bogus"))


# =============================================================== stochastic volatility
def test_expou_volatility_is_stationary_at_any_persistence():
    """ln(sigma) keeps standard deviation xi regardless of phi (the sqrt(1-phi^2) scaling)."""
    xi = 0.4
    z = normals(np.random.default_rng(26), 60_000)
    for phi in (0.0, 0.5, 0.95):
        cfg = MarketConfig(vol_process="expou", vol_of_vol=xi, vol_persistence=phi, volatility=0.02)
        ln_sigma = np.log(volatility_path(cfg, z) / cfg.volatility)
        assert ln_sigma.std() == pytest.approx(xi, rel=0.08), f"phi={phi}"
        assert ln_sigma.mean() == pytest.approx(0.0, abs=0.03)


def test_expou_produces_volatility_clustering():
    """Persistent log-vol gives autocorrelated volatility; the i.i.d. model gives none."""
    z = normals(np.random.default_rng(27), 40_000)
    base = dict(volatility=0.02, vol_of_vol=0.5)

    clustered = volatility_path(MarketConfig(vol_process="expou", vol_persistence=0.9, **base), z)
    iid = volatility_path(MarketConfig(vol_process="iid_lognormal", **base), z)

    def acf1(x):
        d = x - x.mean()
        return float((d[:-1] * d[1:]).mean() / d.var())

    assert acf1(clustered) > 0.7
    assert abs(acf1(iid)) < 0.03


def test_vol_process_none_is_constant():
    z = normals(np.random.default_rng(28), 100)
    sig = volatility_path(MarketConfig(vol_process="none", vol_of_vol=0.5, volatility=0.03), z)
    assert np.allclose(sig, 0.03)


def test_unknown_vol_process_raises():
    with pytest.raises(ValueError):
        volatility_path(MarketConfig(vol_process="bogus", vol_of_vol=0.3), np.zeros(5))


def test_leverage_correlation_is_realised():
    """rho < 0 => volatility rises as the price falls (the leverage effect)."""
    cfg = MarketConfig(
        price_process="gbm",
        rng_mode="streams",
        vol_process="expou",
        vol_of_vol=0.5,
        vol_persistence=0.8,
        leverage_corr=-0.8,
        volatility=0.02,
    )
    paths = generate_market_paths(cfg, 40_000, NoiseStreams.from_seed(29))
    ret = paths.returns()
    d_ln_sigma = np.diff(np.log(paths.sigma))
    n = min(ret.size, d_ln_sigma.size)
    assert float(np.corrcoef(ret[:n], d_ln_sigma[:n])[0, 1]) < -0.3


# =============================================================== imbalance scaling
@pytest.mark.parametrize("phi, min_ratio", [(0.5, 1.2), (0.85, 2.4)])
def test_imbalance_ou_scaling_strengthens_a_stationary_factor(phi, min_ratio):
    """The stationary innovation makes the latent factor unit-variance before clipping."""
    common = dict(price_process="gbm", rng_mode="streams", imbalance_persistence=phi)
    stationary = generate_market_paths(
        MarketConfig(imbalance_ou_scaling=True, **common), 60_000, NoiseStreams.from_seed(30)
    ).imbalance
    historical = generate_market_paths(
        MarketConfig(imbalance_ou_scaling=False, **common), 60_000, NoiseStreams.from_seed(30)
    ).imbalance

    # a unit-variance Gaussian clipped to [-1, 1] has standard deviation ~0.72, and the
    # stationary scaling reaches that at every phi
    assert stationary.std() == pytest.approx(0.72, rel=0.06)
    # the historical scaling is weaker by sqrt((1 - phi) / (1 + phi)) and phi-dependent;
    # at the default phi = 0.85 the stationary factor is ~2.5x stronger
    assert historical.std() == pytest.approx(np.sqrt((1 - phi) / (1 + phi)), rel=0.15)
    assert stationary.std() > min_ratio * historical.std()


# =============================================================== full market paths
def test_generate_market_paths_shapes_and_ranges():
    cfg = MarketConfig(price_process="gbm", rng_mode="streams", n_levels=7)
    p = generate_market_paths(cfg, 30, NoiseStreams.from_seed(31))
    assert p.mid.shape == (30,)
    assert p.sigma.shape == (29,)
    assert p.imbalance.shape == (30,)
    assert p.depth_bid.shape == (30, 7) and p.depth_ask.shape == (30, 7)
    assert np.all(p.spread >= cfg.tick_size)
    assert np.all(p.volume >= 1.0)
    assert np.all(np.abs(p.imbalance) <= 1.0)
    assert p.mid[0] == pytest.approx(cfg.initial_price)


# =============================================================== simulator integration
def test_legacy_mode_is_bit_identical():
    """Golden values captured at the pre-Brownian baseline; guards every published result."""
    cfg = MarketConfig(volatility=0.02, vol_of_vol=0.2, imbalance_alpha=0.05, drift=0.0005)
    sim = MarketSimulator(cfg, rng=np.random.default_rng(11))
    sim.reset(horizon=25)
    mids, spreads, imbs = [], [], []
    for _ in range(5):
        mids.append(sim.mid)
        spreads.append(sim.spread)
        imbs.append(sim.imbalance_factor)
        sim.advance()

    assert mids == pytest.approx(
        [100.0, 96.8493876298, 97.3513522461, 97.8294340018, 95.6756844192], abs=1e-9
    )
    assert spreads[:3] == pytest.approx([0.02, 0.0155534454, 0.0277044413], abs=1e-9)
    assert imbs[1:4] == pytest.approx([-0.0992419396, -0.1626312746, 0.0986693144], abs=1e-9)


def test_legacy_env_episode_is_bit_identical():
    env = ExecutionEnv(
        MarketConfig(volatility=0.02, vol_of_vol=0.2, imbalance_alpha=0.05),
        ExecutionConfig(horizon=12, latency_steps=0.3, commission_bps=1.0),
    )
    env.reset(seed=42)
    rewards, info = [], {}
    for a in [0.0, 0.2, 0.0, 0.5, 0.1, 0.0, 0.3, 0.9, 0.0, 0.4, 0.2, 1.0]:
        _, r, term, trunc, info = env.step(np.array([a], dtype=np.float32))
        rewards.append(r)
        if term or trunc:
            break
    assert rewards[:4] == pytest.approx(
        [-0.0, -92.9874022545, -1.1609902897, -301.3391192043], abs=1e-6
    )
    assert info["episode_summary"]["implementation_shortfall_bps"] == pytest.approx(
        731.9389291146, abs=1e-6
    )


def _mid_path(fraction, cfg, seed=5, horizon=12):
    env = ExecutionEnv(cfg, ExecutionConfig(horizon=horizon, force_liquidation=False))
    env.reset(seed=seed)
    out = [env.market.mid]
    for _ in range(horizon):
        _, _, term, trunc, _ = env.step(np.array([fraction], dtype=np.float32))
        out.append(env.market.mid)
        if term or trunc:
            break
    return np.asarray(out)


def test_price_path_is_independent_of_agent_actions():
    """With impact switched off, what the agent trades must not move the market at all.

    Under the legacy dynamics it does: executing a trade re-quotes the book, consuming
    normal draws that would otherwise have driven the price, so two strategies on one seed
    drift onto different paths.  The Brownian mode draws the whole exogenous path up front,
    which is what makes paired strategy comparisons genuine common random numbers.
    """
    frozen = dict(permanent_impact=0.0, temporary_impact=0.0)
    brownian = _mid_path(0.0, MarketConfig(price_process="gbm", rng_mode="streams", **frozen))
    brownian_trading = _mid_path(
        0.3, MarketConfig(price_process="gbm", rng_mode="streams", **frozen)
    )
    assert np.allclose(brownian, brownian_trading)

    legacy = _mid_path(0.0, MarketConfig(**frozen))
    legacy_trading = _mid_path(0.3, MarketConfig(**frozen))
    assert not np.allclose(legacy, legacy_trading)  # documents the behaviour being fixed


@pytest.mark.parametrize("process", ["abm", "gbm", "ou"])
def test_env_runs_under_every_price_process(process):
    cfg = MarketConfig(price_process=process, rng_mode="streams", mean_reversion=0.1)
    env = ExecutionEnv(cfg, ExecutionConfig(horizon=10))
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    done = False
    while not done:
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        done = term or trunc
    assert info["episode_summary"]["executed_shares"] == pytest.approx(10_000.0)


def test_permanent_impact_still_moves_the_mid_under_gbm():
    cfg = MarketConfig(price_process="gbm", rng_mode="streams", permanent_impact=1e-4)
    sim = MarketSimulator(cfg, rng=np.random.default_rng(0))
    sim.reset(horizon=10)
    mid0 = sim.mid
    res = sim.execute(Side.SELL, 1_000.0)
    assert res.mid_after < mid0
    assert sim.shift < 0  # the impact persists as an overlay on the exogenous path


def test_impact_persists_across_steps_under_gbm():
    cfg = MarketConfig(price_process="gbm", rng_mode="streams", permanent_impact=1e-4)
    sim = MarketSimulator(cfg, rng=np.random.default_rng(2))
    sim.reset(horizon=10)
    clean = float(sim.paths.mid[1])
    sim.execute(Side.SELL, 1_000.0)
    shift = sim.shift
    sim.advance()
    assert sim.mid == pytest.approx(clean + shift)


# =============================================================== latency models
def test_bridge_latency_is_unbiased_and_adverse_is_not():
    """The adverse model always charges the trader; the bridge is centred on the chord."""
    common = dict(price_process="gbm", rng_mode="streams", volatility=0.02)

    def mean_shift(model):
        shifts = []
        for seed in range(300):
            sim = MarketSimulator(
                MarketConfig(latency_model=model, **common), np.random.default_rng(seed)
            )
            sim.reset(horizon=5)
            shifts.append(sim.apply_latency(Side.SELL, 0.5))
        return float(np.mean(shifts))

    adverse = mean_shift("adverse")
    bridge = mean_shift("bridge")
    assert adverse < 0  # a SELL is always pushed down: latency is a pure cost
    assert abs(bridge) < abs(adverse) / 3


def test_bridge_latency_deviation_is_transient():
    cfg = MarketConfig(price_process="gbm", rng_mode="streams", latency_model="bridge")
    sim = MarketSimulator(cfg, rng=np.random.default_rng(4))
    sim.reset(horizon=8)
    sim.apply_latency(Side.SELL, 0.5)
    assert sim._intrastep_dev != 0.0
    sim.advance()
    assert sim._intrastep_dev == 0.0
    assert sim.mid == pytest.approx(float(sim.paths.mid[1]))


# =============================================================== antithetic variates
def test_antithetic_twin_mirrors_the_log_returns():
    """With constant vol and no drift, log-returns of the twin satisfy r + r' = -sigma^2 dt."""
    cfg = MarketConfig(
        price_process="gbm",
        rng_mode="streams",
        vol_process="none",
        volatility=0.02,
        drift=0.0,
        imbalance_alpha=0.0,
    )
    plain = generate_market_paths(cfg, 20, NoiseStreams.from_seed(33, antithetic=False))
    twin = generate_market_paths(cfg, 20, NoiseStreams.from_seed(33, antithetic=True))
    r1 = np.diff(np.log(plain.mid))
    r2 = np.diff(np.log(twin.mid))
    assert np.allclose(r1 + r2, -(cfg.volatility**2) * cfg.dt)


def test_antithetic_reduces_the_standard_error_of_mean_is():
    """Pairing Z with -Z cancels most of the shared price risk in execution cost."""
    cfg = MarketConfig(price_process="gbm", rng_mode="streams", volatility=0.03)
    factory = lambda: ExecutionEnv(cfg, ExecutionConfig(horizon=10))  # noqa: E731

    plain = evaluate(factory, TWAP(), n_episodes=120, base_seed=100, antithetic=False)
    anti = evaluate(factory, TWAP(), n_episodes=120, base_seed=100, antithetic=True)

    def is_values(res):
        return np.array([m["implementation_shortfall_bps"] for m in res.episode_metrics])

    se_plain = is_values(plain).std(ddof=1) / np.sqrt(120)
    pair_means = is_values(anti).reshape(-1, 2).mean(axis=1)
    se_anti = pair_means.std(ddof=1) / np.sqrt(pair_means.size)
    assert se_anti < se_plain


def test_block_bootstrap_is_wider_for_dependent_pairs():
    """Resampling dependent antithetic pairs individually understates the variance."""
    rng = np.random.default_rng(34)
    pair = rng.normal(0.0, 1.0, 200)
    diff = np.repeat(pair, 2)  # perfectly dependent within each pair
    bench = np.zeros_like(diff)

    lo1, hi1 = paired_bootstrap_ci(diff, bench, n_boot=2_000, rng=0, block=1)
    lo2, hi2 = paired_bootstrap_ci(diff, bench, n_boot=2_000, rng=0, block=2)
    assert (hi2 - lo2) > (hi1 - lo1)


# =============================================================== regimes & AC validation
def test_brownian_regimes_are_registered_and_valid():
    for name in ("abm_ac_optimal", "gbm_medium", "clustered_vol", "leverage"):
        assert name in list_regimes()
        cfg = get_regime(name)
        assert cfg.rng_mode == "streams"
        RunConfig(market=cfg).validate()
    assert get_regime("abm_ac_optimal").price_process == "abm"
    assert get_regime("leverage").leverage_corr < 0


def test_ac_schedule_matches_itself_and_differs_from_twap():
    """The AC-optimality probe: schedule_L1 is zero against itself and positive vs TWAP."""
    cfg = get_regime("abm_ac_optimal")
    factory = lambda: ExecutionEnv(cfg, ExecutionConfig(horizon=10))  # noqa: E731
    results = {
        "AlmgrenChriss": evaluate(factory, AlmgrenChriss(1e-6), n_episodes=12, base_seed=0),
        "TWAP": evaluate(factory, TWAP(), n_episodes=12, base_seed=0),
    }
    table = schedule_gap_table(results, reference="AlmgrenChriss")
    assert table.loc["AlmgrenChriss", "schedule_L1"] == pytest.approx(0.0, abs=1e-12)
    assert table.loc["TWAP", "schedule_L1"] > 0.0
    with pytest.raises(KeyError):
        schedule_gap_table(results, reference="missing")


def test_schedule_gap_handles_ragged_and_empty_input():
    assert schedule_gap([0.5, 0.5], [0.5, 0.5, 0.1]) == pytest.approx(0.0)
    assert np.isnan(schedule_gap([], []))


# =============================================================== config validation
@pytest.mark.parametrize(
    "kwargs",
    [
        {"price_process": "bogus"},
        {"dt": 0.0},
        {"rng_mode": "bogus"},
        {"normal_method": "bogus"},
        {"vol_process": "bogus"},
        {"vol_persistence": 1.0},
        {"leverage_corr": -1.5},
        {"latency_model": "bogus"},
    ],
)
def test_runconfig_rejects_bad_brownian_params(kwargs):
    with pytest.raises(ValueError):
        RunConfig(market=MarketConfig(**kwargs)).validate()


def test_brownian_config_round_trips_through_yaml(tmp_path):
    from rl_execution.utils.config_io import dump_run_config, load_run_config

    rc = RunConfig(market=get_regime("leverage"))
    path = tmp_path / "run.yaml"
    dump_run_config(rc, str(path))
    assert load_run_config(str(path)).to_dict() == rc.to_dict()
