"""The execution environment as batched tensor operations, for many episodes at once.

:class:`BatchedExecutionEnv` steps ``N`` episodes in lockstep on a CPU or a GPU.  It is a
re-implementation of :class:`~rl_execution.envs.ExecutionEnv` -- the same book, fills,
impact, mark-to-market reward and observation -- written over ``[N]`` and ``[N, levels]``
tensors instead of one episode at a time.  That is possible because the whole exogenous
market is drawn before the agent acts and impact is an additive overlay on it (see
:mod:`rl_execution.envs.market`), so a step needs no per-episode branching.

The numpy environment stays the **reference**: every reported number comes from it, and
``tests/test_vec_env.py`` checks this one against it step by step on identical paths
(:func:`reference_paths`).  Two ways to get paths:

* drawn here, in torch (:func:`generate_paths`) -- same distributions as
  :func:`~rl_execution.envs.processes.generate_market_paths`, different random numbers;
  used for training;
* copied from the numpy simulator for given seeds (:func:`reference_paths`) -- identical
  episodes; used for the parity tests and for checkpoint selection on validation seeds.

Episodes run in lockstep for exactly ``horizon`` steps.  An episode that finishes early
(its order fully filled) keeps stepping with zero trades and zero reward until the batch
ends; ``step`` reports which episodes were still alive so a learner can mask those steps.
Not modelled here: latency (``latency_steps`` must be 0) and the ambient market volume,
which only the VWAP baseline uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from rl_execution.config import ActionType, ExecutionConfig, MarketConfig
from rl_execution.envs.execution_env import DEPTH_REFERENCE, MULTIPLIERS

#: Scalar :class:`MarketConfig` fields that may differ between the episodes of one batch.
_PARAM_FIELDS = (
    "initial_price",
    "tick_size",
    "base_spread",
    "spread_vol",
    "base_depth",
    "level_growth",
    "depth_vol",
    "volatility",
    "drift",
    "mean_reversion",
    "vol_of_vol",
    "dt",
    "vol_persistence",
    "leverage_corr",
    "permanent_impact",
    "temporary_impact",
    "imbalance_persistence",
    "imbalance_strength",
    "imbalance_alpha",
)
_VOL_MODES = {"none": 0, "iid_lognormal": 1, "expou": 2}

#: Observation bounds, identical to ``ExecutionEnv.observation_space``.
_OBS_LOW = (0.0, 0.0, -5.0, -1.0, 0.0, 0.0, 0.0)
_OBS_HIGH = (1.0, 1.0, 5.0, 1.0, 5.0, 10.0, 10.0)
OBS_DIM = len(_OBS_LOW)


# --------------------------------------------------------------------------- parameters
@dataclass
class BatchMarket:
    """One market per episode: every scalar parameter as an ``[N]`` tensor.

    ``kappa`` is the OU mean reversion actually applied (0 unless ``price_process`` is
    ``ou``, as in the numpy simulator), ``is_gbm`` selects the geometric process and
    ``vol_mode`` the volatility process (0 none, 1 iid lognormal, 2 exp-OU).
    """

    p: Dict[str, Tensor]
    kappa: Tensor
    is_gbm: Tensor
    vol_mode: Tensor
    stationary_imbalance: Tensor
    n_levels: int

    @property
    def n(self) -> int:
        return int(self.kappa.shape[0])

    @classmethod
    def from_configs(
        cls,
        configs: Union[MarketConfig, Sequence[MarketConfig]],
        n: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "BatchMarket":
        """Stack ``configs`` (or repeat a single config ``n`` times)."""
        if isinstance(configs, MarketConfig):
            if n is None:
                raise ValueError("n is required with a single MarketConfig")
            configs = [configs] * int(n)
        configs = list(configs)
        levels = {int(c.n_levels) for c in configs}
        if len(levels) != 1:
            raise ValueError(f"every market in a batch needs the same n_levels, got {levels}")
        for c in configs:
            if c.price_process not in ("abm", "gbm", "ou"):
                raise ValueError(f"unknown price_process {c.price_process!r}")
            if c.vol_process not in _VOL_MODES:
                raise ValueError(f"unknown vol_process {c.vol_process!r}")

        def col(values) -> Tensor:
            return torch.tensor(values, dtype=dtype, device=device)

        params = {f: col([float(getattr(c, f)) for c in configs]) for f in _PARAM_FIELDS}
        return cls(
            p=params,
            kappa=col([float(c.mean_reversion) if c.price_process == "ou" else 0.0 for c in configs]),
            is_gbm=torch.tensor([c.price_process == "gbm" for c in configs], device=device),
            vol_mode=torch.tensor([_VOL_MODES[c.vol_process] for c in configs], device=device),
            stationary_imbalance=torch.tensor(
                [bool(c.imbalance_ou_scaling) for c in configs], device=device
            ),
            n_levels=levels.pop(),
        )  # fmt: skip


def sample_randomized_markets(
    n: int,
    generator: torch.Generator,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> BatchMarket:
    """``n`` markets from the domain-randomised family, drawn in torch.

    Same distribution as :func:`~rl_execution.experiments.regimes.randomized_market_config`
    (``brownian=False``), whose ranges it reads, so the two cannot drift apart; different
    random numbers.
    """
    from rl_execution.experiments.regimes import (
        _BASE,
        _RANDOMIZED_MEAN_REVERSION,
        _RANDOMIZED_UNIFORM,
    )

    base = BatchMarket.from_configs(_BASE, n=n, device=device, dtype=dtype)
    for name, (low, high, unit) in _RANDOMIZED_UNIFORM.items():
        u = torch.rand(n, generator=generator, device=device, dtype=dtype)
        base.p[name] = (low + (high - low) * u) * unit
    choices = torch.tensor(_RANDOMIZED_MEAN_REVERSION, dtype=dtype, device=device)
    pick = torch.randint(len(choices), (n,), generator=generator, device=device)
    base.p["mean_reversion"] = choices[pick]
    base.kappa = base.p["mean_reversion"].clone()  # OU whenever mean reversion is drawn
    base.vol_mode = torch.full((n,), _VOL_MODES["iid_lognormal"], device=device)
    return base


# --------------------------------------------------------------------------- paths
@dataclass
class BatchPaths:
    """The exogenous market of ``N`` episodes; ``P`` points, ``P - 1`` increments."""

    mid: Tensor  # [N, P]
    sigma: Tensor  # [N, P - 1]
    imbalance: Tensor  # [N, P]
    spread: Tensor  # [N, P]
    depth_bid: Tensor  # [N, P, L]
    depth_ask: Tensor  # [N, P, L]

    @property
    def n_points(self) -> int:
        return int(self.mid.shape[1])


def generate_paths(market: BatchMarket, n_points: int, generator: torch.Generator) -> BatchPaths:
    """Draw ``N`` exogenous markets in torch (the batched ``generate_market_paths``)."""
    p = market.p
    N, P, L = market.n, int(n_points), market.n_levels
    n = P - 1
    dev, dt_ = p["dt"].device, p["dt"].dtype

    def randn(*shape) -> Tensor:
        return torch.randn(*shape, generator=generator, device=dev, dtype=dt_)

    def col(x: Tensor) -> Tensor:
        return x.unsqueeze(1)

    # price shocks, and volatility shocks correlated with the previous step's price shock
    z_price = randn(N, n)
    z_indep = randn(N, n + 1)
    z_lag = torch.cat([z_indep[:, n : n + 1], z_price[:, :-1]], dim=1)
    rho = col(p["leverage_corr"].clamp(-1.0, 1.0))
    z_vol = rho * z_lag + torch.sqrt(1.0 - rho * rho) * z_indep[:, :n]

    sigma_bar, xi = col(p["volatility"]), col(p["vol_of_vol"])
    iid = sigma_bar * torch.exp(xi * z_vol)
    phi_v = col(p["vol_persistence"].clamp(0.0, 0.999999) ** p["dt"])
    inn_v = xi * torch.sqrt(1.0 - phi_v * phi_v)
    y = [xi[:, 0] * z_vol[:, 0]]
    for t in range(1, n):
        y.append(phi_v[:, 0] * y[-1] + inn_v[:, 0] * z_vol[:, t])
    expou = sigma_bar * torch.exp(torch.stack(y, dim=1))
    mode = col(market.vol_mode)
    sigma = torch.where(mode == 1, iid, torch.where(mode == 2, expou, sigma_bar.expand(N, n)))
    sigma = torch.where(xi <= 0, sigma_bar.expand(N, n), sigma)

    # latent imbalance (AR(1), clipped) and the alpha it adds to the next return; the
    # stationary start reuses the first innovation exactly as the numpy simulator does
    z_imb = randn(N, n)
    phi_i = p["imbalance_persistence"].clamp(0.0, 0.999999) ** p["dt"]
    stat = market.stationary_imbalance
    inn_i = torch.where(stat, torch.sqrt(1.0 - phi_i * phi_i), 1.0 - phi_i)
    imb = [torch.where(stat, z_imb[:, 0], torch.zeros_like(phi_i))]
    for t in range(n):
        imb.append(phi_i * imb[-1] + inn_i * z_imb[:, t])
    imbalance = torch.stack(imb, dim=1).clamp(-1.0, 1.0)
    alpha = col(p["imbalance_alpha"]) * imbalance[:, :n]

    # mid price: arithmetic BM is OU with kappa = 0; GBM in log space
    s0, dt, kappa = col(p["initial_price"]), col(p["dt"]), col(market.kappa)
    tick, mu = col(p["tick_size"]), col(p["drift"])
    decay = torch.exp(-kappa * dt)
    safe_k = torch.where(kappa > 0, kappa, torch.ones_like(kappa))
    unit_std = torch.where(
        kappa > 0, torch.sqrt((1.0 - decay * decay) / (2.0 * safe_k)), torch.sqrt(dt)
    )
    level = [s0[:, 0]]
    for t in range(n):
        drift = (mu[:, 0] + alpha[:, t]) * s0[:, 0] * dt[:, 0]
        shock = sigma[:, t] * s0[:, 0] * unit_std[:, 0] * z_price[:, t]
        level.append(s0[:, 0] + (level[-1] - s0[:, 0]) * decay[:, 0] + shock + drift)
    arithmetic = torch.maximum(torch.stack(level, dim=1), tick)
    log_ret = (mu + alpha - 0.5 * sigma * sigma) * dt + sigma * torch.sqrt(dt) * z_price
    zeros = torch.zeros(N, 1, device=dev, dtype=dt_)
    geometric = s0 * torch.exp(torch.cumsum(torch.cat([zeros, log_ret], dim=1), dim=1))
    mid = torch.where(col(market.is_gbm), geometric, arithmetic)

    spread = torch.maximum(
        col(p["base_spread"]) * torch.exp(col(p["spread_vol"]) * randn(N, P)), tick
    )
    spread[:, 0] = p["base_spread"]
    depth = torch.exp(p["depth_vol"].view(N, 1, 1, 1) * randn(N, P, 2, L))
    return BatchPaths(mid, sigma, imbalance, spread, depth[:, :, 0, :], depth[:, :, 1, :])  # fmt: skip


def reference_paths(
    env_factory: Callable[[], Any],
    n_episodes: int,
    base_seed: int,
    antithetic: bool = False,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[List[MarketConfig], BatchPaths]:
    """The exact markets the numpy environment would face on these seeds.

    Seeds and antithetic pairing follow :func:`~rl_execution.backtest.evaluate` (episode
    ``2k`` uses seed ``base_seed + k``, ``2k + 1`` its antithetic twin), and one environment
    from ``env_factory`` is reset for every episode, so a domain-randomised factory yields
    the same sequence of sampled markets as an evaluation would.  Returns each episode's
    :class:`MarketConfig` and the stacked paths.
    """
    env = env_factory()
    configs, mats = [], []
    for i in range(int(n_episodes)):
        seed = base_seed + (i // 2 if antithetic else i)
        options = {"antithetic": bool(i % 2)} if antithetic else None
        env.reset(seed=seed, options=options)
        p = env.market.paths
        configs.append(env.market_config)
        mats.append((p.mid, p.sigma, p.imbalance, p.spread, p.depth_bid, p.depth_ask))

    def stack(k: int) -> Tensor:
        return torch.as_tensor(np.stack([m[k] for m in mats]), dtype=dtype, device=device)

    return configs, BatchPaths(*(stack(k) for k in range(6)))


# --------------------------------------------------------------------------- environment
class BatchedExecutionEnv:
    """``N`` execution episodes stepped together as tensor operations.

    ``market`` is one :class:`MarketConfig` for every episode, a sequence of ``N`` configs,
    or ``"randomized"`` to draw a fresh market per episode from the domain-randomised family
    at every reset.  ``exec_config`` (order size, horizon, side, frictions, reward) is shared.
    """

    def __init__(
        self,
        n_envs: int,
        market: Union[MarketConfig, Sequence[MarketConfig], str, None] = None,
        exec_config: Optional[ExecutionConfig] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
    ) -> None:
        self.n = int(n_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.exec_config = exec_config or ExecutionConfig()
        ec = self.exec_config
        if float(ec.latency_steps) != 0.0:
            raise NotImplementedError("BatchedExecutionEnv does not model latency_steps > 0")
        self.horizon = int(ec.horizon)
        self.total = float(ec.total_inventory)
        self.sign = float(ec.side.sign)
        self.discrete = ec.action_type is ActionType.DISCRETE
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(int(seed))

        self.randomized = isinstance(market, str)
        if isinstance(market, str):
            if market != "randomized":
                raise ValueError(
                    f"market must be a MarketConfig, a sequence or 'randomized', got {market!r}"
                )
            self.market = sample_randomized_markets(self.n, self.generator, self.device, dtype)
        else:
            cfg = market if market is not None else MarketConfig()
            self.market = BatchMarket.from_configs(cfg, n=self.n, device=self.device, dtype=dtype)
        if self.market.n != self.n:
            raise ValueError(f"{self.market.n} market configs for {self.n} environments")

        f = lambda x: torch.tensor(x, dtype=dtype, device=self.device)  # noqa: E731
        self._obs_low, self._obs_high = f(_OBS_LOW), f(_OBS_HIGH)
        self._multipliers = f(MULTIPLIERS)
        self._levels = torch.arange(self.market.n_levels, device=self.device, dtype=dtype)
        self.paths: Optional[BatchPaths] = None
        self.t = 0

    # ------------------------------------------------------------------ helpers
    def _col(self, name: str) -> Tensor:
        return self.market.p[name]

    def _market_view(self) -> Dict[str, Tensor]:
        """Mid, spread, sigma, imbalance and the two book sides at each market index."""
        assert self.paths is not None
        p, k, rows = self.paths, self.k, self._rows
        tick = self._col("tick_size")
        mid = torch.maximum(p.mid[rows, k] + self.shift, tick)
        spread = p.spread[rows, k]
        sigma = p.sigma[rows, torch.clamp(k, max=p.sigma.shape[1] - 1)]
        imb_factor = p.imbalance[rows, k]

        half = torch.maximum(spread / 2.0, tick)
        best_bid, best_ask = mid - half, mid + half
        lv, tk = self._levels.unsqueeze(0), tick.unsqueeze(1)
        bid_prices = best_bid.unsqueeze(1) - lv * tk
        ask_prices = best_ask.unsqueeze(1) + lv * tk
        growth = (1.0 + self._col("level_growth")).unsqueeze(1) ** lv
        skew = self._col("imbalance_strength") * imb_factor.clamp(-1.0, 1.0)
        base = self._col("base_depth").unsqueeze(1) * growth
        bid_sizes = torch.clamp(base * p.depth_bid[rows, k] * (1.0 + skew).unsqueeze(1), min=1.0)
        ask_sizes = torch.clamp(base * p.depth_ask[rows, k] * (1.0 - skew).unsqueeze(1), min=1.0)
        return {
            "mid": mid,
            "sigma": sigma,
            "bid_prices": bid_prices,
            "bid_sizes": bid_sizes,
            "ask_prices": ask_prices,
            "ask_sizes": ask_sizes,
        }

    def _observe(self, view: Dict[str, Tensor]) -> Tensor:
        bid, ask = view["bid_sizes"], view["ask_sizes"]
        sb, sa = bid.sum(1), ask.sum(1)
        imbalance = (sb - sa) / (sb + sa)
        depth5 = bid[:, :5].sum(1) + ask[:, :5].sum(1)
        spread = view["ask_prices"][:, 0] - view["bid_prices"][:, 0]
        vol = self._col("volatility")
        sigma_ref = torch.clamp(vol * torch.sqrt(self._col("dt")), min=1e-12)
        t = float(self.t)
        obs = torch.stack(
            [
                self.remaining / self.total,
                torch.full_like(self.remaining, (self.horizon - t) / self.horizon),
                self.sign * (view["mid"] / self.arrival - 1.0) / (sigma_ref * max(t, 1.0) ** 0.5),
                self.sign * imbalance,
                torch.where(vol > 0, view["sigma"] / torch.where(vol > 0, vol, 1.0), 1.0),
                spread / view["mid"] * 1e4 / 10.0,
                depth5 / DEPTH_REFERENCE,
            ],
            dim=1,
        )
        return torch.clamp(obs, self._obs_low, self._obs_high)

    def _fraction(self, action: Tensor) -> Tensor:
        """Fraction of the remaining inventory each action trades (``action_to_fraction``)."""
        steps_left = float(max(self.horizon - self.t, 1))
        if self.discrete:
            idx = action.reshape(self.n).long().clamp(0, len(MULTIPLIERS) - 1)
            multiplier = self._multipliers[idx]
        else:
            a = action.reshape(self.n).to(self.dtype).clamp(-1.0, 1.0)
            multiplier = torch.where(a <= 0, 1.0 + a, steps_left ** torch.clamp(a, min=0.0))
        return torch.clamp(multiplier / steps_left, max=1.0)

    # ------------------------------------------------------------------ API
    def reset(self, paths: Optional[BatchPaths] = None) -> Tensor:
        """Start ``N`` new episodes, on ``paths`` if given, else on freshly drawn ones."""
        if paths is None:
            if self.randomized:
                self.market = sample_randomized_markets(
                    self.n, self.generator, self.device, self.dtype
                )
            paths = generate_paths(self.market, self.horizon + 2, self.generator)
        if paths.mid.shape[0] != self.n or paths.n_points < self.horizon + 1:
            raise ValueError("paths must cover every environment and horizon + 1 points")
        self.paths = paths
        zeros = torch.zeros(self.n, dtype=self.dtype, device=self.device)
        self._rows = torch.arange(self.n, device=self.device)
        self.k = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.t = 0
        self.shift = zeros.clone()
        self.remaining = torch.full_like(zeros, self.total)
        self.arrival = torch.maximum(paths.mid[:, 0], self._col("tick_size"))
        self.executed_shares = zeros.clone()
        self.executed_notional = zeros.clone()
        self.commission = zeros.clone()
        self.risk_penalty_bps = zeros.clone()
        self.cum_reward = zeros.clone()
        self.sum_sq_traded = zeros.clone()
        self.alive = torch.ones(self.n, dtype=torch.bool, device=self.device)
        return self._observe(self._market_view())

    def step(self, action: Tensor) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        """Advance every episode one decision step.

        Returns ``(obs, reward, done, info)``: ``done`` marks the episodes that ended on this
        step, and ``info["alive"]`` the ones that were still running before it (the rest
        traded nothing and got zero reward).
        """
        if self.paths is None or self.t >= self.horizon:
            raise RuntimeError("call reset() first; every episode lasts exactly `horizon` steps")
        ec, sign = self.exec_config, self.sign
        tick = self._col("tick_size")
        view = self._market_view()
        alive_before = self.alive.clone()

        fraction = self._fraction(action)
        if self.t >= self.horizon - 1:
            fraction = torch.ones_like(fraction)  # the last step liquidates what is left
        shares = torch.clamp(fraction * self.remaining, min=0.0)
        shares = torch.minimum(shares, self.remaining)
        decision_mid = view["mid"]

        # walk one side of the book, sweeping past the displayed depth at one extra tick
        if sign > 0:
            prices, sizes, direction = view["bid_prices"], view["bid_sizes"], -1.0
        else:
            prices, sizes, direction = view["ask_prices"], view["ask_sizes"], 1.0
        before = torch.cumsum(sizes, dim=1) - sizes
        take = torch.minimum(torch.clamp(shares.unsqueeze(1) - before, min=0.0), sizes)
        in_book = take.sum(1)
        swept = torch.clamp(shares - in_book, min=0.0)
        if prices.shape[1] > 1:
            level_tick = (prices[:, 1] - prices[:, 0]).abs()
        else:
            level_tick = torch.full_like(shares, 0.01)
        sweep_price = prices[:, -1] + direction * level_tick
        notional = (take * prices).sum(1) + swept * sweep_price
        # the sweep fills the whole order; ``in_book + swept`` can miss ``shares`` by an
        # ulp, which in float32 (~1e-3 shares) would leave a full liquidation unfinished
        filled = shares
        traded = filled > 0
        avg_price = torch.where(traded, notional / torch.where(traded, filled, 1.0), decision_mid)

        # temporary impact concedes a per-share amount linear in size; permanent impact
        # shifts the mid for the rest of the episode
        eff_price = avg_price - sign * self._col("temporary_impact") * filled
        self.shift = self.shift - sign * self._col("permanent_impact") * filled
        fee = ec.commission_per_share * filled + (ec.commission_bps / 1e4) * eff_price * filled
        self.remaining = torch.clamp(self.remaining - filled, min=0.0)
        self.executed_shares = self.executed_shares + filled
        self.executed_notional = self.executed_notional + eff_price * filled
        self.commission = self.commission + fee
        self.sum_sq_traded = self.sum_sq_traded + (filled / self.total) ** 2

        bps = 1e4 / (self.arrival * self.total)
        fill_cost = (sign * filled * (decision_mid - eff_price) + fee) * bps

        # carry the rest through the next step: variance penalty, then mark to market
        self.t += 1
        terminated = (self.remaining <= 1e-9) | (self.t >= self.horizon)
        going = ~terminated
        p = self.paths
        mid_now = torch.maximum(p.mid[self._rows, self.k] + self.shift, tick)
        s0, dt = self._col("initial_price"), self._col("dt")
        level = torch.where(self.market.is_gbm, torch.maximum(mid_now, tick), s0)
        step_std_bps = view["sigma"] * level * torch.sqrt(dt) / self.arrival * 1e4
        held = self.remaining
        risk = ec.reward.risk_aversion * (step_std_bps * held / self.total) ** 2
        self.k = torch.where(going, torch.clamp(self.k + 1, max=p.n_points - 1), self.k)
        mid_next = torch.maximum(p.mid[self._rows, self.k] + self.shift, tick)
        hold = sign * held * (mid_next - decision_mid) * bps
        zero = torch.zeros_like(hold)
        risk = torch.where(going, risk, zero)
        hold = torch.where(going, hold, zero)

        self.risk_penalty_bps = self.risk_penalty_bps + risk
        reward = (hold - fill_cost - risk) * ec.reward.reward_scale
        self.cum_reward = self.cum_reward + reward
        done = alive_before & terminated
        self.alive = alive_before & going

        obs = self._observe(self._market_view())
        info = {
            "alive": alive_before,
            "filled": filled,
            "exec_price": eff_price,
            "fill_cost_bps": fill_cost,
            "hold_pnl_bps": hold,
            "risk_bps": risk,
        }
        return obs, reward, done, info

    def episode_summary(self) -> Dict[str, Tensor]:
        """Per-episode results (meaningful once every episode has ended)."""
        paper = self.arrival * self.total
        is_cash = self.sign * (paper - self.executed_notional) + self.commission
        is_bps = is_cash / paper * 1e4
        return {
            "implementation_shortfall_bps": is_bps,
            "risk_penalty_bps": self.risk_penalty_bps,
            "objective_bps": is_bps + self.risk_penalty_bps,
            "cum_reward": self.cum_reward,
            "effective_steps": torch.where(
                self.sum_sq_traded > 0, 1.0 / self.sum_sq_traded, torch.zeros_like(is_bps)
            ),
        }
