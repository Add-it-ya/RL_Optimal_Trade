"""``rlx train --agent vecppo``: train PPO on the batched environment from a RunConfig.

The result is an ordinary run folder (config, weights, provenance with the weights'
SHA-256, train log, validation curve), so ``rlx evaluate`` scores it on the numpy reference
environment exactly like the SB3 and DQN runs.
"""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Callable, Optional, Union

import torch

from rl_execution.agents.base import resolve_device
from rl_execution.config import ActionType, MarketConfig, RunConfig
from rl_execution.vec.ppo import BatchedValidator, VecPPOConfig, train_vec_ppo
from rl_execution.vec.torch_env import BatchedExecutionEnv, reference_paths

#: Batch size used when ``rlx train --agent vecppo`` is given no ``--n-envs``.
DEFAULT_VEC_ENVS = 256


def vec_config(rc: RunConfig) -> VecPPOConfig:
    """Hyperparameters from ``rc.agent_kwargs`` (unknown keys rejected) and ``rc.n_envs``."""
    known = {f.name for f in fields(VecPPOConfig)}
    unknown = set(rc.agent_kwargs) - known
    if unknown:
        raise ValueError(f"unknown vecppo agent_kwargs {sorted(unknown)}; known: {sorted(known)}")
    return replace(VecPPOConfig(**rc.agent_kwargs), n_envs=int(rc.n_envs))


def train_vec_run(
    rc: RunConfig, progress: bool = False, log: Optional[Callable[[str], None]] = None
):
    """Train the run described by ``rc`` and return a :class:`~rl_execution.training.TrainingResult`."""
    from rl_execution.experiments.regimes import get_regime
    from rl_execution.training import ENV_SEED_STRIDE, TrainingResult, make_env_factory
    from rl_execution.utils.seeding import set_global_seeds

    device = torch.device(resolve_device(rc.device))
    # deterministic algorithms are only guaranteed (and only cheap) on the CPU
    set_global_seeds(rc.seed, deterministic_torch=device.type == "cpu")
    ec = replace(rc.execution, action_type=ActionType.CONTINUOUS)
    market: Union[str, MarketConfig]
    if rc.randomized:
        market = "randomized"
    else:
        market = get_regime(rc.regime) if rc.regime else rc.market
    cfg = vec_config(rc)
    env = BatchedExecutionEnv(cfg.n_envs, market, ec, device=device, seed=rc.seed * ENV_SEED_STRIDE)

    # validation: the numpy simulator's own episodes for the validation seeds, replayed in
    # one batch (antithetic pairs share a market, so not for the randomised family)
    factory = make_env_factory(
        "vecppo",
        ec,
        regime=rc.regime,
        randomized=rc.randomized,
        market_config=rc.market,
        seed=rc.val_base_seed,
    )
    configs, paths = reference_paths(
        factory, rc.val_episodes, rc.val_base_seed, antithetic=not rc.randomized, device=device
    )
    validator = BatchedValidator(
        BatchedExecutionEnv(len(configs), configs, ec, device=device), paths
    )

    out = train_vec_ppo(
        env,
        rc.timesteps,
        cfg,
        validator=validator,
        eval_every=rc.eval_every,
        log=log if (progress and log) else (print if progress else None),
    )
    return TrainingResult(
        agent=out["agent"],
        episodes=out["episodes"],
        validation=out["validation"],
        selected_step=out["selected_step"],
    )
