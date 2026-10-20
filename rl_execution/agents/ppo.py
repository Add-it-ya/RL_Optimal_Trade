"""PPO agent: a thin wrapper around Stable-Baselines3's battle-tested implementation.

It exposes the common :class:`BaseAgent` interface so PPO trains and backtests exactly like
the from-scratch DQN agents.  PPO acts on the continuous, TWAP-relative action space and is
normally trained on a ``DummyVecEnv`` of several environments (see
:func:`rl_execution.training.train_agent`): this environment is cheap, so stepping 16 copies
in one process beats spawning workers.

Defaults suit the execution task: ``gamma = 1`` (finite horizon, time left is observed), a
small initial action noise (``log_std_init = -1``, i.e. a standard deviation of ~0.37
around the TWAP pace instead of SB3's 1.0, which would mostly hit the "wait" and
"liquidate now" ends of the action range), and a learning rate that decays linearly to 0.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Optional

import numpy as np

from rl_execution.agents.base import BaseAgent, resolve_device
from rl_execution.config import ActionType


class LinearSchedule:
    """``start * progress_remaining`` -- SB3 calls it with progress going from 1 to 0."""

    def __init__(self, start: float) -> None:
        self.start = float(start)

    def __call__(self, progress_remaining: float) -> float:
        return self.start * float(progress_remaining)


DEFAULT_HP: Dict[str, Any] = dict(
    n_steps=256,  # per environment; x n_envs transitions per update
    batch_size=1024,
    n_epochs=10,
    gamma=1.0,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    max_grad_norm=0.5,
)
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_LOG_STD_INIT = -1.0


class PPOAgent(BaseAgent):
    """Adapter around a Stable-Baselines3 ``PPO`` model."""

    name = "PPO"
    action_type = ActionType.CONTINUOUS

    def __init__(
        self,
        env=None,
        model=None,
        seed: Optional[int] = None,
        net_arch=(64, 64),
        log_std_init: float = DEFAULT_LOG_STD_INIT,
        learning_rate: Any = DEFAULT_LEARNING_RATE,
        verbose: int = 0,
        device: str = "cpu",
        **hp,
    ):
        if model is not None:
            self.model = model
            return

        from stable_baselines3 import PPO

        params = dict(DEFAULT_HP)
        params.update(hp)
        if isinstance(learning_rate, (int, float)):
            learning_rate = LinearSchedule(learning_rate)
        self.model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=verbose,
            device=resolve_device(device),
            learning_rate=learning_rate,
            policy_kwargs=dict(net_arch=list(net_arch), log_std_init=log_std_init),
            **params,
        )

    # ------------------------------------------------------------------ weights
    def get_weights(self) -> Dict[str, Any]:
        """A detached copy of the policy's weights (for checkpoint selection)."""
        return copy.deepcopy(self.model.policy.state_dict())

    def set_weights(self, weights: Dict[str, Any]) -> None:
        self.model.policy.load_state_dict(weights)

    # ------------------------------------------------------------------ training
    def train(
        self,
        total_timesteps: int,
        progress: bool = False,
        callback: Optional[Callable[[int], None]] = None,
        **kwargs,
    ) -> "PPOAgent":
        """Train for ``total_timesteps`` env steps; ``callback(steps_so_far)`` runs after each
        vectorised step (so ``steps_so_far`` advances by the number of environments)."""
        sb3_callback = None
        if callback is not None:
            from stable_baselines3.common.callbacks import BaseCallback

            forward: Callable[[int], None] = callback

            class _Forward(BaseCallback):
                def _on_step(self) -> bool:
                    forward(int(self.num_timesteps))
                    return True

            sb3_callback = _Forward()
        self.model.learn(
            total_timesteps=int(total_timesteps), progress_bar=progress, callback=sb3_callback
        )
        return self

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        action, _ = self.model.predict(
            np.asarray(obs, dtype=np.float32), deterministic=deterministic
        )
        return action

    def save(self, path: str) -> None:
        """Save to ``path`` exactly (SB3 would otherwise append ``.zip`` to a bare name)."""
        with open(path, "wb") as f:
            self.model.save(f)

    @classmethod
    def load(cls, path: str, env=None) -> "PPOAgent":
        from stable_baselines3 import PPO

        return cls(model=PPO.load(path, env=env, device="cpu"))
