"""PPO agent: a thin wrapper around Stable-Baselines3's battle-tested implementation.

It exposes the common :class:`BaseAgent` interface so PPO trains and backtests exactly like
the from-scratch DQN agents.  PPO here acts on the continuous action space.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from rl_execution.agents.base import BaseAgent, resolve_device

# default, CPU-friendly hyper-parameters
DEFAULT_HP: Dict[str, Any] = dict(
    n_steps=512,
    batch_size=128,
    gae_lambda=0.95,
    gamma=0.99,
    ent_coef=0.0,
    learning_rate=3e-4,
    n_epochs=10,
)


class PPOAgent(BaseAgent):
    """Adapter around a Stable-Baselines3 ``PPO`` model."""

    name = "PPO"

    def __init__(
        self,
        env=None,
        model=None,
        seed: Optional[int] = None,
        net_arch=(64, 64),
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
        self.model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=verbose,
            device=resolve_device(device),
            policy_kwargs=dict(net_arch=list(net_arch)),
            **params,
        )

    def train(self, total_timesteps: int, progress: bool = False, **kwargs) -> "PPOAgent":
        self.model.learn(total_timesteps=int(total_timesteps), progress_bar=progress)
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
