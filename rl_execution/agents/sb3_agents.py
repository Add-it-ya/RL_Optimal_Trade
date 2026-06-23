"""Stable-Baselines3 agent wrappers (PPO, A2C, SAC, and SB3's DQN).

These wrap battle-tested SB3 implementations behind the common :class:`BaseAgent`
interface so they can be trained and backtested identically to the from-scratch agents.
SAC requires a continuous action space; SB3-DQN requires a discrete one; PPO and A2C
support both.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from rl_execution.agents.base import BaseAgent, resolve_device

# default, CPU-friendly hyper-parameters per algorithm
_DEFAULT_HP: Dict[str, Dict[str, Any]] = {
    "ppo": dict(n_steps=512, batch_size=128, gae_lambda=0.95, gamma=0.99,
                ent_coef=0.0, learning_rate=3e-4, n_epochs=10),
    "a2c": dict(n_steps=16, gamma=0.99, learning_rate=7e-4, ent_coef=0.0),
    "sac": dict(buffer_size=100_000, batch_size=256, learning_rate=3e-4,
                gamma=0.99, train_freq=1, learning_starts=1_000),
    "dqn": dict(buffer_size=100_000, batch_size=128, learning_rate=1e-3,
                gamma=0.99, train_freq=4, target_update_interval=500,
                learning_starts=1_000, exploration_fraction=0.5,
                exploration_final_eps=0.05),
}


def _algo_class(algo: str):
    algo = algo.lower()
    from stable_baselines3 import A2C, DQN, PPO, SAC

    table = {"ppo": PPO, "a2c": A2C, "sac": SAC, "dqn": DQN}
    if algo not in table:
        raise KeyError(f"Unsupported SB3 algo '{algo}'. Choose from {sorted(table)}.")
    return table[algo]


class SB3Agent(BaseAgent):
    """Thin adapter around a Stable-Baselines3 model."""

    def __init__(
        self,
        algo: str,
        env=None,
        model=None,
        seed: Optional[int] = None,
        net_arch=(64, 64),
        verbose: int = 0,
        device: str = "cpu",
        **hp,
    ):
        self.algo = algo.lower()
        self.name = {"ppo": "PPO", "a2c": "A2C", "sac": "SAC", "dqn": "SB3-DQN"}[self.algo]
        if model is not None:
            self.model = model
            return

        cls = _algo_class(self.algo)
        params = dict(_DEFAULT_HP[self.algo])
        params.update(hp)
        policy_kwargs = dict(net_arch=list(net_arch))
        self.model = cls(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=verbose,
            device=resolve_device(device),
            policy_kwargs=policy_kwargs,
            **params,
        )

    def train(self, total_timesteps: int, progress: bool = False, **kwargs) -> "SB3Agent":
        self.model.learn(total_timesteps=int(total_timesteps), progress_bar=progress)
        return self

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        action, _ = self.model.predict(np.asarray(obs, dtype=np.float32),
                                       deterministic=deterministic)
        return action

    def save(self, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, path: str, env=None, algo: str = "ppo") -> "SB3Agent":
        model = _algo_class(algo).load(path, env=env, device="cpu")
        return cls(algo, model=model)
