"""A simple fixed-size uniform experience replay buffer for value-based agents."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Batch(NamedTuple):
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    """Ring-buffer storing ``(s, a, r, s', done)`` transitions."""

    def __init__(self, capacity: int, obs_dim: int, rng: np.random.Generator | None = None):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.pos = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done) -> None:
        i = self.pos
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def sample(self, batch_size: int) -> Batch:
        high = len(self)
        idx = self.rng.integers(0, high, size=batch_size)
        return Batch(
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )
