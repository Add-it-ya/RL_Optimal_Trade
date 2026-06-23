"""From-scratch DQN and Double DQN agents (PyTorch).

Both share the same training loop; ``double=True`` switches the temporal-difference target
from the vanilla DQN ``max_a' Q_target(s', a')`` to the Double-DQN decoupled form
``Q_target(s', argmax_a' Q_online(s', a'))`` (van Hasselt et al., 2016), which reduces the
maximisation bias of Q-learning.

Only a discrete action space is supported (one Q-value per discrete inventory fraction).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from rl_execution.agents.base import BaseAgent, resolve_device
from rl_execution.agents.replay_buffer import ReplayBuffer


def _torch():
    import torch  # local import so the simulation layers don't require torch
    return torch


class QNetwork:
    """Lazily-built MLP Q-network wrapper (keeps torch import local)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden=(64, 64)):
        torch = _torch()
        import torch.nn as nn

        layers = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, n_actions)]
        self.net = nn.Sequential(*layers)

    def __call__(self, x):
        return self.net(x)


class DQNAgent(BaseAgent):
    """Deep Q-Network with an optional Double-DQN target."""

    def __init__(
        self,
        env,
        double: bool = False,
        hidden=(64, 64),
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 100_000,
        batch_size: int = 128,
        learning_starts: int = 1_000,
        train_freq: int = 1,
        target_update_interval: int = 500,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_fraction: float = 0.5,
        max_grad_norm: float = 10.0,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        torch = _torch()
        import torch.nn as nn

        self.env = env
        self.double = double
        self.name = "DoubleDQN" if double else "DQN"
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update_interval = target_update_interval
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_fraction = eps_fraction
        self.max_grad_norm = max_grad_norm

        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.n_actions = int(env.action_space.n)
        self.device = torch.device(resolve_device(device))

        if seed is not None:
            torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.q = QNetwork(self.obs_dim, self.n_actions, hidden)
        self.q_target = QNetwork(self.obs_dim, self.n_actions, hidden)
        self.q.net.to(self.device)
        self.q_target.net.to(self.device)
        self.q_target.net.load_state_dict(self.q.net.state_dict())
        self.optimizer = torch.optim.Adam(self.q.net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber
        self.buffer = ReplayBuffer(buffer_size, self.obs_dim, rng=self.rng)
        self._hidden = hidden
        self.train_log: list[float] = []

    # ------------------------------------------------------------------ acting
    def _epsilon(self, step: int, total: int) -> float:
        frac = min(step / max(self.eps_fraction * total, 1), 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        torch = _torch()
        if (not deterministic) and self.rng.random() < self.eps_end:
            return int(self.rng.integers(0, self.n_actions))
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1).to(self.device)
            q = self.q(x)
            return int(torch.argmax(q, dim=1).item())

    # ------------------------------------------------------------------ training
    def train(self, total_timesteps: int, progress: bool = False, **kwargs) -> "DQNAgent":
        torch = _torch()
        env = self.env
        obs, _ = env.reset()
        ep_reward = 0.0

        iterator = range(int(total_timesteps))
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc=self.name, leave=False)
            except ImportError:
                pass

        for step in iterator:
            eps = self._epsilon(step, total_timesteps)
            if self.rng.random() < eps:
                action = int(self.rng.integers(0, self.n_actions))
            else:
                action = self.predict(obs, deterministic=True)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            self.buffer.add(obs, action, reward, next_obs, terminated)
            ep_reward += reward
            obs = next_obs
            if done:
                obs, _ = env.reset()
                ep_reward = 0.0

            if step >= self.learning_starts and step % self.train_freq == 0:
                self._update()
            if step % self.target_update_interval == 0:
                self.q_target.net.load_state_dict(self.q.net.state_dict())

        return self

    def _update(self) -> None:
        torch = _torch()
        batch = self.buffer.sample(self.batch_size)
        obs = torch.as_tensor(batch.obs).to(self.device)
        actions = torch.as_tensor(batch.actions).long().unsqueeze(1).to(self.device)
        rewards = torch.as_tensor(batch.rewards).unsqueeze(1).to(self.device)
        next_obs = torch.as_tensor(batch.next_obs).to(self.device)
        dones = torch.as_tensor(batch.dones).unsqueeze(1).to(self.device)

        q_values = self.q(obs).gather(1, actions)
        with torch.no_grad():
            if self.double:
                next_actions = torch.argmax(self.q(next_obs), dim=1, keepdim=True)
                next_q = self.q_target(next_obs).gather(1, next_actions)
            else:
                next_q = self.q_target(next_obs).max(dim=1, keepdim=True)[0]
            target = rewards + self.gamma * (1.0 - dones) * next_q

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.train_log.append(float(loss.item()))

    # ------------------------------------------------------------------ io
    def save(self, path: str) -> None:
        torch = _torch()
        torch.save(
            {
                "state_dict": self.q.net.state_dict(),
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
                "hidden": self._hidden,
                "double": self.double,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, env=None) -> "DQNAgent":
        torch = _torch()
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(env, double=ckpt["double"], hidden=tuple(ckpt["hidden"]))
        agent.q.net.load_state_dict(ckpt["state_dict"])
        agent.q_target.net.load_state_dict(ckpt["state_dict"])
        return agent
