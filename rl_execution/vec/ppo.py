"""PPO on the batched environment: one rollout is one episode of every environment.

A compact, CleanRL-style PPO whose defaults mirror the Stable-Baselines3 setup used for the
numpy runs (:mod:`rl_execution.agents.ppo`), so the two can be compared on E1: separate
64-64 tanh actor and critic, orthogonal initialisation (gain sqrt(2), 0.01 for the action
mean, 1 for the value), a state-independent log-std starting at -1, Adam (eps 1e-5) with a
linear learning-rate decay, gamma 1, GAE 0.95, clip 0.2, 10 epochs, per-minibatch advantage
normalisation, value coefficient 0.5 and gradient clipping at 0.5.

Every rollout runs one full episode in each of ``n_envs`` environments (``horizon`` steps).
Steps after an episode has ended are masked out of the advantages and the loss, and only
real steps count towards ``total_timesteps``.  With 256 environments a rollout holds about
5,000 transitions, close to SB3's 16 x 256, so the optimisation is comparable; larger
batches are what a GPU is for, but may need their own tuning.

Checkpoint selection mirrors :class:`~rl_execution.training.CheckpointSelector`: the
deterministic policy is scored on the validation seeds every ``eval_every`` steps and the
best weights are kept.  The validation episodes are the numpy simulator's own markets for
those seeds (:func:`~rl_execution.vec.torch_env.reference_paths`), replayed in one batch,
so selection costs milliseconds instead of minutes and still sees exactly the episodes the
numpy pipeline would.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from rl_execution.agents.base import BaseAgent
from rl_execution.config import ActionType
from rl_execution.vec.torch_env import OBS_DIM, BatchedExecutionEnv, BatchPaths

#: File-format tag written into saved actors.
FORMAT = "rlx-vecppo-v1"


@dataclass
class VecPPOConfig:
    """Hyperparameters of :func:`train_vec_ppo` (defaults mirror the SB3 runs)."""

    n_envs: int = 256
    learning_rate: float = 3e-4
    lr_end_fraction: float = 0.0
    n_epochs: int = 10
    minibatch_size: int = 1024
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: Sequence[int] = (64, 64)
    log_std_init: float = -1.0


# --------------------------------------------------------------------------- networks
def _mlp(sizes: Sequence[int], out: int, out_gain: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        lin = nn.Linear(a, b)
        nn.init.orthogonal_(lin.weight, math.sqrt(2.0))
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]
    head = nn.Linear(sizes[-1], out)
    nn.init.orthogonal_(head.weight, out_gain)
    nn.init.zeros_(head.bias)
    return nn.Sequential(*layers, head)


class ActorCritic(nn.Module):
    """Gaussian policy (mean network + state-independent log-std) and a value network."""

    def __init__(self, hidden: Sequence[int] = (64, 64), log_std_init: float = -1.0) -> None:
        super().__init__()
        sizes = [OBS_DIM, *hidden]
        self.actor = _mlp(sizes, 1, 0.01)
        self.critic = _mlp(sizes, 1, 1.0)
        self.log_std = nn.Parameter(torch.full((1,), float(log_std_init)))

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(obs).squeeze(-1)
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


class TorchPolicyAgent(BaseAgent):
    """A trained actor in the :class:`BaseAgent` interface, for the numpy pipeline.

    ``predict`` returns the deterministic action (the Gaussian mean, clipped to [-1, 1] as
    SB3 does), so ``rlx evaluate`` scores it on the reference environment like any other run.
    """

    name = "VecPPO"
    action_type = ActionType.CONTINUOUS

    def __init__(self, model: ActorCritic, arch: Optional[Dict[str, Any]] = None) -> None:
        self.model = model.cpu().eval()
        self.arch = dict(arch or {})

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(-1, OBS_DIM)
            mean = self.model.actor(x).squeeze(-1)
            if not deterministic:
                mean = mean + self.model.log_std.exp() * torch.randn_like(mean)
        action = np.clip(mean.numpy().astype(np.float32), -1.0, 1.0)
        return action.reshape(-1) if action.size > 1 else action.reshape(1)

    def get_weights(self) -> Dict[str, Any]:
        return copy.deepcopy(self.model.state_dict())

    def set_weights(self, weights: Dict[str, Any]) -> None:
        self.model.load_state_dict(weights)

    def save(self, path: str) -> None:
        torch.save(
            {"format": FORMAT, "arch": self.arch, "state_dict": self.model.state_dict()}, path
        )

    @classmethod
    def load(cls, path: str, env=None) -> "TorchPolicyAgent":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if blob.get("format") != FORMAT:
            raise ValueError(f"{path} is not a saved VecPPO actor ({blob.get('format')!r})")
        arch = blob["arch"]
        model = ActorCritic(tuple(arch["hidden"]), float(arch["log_std_init"]))
        model.load_state_dict(blob["state_dict"])
        return cls(model, arch)


# --------------------------------------------------------------------------- validation
class BatchedValidator:
    """Scores a policy on fixed validation episodes in one batched rollout."""

    def __init__(self, env: BatchedExecutionEnv, paths: BatchPaths) -> None:
        self.env, self.paths = env, paths

    @torch.no_grad()
    def score(self, model: ActorCritic) -> Dict[str, float]:
        env = self.env
        obs = env.reset(self.paths)
        for _ in range(env.horizon):
            action = model.actor(obs.float()).squeeze(-1)
            obs, _, _, _ = env.step(action)
        s = env.episode_summary()
        return {
            "objective_bps": float(s["objective_bps"].mean()),
            "implementation_shortfall_bps": float(s["implementation_shortfall_bps"].mean()),
            "effective_steps": float(s["effective_steps"].mean()),
        }


# --------------------------------------------------------------------------- training
def train_vec_ppo(
    env: BatchedExecutionEnv,
    total_timesteps: int,
    config: Optional[VecPPOConfig] = None,
    validator: Optional[BatchedValidator] = None,
    eval_every: Optional[int] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Train PPO on ``env`` for ``total_timesteps`` real environment steps.

    Returns ``{"agent", "episodes", "validation", "selected_step"}``: the agent carries the
    selected weights when a validator is given (else the final ones); ``episodes`` has one
    row per rollout (mean return and objective of its episodes); ``validation`` the curve.
    """
    cfg = config or VecPPOConfig()
    dev = env.device
    model = ActorCritic(cfg.hidden, cfg.log_std_init).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)
    H, N = env.horizon, env.n

    episodes: List[Dict[str, float]] = []
    validation: List[Dict[str, float]] = []
    best = {"objective": float("inf"), "step": None, "weights": None}

    def validate(step: int) -> None:
        if validator is None:
            return
        row = {"step": int(step), **validator.score(model)}
        validation.append(row)
        if row["objective_bps"] < best["objective"]:
            best.update(objective=row["objective_bps"], step=int(step))
            best["weights"] = copy.deepcopy(model.state_dict())
        if log:
            log(f"step {step:>10,}  validation objective {row['objective_bps']:.3f} bps")

    validate(0)
    next_eval = int(eval_every) if eval_every else None
    steps = 0
    while steps < total_timesteps:
        frac = max(1.0 - steps / float(total_timesteps), 0.0)
        lr = cfg.learning_rate * (cfg.lr_end_fraction + (1.0 - cfg.lr_end_fraction) * frac)
        for group in opt.param_groups:
            group["lr"] = lr

        # ---- rollout: one episode in every environment ----
        obs_buf = torch.zeros(H, N, OBS_DIM, device=dev)
        act_buf = torch.zeros(H, N, device=dev)
        logp_buf = torch.zeros(H, N, device=dev)
        val_buf = torch.zeros(H + 1, N, device=dev)
        rew_buf = torch.zeros(H, N, device=dev)
        done_buf = torch.zeros(H, N, device=dev)
        alive_buf = torch.zeros(H, N, dtype=torch.bool, device=dev)
        obs = env.reset()
        with torch.no_grad():
            for t in range(H):
                o = obs.float()
                dist = model.dist(o)
                action = dist.sample()
                obs_buf[t], act_buf[t] = o, action
                logp_buf[t], val_buf[t] = dist.log_prob(action), model.value(o)
                obs, reward, done, info = env.step(action)
                rew_buf[t], done_buf[t] = reward.float(), done.float()
                alive_buf[t] = info["alive"]
            # every episode has ended after `horizon` steps: no bootstrap value
            adv = torch.zeros(H, N, device=dev)
            last = torch.zeros(N, device=dev)
            for t in reversed(range(H)):
                nonterminal = 1.0 - done_buf[t]
                delta = rew_buf[t] + cfg.gamma * val_buf[t + 1] * nonterminal - val_buf[t]
                last = delta + cfg.gamma * cfg.gae_lambda * nonterminal * last
                adv[t] = last
            returns = adv + val_buf[:H]
        summary = env.episode_summary()
        n_real = int(alive_buf.sum())
        steps += n_real
        episodes.append(
            {
                "step": steps,
                "return": float(summary["cum_reward"].mean()),
                "objective_bps": float(summary["objective_bps"].mean()),
            }
        )

        # ---- update on the real steps only ----
        mask = alive_buf.reshape(-1)
        b_obs = obs_buf.reshape(-1, OBS_DIM)[mask]
        b_act, b_logp = act_buf.reshape(-1)[mask], logp_buf.reshape(-1)[mask]
        b_adv, b_ret = adv.reshape(-1)[mask], returns.reshape(-1)[mask]
        size = b_obs.shape[0]
        mb = max(1, min(int(cfg.minibatch_size), size))
        for _ in range(cfg.n_epochs):
            perm = torch.randperm(size, device=dev)
            for start in range(0, size, mb):
                idx = perm[start : start + mb]
                dist = model.dist(b_obs[idx])
                logp = dist.log_prob(b_act[idx])
                ratio = torch.exp(logp - b_logp[idx])
                a = b_adv[idx]
                if a.numel() > 1:
                    a = (a - a.mean()) / (a.std() + 1e-8)
                pg = -torch.min(
                    a * ratio, a * torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
                ).mean()
                v_loss = ((model.value(b_obs[idx]) - b_ret[idx]) ** 2).mean()
                loss = pg + cfg.vf_coef * v_loss - cfg.ent_coef * dist.entropy().mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()

        if next_eval is not None and steps >= next_eval:
            validate(steps)  # like CheckpointSelector: score once the interval has passed
            while next_eval <= steps:
                next_eval += int(eval_every)  # type: ignore[arg-type]

    if validator is not None and (not validation or validation[-1]["step"] != steps):
        validate(steps)
    if best["weights"] is not None:
        model.load_state_dict(best["weights"])
    arch = {"hidden": list(cfg.hidden), "log_std_init": cfg.log_std_init, "config": asdict(cfg)}
    return {
        "agent": TorchPolicyAgent(model, arch),
        "episodes": episodes,
        "validation": validation,
        "selected_step": best["step"],
    }
