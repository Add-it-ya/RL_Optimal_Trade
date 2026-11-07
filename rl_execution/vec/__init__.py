"""Batched (GPU-capable) execution environment and the PPO that trains on it.

The numpy :class:`~rl_execution.envs.ExecutionEnv` stays the reference: trained policies
are exported and evaluated there, and the batched environment is tested against it step by
step.  Torch is imported here, so this package needs the ``rl`` extra.
"""

from rl_execution.vec.torch_env import (
    OBS_DIM,
    BatchedExecutionEnv,
    BatchMarket,
    BatchPaths,
    generate_paths,
    reference_paths,
    sample_randomized_markets,
)

__all__ = [
    "OBS_DIM",
    "BatchedExecutionEnv",
    "BatchMarket",
    "BatchPaths",
    "generate_paths",
    "reference_paths",
    "sample_randomized_markets",
]
