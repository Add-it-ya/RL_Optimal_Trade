"""Deterministic global seeding for reproducible runs.

RL is notoriously seed-sensitive, so a result is only citeable if it can be reproduced
from a single integer.  :func:`set_global_seeds` seeds every source of randomness we
touch -- Python's ``random``, NumPy, PyTorch (incl. CUDA + cuDNN) and Stable-Baselines3 --
in one call.  The heavy frameworks are imported lazily so the core simulation layers stay
usable without a deep-learning backend installed.

The environment already derives its market RNG from the seeded ``np_random``
(:meth:`ExecutionEnv.reset`), so env-level reproducibility is intact; this adds the
*agent / framework-level* determinism on top.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seeds(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed all RNGs we depend on.

    Parameters
    ----------
    seed:
        The seed applied to every backend.
    deterministic_torch:
        When ``True`` (default) ask PyTorch for deterministic algorithms and disable
        cuDNN autotuning.  Some ops have no deterministic implementation; we pass
        ``warn_only=True`` so those degrade to a warning instead of raising.  Perfect
        determinism is only guaranteed on CPU (the recommended device here).
    """
    # PYTHONHASHSEED only takes effect for *child* processes started after this point
    # (the current interpreter's hash seed is fixed at startup); set it anyway so any
    # subprocess we spawn inherits a reproducible hashing seed.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            # warn_only: some kernels lack a deterministic variant; warn rather than abort.
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        from stable_baselines3.common.utils import set_random_seed

        set_random_seed(seed)
    except ImportError:
        pass
