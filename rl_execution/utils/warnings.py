"""Scoped suppression of known-benign third-party warning noise.

The CLI entry points previously called ``warnings.filterwarnings("ignore")``, which hid
*every* warning -- including genuine ones raised by this package. This helper narrows the
suppression to specific, known-noisy third-party categories so that warnings originating in
``rl_execution`` stay visible (in the terminal and in CI).
"""

from __future__ import annotations

import warnings


def configure_cli_warnings() -> None:
    """Silence only known, benign deprecation/user warnings from optional dependencies."""
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"gymnasium.*")
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning, module=r"gymnasium.*")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"stable_baselines3.*")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"gymnasium.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"pandas.*")
