"""Classical (non-learning) execution strategies."""

from rl_execution.baselines.almgren_chriss import AlmgrenChriss
from rl_execution.baselines.base import BaseStrategy
from rl_execution.baselines.pov import POV
from rl_execution.baselines.random_strategy import RandomStrategy
from rl_execution.baselines.twap import TWAP
from rl_execution.baselines.vwap import VWAP


def make_baseline(name: str, **kwargs) -> BaseStrategy:
    """Factory: build a baseline strategy by (case-insensitive) name."""
    table = {
        "twap": TWAP,
        "vwap": VWAP,
        "pov": POV,
        "random": RandomStrategy,
        "almgren_chriss": AlmgrenChriss,
        "almgrenchriss": AlmgrenChriss,
        "ac": AlmgrenChriss,
    }
    key = name.lower().replace("-", "_").replace(" ", "")
    if key not in table:
        raise KeyError(f"Unknown baseline '{name}'. Available: {sorted(set(table))}")
    return table[key](**kwargs)


ALL_BASELINES = ["TWAP", "VWAP", "POV", "Random", "AlmgrenChriss"]

__all__ = [
    "BaseStrategy",
    "TWAP",
    "VWAP",
    "POV",
    "RandomStrategy",
    "AlmgrenChriss",
    "make_baseline",
    "ALL_BASELINES",
]
