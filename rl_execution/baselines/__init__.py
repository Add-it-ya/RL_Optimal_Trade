"""Classical (non-learning) execution strategies and the null-hypothesis policies."""

from rl_execution.baselines.almgren_chriss import AlmgrenChriss
from rl_execution.baselines.base import BaseStrategy
from rl_execution.baselines.nulls import DumpAtStart, SignalRule, WaitThenDump
from rl_execution.baselines.twap import TWAP
from rl_execution.baselines.vwap import VWAP


def make_baseline(name: str, **kwargs) -> BaseStrategy:
    """Factory: build a baseline strategy by (case-insensitive) name."""
    table = {
        "twap": TWAP,
        "vwap": VWAP,
        "almgren_chriss": AlmgrenChriss,
        "almgrenchriss": AlmgrenChriss,
        "ac": AlmgrenChriss,
        "dumpatstart": DumpAtStart,
        "waitthendump": WaitThenDump,
        "signalrule": SignalRule,
    }
    key = name.lower().replace("-", "_").replace(" ", "")
    key = key if key in table else key.replace("_", "")
    if key not in table:
        raise KeyError(f"Unknown baseline '{name}'. Available: {sorted(set(table))}")
    return table[key](**kwargs)


ALL_BASELINES = ["TWAP", "VWAP", "AlmgrenChriss", "DumpAtStart", "WaitThenDump", "SignalRule"]

__all__ = [
    "BaseStrategy",
    "TWAP",
    "VWAP",
    "AlmgrenChriss",
    "DumpAtStart",
    "WaitThenDump",
    "SignalRule",
    "make_baseline",
    "ALL_BASELINES",
]
