"""Config construction: types are coerced (and junk rejected) before validation runs.

Range and membership checks live in ``RunConfig.validate``; these tests pin the layer
below it, where a YAML typo used to arrive as a truthy string and pass unnoticed.
"""

import numpy as np
import pytest

from rl_execution.config import MarketConfig, RunConfig
from rl_execution.utils import load_config


# =============================================================== booleans
def test_bool_fields_reject_non_boolean_strings():
    """The original bug: a missing space before a YAML comment.

    ``bool('false# oops')`` is True, so the stationary imbalance scaling switched on and
    validation passed -- an invalid run with no error anywhere.
    """
    with pytest.raises(ValueError, match="imbalance_ou_scaling"):
        MarketConfig.from_dict({"imbalance_ou_scaling": "false# oops"})


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, True),
        (False, False),
        ("true", True),
        ("True", True),
        (" FALSE ", False),
        ("yes", True),
        ("no", False),
        ("1", True),
        ("0", False),
        (1, True),
        (0, False),
        (np.bool_(True), True),
    ],
)
def test_bool_fields_accept_yaml_booleans(value, expected):
    cfg = MarketConfig.from_dict({"imbalance_ou_scaling": value})
    assert cfg.imbalance_ou_scaling is expected


@pytest.mark.parametrize("value", ["", "maybe", "2", 2, None, [], 0.5])
def test_bool_fields_reject_everything_else(value):
    with pytest.raises(ValueError):
        MarketConfig.from_dict({"imbalance_ou_scaling": value})


# =============================================================== numbers
def test_numeric_fields_coerce_from_strings():
    cfg = MarketConfig.from_dict({"dt": "0.5", "n_levels": "8"})
    assert cfg.dt == 0.5 and isinstance(cfg.dt, float)
    assert cfg.n_levels == 8 and isinstance(cfg.n_levels, int)


def test_numeric_fields_reject_junk():
    with pytest.raises(ValueError, match="volatility"):
        MarketConfig.from_dict({"volatility": "0.02 # per step"})
    with pytest.raises(ValueError, match="n_levels"):
        MarketConfig.from_dict({"n_levels": 10.5})


def test_numpy_scalars_become_python_floats():
    """Regime samplers hand back numpy scalars; configs stay JSON-serialisable."""
    cfg = MarketConfig.from_dict({"volatility": np.float64(0.03)})
    assert type(cfg.volatility) is float


# =============================================================== enum-like strings
@pytest.mark.parametrize(
    "field, value, expected",
    [
        ("price_process", " GBM ", "gbm"),
        ("normal_method", "INVERSE_CDF", "inverse_cdf"),
        ("vol_process", " expou", "expou"),
    ],
)
def test_enum_fields_are_case_and_whitespace_tolerant(field, value, expected):
    cfg = MarketConfig.from_dict({field: value})
    assert getattr(cfg, field) == expected
    RunConfig(market=cfg).validate()  # normalised values pass the membership checks


def test_enum_fields_reject_non_strings():
    with pytest.raises(ValueError, match="price_process"):
        MarketConfig.from_dict({"price_process": 3})


# =============================================================== nothing else moved
def test_default_yaml_still_loads():
    market, execution = load_config("config/default.yaml")
    assert market.price_process == "abm"
    assert market.imbalance_ou_scaling is False
    RunConfig(market=market, execution=execution).validate()


def test_defaults_round_trip_unchanged():
    cfg = MarketConfig()
    assert MarketConfig.from_dict(cfg.to_dict()) == cfg
