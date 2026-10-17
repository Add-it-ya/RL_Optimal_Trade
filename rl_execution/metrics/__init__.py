"""Execution-quality metrics and statistical-rigor utilities."""

from rl_execution.metrics.metrics import (
    METRIC_COLUMNS,
    aggregate_metrics,
    compute_episode_metrics,
)
from rl_execution.metrics.stats import (
    adjust_pvalues,
    benjamini_hochberg,
    bootstrap_ci,
    holm_bonferroni,
    paired_bootstrap_ci,
)

__all__ = [
    "compute_episode_metrics",
    "aggregate_metrics",
    "METRIC_COLUMNS",
    "bootstrap_ci",
    "paired_bootstrap_ci",
    "adjust_pvalues",
    "holm_bonferroni",
    "benjamini_hochberg",
]
