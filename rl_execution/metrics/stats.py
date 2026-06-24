"""Statistical-rigor utilities for defensible result reporting (Step 2).

A single headline number (``DoubleDQN -440 bps``) from one seed, tuned by hand and
evaluated in-sample, is a red flag to a quant panel rather than evidence.  This module
supplies the building blocks that turn point estimates into *interval* estimates with
calibrated uncertainty and guard against data snooping:

* :func:`bootstrap_ci` / :func:`paired_bootstrap_ci` -- percentile bootstrap confidence
  intervals.  The paired variant resamples episode indices *jointly* so it preserves the
  common-random-number pairing the backtester relies on (the whole point of paired
  evaluation is that the shared price-path risk cancels).
* :func:`holm_bonferroni` / :func:`benjamini_hochberg` -- multiple-testing corrections so a
  "significant" regime is not just the lucky tail of many tests across the regime x agent
  grid.  Uses :mod:`statsmodels` when present and an exact NumPy fallback otherwise.
* :func:`deflated_sharpe` (Bailey & Lopez de Prado) -- the probabilistic Sharpe ratio
  deflated by the expected maximum Sharpe under ``n_trials`` independent tries, i.e. a
  Sharpe estimate that already pays for the search that produced it.

Everything is deterministic given an explicit ``rng`` (an ``int`` seed or a
``numpy.random.Generator``), so reported intervals reproduce exactly.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

RngLike = Union[int, np.random.Generator, None]
ArrayLike = Union[Sequence[float], np.ndarray]
Stat = Callable[..., np.ndarray]

# Euler-Mascheroni constant, used in the expected-maximum-Sharpe approximation.
_EULER_GAMMA = 0.5772156649015329


def _as_1d(arr: ArrayLike) -> np.ndarray:
    return np.asarray(arr, dtype=float).ravel()


def _apply_stat(samples: np.ndarray, stat: Stat) -> np.ndarray:
    """Apply ``stat`` row-wise to a ``(n_boot, n)`` matrix of resamples.

    Fast path uses the ``axis=1`` reduction supported by NumPy callables (``np.mean``,
    ``np.median``, ...); anything else is applied row-by-row.
    """
    try:
        return np.asarray(stat(samples, axis=1), dtype=float)
    except TypeError:
        return np.asarray([float(stat(row)) for row in samples], dtype=float)


def bootstrap_ci(
    arr: ArrayLike,
    stat: Stat = np.mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng: RngLike = None,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for ``stat`` of ``arr`` at confidence ``1 - alpha``.

    Resamples ``arr`` with replacement ``n_boot`` times and returns the
    ``(alpha/2, 1 - alpha/2)`` quantiles of the bootstrap distribution of ``stat``.
    Degenerate inputs return ``(nan, nan)`` (empty) or a zero-width interval (single point).
    """
    a = _as_1d(arr)
    n = a.size
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        v = float(stat(a))
        return (v, v)
    gen = np.random.default_rng(rng)
    idx = gen.integers(0, n, size=(n_boot, n))
    stats = _apply_stat(a[idx], stat)
    return (float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2)))


def paired_bootstrap_ci(
    strat: ArrayLike,
    bench: ArrayLike,
    stat: Stat = np.mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng: RngLike = None,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for ``stat(strat - bench)`` preserving CRN pairing.

    ``strat`` and ``bench`` are aligned per-episode series (same price paths). A single set
    of resampled *episode indices* is applied to the per-episode difference, so the
    common-random-number pairing is never broken.  Returns the CI of the paired delta
    (for IS-style costs, ``< 0`` means the strategy beats the benchmark).
    """
    s = _as_1d(strat)
    b = _as_1d(bench)
    n = int(min(s.size, b.size))
    if n == 0:
        return (float("nan"), float("nan"))
    diff: np.ndarray = s[:n] - b[:n]
    if n == 1:
        v = float(stat(diff))
        return (v, v)
    gen = np.random.default_rng(rng)
    idx = gen.integers(0, n, size=(n_boot, n))
    stats = _apply_stat(diff[idx], stat)
    return (float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2)))


# --------------------------------------------------------------------------- multiple testing
def adjust_pvalues(
    pvals: ArrayLike, method: str = "holm", alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Family-wise / FDR correction of ``pvals``; returns ``(reject, p_adjusted)``.

    ``method`` is one of ``holm`` (Holm-Bonferroni, FWER), ``bh`` / ``fdr_bh``
    (Benjamini-Hochberg, FDR) or ``bonferroni``.  Uses
    :func:`statsmodels.stats.multitest.multipletests` when statsmodels is installed and an
    exact NumPy implementation otherwise, so the core package needs no extra dependency.
    """
    p = _as_1d(pvals)
    if p.size == 0:
        return (np.array([], dtype=bool), np.array([], dtype=float))
    try:
        from statsmodels.stats.multitest import multipletests

        sm = {"bh": "fdr_bh", "fdr_bh": "fdr_bh", "holm": "holm", "bonferroni": "bonferroni"}
        reject, p_adj, _, _ = multipletests(p, alpha=alpha, method=sm.get(method, method))
        return (np.asarray(reject, dtype=bool), np.asarray(p_adj, dtype=float))
    except ImportError:
        return _adjust_pvalues_numpy(p, method, alpha)


def _adjust_pvalues_numpy(
    p: np.ndarray, method: str, alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    m = method.lower()
    if m == "bonferroni":
        adj_sorted = np.minimum(ranked * n, 1.0)
    elif m == "holm":
        adj_sorted = np.maximum.accumulate(np.minimum((n - np.arange(n)) * ranked, 1.0))
    elif m in ("bh", "fdr_bh"):
        ranks = np.arange(1, n + 1)
        adj_sorted = np.minimum.accumulate((ranked * n / ranks)[::-1])[::-1]
        adj_sorted = np.minimum(adj_sorted, 1.0)
    else:
        raise ValueError(f"Unknown correction method {method!r}")
    p_adj = np.empty_like(adj_sorted)
    p_adj[order] = adj_sorted
    return (p_adj < alpha, p_adj)


def holm_bonferroni(pvals: ArrayLike, alpha: float = 0.05) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    return adjust_pvalues(pvals, method="holm", alpha=alpha)[1]


def benjamini_hochberg(pvals: ArrayLike, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (controls the false-discovery rate)."""
    return adjust_pvalues(pvals, method="bh", alpha=alpha)[1]


# --------------------------------------------------------------------------- deflated Sharpe
def probabilistic_sharpe_ratio(
    sr: float,
    sr_benchmark: float = 0.0,
    n_obs: int = 252,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Probabilistic Sharpe ratio: ``P(true SR > sr_benchmark)`` given a non-normal return
    distribution (Bailey & Lopez de Prado).  ``kurt`` is the (non-excess) kurtosis (3 = normal).
    """
    from scipy.stats import norm

    denom = math.sqrt(max(1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr, 1e-12))
    z = (sr - sr_benchmark) * math.sqrt(max(n_obs - 1, 1)) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """Expected maximum of ``n_trials`` iid Sharpe estimates with cross-trial std ``sr_std``
    (the deflation threshold ``SR_0`` in the deflated-Sharpe construction).
    """
    if n_trials <= 1 or sr_std <= 0.0:
        return 0.0
    from scipy.stats import norm

    return float(
        sr_std
        * (
            (1.0 - _EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n_trials)
            + _EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        )
    )


def deflated_sharpe(
    sr: float,
    n_trials: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    n_obs: int = 252,
    sr_std: Optional[float] = None,
) -> float:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado).

    The probabilistic Sharpe ratio measured against the *expected maximum* Sharpe one would
    obtain from ``n_trials`` independent strategy configurations -- so a Sharpe that looks
    impressive only because many variants were tried is deflated toward 0.5 (no skill).

    ``sr_std`` is the standard deviation of the Sharpe estimates across the trials; when it is
    unavailable (only the winning trial's stats are kept) it falls back to the analytic Sharpe
    standard error ``sqrt((1 - skew*SR + ((kurt-1)/4) SR^2) / (n_obs - 1))`` (Lo), a reasonable
    conservative proxy.  Returns a probability in ``[0, 1]``; ``> 0.95`` is the usual bar.
    """
    if sr_std is None:
        var = max(1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr, 1e-12)
        sr_std = math.sqrt(var / max(n_obs - 1, 1))
    sr0 = expected_max_sharpe(n_trials, sr_std)
    return probabilistic_sharpe_ratio(sr, sr0, n_obs, skew, kurt)
