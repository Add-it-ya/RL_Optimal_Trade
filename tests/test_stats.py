"""Statistical-utility tests: bootstrap CI coverage, paired-bootstrap pairing, multiple-testing
correction and the deflated / probabilistic Sharpe ratio.  These use only NumPy/SciPy (the
core dependencies), so the multiple-testing tests exercise the NumPy fallback when statsmodels
is absent and the statsmodels path when it is installed.
"""

import sys

import numpy as np
import pytest

from rl_execution.metrics.stats import (
    _adjust_pvalues_numpy,
    adjust_pvalues,
    benjamini_hochberg,
    bootstrap_ci,
    deflated_sharpe,
    holm_bonferroni,
    paired_bootstrap_ci,
    probabilistic_sharpe_ratio,
)


def test_bootstrap_ci_coverage():
    """A 95% bootstrap CI for the mean covers the true mean ~95% of the time."""
    true_mean = 1.5
    rng = np.random.default_rng(12345)
    trials = 200
    covered = 0
    for _ in range(trials):
        sample = rng.normal(true_mean, 2.0, size=80)
        lo, hi = bootstrap_ci(sample, n_boot=1000, rng=rng)
        covered += int(lo <= true_mean <= hi)
    # Nominal 0.95; a wide tolerance band keeps the test from flaking on the finite trial count.
    assert 0.88 <= covered / trials <= 1.0


def test_bootstrap_ci_is_deterministic_given_rng():
    arr = np.arange(50.0)
    assert bootstrap_ci(arr, n_boot=500, rng=0) == bootstrap_ci(arr, n_boot=500, rng=0)


def test_bootstrap_ci_degenerate_inputs():
    assert all(np.isnan(bootstrap_ci([], n_boot=10)))
    assert bootstrap_ci([3.0], n_boot=10) == (3.0, 3.0)


def test_paired_bootstrap_preserves_pairing():
    """Identical paired series -> the paired delta CI collapses to exactly 0 (noise cancels)."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 100, size=200)  # huge per-episode variance
    assert paired_bootstrap_ci(x, x, n_boot=2000, rng=1) == (0.0, 0.0)


def test_paired_bootstrap_detects_constant_shift():
    rng = np.random.default_rng(1)
    bench = rng.normal(0, 50, size=300)
    strat = bench - 5.0  # strategy beats benchmark by exactly 5 every episode
    lo, hi = paired_bootstrap_ci(strat, bench, n_boot=2000, rng=2)
    assert lo == hi == -5.0  # degenerate (constant) difference


def test_holm_bonferroni_matches_known_values():
    adj = holm_bonferroni([0.01, 0.02, 0.03, 0.04])
    assert np.allclose(adj, [0.04, 0.06, 0.06, 0.06])


def test_holm_is_at_least_as_conservative_as_bh():
    p = np.array([0.001, 0.01, 0.02, 0.2, 0.5])
    assert np.all(holm_bonferroni(p) >= benjamini_hochberg(p) - 1e-12)


def test_adjust_pvalues_reject_flags_and_empty():
    reject, adj = adjust_pvalues([0.001, 0.5, 0.9], method="holm", alpha=0.05)
    assert list(reject) == [True, False, False]
    assert adj.shape == (3,)
    r_empty, a_empty = adjust_pvalues([], method="bh")
    assert r_empty.size == 0 and a_empty.size == 0


def test_deflated_sharpe_monotonic_in_trials():
    """Searching more trials deflates the same Sharpe toward 0.5 (lower DSR)."""
    base = deflated_sharpe(0.3, n_trials=1, n_obs=120)
    many = deflated_sharpe(0.3, n_trials=100, n_obs=120)
    assert 0.0 <= many < base <= 1.0


def test_probabilistic_sharpe_increases_with_sharpe():
    lo = probabilistic_sharpe_ratio(0.1, 0.0, n_obs=40)
    hi = probabilistic_sharpe_ratio(0.5, 0.0, n_obs=40)
    assert 0.0 <= lo < hi <= 1.0


def test_correction_uses_numpy_fallback_without_statsmodels(monkeypatch):
    """With statsmodels unavailable the correction degrades to the exact NumPy implementation."""
    monkeypatch.setitem(sys.modules, "statsmodels.stats.multitest", None)
    reject, adj = adjust_pvalues([0.01, 0.02, 0.03, 0.04], method="holm")
    assert np.allclose(adj, [0.04, 0.06, 0.06, 0.06])
    assert reject.dtype == bool


def test_numpy_fallback_methods_and_validation():
    p = np.array([0.04, 0.01, 0.03, 0.02])  # deliberately unsorted
    _, bonf = _adjust_pvalues_numpy(p, "bonferroni", 0.05)
    assert np.allclose(bonf, np.minimum(p * 4, 1.0))
    _, bh = _adjust_pvalues_numpy(p, "bh", 0.05)
    assert np.all((bh >= 0.0) & (bh <= 1.0))
    with pytest.raises(ValueError):
        _adjust_pvalues_numpy(p, "nope", 0.05)
