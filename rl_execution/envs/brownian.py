"""Brownian-motion primitives: RNG substreams, Wiener increments and bridges.

This module owns *how random numbers are made* for the simulator, so that the market
dynamics in :mod:`rl_execution.envs.processes` can be written as plain mathematics.

Three ideas live here.

**Named substreams.**  A single shared ``Generator`` couples every noise source through
the order in which draws happen: adding one call anywhere shifts the entire downstream
sequence, so seeds stop being comparable across code versions, and — worse — the *agent's
own actions* can consume draws and thereby perturb the market path it faces.
:class:`NoiseStreams` derives one independent PCG64 generator per
:class:`Stream` from a single seed via ``SeedSequence.spawn``, which removes the coupling.

**Wiener increments.**  A standard Brownian motion satisfies ``W_0 = 0`` with independent
increments ``W_{t+dt} - W_t ~ N(0, dt)``.  :class:`BrownianEngine` draws the whole
increment vector for an episode in one vectorised call (also much faster than per-step
scalar draws) and exposes the cumulative path.

**Brownian bridge.**  Conditional on both endpoints of a segment, the value at an interior
fraction ``delta`` is Gaussian with mean ``a + delta (b - a)`` and variance
``delta (1 - delta) V`` where ``V`` is the segment's total variance.  This is the correct
object for anything that happens *between* two decision steps — latency fills above all.

Normals can be generated either by NumPy's ziggurat (fastest) or by inverse-CDF
transform of uniforms.  The latter is monotone in the underlying uniform, which is what
makes antithetic pairing exact and what quasi-Monte-Carlo constructions require.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Optional, Sequence, Union

import numpy as np

__all__ = [
    "Stream",
    "NoiseStreams",
    "make_streams",
    "normals",
    "correlate",
    "correlated_normals",
    "BrownianEngine",
    "bridge_value",
    "brownian_path",
]

RngLike = Union[int, np.random.Generator, None]

#: Methods available for turning uniforms into standard normals.
NORMAL_METHODS = ("ziggurat", "inverse_cdf")


class Stream(IntEnum):
    """Named, independent sources of randomness in the simulator.

    Each member gets its own generator, so a change to one noise source cannot perturb
    the realisations of another for a given seed.
    """

    PRICE = 0
    VOL = 1
    IMBALANCE = 2
    SPREAD = 3
    DEPTH = 4
    VOLUME = 5
    LATENCY = 6
    EXEC_ORDER = 7


# --------------------------------------------------------------------------- normals
def _ndtri(u: np.ndarray) -> np.ndarray:
    """Inverse standard-normal CDF, clipped away from the open interval's endpoints."""
    try:
        from scipy.special import ndtri
    except ImportError as exc:  # pragma: no cover - scipy is a core dependency
        raise ImportError("method='inverse_cdf' requires SciPy (scipy.special.ndtri).") from exc

    eps = np.finfo(float).tiny
    return np.asarray(ndtri(np.clip(u, eps, 1.0 - np.finfo(float).epsneg)), dtype=float)


def normals(
    rng: np.random.Generator,
    size: Union[int, Sequence[int], None] = None,
    method: str = "ziggurat",
) -> np.ndarray:
    """Draw standard normals from ``rng``.

    ``method='ziggurat'`` uses NumPy's ``standard_normal`` (fast, the default).
    ``method='inverse_cdf'`` maps uniforms through the inverse normal CDF; it is slower but
    *monotone* in the uniform, which exact antithetic pairing and QMC constructions rely on.
    """
    if method == "ziggurat":
        return np.asarray(rng.standard_normal(size), dtype=float)
    if method == "inverse_cdf":
        return _ndtri(np.asarray(rng.random(size), dtype=float))
    raise ValueError(f"Unknown normal method '{method}'. Expected one of {NORMAL_METHODS}.")


def correlate(z1: np.ndarray, z2: np.ndarray, rho: float) -> np.ndarray:
    """Combine two independent standard normals into one correlated with ``z1`` by ``rho``.

    Returns ``rho * z1 + sqrt(1 - rho**2) * z2``, which is again standard normal and has
    ``corr(z1, result) = rho``.  This is the two-dimensional case of a Cholesky factor.
    """
    r = float(np.clip(rho, -1.0, 1.0))
    return r * np.asarray(z1, dtype=float) + np.sqrt(1.0 - r * r) * np.asarray(z2, dtype=float)


def correlated_normals(
    rng: np.random.Generator,
    corr: np.ndarray,
    size: int,
    method: str = "ziggurat",
) -> np.ndarray:
    """Draw ``size`` samples of a correlated standard-normal vector.

    ``corr`` is a ``(d, d)`` correlation matrix; the return has shape ``(d, size)`` and
    empirical correlation converging to ``corr``.  Uses the Cholesky factor ``L`` of
    ``corr`` applied to i.i.d. normals: ``L @ Z``.
    """
    c = np.asarray(corr, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("corr must be a square matrix.")
    chol = np.linalg.cholesky(c)
    z = normals(rng, (c.shape[0], int(size)), method=method)
    return np.asarray(chol @ z, dtype=float)


# --------------------------------------------------------------------------- streams
@dataclass
class NoiseStreams:
    """A bundle of independent generators, one per :class:`Stream`.

    Two construction modes:

    * :meth:`from_seed` -- ``SeedSequence(seed).spawn(...)`` gives statistically independent
      substreams.  Adding a draw to one stream leaves the others bit-identical.
    * :meth:`shared` -- every :class:`Stream` maps to the *same* generator.  This reproduces
      the historical single-generator draw order exactly and exists so legacy results stay
      reproducible.

    ``antithetic`` negates every normal drawn through :meth:`normals`.  Pairing an episode
    seeded ``s`` with ``antithetic=False`` against the same seed with ``antithetic=True``
    yields the exact ``Z`` / ``-Z`` pair used for antithetic variance reduction.
    """

    gens: Dict[Stream, np.random.Generator]
    shared_generator: bool = False
    antithetic: bool = False
    method: str = "ziggurat"
    seed: Optional[int] = field(default=None, compare=False)

    # -- construction ---------------------------------------------------------------
    @classmethod
    def from_seed(
        cls, seed: int, *, antithetic: bool = False, method: str = "ziggurat"
    ) -> "NoiseStreams":
        """Independent substreams spawned from one integer seed."""
        children = np.random.SeedSequence(int(seed)).spawn(len(Stream))
        gens = {s: np.random.default_rng(c) for s, c in zip(Stream, children)}
        return cls(
            gens, shared_generator=False, antithetic=antithetic, method=method, seed=int(seed)
        )

    @classmethod
    def shared(
        cls, rng: np.random.Generator, *, antithetic: bool = False, method: str = "ziggurat"
    ) -> "NoiseStreams":
        """Legacy mode: all streams are the same generator, preserving draw order."""
        return cls(
            {s: rng for s in Stream}, shared_generator=True, antithetic=antithetic, method=method
        )

    # -- access ---------------------------------------------------------------------
    def __getitem__(self, stream: Stream) -> np.random.Generator:
        return self.gens[stream]

    def normals(
        self,
        stream: Stream,
        size: Union[int, Sequence[int], None] = None,
        method: Optional[str] = None,
    ) -> np.ndarray:
        """Standard normals from ``stream``, sign-flipped when ``antithetic`` is set."""
        z = normals(self.gens[stream], size, method=method or self.method)
        return -z if self.antithetic else z

    def lognormals(
        self, stream: Stream, sigma: float, size: Union[int, Sequence[int], None] = None
    ) -> np.ndarray:
        """``exp(sigma * Z)`` multiplicative noise from ``stream``."""
        return np.exp(float(sigma) * self.normals(stream, size))


def make_streams(
    rng: np.random.Generator,
    mode: str = "legacy",
    *,
    antithetic: bool = False,
    method: str = "ziggurat",
) -> NoiseStreams:
    """Build a :class:`NoiseStreams` from an existing generator.

    ``mode='legacy'`` shares ``rng`` across all streams (historical draw order preserved).
    ``mode='streams'`` consumes a single integer from ``rng`` and spawns independent
    substreams from it, so the bundle is still a deterministic function of ``rng``'s state.
    """
    if mode == "streams":
        seed = int(rng.integers(0, 2**63 - 1))
        return NoiseStreams.from_seed(seed, antithetic=antithetic, method=method)
    if mode == "legacy":
        return NoiseStreams.shared(rng, antithetic=antithetic, method=method)
    raise ValueError(f"Unknown rng_mode '{mode}'. Expected 'legacy' or 'streams'.")


# --------------------------------------------------------------------------- brownian
def bridge_value(
    start: float,
    end: float,
    delta: float,
    total_variance: float,
    z: float,
) -> float:
    """Brownian-bridge value at fraction ``delta`` of a segment with known endpoints.

    Conditional on ``W_0 = start`` and ``W_T = end``, ``W_{delta T}`` is Gaussian with
    mean ``start + delta (end - start)`` and variance ``delta (1 - delta) * total_variance``.
    ``z`` is a standard normal supplied by the caller (so the caller controls which stream
    it comes from and whether it participates in antithetic pairing).

    The units of ``total_variance`` set the space the bridge lives in: pass ``dt`` to bridge
    a Wiener path, ``sigma**2 * dt`` to bridge a log-price, ``(sigma * S)**2 * dt`` to bridge
    an arithmetic price.
    """
    d = float(np.clip(delta, 0.0, 1.0))
    mean = float(start) + d * (float(end) - float(start))
    var = d * (1.0 - d) * max(float(total_variance), 0.0)
    if var <= 0.0:
        return mean
    return mean + np.sqrt(var) * float(z)


def brownian_path(
    rng: np.random.Generator,
    n_steps: int,
    dt: float = 1.0,
    *,
    antithetic: bool = False,
    method: str = "ziggurat",
) -> np.ndarray:
    """Convenience: a single Wiener path ``W_0..W_n`` with ``W_0 = 0``."""
    return BrownianEngine.draw(rng, n_steps, dt=dt, antithetic=antithetic, method=method).path


class BrownianEngine:
    """Pre-generated Wiener increments for one episode.

    Holds the standard normals ``z``, the increments ``dW = z * sqrt(dt)`` and the
    cumulative path ``W`` (length ``n_steps + 1``, starting at zero).  Drawing the whole
    vector up front is both faster than per-step scalar draws and a prerequisite for
    decoupling the market path from the agent's actions.
    """

    def __init__(self, z: np.ndarray, dt: float = 1.0) -> None:
        self.z = np.asarray(z, dtype=float).ravel()
        self.dt = float(dt)
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        self.dW = self.z * np.sqrt(self.dt)
        self.path = np.concatenate([[0.0], np.cumsum(self.dW)])

    # -- construction ---------------------------------------------------------------
    @classmethod
    def draw(
        cls,
        rng: np.random.Generator,
        n_steps: int,
        dt: float = 1.0,
        *,
        antithetic: bool = False,
        method: str = "ziggurat",
    ) -> "BrownianEngine":
        z = normals(rng, int(n_steps), method=method)
        return cls(-z if antithetic else z, dt=dt)

    # -- views ----------------------------------------------------------------------
    @property
    def n_steps(self) -> int:
        return int(self.z.size)

    @property
    def time_grid(self) -> np.ndarray:
        return np.arange(self.n_steps + 1, dtype=float) * self.dt

    def quadratic_variation(self) -> float:
        """``sum (dW)^2`` -- converges to elapsed time ``T`` as ``dt -> 0``."""
        return float(np.sum(self.dW**2))

    # -- bridge ---------------------------------------------------------------------
    def bridge(self, i: int, delta: float, z: float) -> float:
        """Value of the path inside step ``i`` at fraction ``delta``, given both endpoints."""
        if not 0 <= i < self.n_steps:
            raise IndexError(f"step {i} out of range for {self.n_steps} increments")
        return bridge_value(self.path[i], self.path[i + 1], delta, self.dt, z)

    def refine(self, i: int, k: int, rng: np.random.Generator) -> np.ndarray:
        """``k`` equally spaced interior points of step ``i``, drawn as a Brownian bridge.

        Successive refinement conditions each new point on its immediate neighbours, so the
        returned sub-path has the correct joint law (not merely the correct marginals).
        """
        if k <= 0:
            return np.array([])
        left, right = float(self.path[i]), float(self.path[i + 1])
        grid = np.linspace(0.0, 1.0, k + 2)  # includes both endpoints
        out = np.empty(k + 2)
        out[0], out[-1] = left, right
        for j in range(1, k + 1):
            # condition on the last fixed point and the segment end
            done, remaining = grid[j - 1], 1.0 - grid[j - 1]
            delta = (grid[j] - done) / remaining if remaining > 0 else 0.0
            out[j] = bridge_value(
                out[j - 1], right, delta, remaining * self.dt, float(rng.standard_normal())
            )
        return out[1:-1]
