# Brownian Motion in the Market Simulator — Implementation Plan

A design and delivery plan for putting an explicit **Wiener process / Brownian motion**
layer underneath the synthetic market, and for restructuring how random numbers are
generated so that the simulation is mathematically principled, reproducible, and
lower-variance to evaluate.

Scope: `rl_execution/envs/`, `rl_execution/config.py`, `rl_execution/experiments/regimes.py`,
`rl_execution/backtest/engine.py`, `rl_execution/data/lob_data.py`, plus new tests and docs.

---

## 1. Where randomness lives today

Every stochastic draw in the codebase, and what it feeds:

| Site | Draw | Feeds |
|---|---|---|
| [market.py:171](rl_execution/envs/market.py:171) | `rng.normal(0, sigma)` | mid-price shock |
| [market.py:161](rl_execution/envs/market.py:161) | `exp(rng.normal(0, vol_of_vol))` | per-step σ |
| [market.py:179](rl_execution/envs/market.py:179) | `rng.normal(0, 1)` | AR(1) imbalance innovation |
| [market.py:186](rl_execution/envs/market.py:186) | `exp(rng.normal(0, spread_vol))` | bid-ask spread |
| [market.py:87](rl_execution/envs/market.py:87) | `exp(rng.normal(0, volume_vol))` | ambient traded volume |
| [market.py:205](rl_execution/envs/market.py:205) | `abs(rng.normal())` | latency price slip |
| [lob.py:115](rl_execution/envs/lob.py:115) | `rng.normal(0, depth_vol, size=n)` ×2 | resting depth per level |
| [execution_env.py:118](rl_execution/envs/execution_env.py:118) | `np_random.integers(...)` | per-episode market seed |
| [multi_agent.py:146](rl_execution/envs/multi_agent.py:146) | `rng.shuffle(order)` | execution priority |
| [lob_data.py:171](rl_execution/data/lob_data.py:171) | `rng.integers(...)` | replay window start |
| [regimes.py:74](rl_execution/experiments/regimes.py:74) | `rng.uniform/choice` | domain randomisation |

All of these share **one** `np.random.Generator` per episode, drawn in program order.

### 1.1 Gaps this creates

**G1 — There is no time.** [`advance()`](rl_execution/envs/market.py:154) is an
Euler–Maruyama step with an implicit `dt = 1`. `volatility` is "std of per-step return", so
changing `horizon` from 20 to 80 silently changes the total variance of an episode by 4×
instead of holding it fixed. There is no `√dt` scaling anywhere, so results are not
comparable across time resolutions and the `volatility` config value has no calendar meaning.

**G2 — The price process is not a named process.** `new_mid = mid * (1 + ret)` is a
multiplicative random walk, not GBM: no Itô correction, no exactness, and positivity is
enforced by clamping with `max(..., tick_size)` rather than by construction. The log-price
is never modelled.

**G3 — Model mismatch with the Almgren–Chriss baseline.**
[almgren_chriss.py:45](rl_execution/baselines/almgren_chriss.py:45) sets
`sigma = volatility * initial_price`, i.e. it assumes **arithmetic** Brownian motion, which
is the setting in which its closed form is provably optimal. The simulator is
multiplicative. AC is therefore benchmarked slightly outside its own model, and we lose the
strongest validation available: *"in the regime where the optimal policy is known in closed
form, does the RL agent recover it?"*

**G4 — Agent actions perturb the exogenous price path.** `execute()` calls
[`_make_snapshot()`](rl_execution/envs/market.py:136), which consumes `2 × n_levels` normal
draws from the shared stream. A strategy that trades zero shares on a step
([early return, market.py:113](rl_execution/envs/market.py:113)) consumes **no** draws,
while one that trades consumes 20. From the first differing action onward, the two
strategies see *different price paths* despite identical seeds. This weakens the common-
random-numbers claim in [`paired_is_table`](rl_execution/backtest/engine.py:186) — the
paired IS difference no longer fully cancels shared path risk, which inflates the variance
of every headline number in `results/` and widens the bootstrap CIs.

**G5 — All noise sources are independent.** No leverage effect (price/vol correlation), no
volume–volatility correlation, no spread widening in stress. Real microstructure has all
three.

**G6 — No volatility clustering.** `vol_of_vol` is i.i.d. lognormal noise on σ, so σ has
zero autocorrelation. But `recent_volatility()` is fed to the agent as observation feature
#5 ([execution_env.py:104](rl_execution/envs/execution_env.py:104)) — a feature with no
persistence carries almost no forecast value, so the agent cannot learn "trade less when
volatility is elevated". This is a *live* limitation of the current results.

**G7 — Imbalance innovation is mis-scaled.** [market.py:179](rl_execution/envs/market.py:179)
uses `x ← φx + (1−φ)ε`. The stationary AR(1) / discrete-OU scaling is `√(1−φ²)`. With
`φ = 0.85`, stationary std is `0.15/√(1−0.85²) ≈ 0.285`, not 1. So the effective book skew
is `imbalance_strength × 0.285 ≈ 0.11`, not the configured `0.40`, and `imbalance_alpha`
is ~3.5× weaker than the config implies.

**G8 — Latency is not a Brownian model.** [market.py:205](rl_execution/envs/market.py:205)
uses `|N(0,1)|` and forces the sign adverse. That is a folded normal with mean `σ√(2δ/π)`,
i.e. a deterministic cost with noise — never favourable. The correct object for "price at a
fractional time inside a step" is a **Brownian bridge**.

**G9 — Draw-order fragility.** One shared stream means adding a single new noise source
anywhere reshuffles every downstream draw. Seeds are not stable across code versions, so
`results/*.csv` cannot be regenerated after any change to the simulator internals.

**G10 — No variance reduction.** Evaluation is plain Monte Carlo over
`seed = base_seed + i` ([engine.py:123](rl_execution/backtest/engine.py:123)). Antithetic
variates and QMC are both directly applicable and would tighten every CI in the report.

---

## 2. Target design

Two new modules, both dependency-free beyond NumPy.

```
rl_execution/envs/
├── brownian.py     # NEW — RNG substreams, Wiener increments, bridge, antithetic, Cholesky
├── processes.py    # NEW — ABM / GBM / OU price processes, exp-OU stochastic volatility
├── market.py       # MODIFIED — delegates dynamics to a process object
├── lob.py          # MODIFIED — takes a depth substream instead of the shared rng
└── execution_env.py# MODIFIED — builds the stream bundle from the episode seed
```

### 2.1 `brownian.py` — random number generation

**Named substreams.** Replace the single `Generator` with a bundle derived from one
`SeedSequence` via `spawn()`, which produces provably independent streams:

```python
class Stream(IntEnum):
    PRICE = 0; VOL = 1; IMBALANCE = 2; SPREAD = 3
    DEPTH = 4; VOLUME = 5; LATENCY = 6; EXEC_ORDER = 7

@dataclass
class NoiseStreams:
    """One independent PCG64 generator per named noise source."""
    gens: dict[Stream, np.random.Generator]

    @classmethod
    def from_seed(cls, seed: int) -> "NoiseStreams":
        children = np.random.SeedSequence(seed).spawn(len(Stream))
        return cls({s: np.random.default_rng(c) for s, c in zip(Stream, children)})

    def __getitem__(self, s: Stream) -> np.random.Generator:
        return self.gens[s]
```

This fixes **G9** (a new draw in `DEPTH` cannot perturb `PRICE`) and is a prerequisite for
fixing **G4**.

**The Brownian engine.** Pre-draws the whole episode's exogenous path at `reset()`:

```python
class BrownianEngine:
    """Wiener increments dW ~ N(0, dt), pre-generated for the episode."""

    def __init__(self, rng, n_steps, dt=1.0, antithetic=False, method="ziggurat"):
        z = self._normals(rng, n_steps, method)      # standard normals
        if antithetic:
            z = -z if self._is_odd_replicate else z  # paired sign flip
        self.dW = z * np.sqrt(dt)
        self.dt = dt

    def path(self) -> np.ndarray:                     # W_t, W_0 = 0
        return np.concatenate([[0.0], np.cumsum(self.dW)])

    def bridge(self, i, delta, rng) -> float:
        """W at fraction `delta` inside step i, conditional on both endpoints."""
        w0, w1 = self.W[i], self.W[i + 1]
        mean = w0 + delta * (w1 - w0)
        std  = np.sqrt(delta * (1.0 - delta) * self.dt)
        return mean + std * rng.standard_normal()
```

Generating the whole increment vector in one bulk `standard_normal(n)` call is also
materially faster than `n` scalar `rng.normal()` calls in a Python loop.

**How the normals are made** (relevant to "generating random numbers"):

1. **Bit source** — PCG64 (NumPy's `default_rng`), period 2¹²⁸, jump-ahead capable,
   with `SeedSequence` giving independent substreams.
2. **Uniform → normal** — two selectable methods:
   * `"ziggurat"` — NumPy's `standard_normal`; fastest, the default.
   * `"inverse_cdf"` — `ndtri(u)` on `rng.random()`; monotone in `u`, which is **required**
     for QMC (§5.3) and makes antithetic pairing exact (`u ↔ 1−u ⇒ z ↔ −z`).
3. **Correlation** — a Cholesky factor `L` of the target correlation matrix applied to a
   vector of i.i.d. normals, `Z_corr = L Z`. For the 2-D price/vol case this reduces to
   `Z_v = ρ Z_S + √(1−ρ²) Z_⊥`.
4. **Scaling** — increments are `σ √dt · Z`; the path is a cumulative sum.

### 2.2 `processes.py` — the price processes

A small protocol plus concrete processes, selected by config:

```python
class PriceProcess(Protocol):
    def reset(self, s0: float, n_steps: int, streams: NoiseStreams) -> None: ...
    def step(self, i: int) -> float: ...          # exogenous mid at step i (no impact)
    def sigma(self, i: int) -> float: ...         # instantaneous vol, for AC and features
```

| Process | Update | Why we want it |
|---|---|---|
| `ArithmeticBM` | `S_{t+dt} = S_t + μ dt + σ√dt Z` | Exactly the Almgren–Chriss world → closed-form optimality benchmark (fixes **G3**) |
| `GeometricBM` | `S_{t+dt} = S_t exp((μ − ½σ²)dt + σ√dt Z)` | Exact in distribution, positive by construction, no tick clamping (fixes **G2**) |
| `OrnsteinUhlenbeck` | `S_{t+dt} = S̄ + (S_t − S̄)e^{−κdt} + σ√((1−e^{−2κdt})/2κ) Z` | Exact discretisation of the `sideways` regime (currently an ad-hoc pull term) |
| `ExpOUVolatility` | `ln σ_{t+dt} = ln σ̄ + φ(ln σ_t − ln σ̄) + ξ√(1−φ²) Z_v`, `corr(Z_S, Z_v) = ρ` | Volatility **clustering** + leverage effect (fixes **G5**, **G6**) |
| `Heston` *(stretch)* | CIR variance, full-truncation or QE scheme | Textbook stochastic vol; only if the exp-OU version proves insufficient |

All use **exact** discretisations where one exists (GBM, OU, exp-OU), so there is no
discretisation bias to defend at review time.

### 2.3 Impact as an overlay (fixes G4)

The exogenous Brownian path is generated once at `reset()` from the `PRICE`/`VOL` streams
and is **independent of what the agent does**. Permanent impact becomes an additive shift on
top of it — exactly the pattern
[`HistoricalMarketSimulator`](rl_execution/data/lob_data.py:150) already uses with
`self.shift`:

```
mid_t = exogenous_path[t] + cumulative_permanent_impact
```

Consequence: two strategies on the same seed see the **identical** price path regardless of
their trading, so `paired_is_table` becomes true common random numbers and its CIs tighten
without any extra episodes.

### 2.4 Config surface

Added to [`MarketConfig`](rl_execution/config.py:40):

```yaml
market:
  price_process: legacy       # legacy | abm | gbm | ou | gbm_expou
  dt: 1.0                     # time per decision step, in volatility units
  rng_mode: legacy            # legacy | streams
  vol_process: none           # none | iid_lognormal (current) | expou
  vol_persistence: 0.90       # φ of the log-vol OU  (0 = no clustering)
  leverage_corr: -0.3         # ρ between price and vol Brownians
  imbalance_ou_scaling: false # true ⇒ √(1−φ²) innovation (fixes G7)
  latency_model: adverse      # adverse (current) | bridge
  antithetic: false           # evaluation-time variance reduction
```

Defaults are chosen so that **`legacy` reproduces today's numbers bit-for-bit**. Everything
new is opt-in.

---

## 3. Backward compatibility (non-negotiable)

`results/*.csv`, `results/summary.json` and `reports/REPORT.md` are all tied to the current
draw order. The plan therefore:

* Keeps `price_process: legacy` and `rng_mode: legacy` as defaults through Phases 0–4.
* Adds a golden-value test (`test_legacy_mode_bit_identical`) pinning a short episode's mid
  path, so any accidental change to the legacy stream fails CI.
* Notes that adding fields to `MarketConfig` changes `config_hash`
  ([provenance.py](rl_execution/utils/provenance.py)) for every run — expected, and the
  reason the switch happens in one commit rather than drifting.
* `MarketConfig.from_dict` already filters unknown keys, so old YAML configs keep loading.

---

## 4. Delivery phases

### Phase 0 — Substreams, no behaviour change
* Add `brownian.py` with `Stream`, `NoiseStreams`.
* `MarketSimulator` accepts either a `Generator` (legacy) or a `NoiseStreams`.
* `build_book` takes the `DEPTH` stream.
* Tests: substream independence; legacy golden path unchanged.
* **Risk:** low. **Files:** `brownian.py`, `market.py`, `lob.py`, `execution_env.py`.

### Phase 1 — Brownian engine
* `BrownianEngine`: bulk increments, `path()`, `bridge()`, antithetic flag, `inverse_cdf`
  normals, `correlated_normals(rho)`.
* Statistical test suite (§6).
* **Risk:** low — pure addition, nothing wired in yet.

### Phase 2 — Price processes behind a flag
* `processes.py` with ABM / GBM / OU + `make_price_process(config)`.
* `MarketSimulator.reset()` pre-generates the exogenous path; `advance()` reads from it;
  `execute()` accumulates `self.shift` instead of overwriting `self.mid` (§2.3).
* `MarketConfig.price_process`, `dt`; range checks in `_validate_manual` and
  [`utils/validation.py`](rl_execution/utils/validation.py).
* Update `config/default.yaml` with the new keys and comments.
* **Risk:** medium — this is the change that decouples agent actions from the path.

### Phase 3 — Stochastic volatility, imbalance, latency
* `ExpOUVolatility` with leverage correlation ρ; `vol_process: expou`.
* `imbalance_ou_scaling: true` path using `√(1−φ²)` (**G7**).
* `latency_model: bridge` — `apply_latency` conditions on the step's Brownian endpoints
  instead of drawing a folded normal (**G8**); keep `adverse` selectable and document it as
  an *adverse-selection* model rather than a price model.
* `dt` respected everywhere: σ√dt, μ·dt, κ·dt.
* **Risk:** medium — changes the meaning of `volatility`; needs a migration note in
  `CHANGELOG.md`.

### Phase 4 — New regimes and validation experiments
Add to [`REGIMES`](rl_execution/experiments/regimes.py:28):

| Regime | Config | Question it answers |
|---|---|---|
| `abm_ac_optimal` | `price_process=abm`, `imbalance_alpha=0`, `drift=0`, linear impact | **Does RL recover the provably optimal AC schedule?** |
| `gbm_medium` | `price_process=gbm` | Do the strategy rankings survive a change of price model? |
| `clustered_vol` | `vol_process=expou`, `vol_persistence=0.9` | Does the agent finally exploit the volatility feature? |
| `leverage` | `leverage_corr=-0.5` | Does a SELL order suffer more when vol spikes on down-moves? |

Headline figure: RL schedule vs AC closed-form schedule under `abm_ac_optimal`, with the L1
distance between them reported. A near-zero gap is direct evidence the learner is sound;
`randomized_market_config` also gains the new axes so domain randomisation covers them.

### Phase 5 — Variance reduction in evaluation
* `evaluate(..., antithetic=True)` pairs episodes `(2k, 2k+1)` with `Z` and `−Z`.
* **Important:** antithetic pairs are not i.i.d., so
  [`paired_bootstrap_ci`](rl_execution/metrics/stats.py:80) must resample **pairs as blocks**,
  not individual episodes, or the CI is anti-conservative. Add `block=2` support.
* Report the achieved standard-error reduction on IS as a table — a concrete,
  defensible efficiency claim.
* Optional: Sobol (`scipy.stats.qmc`) + Brownian-bridge dimension ordering (§5.3) for
  point estimates only, never for the bootstrap.

### Phase 6 — Data, docs, dashboard
* `synthetic_lob_dataframe` gains a `price_process` argument so replay datasets can be exact
  GBM paths.
* `scripts/brownian_demo.py`: sample paths with the `±σ√t` envelope, return QQ-plot,
  ACF of `|returns|` (clustering), quadratic-variation convergence, antithetic pair overlay.
* `rl_execution/viz/plots.py`: `plot_brownian_paths`, `plot_return_qq`, `plot_vol_acf`.
* README "Environment" row and `reports/REPORT.md` methodology section updated with the SDEs.

---

## 5. Mathematical reference for the implementation

### 5.1 The processes

Wiener process: `W_0 = 0`, independent increments, `W_{t+dt} − W_t ~ N(0, dt)`, continuous
paths. In code, `dW = √dt · Z` with `Z ~ N(0,1)`, and `W = cumsum(dW)`.

* **ABM:** `dS = μ dt + σ dW` → `S_{t+dt} = S_t + μdt + σ√dt Z`. Gaussian, can go negative
  (acceptable over a 20-step horizon at `S₀ = 100`, `σ = 2%`), matches Almgren–Chriss.
* **GBM:** `dS = μS dt + σS dW` → `S_{t+dt} = S_t exp((μ − ½σ²)dt + σ√dt Z)`. The `−½σ²`
  Itô correction is what the current `mid*(1+ret)` update is missing; without it,
  `E[S_T] ≠ S_0 e^{μT}` and the process has a systematic upward bias in the mean.
* **OU:** `dX = κ(X̄ − X)dt + σ dW`, exact discretisation as in §2.2. Note that the existing
  `imbalance_persistence` **is** `e^{−κdt}` — the imbalance factor is already an OU process,
  just with the wrong innovation scale (**G7**).
* **Exp-OU vol:** `d ln σ = κ_v(ln σ̄ − ln σ)dt + ξ dW_v`, `corr(dW, dW_v) = ρ`. Positive by
  construction, produces clustering, and `ρ < 0` gives the leverage effect.

### 5.2 Brownian bridge

Given `W_0 = a` and `W_T = b`, for `0 < δ < 1`:

```
W_{δT} ~ N( a + δ(b − a),  δ(1 − δ)T )
```

Uses: (i) principled intra-step latency fills, (ii) refining a coarse path into sub-steps
for VWAP-style intra-step benchmarks, (iii) QMC dimension ordering.

### 5.3 QMC + bridge ordering (Phase 5, optional)

Sobol points are most uniform in their **leading** dimensions. Constructing the path by
bridge — terminal value first, then the midpoint, then quarters — puts most of the variance
into those leading dimensions, which is what makes QMC pay off. Requires the `inverse_cdf`
normal method (§2.1). Restrict to point estimates; the bootstrap machinery assumes i.i.d.
samples.

---

## 6. Test plan — `tests/test_brownian.py`

All tests seeded, with tolerances sized for the sample count (target < 5 s total).

**Wiener properties**
* `test_increments_are_gaussian` — KS test of `dW/√dt` against N(0,1).
* `test_variance_grows_linearly_in_time` — `Var(W_t) ≈ t` across several `t`.
* `test_increments_are_uncorrelated` — lag-1..5 autocorrelation ≈ 0.
* `test_quadratic_variation_converges` — `Σ(ΔW)² → T` as `dt → 0`.
* `test_time_rescaling_invariance` — halving `dt` and doubling steps leaves the terminal
  distribution unchanged (two-sample KS).

**Processes**
* `test_gbm_is_strictly_positive` — no clamping needed on 10⁴ paths.
* `test_gbm_log_returns_have_ito_corrected_mean` — mean ≈ `(μ − ½σ²)dt`.
* `test_gbm_is_martingale_under_zero_drift` — `E[S_T/S_0] ≈ 1`.
* `test_abm_matches_closed_form_moments` — mean and variance vs analytic.
* `test_ou_stationary_mean_and_variance` — long-run mean `X̄`, variance `σ²/2κ`.
* `test_expou_produces_volatility_clustering` — ACF of `|returns|` significantly > 0 at
  `vol_persistence=0.9` and ≈ 0 at `0.0`.
* `test_leverage_correlation_is_realised` — empirical `corr(ΔS, Δσ) ≈ ρ`.

**Bridge**
* `test_bridge_matches_endpoints` — `δ→0` and `δ→1` return the endpoints.
* `test_bridge_midpoint_variance` — `Var = T/4` at `δ = 0.5`.
* `test_latency_bridge_is_unbiased` — mean slip ≈ 0, unlike the `adverse` model.

**RNG plumbing**
* `test_substreams_are_independent` — cross-correlation ≈ 0 between streams.
* `test_substream_isolation_under_added_draws` — extra draws from `DEPTH` leave the `PRICE`
  sequence identical. *(This is the regression guard for **G9**.)*
* `test_inverse_cdf_matches_ziggurat_distribution` — same law, different generator.
* `test_antithetic_pairs_sum_to_zero` — `Z + Z' = 0` exactly.
* `test_correlated_normals_hit_target_correlation` — Cholesky check for 2-D and 3-D.

**Integration**
* `test_legacy_mode_bit_identical` — golden mid path; guards the committed results.
* `test_price_path_independent_of_agent_actions` — a zero-trading and a full-trading
  strategy on the same seed see the **same** exogenous mid path. *(Regression guard for **G4**.)*
* `test_antithetic_reduces_is_standard_error` — measured SE with antithetic < plain MC.
* `test_rl_env_still_passes_gym_checker` — under each `price_process`.

---

## 7. Risks and open decisions

| Risk | Mitigation |
|---|---|
| Existing results invalidated | `legacy` defaults + golden-value test through Phase 4; regenerate `results/` in one deliberate commit with a `CHANGELOG.md` migration note |
| `volatility` changes meaning once `dt ≠ 1` | Document as "σ per unit time"; keep `dt = 1.0` default so the number is unchanged |
| AC baseline σ is model-dependent | Under `abm` use `σ_abs = volatility × initial_price` (as today); under `gbm` use `σ × current mid`. Make this explicit in `_optimal_trades` |
| Extra RNG cost per step | Bulk pre-draw at `reset()` is *faster* than the current per-step scalar draws; benchmark in Phase 2 |
| Sobol requires SciPy | SciPy is already a dependency (`stats` used in `engine.py`); still gate the import |
| Bootstrap invalid under antithetic | Block-resample pairs (Phase 5); documented in the `paired_bootstrap_ci` docstring |
| Torch determinism on Windows | Unchanged — this plan touches only NumPy-side randomness |

**Open decisions to settle before Phase 2:**
1. Does `legacy` stay the default permanently, or does `gbm` become the default at a tagged
   release with regenerated results? *(Recommendation: flip to `gbm` at Phase 4, regenerate
   everything once, keep `legacy` selectable for reproducing the old report.)*
2. Exp-OU volatility versus full Heston. *(Recommendation: exp-OU — positive by
   construction, exactly discretisable, one fewer scheme to defend. Heston only if a
   reviewer asks for it.)*
3. Whether `imbalance_ou_scaling` becomes the default. It makes `imbalance_strength` and
   `imbalance_alpha` mean what the config says, but roughly triples the effective alpha and
   will change every reported number. *(Recommendation: enable it in the same commit as the
   `gbm` default flip, and re-tune `imbalance_alpha` down so the exploitable edge stays at
   today's economic magnitude.)*

---

## 8. Effort summary

| Phase | Deliverable | Est. |
|---|---|---|
| 0 | Substreams, no behaviour change | 0.5 d |
| 1 | `BrownianEngine` + statistical tests | 1 d |
| 2 | ABM/GBM/OU behind a flag; impact as overlay | 1.5 d |
| 3 | Exp-OU vol, leverage, OU imbalance, bridge latency | 1.5 d |
| 4 | New regimes + AC-optimality validation experiment | 1 d |
| 5 | Antithetic + block bootstrap (+ optional QMC) | 1 d |
| 6 | Demo script, plots, README/report updates | 0.5 d |

**Highest value per unit of effort:** Phase 2 (fixes **G4**, which quietly weakens every
statistical claim currently in the report) and Phase 4 (the AC-optimality experiment, the
single most persuasive validation this project can show).
