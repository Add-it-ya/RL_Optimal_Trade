# Roadmap

**Goal.** A learned execution policy whose quality can be *checked*, not just reported:

1. With no predictive signal, it recovers the closed-form Almgren-Chriss optimum.
2. With a realistic signal, it beats Almgren-Chriss on the same mean-variance objective, and
   the gain is reported as a function of signal strength with confidence intervals.
3. It degrades gracefully on market regimes it was not trained on.

---

## Where things stand (v0.2)

The lean core is done: Brownian-only market dynamics, PPO and DQN / Double DQN, one
`rlx train | evaluate | report` command line, and run folders whose weights are pinned by a
SHA-256 hash. The v0.1 results were withdrawn. They could not be regenerated from the
repository, and the published PPO row was identical to a policy that never trades and dumps
the whole order at the final step.

## Why the v0.1 agents did not learn execution

It was not a compute problem. Measured on the v0.1 simulator, 1,500 episodes per row on the
same seeds (sell 10,000 shares over 20 steps):

| regime | strategy | mean IS (bps) | sd IS (bps) |
|---|---|---|---|
| `normal_liquidity` | TWAP | −40.1 | 856.5 |
| | dump everything at t = 0 | 17.5 | **0.3** |
| | wait, then dump at the end | −111.4 | 1807.2 |
| `abm_ac_optimal` | TWAP | 31.7 | 495.2 |
| | dump everything at t = 0 | 14.2 | **0.3** |

- **Mis-scaled economics.** Dumping the entire order at once costs only ~15 bps, while
  per-episode price noise is 500–1,800 bps. The part of the reward an agent can control is
  a few percent of its standard deviation at most, so the learning signal is buried. The only "solutions"
  left are degenerate: dump immediately, or hold and speculate.
- **The reward was not the objective.** Temporary impact was charged twice (in the fill price
  and again as a penalty). The risk term used a realised-volatility estimate that is zero at
  the start of every episode. Its risk aversion was never linked to the Almgren-Chriss
  baseline's.
- **The action started in the worst place.** A Box [0, 1] "fraction of remaining
  inventory": an untrained Gaussian policy outputs ≈ 0, which clips to "do nothing", so PPO
  starts as wait-then-dump.
- **Thin training and evaluation.** One environment, one training seed, γ = 0.99 on a
  20-step horizon, no checkpoint selection on held-out seeds, and one evaluation seed block
  whose noise exceeded several of the reported effects.

---

## Phase 1: fix the problem formulation

**1.1 Recalibrate the market** so impact and risk trade off. A step is a ~20-minute bucket.
Starting values, to be confirmed by a calibration script:

| parameter | v0.1 | target |
|---|---|---|
| per-step volatility | 0.02 | ~0.004 |
| temporary impact η | 1e-5 | ~1.5e-4 |
| permanent impact γ | 5e-6 | ~2e-5 |
| imbalance signal | IC ≈ 0.43 | IC 0.05–0.10 (realistic) |

Pinned by a test: dump-at-arrival cost 100–200 bps, TWAP cost 10–30 bps, TWAP IS sd
80–130 bps, and an Almgren-Chriss schedule that differs from TWAP by an L1 distance of at
least 0.2. Without that last one, "recovering AC" would be indistinguishable from staying
at TWAP.

**1.2 Make the reward equal the objective.** Per step, using mark-to-market accounting:

```
r_t = − (cost of this fill vs the pre-trade mid)
      + (P&L of the inventory still held over the next step)
      − λ · (σ_t · x_{t+1})²
```

Summed over an episode this is exactly `−IS − λ·Σσ²x²`, the Almgren-Chriss mean-variance
objective, enforced by an identity test. One λ is shared with the Almgren-Chriss baseline, so
in the no-signal regime AC *is* the optimum of the RL reward.

**1.3 Make the action relative to TWAP.** `a ∈ [−1, 1]` maps to a multiplier
`m = (1 + a)² ∈ [0, 4]` on the TWAP rate. An untrained policy now starts at TWAP.
DoubleDQN uses the grid `{0, ¼, ½, ¾, 1, 1¼, 1½, 2, 3, 4}`.

**1.4 Clean up the observation.** All features are O(1) and scale-free: inventory, time,
price move in σ units, imbalance, volatility ratio, spread, and depth against a fixed
reference.

**1.5 Add baselines that expose cheating.** `DumpAtStart`, `WaitThenDump`, and a tuned
two-line `SignalRule` alongside TWAP, VWAP and Almgren-Chriss. Also add an
effective-steps-traded metric, and pad mean schedules to the horizon instead of truncating
them.

## Phase 2: train properly

- PPO with 8–16 vectorised envs, γ = 1, a hand-scaled observation and reward, 2–5M steps per
  seed. DoubleDQN with γ = 1, Polyak target updates and 1–2M steps.
- **Ten training seeds** per configuration; the training seed is the unit of analysis.
- **Three disjoint seed blocks**: training, validation (checkpoint selection), and test
  (touched only by `rlx evaluate`).
- Degenerate-policy guard: any seed whose best checkpoint trades in fewer than five
  effective steps is flagged in the report, not silently dropped.

## Phase 3: three experiments, judged against criteria fixed in advance

| exp | market | question |
|---|---|---|
| **E1** ground truth | arithmetic BM, no signal, constant volatility | Does the learner recover the Almgren-Chriss optimum? |
| **E2** signal sweep | IC ∈ {0, 0.025, 0.05, 0.10, 0.20} | How does the edge over AC and over a two-line rule scale with signal strength? |
| **E3** robustness | train on a randomised family; test in- and out-of-distribution (2× vol, 2× impact, GBM, clustered vol, leverage) | How much does performance degrade off-distribution? |

| check | pass |
|---|---|
| E1 | mean objective within max(0.5 bps, 95% CI) of AC; median-seed schedule L1 to AC < 0.10 |
| E2 | the improvement at IC = 0 has a CI covering zero (no simulator exploit) and rises with IC; results vs `SignalRule` reported whatever they show |
| E3 | in- and out-of-distribution reported separately, degradation in bps with a CI |
| language | Holm-adjusted across the reported family; no "wins in N/N regimes" claims |

## Phase 4: GPU-batched environment

Because the whole exogenous market is drawn at reset and impact is an additive overlay, a
step reduces to a few tensor operations. Planned:

- a batched PyTorch environment that steps thousands of episodes at once on a GPU;
- a compact PPO trained on it;
- three parity tests against the NumPy reference environment:
  - identical fills, rewards and observations given the same paths;
  - matching path statistics;
  - the GPU PPO reproducing the E1 result;
- a throughput benchmark. The target is ≥ 1M environment steps per second.

Reported numbers keep coming from the reference environment; the GPU path makes E2's sweep
cheap.

## Phase 5: results and write-up

Tables with mean, spread, tail (CVaR95) and the paired gap to AC with confidence intervals;
schedule-vs-AC, cost-vs-risk, improvement-vs-signal and throughput figures; a short "what
went wrong in v0.1" section. The curated results and the small weight files will be
committed with their hashes, plus a script that re-evaluates them and checks the published
table.

## Out of scope for now

Validation on real limit-order-book data, multi-agent interaction, and live or paper
trading. Each is worth doing once the results above hold.
