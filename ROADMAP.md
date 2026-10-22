# Roadmap

**Goal.** A learned execution policy whose quality can be *checked*, not just reported:

1. With no predictive signal, it recovers the closed-form Almgren-Chriss optimum.
2. With a realistic signal, it beats Almgren-Chriss on the same mean-variance objective, and
   the gain is reported as a function of signal strength with confidence intervals.
3. It degrades gracefully on market regimes it was not trained on.

---

## Where things stand (v0.2)

**Phases 1, 2 and 3 are done**: all three experiments have run (see below).
Phase 1 brought the calibrated market, the exact
objective, the TWAP-relative action, the cleaner observation and the null baselines.
Phase 2 brought vectorised PPO, DoubleDQN fixes, O(1) rewards and checkpoint selection on
validation seeds. First evidence that training now works, on the signal-free
`abm_ac_optimal` regime with the default 2M-step budget: four PPO runs reached a validation
objective of 29.1–29.6 bps against 28.9 for the Almgren-Chriss optimum and 32.8 for TWAP
(their untrained starting point), closing 82–95% of the gap. DoubleDQN first did
not learn at all: the value gaps between neighbouring trade sizes are tiny next to the
reward noise, so its greedy action flipped with estimation error. A dueling head, batch
1024 and a learning rate of 5e-5 fixed that. Two 2M-step seeds kept checkpoints at 29.7 bps
(untrained: 45.9 and 187.7), and most checkpoints after 300k sit below TWAP. It is noisier
than PPO, front-loads more than AC, and takes ~27 minutes per seed.

**E1 result** (ten seeds per agent, 2M steps each, scored on 5,000 test episodes the agents
never saw; AC 28.87 bps, TWAP 32.83). PPO ends up **0.47 bps** above the Almgren-Chriss
optimum (95% CI 0.23–0.79 across seeds). That is inside the 0.5 bps tolerance but still
measurably worse than AC. Its average schedule has the right front-loaded shape but is
flatter than AC's, at a median L1 distance of 0.125 against a 0.10 bar. By the criteria
fixed in advance that reads as *directionally right, under-tuned*. Four of the ten seeds do
clear 0.10, and they are the cheapest ones. DoubleDQN ends up **1.73 bps** off (1.52–1.96,
median L1 0.197), so it fails the cost criterion. Its greedy action reacts to noise in the
observation, so its schedule changes from episode to episode in a market where nothing is
predictable; uneven trade sizes cost more under impact that grows faster than the trade.
No seed of either agent collapsed to a block trade. Running the evaluation also exposed
a bug: `rlx evaluate` gave DoubleDQN the continuous action space, which made it look
exactly like dumping the order at once. That's fixed, and a mismatch now raises an error.

**A longer budget helps (exploratory, not part of the pre-registered E1).** Retrained for
5M steps, the same ten PPO seeds come within **0.20 bps** of AC (0.07–0.35), and eight of
the ten clear the 0.10 schedule bar. Every seed improves, most of all those that were worst
at 2M. A 2M run that follows the 5M learning-rate schedule recovers less than half of that
gain, so it is the extra training steps, not the slower decay. E2 and E3 therefore train
PPO for 5M steps.

**E3 result** (one PPO policy trained on a randomised family of markets, ten seeds; scored
on 3 regimes inside that family and 9 outside it). Measured as the gap to Almgren-Chriss
in each regime, PPO gets **5.5 bps** worse out of distribution (95% CI 5.0–6.0), against
6.2 for TWAP. The average hides two regimes. At twice the training volatility every seed is
20–37 bps worse than AC, because the policy keeps its usual pace where AC speeds up sharply;
the volatility is outside anything it was trained on, and it does not extrapolate. In the
mean-reverting market, which the family does include, every seed beats AC by 3–6 bps,
because it learned the reversion AC ignores. Without those two regimes the degradation is
0.8 bps (0.7–1.1) against 0.3 for TWAP. A per-regime refit of the two-line `SignalRule`
beats the single PPO policy in most regimes; that rule is told each regime's parameters,
so E2 is where the value of the signal is measured cleanly.

**E2 result** (the E1 market with its order-book signal switched on at five strengths; ten
PPO seeds each, plus DoubleDQN at IC 0.05). PPO's saving over Almgren-Chriss grows with the
signal, by about 3.6 bps per 0.05 of IC across all 50 seeds. It grows roughly with the
square of the IC: +2.95 bps at IC 0.10, +13.4 at 0.20. With no signal PPO is still 0.2 bps
*worse* than AC (0.07–0.35), so the check written for this point ("the CI covers zero")
fails as stated, but on the expensive side. There is no sign of a simulator exploit. The
honest headline is the comparison with the two-line `SignalRule`, fitted on validation
seeds. At realistic signal strengths (IC 0.05–0.10), PPO is up to 0.5 bps worse than it at
IC 0.05 and level with it at 0.10. It only pulls ahead at IC 0.20 (+0.97 bps, 0.74–1.17),
two to four times stronger than real order-book signals. DoubleDQN trails both. So a
learned policy recovers the value of a realistic signal about as well as a tuned linear
rule, and does better only when the signal is strong.

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

**1.3 Make the action relative to TWAP.** `a ∈ [−1, 1]` sets a multiplier on the TWAP
pace: −1 waits, 0 is TWAP, +1 liquidates everything now (`1 + a` below zero,
`steps_left ** a` above). Every schedule is reachable, so the baselines run through the
same action interface as the agents, and an untrained policy starts at TWAP. DoubleDQN
uses the grid `{0, ¼, ½, ¾, 1, 1¼, 1½, 2, 3, 4}`.

**1.4 Clean up the observation.** All features are O(1) and scale-free: inventory, time,
price move in σ units, imbalance, volatility ratio, spread, and depth against a fixed
reference.

**1.5 Add baselines that expose cheating.** `DumpAtStart`, `WaitThenDump`, and a tuned
two-line `SignalRule` alongside TWAP, VWAP and Almgren-Chriss. Also add an
effective-steps-traded metric, and pad mean schedules to the horizon instead of truncating
them.

## Phase 2: train properly

- PPO with 16 vectorised envs in one process, γ = 1, a hand-scaled observation and reward,
  2M steps per seed (~5 minutes on one core). DoubleDQN with γ = 1, Polyak target updates,
  a dueling head, batch 1024 and learning rate 5e-5 (~27 minutes per 2M-step seed).
- **Ten training seeds** per configuration; the training seed is the unit of analysis.
- **Three disjoint seed blocks**: training, validation (checkpoint selection), and test
  (touched only by `rlx evaluate`).
- Degenerate-policy guard: any seed whose best checkpoint trades in fewer than five
  effective steps is flagged in the report, not silently dropped.

## Phase 3: three experiments, judged against criteria fixed in advance

| exp | market | question |
|---|---|---|
| **E1** ground truth | arithmetic BM, no signal, constant volatility | Does the learner recover the Almgren-Chriss optimum? |
| **E2** signal sweep | the E1 market with its order-book signal switched on, IC ∈ {0, 0.025, 0.05, 0.10, 0.20} | How does the edge over AC and over a two-line rule scale with signal strength? |
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
