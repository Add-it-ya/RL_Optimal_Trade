# E1b (exploratory): reading rule, fixed before any 5M result existed

Written 2026-10-20 14:23 +0530. The four 5M PPO runs (seeds 0-3, experiment `e1b_5m`) started
14:18:47-14:18:49 from `../RL-Optimal-Trade-Execution-e1` (afbb38b, clean); at the time of
writing `results/e1b_5m/` does not exist. E1's pre-registered verdict (§6.2/§9 of
IMPROVEMENT_PLAN.md) is not touched by anything below.

Question: does PPO's gap to Almgren-Chriss on `abm_ac_optimal` shrink when the training
budget goes from 2M (E1) to 5M steps? Everything else is identical (defaults, eval_every
100k, 1,000 validation episodes, linear lr 3e-4 -> 0 over the budget, so the 5M runs also
anneal more slowly).

Baseline for the comparison: E1's PPO seeds 0-3 on the same 5,000 antithetic test episodes
(seeds 2,000,000+): gap to AC 0.128 / 1.638 / 0.653 / 0.204 bps (mean 0.656), L1 0.078 /
0.313 / 0.204 / 0.135 (median 0.169), selected at 1.9M / 0.9M / 1.9M / 1.9M. Seed 1 is E1's
worst seed and peaked at 0.9M, then got worse: an instability, not obviously a budget
limit. The seeds 0-3 subset is worse than E1's 10-seed mean (0.47).

Primary measure: per seed, dJ = J(5M) - J(2M), paired episode by episode on the identical
test block (AC's per-episode objective must be bit-identical in both result sets), 95% CI by
block-2 bootstrap; then the mean of the four dJ.

Reading:
- "Budget binds": mean dJ < 0, at least 3 of 4 seeds have dJ < 0, and the median L1 over
  seeds 0-3 falls.
- "Budget does not bind": mean dJ >= 0, or at most 2 of 4 seeds improve. The residual gap
  then has to be explained by something other than the number of steps.
- Anything else: "mixed", reported as such with the per-seed numbers.
- Always report: selected step and its fraction of the budget, the validation curve at
  0.5M intervals, eff steps, first-trade fraction, and whether any 5M seed gets under the
  E1 L1 bar of 0.10. n = 4, so no across-seed claim beyond the sign count and a crude CI.

10M follow-up (`e1b_10m`, seeds 0-1): run it only if at least 2 of 4 5M seeds select a
checkpoint in the last 15% of the budget (step >= 4.25M) and mean dJ < 0.
