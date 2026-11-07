# Results: `e1`

- Test episodes per strategy: 5000 (seeds from 2,000,000, antithetic pairs)
- Trained runs: 20; code version(s): afbb38b098
- `J_bps` is the objective the agents are rewarded on: implementation shortfall plus the Almgren-Chriss penalty on carried variance (lower is better). `eff_steps` is how many steps the order is effectively spread over (20 = TWAP, 1 = one block trade). `IS_p95` and `IS_cvar95` are the 95th percentile of implementation shortfall and the mean of its worst 5% (bps).
- `L1_vs_AC` is the L1 distance between the strategy's mean schedule (fraction of the order traded at each step) and Almgren-Chriss's: 0 = the same shape, 2 = disjoint. For multi-seed agents it is the median across seeds.
- `vs_AlmgrenChriss` is the mean paired difference in `objective_bps` (negative = better than AlmgrenChriss). For multi-seed agents the 95% CI and p-value are across training seeds; `p_holm` is Holm-adjusted over every row of every table.
- `n_degenerate` counts trained seeds whose policy trades in fewer than 5 effective steps -- a block trade, not an execution schedule. They are reported, never dropped.

## abm_ac_optimal

| strategy | n_seeds | n_degenerate | J_bps | IS_bps | IS_std | IS_p95 | IS_cvar95 | eff_steps | L1_vs_AC | vs_AlmgrenChriss | ci_low | ci_high | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SignalRule | — | — | 28.87 | 21.5 | 71.6 | 138.4 | 168.4 | 14.7 | 0.000 | 0.00 | 0.00 | 0.00 | 1.0 |
| AlmgrenChriss | — | — | 28.87 | 21.5 | 71.6 | 138.4 | 168.4 | 14.7 | 0.000 | 0.00 | — | — | — |
| PPO | 10 | 0 | 29.34 | 20.7 | 77.4 | 148.8 | 183.7 | 16.2 | 0.125 | 0.47 | 0.23 | 0.79 | 0.0 |
| DoubleDQN | 10 | 0 | 30.60 | 22.8 | 73.8 | 145.3 | 176.4 | 13.5 | 0.197 | 1.73 | 1.52 | 1.96 | 4.78e-07 |
| VWAP | — | — | 32.53 | 19.1 | 96.5 | 178.2 | 217.1 | 19.6 | 0.479 | 3.66 | 3.66 | 3.66 | 0.0 |
| TWAP | — | — | 32.83 | 19.0 | 98.1 | 180.5 | 220.0 | 20.0 | 0.503 | 3.96 | 3.96 | 3.96 | 0.0 |
| DumpAtStart | — | — | 154.38 | 154.4 | 0.7 | 155.7 | 155.9 | 1.0 | 1.759 | 125.51 | 125.51 | 125.52 | 0.0 |
| WaitThenDump | — | — | 196.94 | 154.4 | 173.6 | 447.3 | 512.5 | 1.0 | 1.959 | 168.07 | 168.07 | 168.08 | 0.0 |

SignalRule coefficient fitted on validation seeds: beta = 0.0.

![abm_ac_optimal schedule](figures/abm_ac_optimal_schedule.png)
![abm_ac_optimal risk_vs_cost](figures/abm_ac_optimal_risk_vs_cost.png)
![abm_ac_optimal is_distribution](figures/abm_ac_optimal_is_distribution.png)

## Checkpoint selection

Mean objective (bps) on the validation seeds before training, at the checkpoint each run kept, and at the end of training. The test seeds above were never used for selection.

| run | J_untrained | J_selected | selected_step | J_final |
|---|---|---|---|---|
| DoubleDQN/s0 | 45.4 | 30.3 | 600,000 | 31.7 |
| DoubleDQN/s1 | 188.2 | 30.5 | 2,000,000 | 30.5 |
| DoubleDQN/s2 | 36.6 | 30.5 | 1,900,000 | 30.7 |
| DoubleDQN/s3 | 80.6 | 30.5 | 800,000 | 33.7 |
| DoubleDQN/s4 | 151.9 | 30.7 | 1,200,000 | 31.3 |
| DoubleDQN/s5 | 43.3 | 30.7 | 1,300,000 | 31.6 |
| DoubleDQN/s6 | 80.0 | 29.9 | 900,000 | 32.1 |
| DoubleDQN/s7 | 32.8 | 29.9 | 800,000 | 33.5 |
| DoubleDQN/s8 | 31.0 | 30.5 | 800,000 | 33.4 |
| DoubleDQN/s9 | 30.1 | 30.1 | 0 | 32.1 |
| PPO/s0 | 32.8 | 29.0 | 1,900,000 | 29.2 |
| PPO/s1 | 32.8 | 30.4 | 900,000 | 31.1 |
| PPO/s2 | 32.8 | 29.6 | 1,900,000 | 29.9 |
| PPO/s3 | 32.8 | 29.2 | 1,900,000 | 29.2 |
| PPO/s4 | 32.9 | 29.3 | 1,300,000 | 29.9 |
| PPO/s5 | 32.9 | 29.8 | 1,200,000 | 32.4 |
| PPO/s6 | 32.9 | 29.1 | 1,700,000 | 29.4 |
| PPO/s7 | 32.9 | 29.3 | 1,800,000 | 30.1 |
| PPO/s8 | 32.7 | 29.0 | 1,200,000 | 30.3 |
| PPO/s9 | 32.9 | 29.0 | 1,000,000 | 29.7 |

![validation curves](figures/validation_curves.png)
