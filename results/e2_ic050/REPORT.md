# Results: `e2_ic050`

- Test episodes per strategy: 5000 (seeds from 2,000,000, antithetic pairs)
- Trained runs: 20; code version(s): f8efbd7fda
- `J_bps` is the objective the agents are rewarded on: implementation shortfall plus the Almgren-Chriss penalty on carried variance (lower is better). `eff_steps` is how many steps the order is effectively spread over (20 = TWAP, 1 = one block trade). `IS_p95` and `IS_cvar95` are the 95th percentile of implementation shortfall and the mean of its worst 5% (bps).
- `L1_vs_AC` is the L1 distance between the strategy's mean schedule (fraction of the order traded at each step) and Almgren-Chriss's: 0 = the same shape, 2 = disjoint. For multi-seed agents it is the median across seeds.
- `vs_AlmgrenChriss` is the mean paired difference in `objective_bps` (negative = better than AlmgrenChriss). For multi-seed agents the 95% CI and p-value are across training seeds; `p_holm` is Holm-adjusted over every row of every table.
- `n_degenerate` counts trained seeds whose policy trades in fewer than 5 effective steps -- a block trade, not an execution schedule. They are reported, never dropped.

## signal_ic050

| strategy | n_seeds | n_degenerate | J_bps | IS_bps | IS_std | IS_p95 | IS_cvar95 | eff_steps | L1_vs_AC | vs_AlmgrenChriss | ci_low | ci_high | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SignalRule | — | — | 28.20 | 20.4 | 74.6 | 140.4 | 171.0 | 14.3 | 0.030 | -0.67 | -1.21 | -0.12 | 0.0 |
| PPO | 10 | 0 | 28.67 | 19.7 | 79.7 | 148.1 | 182.7 | 14.9 | 0.121 | -0.20 | -0.41 | 0.01 | 0.1 |
| AlmgrenChriss | — | — | 28.87 | 21.5 | 71.9 | 138.7 | 168.8 | 14.7 | 0.000 | 0.00 | — | — | — |
| DoubleDQN | 10 | 0 | 30.14 | 21.5 | 78.3 | 147.0 | 182.8 | 13.2 | 0.145 | 1.27 | 0.95 | 1.62 | 0.000218 |
| VWAP | — | — | 32.53 | 19.1 | 97.1 | 179.2 | 217.8 | 19.6 | 0.479 | 3.66 | 3.66 | 3.66 | 0.0 |
| TWAP | — | — | 32.83 | 19.0 | 98.7 | 181.7 | 220.6 | 20.0 | 0.503 | 3.96 | 3.96 | 3.96 | 0.0 |
| DumpAtStart | — | — | 154.38 | 154.4 | 0.7 | 155.7 | 155.9 | 1.0 | 1.759 | 125.51 | 125.51 | 125.52 | 0.0 |
| WaitThenDump | — | — | 196.94 | 154.4 | 175.1 | 445.0 | 515.6 | 1.0 | 1.959 | 168.07 | 168.07 | 168.08 | 0.0 |

SignalRule coefficient fitted on validation seeds: beta = 1.0.

![signal_ic050 schedule](figures/signal_ic050_schedule.png)
![signal_ic050 risk_vs_cost](figures/signal_ic050_risk_vs_cost.png)
![signal_ic050 is_distribution](figures/signal_ic050_is_distribution.png)

## Checkpoint selection

Mean objective (bps) on the validation seeds before training, at the checkpoint each run kept, and at the end of training. The test seeds above were never used for selection.

| run | J_untrained | J_selected | selected_step | J_final |
|---|---|---|---|---|
| DoubleDQN/s0 | 47.7 | 29.1 | 1,000,000 | 31.5 |
| DoubleDQN/s1 | 187.0 | 28.6 | 300,000 | 29.1 |
| DoubleDQN/s2 | 36.5 | 28.9 | 900,000 | 30.1 |
| DoubleDQN/s3 | 80.6 | 28.5 | 900,000 | 30.1 |
| DoubleDQN/s4 | 152.3 | 28.1 | 1,400,000 | 29.4 |
| DoubleDQN/s5 | 44.4 | 28.5 | 700,000 | 29.6 |
| DoubleDQN/s6 | 80.0 | 29.0 | 800,000 | 31.0 |
| DoubleDQN/s7 | 32.8 | 28.5 | 900,000 | 28.9 |
| DoubleDQN/s8 | 31.0 | 28.8 | 1,100,000 | 29.6 |
| DoubleDQN/s9 | 30.0 | 28.6 | 1,000,000 | 30.4 |
| PPO/s0 | 32.8 | 27.6 | 3,300,000 | 27.9 |
| PPO/s1 | 32.8 | 27.9 | 1,800,000 | 28.4 |
| PPO/s2 | 32.8 | 27.5 | 1,900,000 | 27.9 |
| PPO/s3 | 32.8 | 27.8 | 2,500,000 | 28.4 |
| PPO/s4 | 32.9 | 27.4 | 5,000,000 | 27.4 |
| PPO/s5 | 32.9 | 27.6 | 1,300,000 | 27.9 |
| PPO/s6 | 32.9 | 27.8 | 5,000,000 | 27.8 |
| PPO/s7 | 32.9 | 27.6 | 1,300,000 | 28.6 |
| PPO/s8 | 32.7 | 27.6 | 5,000,000 | 27.6 |
| PPO/s9 | 32.9 | 27.9 | 4,800,000 | 27.9 |

![validation curves](figures/validation_curves.png)
