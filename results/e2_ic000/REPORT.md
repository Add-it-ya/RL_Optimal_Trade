# Results: `e1b_5m`

- Test episodes per strategy: 5000 (seeds from 2,000,000, antithetic pairs)
- Trained runs: 10; code version(s): afbb38b098
- `J_bps` is the objective the agents are rewarded on: implementation shortfall plus the Almgren-Chriss penalty on carried variance (lower is better). `eff_steps` is how many steps the order is effectively spread over (20 = TWAP, 1 = one block trade). `IS_p95` and `IS_cvar95` are the 95th percentile of implementation shortfall and the mean of its worst 5% (bps).
- `L1_vs_AC` is the L1 distance between the strategy's mean schedule (fraction of the order traded at each step) and Almgren-Chriss's: 0 = the same shape, 2 = disjoint. For multi-seed agents it is the median across seeds.
- `vs_AlmgrenChriss` is the mean paired difference in `objective_bps` (negative = better than AlmgrenChriss). For multi-seed agents the 95% CI and p-value are across training seeds; `p_holm` is Holm-adjusted over every row of every table.
- `n_degenerate` counts trained seeds whose policy trades in fewer than 5 effective steps -- a block trade, not an execution schedule. They are reported, never dropped.

## signal_ic000

| strategy | n_seeds | n_degenerate | J_bps | IS_bps | IS_std | IS_p95 | IS_cvar95 | eff_steps | L1_vs_AC | vs_AlmgrenChriss | ci_low | ci_high | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SignalRule | — | — | 28.87 | 21.5 | 71.6 | 138.4 | 168.4 | 14.7 | 0.000 | 0.00 | 0.00 | 0.00 | 1.0 |
| AlmgrenChriss | — | — | 28.87 | 21.5 | 71.6 | 138.4 | 168.4 | 14.7 | 0.000 | 0.00 | — | — | — |
| PPO | 10 | 0 | 29.07 | 21.2 | 74.0 | 143.5 | 177.5 | 15.3 | 0.065 | 0.20 | 0.07 | 0.35 | 0.0 |
| VWAP | — | — | 32.53 | 19.1 | 96.5 | 178.2 | 217.1 | 19.6 | 0.479 | 3.66 | 3.66 | 3.66 | 0.0 |
| TWAP | — | — | 32.83 | 19.0 | 98.1 | 180.5 | 220.0 | 20.0 | 0.503 | 3.96 | 3.96 | 3.96 | 0.0 |
| DumpAtStart | — | — | 154.38 | 154.4 | 0.7 | 155.7 | 155.9 | 1.0 | 1.759 | 125.51 | 125.51 | 125.52 | 0.0 |
| WaitThenDump | — | — | 196.94 | 154.4 | 173.6 | 447.3 | 512.5 | 1.0 | 1.959 | 168.07 | 168.07 | 168.08 | 0.0 |

SignalRule coefficient fitted on validation seeds: beta = 0.0.

![signal_ic000 schedule](figures/signal_ic000_schedule.png)
![signal_ic000 risk_vs_cost](figures/signal_ic000_risk_vs_cost.png)
![signal_ic000 is_distribution](figures/signal_ic000_is_distribution.png)

## Checkpoint selection

Mean objective (bps) on the validation seeds before training, at the checkpoint each run kept, and at the end of training. The test seeds above were never used for selection.

| run | J_untrained | J_selected | selected_step | J_final |
|---|---|---|---|---|
| PPO/s0 | 32.8 | 28.9 | 4,400,000 | 28.9 |
| PPO/s1 | 32.8 | 29.7 | 2,700,000 | 29.7 |
| PPO/s2 | 32.8 | 29.1 | 3,600,000 | 29.5 |
| PPO/s3 | 32.8 | 29.1 | 1,800,000 | 29.8 |
| PPO/s4 | 32.9 | 28.9 | 4,100,000 | 29.3 |
| PPO/s5 | 32.9 | 29.2 | 1,000,000 | 33.4 |
| PPO/s6 | 32.9 | 29.0 | 2,300,000 | 29.2 |
| PPO/s7 | 32.9 | 29.2 | 1,300,000 | 29.7 |
| PPO/s8 | 32.7 | 29.0 | 4,800,000 | 29.0 |
| PPO/s9 | 32.9 | 29.0 | 4,600,000 | 29.1 |

![validation curves](figures/validation_curves.png)
