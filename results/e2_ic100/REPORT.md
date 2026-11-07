# Results: `e2_ic100`

- Test episodes per strategy: 5000 (seeds from 2,000,000, antithetic pairs)
- Trained runs: 10; code version(s): f8efbd7fda
- `J_bps` is the objective the agents are rewarded on: implementation shortfall plus the Almgren-Chriss penalty on carried variance (lower is better). `eff_steps` is how many steps the order is effectively spread over (20 = TWAP, 1 = one block trade). `IS_p95` and `IS_cvar95` are the 95th percentile of implementation shortfall and the mean of its worst 5% (bps).
- `L1_vs_AC` is the L1 distance between the strategy's mean schedule (fraction of the order traded at each step) and Almgren-Chriss's: 0 = the same shape, 2 = disjoint. For multi-seed agents it is the median across seeds.
- `vs_AlmgrenChriss` is the mean paired difference in `objective_bps` (negative = better than AlmgrenChriss). For multi-seed agents the 95% CI and p-value are across training seeds; `p_holm` is Holm-adjusted over every row of every table.
- `n_degenerate` counts trained seeds whose policy trades in fewer than 5 effective steps -- a block trade, not an execution schedule. They are reported, never dropped.

## signal_ic100

| strategy | n_seeds | n_degenerate | J_bps | IS_bps | IS_std | IS_p95 | IS_cvar95 | eff_steps | L1_vs_AC | vs_AlmgrenChriss | ci_low | ci_high | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SignalRule | — | — | 25.92 | 18.1 | 76.8 | 138.2 | 168.0 | 14.3 | 0.030 | -2.95 | -3.51 | -2.40 | 2.61e-25 |
| PPO | 10 | 0 | 25.92 | 15.9 | 87.1 | 147.3 | 184.5 | 13.0 | 0.129 | -2.95 | -3.10 | -2.78 | 8.03e-11 |
| AlmgrenChriss | — | — | 28.87 | 21.5 | 73.8 | 142.7 | 172.6 | 14.7 | 0.000 | 0.00 | — | — | — |
| VWAP | — | — | 32.53 | 19.1 | 100.0 | 184.4 | 223.0 | 19.6 | 0.479 | 3.66 | 3.66 | 3.66 | 0.0 |
| TWAP | — | — | 32.83 | 19.0 | 101.6 | 185.6 | 225.9 | 20.0 | 0.503 | 3.96 | 3.96 | 3.96 | 0.0 |
| DumpAtStart | — | — | 154.38 | 154.4 | 0.7 | 155.7 | 155.9 | 1.0 | 1.759 | 125.51 | 125.51 | 125.52 | 0.0 |
| WaitThenDump | — | — | 196.94 | 154.4 | 180.7 | 451.0 | 526.4 | 1.0 | 1.959 | 168.07 | 168.07 | 168.08 | 0.0 |

SignalRule coefficient fitted on validation seeds: beta = 1.0.

![signal_ic100 schedule](figures/signal_ic100_schedule.png)
![signal_ic100 risk_vs_cost](figures/signal_ic100_risk_vs_cost.png)
![signal_ic100 is_distribution](figures/signal_ic100_is_distribution.png)

## Checkpoint selection

Mean objective (bps) on the validation seeds before training, at the checkpoint each run kept, and at the end of training. The test seeds above were never used for selection.

| run | J_untrained | J_selected | selected_step | J_final |
|---|---|---|---|---|
| PPO/s0 | 32.8 | 23.8 | 2,400,000 | 24.4 |
| PPO/s1 | 32.8 | 24.2 | 900,000 | 24.6 |
| PPO/s2 | 32.8 | 24.2 | 5,000,000 | 24.2 |
| PPO/s3 | 32.8 | 23.8 | 4,800,000 | 23.9 |
| PPO/s4 | 32.9 | 24.0 | 1,800,000 | 24.3 |
| PPO/s5 | 32.9 | 23.9 | 4,400,000 | 24.1 |
| PPO/s6 | 32.9 | 24.2 | 2,800,000 | 24.4 |
| PPO/s7 | 32.9 | 24.0 | 4,400,000 | 24.2 |
| PPO/s8 | 32.7 | 24.2 | 1,300,000 | 24.4 |
| PPO/s9 | 32.9 | 23.9 | 2,700,000 | 24.7 |

![validation curves](figures/validation_curves.png)
