# Results: `e2t_ic100`

- Test episodes per strategy: 5000 (seeds from 2,000,000, antithetic pairs)
- Trained runs: 10; code version(s): 20f2f4ac66
- `J_bps` is the objective the agents are rewarded on: implementation shortfall plus the Almgren-Chriss penalty on carried variance (lower is better). `eff_steps` is how many steps the order is effectively spread over (20 = TWAP, 1 = one block trade). `IS_p95` and `IS_cvar95` are the 95th percentile of implementation shortfall and the mean of its worst 5% (bps).
- `L1_vs_AC` is the L1 distance between the strategy's mean schedule (fraction of the order traded at each step) and Almgren-Chriss's: 0 = the same shape, 2 = disjoint. For multi-seed agents it is the median across seeds.
- `vs_AlmgrenChriss` is the mean paired difference in `objective_bps` (negative = better than AlmgrenChriss). For multi-seed agents the 95% CI and p-value are across training seeds; `p_holm` is Holm-adjusted over every row of every table.
- `n_degenerate` counts trained seeds whose policy trades in fewer than 5 effective steps -- a block trade, not an execution schedule. They are reported, never dropped.

## signal_ic100

| strategy | n_seeds | n_degenerate | J_bps | IS_bps | IS_std | IS_p95 | IS_cvar95 | eff_steps | L1_vs_AC | vs_AlmgrenChriss | ci_low | ci_high | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VecPPO | 10 | 0 | 25.49 | 16.1 | 84.4 | 143.7 | 178.2 | 13.2 | 0.114 | -3.38 | -3.44 | -3.32 | 3.79e-15 |
| SignalRule | — | — | 25.92 | 18.1 | 76.8 | 138.2 | 168.0 | 14.3 | 0.030 | -2.95 | -3.51 | -2.40 | 2.61e-25 |
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
| VecPPO/s0 | 32.9 | 23.7 | 13,007,029 | 23.9 |
| VecPPO/s1 | 32.9 | 23.8 | 21,039,445 | 23.8 |
| VecPPO/s2 | 32.8 | 23.7 | 24,065,957 | 24.0 |
| VecPPO/s3 | 32.8 | 23.7 | 21,035,966 | 23.7 |
| VecPPO/s4 | 32.8 | 23.7 | 44,057,739 | 23.7 |
| VecPPO/s5 | 32.9 | 23.6 | 19,071,549 | 23.7 |
| VecPPO/s6 | 32.9 | 23.7 | 26,032,843 | 23.8 |
| VecPPO/s7 | 32.5 | 23.8 | 33,075,183 | 23.8 |
| VecPPO/s8 | 32.9 | 23.8 | 43,069,982 | 24.0 |
| VecPPO/s9 | 32.9 | 23.7 | 42,007,795 | 23.8 |

![validation curves](figures/validation_curves.png)
