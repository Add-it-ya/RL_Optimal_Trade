# Results: `e2t_ic050`

- Test episodes per strategy: 5000 (seeds from 2,000,000, antithetic pairs)
- Trained runs: 10; code version(s): 20f2f4ac66
- `J_bps` is the objective the agents are rewarded on: implementation shortfall plus the Almgren-Chriss penalty on carried variance (lower is better). `eff_steps` is how many steps the order is effectively spread over (20 = TWAP, 1 = one block trade). `IS_p95` and `IS_cvar95` are the 95th percentile of implementation shortfall and the mean of its worst 5% (bps).
- `L1_vs_AC` is the L1 distance between the strategy's mean schedule (fraction of the order traded at each step) and Almgren-Chriss's: 0 = the same shape, 2 = disjoint. For multi-seed agents it is the median across seeds.
- `vs_AlmgrenChriss` is the mean paired difference in `objective_bps` (negative = better than AlmgrenChriss). For multi-seed agents the 95% CI and p-value are across training seeds; `p_holm` is Holm-adjusted over every row of every table.
- `n_degenerate` counts trained seeds whose policy trades in fewer than 5 effective steps -- a block trade, not an execution schedule. They are reported, never dropped.

## signal_ic050

| strategy | n_seeds | n_degenerate | J_bps | IS_bps | IS_std | IS_p95 | IS_cvar95 | eff_steps | L1_vs_AC | vs_AlmgrenChriss | ci_low | ci_high | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SignalRule | — | — | 28.20 | 20.4 | 74.6 | 140.4 | 171.0 | 14.3 | 0.030 | -0.67 | -1.21 | -0.12 | 0.0 |
| VecPPO | 10 | 0 | 28.55 | 19.6 | 79.6 | 146.8 | 180.2 | 14.9 | 0.110 | -0.32 | -0.44 | -0.20 | 0.00178 |
| AlmgrenChriss | — | — | 28.87 | 21.5 | 71.9 | 138.7 | 168.8 | 14.7 | 0.000 | 0.00 | — | — | — |
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
| VecPPO/s0 | 32.9 | 27.5 | 50,041,540 | 27.5 |
| VecPPO/s1 | 32.9 | 27.4 | 46,027,296 | 27.4 |
| VecPPO/s2 | 32.8 | 27.3 | 46,026,566 | 27.3 |
| VecPPO/s3 | 32.8 | 27.5 | 49,058,473 | 27.5 |
| VecPPO/s4 | 32.8 | 27.3 | 24,073,075 | 27.3 |
| VecPPO/s5 | 32.9 | 27.5 | 40,047,680 | 27.5 |
| VecPPO/s6 | 32.9 | 27.7 | 50,041,678 | 27.7 |
| VecPPO/s7 | 32.5 | 27.5 | 41,029,711 | 27.6 |
| VecPPO/s8 | 32.9 | 27.4 | 33,001,783 | 27.4 |
| VecPPO/s9 | 32.9 | 27.5 | 41,031,294 | 27.5 |

![validation curves](figures/validation_curves.png)
