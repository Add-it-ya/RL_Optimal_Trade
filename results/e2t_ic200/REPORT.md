# Results: `e2t_ic200`

- Test episodes per strategy: 5000 (seeds from 2,000,000, antithetic pairs)
- Trained runs: 10; code version(s): 20f2f4ac66
- `J_bps` is the objective the agents are rewarded on: implementation shortfall plus the Almgren-Chriss penalty on carried variance (lower is better). `eff_steps` is how many steps the order is effectively spread over (20 = TWAP, 1 = one block trade). `IS_p95` and `IS_cvar95` are the 95th percentile of implementation shortfall and the mean of its worst 5% (bps).
- `L1_vs_AC` is the L1 distance between the strategy's mean schedule (fraction of the order traded at each step) and Almgren-Chriss's: 0 = the same shape, 2 = disjoint. For multi-seed agents it is the median across seeds.
- `vs_AlmgrenChriss` is the mean paired difference in `objective_bps` (negative = better than AlmgrenChriss). For multi-seed agents the 95% CI and p-value are across training seeds; `p_holm` is Holm-adjusted over every row of every table.
- `n_degenerate` counts trained seeds whose policy trades in fewer than 5 effective steps -- a block trade, not an execution schedule. They are reported, never dropped.

## signal_ic200

| strategy | n_seeds | n_degenerate | J_bps | IS_bps | IS_std | IS_p95 | IS_cvar95 | eff_steps | L1_vs_AC | vs_AlmgrenChriss | ci_low | ci_high | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VecPPO | 10 | 0 | 15.17 | 4.4 | 101.1 | 142.1 | 175.1 | 9.9 | 0.129 | -13.70 | -13.79 | -13.60 | 9.99e-19 |
| SignalRule | — | — | 16.43 | 6.8 | 96.0 | 139.7 | 168.9 | 10.8 | 0.124 | -12.44 | -13.68 | -11.24 | 2.05e-81 |
| AlmgrenChriss | — | — | 28.87 | 21.5 | 81.6 | 156.2 | 188.0 | 14.7 | 0.000 | 0.00 | — | — | — |
| VWAP | — | — | 32.53 | 19.1 | 111.9 | 201.0 | 246.7 | 19.6 | 0.479 | 3.66 | 3.66 | 3.66 | 0.0 |
| TWAP | — | — | 32.83 | 19.0 | 113.5 | 204.1 | 249.8 | 20.0 | 0.503 | 3.96 | 3.96 | 3.96 | 0.0 |
| DumpAtStart | — | — | 154.38 | 154.4 | 0.7 | 155.7 | 155.9 | 1.0 | 1.759 | 125.51 | 125.51 | 125.52 | 0.0 |
| WaitThenDump | — | — | 196.94 | 154.4 | 203.5 | 490.3 | 569.0 | 1.0 | 1.959 | 168.07 | 168.07 | 168.08 | 0.0 |

SignalRule coefficient fitted on validation seeds: beta = 2.0.

![signal_ic200 schedule](figures/signal_ic200_schedule.png)
![signal_ic200 risk_vs_cost](figures/signal_ic200_risk_vs_cost.png)
![signal_ic200 is_distribution](figures/signal_ic200_is_distribution.png)

## Checkpoint selection

Mean objective (bps) on the validation seeds before training, at the checkpoint each run kept, and at the end of training. The test seeds above were never used for selection.

| run | J_untrained | J_selected | selected_step | J_final |
|---|---|---|---|---|
| VecPPO/s0 | 32.9 | 12.4 | 12,038,071 | 12.7 |
| VecPPO/s1 | 32.9 | 12.4 | 31,050,990 | 12.5 |
| VecPPO/s2 | 32.8 | 12.5 | 42,031,958 | 12.5 |
| VecPPO/s3 | 32.8 | 12.2 | 19,081,569 | 12.3 |
| VecPPO/s4 | 32.8 | 12.4 | 23,020,978 | 12.6 |
| VecPPO/s5 | 32.9 | 12.6 | 48,011,116 | 12.6 |
| VecPPO/s6 | 32.9 | 12.1 | 50,064,168 | 12.1 |
| VecPPO/s7 | 32.5 | 13.0 | 27,010,786 | 13.1 |
| VecPPO/s8 | 32.9 | 12.2 | 50,047,907 | 12.2 |
| VecPPO/s9 | 32.9 | 12.2 | 43,003,667 | 12.4 |

![validation curves](figures/validation_curves.png)
