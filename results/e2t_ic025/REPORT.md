# Results: `e2t_ic025`

- Test episodes per strategy: 5000 (seeds from 2,000,000, antithetic pairs)
- Trained runs: 10; code version(s): 20f2f4ac66
- `J_bps` is the objective the agents are rewarded on: implementation shortfall plus the Almgren-Chriss penalty on carried variance (lower is better). `eff_steps` is how many steps the order is effectively spread over (20 = TWAP, 1 = one block trade). `IS_p95` and `IS_cvar95` are the 95th percentile of implementation shortfall and the mean of its worst 5% (bps).
- `L1_vs_AC` is the L1 distance between the strategy's mean schedule (fraction of the order traded at each step) and Almgren-Chriss's: 0 = the same shape, 2 = disjoint. For multi-seed agents it is the median across seeds.
- `vs_AlmgrenChriss` is the mean paired difference in `objective_bps` (negative = better than AlmgrenChriss). For multi-seed agents the 95% CI and p-value are across training seeds; `p_holm` is Holm-adjusted over every row of every table.
- `n_degenerate` counts trained seeds whose policy trades in fewer than 5 effective steps -- a block trade, not an execution schedule. They are reported, never dropped.

## signal_ic025

| strategy | n_seeds | n_degenerate | J_bps | IS_bps | IS_std | IS_p95 | IS_cvar95 | eff_steps | L1_vs_AC | vs_AlmgrenChriss | ci_low | ci_high | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SignalRule | — | — | 28.80 | 21.4 | 72.2 | 138.3 | 169.1 | 14.6 | 0.007 | -0.07 | -0.34 | 0.20 | 0.6 |
| AlmgrenChriss | — | — | 28.87 | 21.5 | 71.6 | 137.8 | 168.2 | 14.7 | 0.000 | 0.00 | — | — | — |
| VecPPO | 10 | 0 | 28.95 | 21.0 | 74.5 | 143.6 | 175.0 | 15.1 | 0.062 | 0.08 | 0.02 | 0.14 | 0.1 |
| VWAP | — | — | 32.53 | 19.1 | 96.5 | 178.4 | 217.0 | 19.6 | 0.479 | 3.66 | 3.66 | 3.66 | 0.0 |
| TWAP | — | — | 32.83 | 19.0 | 98.1 | 179.9 | 219.6 | 20.0 | 0.503 | 3.96 | 3.96 | 3.96 | 0.0 |
| DumpAtStart | — | — | 154.38 | 154.4 | 0.7 | 155.7 | 155.9 | 1.0 | 1.759 | 125.51 | 125.51 | 125.52 | 0.0 |
| WaitThenDump | — | — | 196.94 | 154.4 | 173.8 | 445.6 | 512.8 | 1.0 | 1.959 | 168.07 | 168.07 | 168.08 | 0.0 |

SignalRule coefficient fitted on validation seeds: beta = 0.5.

![signal_ic025 schedule](figures/signal_ic025_schedule.png)
![signal_ic025 risk_vs_cost](figures/signal_ic025_risk_vs_cost.png)
![signal_ic025 is_distribution](figures/signal_ic025_is_distribution.png)

## Checkpoint selection

Mean objective (bps) on the validation seeds before training, at the checkpoint each run kept, and at the end of training. The test seeds above were never used for selection.

| run | J_untrained | J_selected | selected_step | J_final |
|---|---|---|---|---|
| VecPPO/s0 | 32.9 | 28.4 | 50,041,991 | 28.4 |
| VecPPO/s1 | 32.9 | 28.5 | 41,031,826 | 28.5 |
| VecPPO/s2 | 32.8 | 28.4 | 50,042,669 | 28.4 |
| VecPPO/s3 | 32.8 | 28.4 | 34,067,493 | 28.4 |
| VecPPO/s4 | 32.9 | 28.5 | 27,024,585 | 28.5 |
| VecPPO/s5 | 32.9 | 28.4 | 24,073,577 | 28.5 |
| VecPPO/s6 | 32.9 | 28.4 | 27,023,216 | 28.5 |
| VecPPO/s7 | 32.5 | 28.4 | 28,005,655 | 28.4 |
| VecPPO/s8 | 32.9 | 28.4 | 48,076,850 | 28.4 |
| VecPPO/s9 | 32.9 | 28.5 | 15,063,660 | 28.5 |

![validation curves](figures/validation_curves.png)
