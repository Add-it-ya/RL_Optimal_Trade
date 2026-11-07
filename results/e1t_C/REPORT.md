# Results: `e1t_C`

- Test episodes per strategy: 5000 (seeds from 2,000,000, antithetic pairs)
- Trained runs: 10; code version(s): 20f2f4ac66
- `J_bps` is the objective the agents are rewarded on: implementation shortfall plus the Almgren-Chriss penalty on carried variance (lower is better). `eff_steps` is how many steps the order is effectively spread over (20 = TWAP, 1 = one block trade). `IS_p95` and `IS_cvar95` are the 95th percentile of implementation shortfall and the mean of its worst 5% (bps).
- `L1_vs_AC` is the L1 distance between the strategy's mean schedule (fraction of the order traded at each step) and Almgren-Chriss's: 0 = the same shape, 2 = disjoint. For multi-seed agents it is the median across seeds.
- `vs_AlmgrenChriss` is the mean paired difference in `objective_bps` (negative = better than AlmgrenChriss). For multi-seed agents the 95% CI and p-value are across training seeds; `p_holm` is Holm-adjusted over every row of every table.
- `n_degenerate` counts trained seeds whose policy trades in fewer than 5 effective steps -- a block trade, not an execution schedule. They are reported, never dropped.

## abm_ac_optimal

| strategy | n_seeds | n_degenerate | J_bps | IS_bps | IS_std | IS_p95 | IS_cvar95 | eff_steps | L1_vs_AC | vs_AlmgrenChriss | ci_low | ci_high | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SignalRule | — | — | 28.87 | 21.5 | 71.6 | 138.4 | 168.4 | 14.7 | 0.000 | 0.00 | 0.00 | 0.00 | 1.0 |
| AlmgrenChriss | — | — | 28.87 | 21.5 | 71.6 | 138.4 | 168.4 | 14.7 | 0.000 | 0.00 | — | — | — |
| VecPPO | 10 | 0 | 28.96 | 21.4 | 72.5 | 141.0 | 173.2 | 15.0 | 0.039 | 0.09 | 0.05 | 0.12 | 0.0012 |
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
| VecPPO/s0 | 32.9 | 28.9 | 26,038,955 | 29.0 |
| VecPPO/s1 | 32.9 | 28.9 | 26,040,270 | 28.9 |
| VecPPO/s2 | 32.8 | 28.9 | 32,020,093 | 29.0 |
| VecPPO/s3 | 32.8 | 28.9 | 29,070,165 | 28.9 |
| VecPPO/s4 | 32.9 | 28.9 | 49,060,728 | 28.9 |
| VecPPO/s5 | 32.9 | 28.8 | 45,046,116 | 28.8 |
| VecPPO/s6 | 32.9 | 28.9 | 27,023,908 | 28.9 |
| VecPPO/s7 | 32.5 | 28.9 | 50,042,740 | 28.9 |
| VecPPO/s8 | 32.9 | 29.0 | 39,065,625 | 29.1 |
| VecPPO/s9 | 32.9 | 28.9 | 29,072,846 | 29.0 |

![validation curves](figures/validation_curves.png)
