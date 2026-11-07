# Results: `e1t_D`

- Test episodes per strategy: 2000 (seeds from 1,500,000, antithetic pairs)
- Trained runs: 10; code version(s): 20f2f4ac66
- `J_bps` is the objective the agents are rewarded on: implementation shortfall plus the Almgren-Chriss penalty on carried variance (lower is better). `eff_steps` is how many steps the order is effectively spread over (20 = TWAP, 1 = one block trade). `IS_p95` and `IS_cvar95` are the 95th percentile of implementation shortfall and the mean of its worst 5% (bps).
- `L1_vs_AC` is the L1 distance between the strategy's mean schedule (fraction of the order traded at each step) and Almgren-Chriss's: 0 = the same shape, 2 = disjoint. For multi-seed agents it is the median across seeds.
- `vs_AlmgrenChriss` is the mean paired difference in `objective_bps` (negative = better than AlmgrenChriss). For multi-seed agents the 95% CI and p-value are across training seeds; `p_holm` is Holm-adjusted over every row of every table.
- `n_degenerate` counts trained seeds whose policy trades in fewer than 5 effective steps -- a block trade, not an execution schedule. They are reported, never dropped.

## abm_ac_optimal

| strategy | n_seeds | n_degenerate | J_bps | IS_bps | IS_std | IS_p95 | IS_cvar95 | eff_steps | L1_vs_AC | vs_AlmgrenChriss | ci_low | ci_high | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AlmgrenChriss | — | — | 28.87 | 21.5 | 70.3 | 139.8 | 169.7 | 14.7 | 0.000 | 0.00 | — | — | — |
| SignalRule | — | — | 28.87 | 21.5 | 70.3 | 139.8 | 169.7 | 14.7 | 0.000 | 0.00 | 0.00 | 0.00 | 1.0 |
| VecPPO | 10 | 0 | 28.89 | 21.3 | 71.6 | 141.9 | 172.8 | 15.1 | 0.037 | 0.02 | -0.03 | 0.07 | 1.0 |
| VWAP | — | — | 32.53 | 19.1 | 95.4 | 178.2 | 219.8 | 19.6 | 0.479 | 3.66 | 3.66 | 3.66 | 0.0 |
| TWAP | — | — | 32.83 | 19.0 | 97.0 | 181.3 | 223.3 | 20.0 | 0.503 | 3.96 | 3.96 | 3.96 | 0.0 |
| DumpAtStart | — | — | 154.38 | 154.4 | 0.7 | 155.7 | 155.9 | 1.0 | 1.759 | 125.51 | 125.50 | 125.52 | 0.0 |
| WaitThenDump | — | — | 196.94 | 154.4 | 173.0 | 437.0 | 511.7 | 1.0 | 1.959 | 168.07 | 168.06 | 168.08 | 0.0 |

SignalRule coefficient fitted on validation seeds: beta = 0.0.

![abm_ac_optimal schedule](figures/abm_ac_optimal_schedule.png)
![abm_ac_optimal risk_vs_cost](figures/abm_ac_optimal_risk_vs_cost.png)
![abm_ac_optimal is_distribution](figures/abm_ac_optimal_is_distribution.png)

## Checkpoint selection

Mean objective (bps) on the validation seeds before training, at the checkpoint each run kept, and at the end of training. The test seeds above were never used for selection.

| run | J_untrained | J_selected | selected_step | J_final |
|---|---|---|---|---|
| VecPPO/s0 | 32.9 | 28.9 | 160,198,321 | 28.9 |
| VecPPO/s1 | 32.9 | 28.9 | 64,190,162 | 28.9 |
| VecPPO/s2 | 32.8 | 28.8 | 96,301,676 | 28.9 |
| VecPPO/s3 | 32.8 | 29.0 | 56,325,093 | 29.0 |
| VecPPO/s4 | 32.9 | 28.9 | 144,145,646 | 29.0 |
| VecPPO/s5 | 32.9 | 28.9 | 84,176,730 | 28.9 |
| VecPPO/s6 | 32.9 | 28.9 | 180,189,324 | 28.9 |
| VecPPO/s7 | 32.5 | 28.9 | 160,196,684 | 28.9 |
| VecPPO/s8 | 32.9 | 28.9 | 200,174,953 | 28.9 |
| VecPPO/s9 | 32.9 | 28.9 | 116,292,208 | 28.9 |

![validation curves](figures/validation_curves.png)
