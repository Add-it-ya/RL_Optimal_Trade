# Results: `e3`

- Test episodes per strategy: 5000 (seeds from 2,000,000, antithetic pairs)
- Trained runs: 10; code version(s): f8efbd7fda
- `J_bps` is the objective the agents are rewarded on: implementation shortfall plus the Almgren-Chriss penalty on carried variance (lower is better). `eff_steps` is how many steps the order is effectively spread over (20 = TWAP, 1 = one block trade). `IS_p95` and `IS_cvar95` are the 95th percentile of implementation shortfall and the mean of its worst 5% (bps).
- `L1_vs_AC` is the L1 distance between the strategy's mean schedule (fraction of the order traded at each step) and Almgren-Chriss's: 0 = the same shape, 2 = disjoint. For multi-seed agents it is the median across seeds.
- `vs_AlmgrenChriss` is the mean paired difference in `objective_bps` (negative = better than AlmgrenChriss). For multi-seed agents the 95% CI and p-value are across training seeds; `p_holm` is Holm-adjusted over every row of every table.
- `n_degenerate` counts trained seeds whose policy trades in fewer than 5 effective steps -- a block trade, not an execution schedule. They are reported, never dropped.

## In distribution

### normal_liquidity

| strategy | n_seeds | n_degenerate | J_bps | IS_bps | IS_std | IS_p95 | IS_cvar95 | eff_steps | L1_vs_AC | vs_AlmgrenChriss | ci_low | ci_high | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SignalRule | — | — | 27.47 | 19.7 | 75.7 | 138.6 | 169.4 | 14.3 | 0.030 | -1.84 | -2.39 | -1.29 | 8.15e-10 |
| PPO | 10 | 0 | 28.93 | 19.3 | 82.9 | 165.7 | 213.7 | 14.1 | 0.149 | -0.39 | -0.64 | -0.12 | 0.1 |
| AlmgrenChriss | — | — | 29.31 | 22.0 | 72.8 | 141.7 | 171.1 | 14.7 | 0.000 | 0.00 | — | — | — |
| VWAP | — | — | 32.82 | 19.4 | 98.5 | 182.1 | 220.5 | 19.6 | 0.479 | 3.51 | 3.51 | 3.51 | 0.0 |
| TWAP | — | — | 33.12 | 19.3 | 100.0 | 184.7 | 223.3 | 20.0 | 0.503 | 3.81 | 3.80 | 3.81 | 0.0 |
| DumpAtStart | — | — | 157.62 | 157.6 | 1.1 | 159.4 | 159.5 | 1.0 | 1.759 | 128.31 | 128.30 | 128.31 | 0.0 |
| WaitThenDump | — | — | 200.33 | 157.8 | 177.8 | 450.5 | 524.2 | 1.0 | 1.959 | 171.02 | 171.01 | 171.03 | 0.0 |

SignalRule coefficient fitted on validation seeds: beta = 1.0.

![normal_liquidity schedule](figures/normal_liquidity_schedule.png)
![normal_liquidity risk_vs_cost](figures/normal_liquidity_risk_vs_cost.png)
![normal_liquidity is_distribution](figures/normal_liquidity_is_distribution.png)

### medium_vol

| strategy | n_seeds | n_degenerate | J_bps | IS_bps | IS_std | IS_p95 | IS_cvar95 | eff_steps | L1_vs_AC | vs_AlmgrenChriss | ci_low | ci_high | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SignalRule | — | — | 28.06 | 19.6 | 78.6 | 144.5 | 176.3 | 14.3 | 0.030 | -1.79 | -2.35 | -1.23 | 5.86e-09 |
| PPO | 10 | 0 | 29.68 | 19.3 | 85.9 | 171.5 | 221.6 | 14.0 | 0.148 | -0.17 | -0.43 | 0.11 | 0.9 |
| AlmgrenChriss | — | — | 29.85 | 21.9 | 75.6 | 144.9 | 177.3 | 14.7 | 0.000 | 0.00 | — | — | — |
| VWAP | — | — | 33.72 | 19.2 | 102.2 | 187.4 | 228.2 | 19.6 | 0.479 | 3.87 | 3.62 | 4.12 | 3.99e-165 |
| TWAP | — | — | 34.08 | 19.1 | 103.8 | 189.4 | 231.5 | 20.0 | 0.503 | 4.23 | 3.97 | 4.48 | 1.22e-188 |
| DumpAtStart | — | — | 157.62 | 157.6 | 1.1 | 159.4 | 159.5 | 1.0 | 1.759 | 127.77 | 127.21 | 128.33 | 0.0 |
| WaitThenDump | — | — | 203.23 | 157.1 | 184.6 | 460.4 | 536.3 | 1.0 | 1.959 | 173.39 | 172.36 | 174.42 | 0.0 |

SignalRule coefficient fitted on validation seeds: beta = 1.0.

![medium_vol schedule](figures/medium_vol_schedule.png)
![medium_vol risk_vs_cost](figures/medium_vol_risk_vs_cost.png)
![medium_vol is_distribution](figures/medium_vol_is_distribution.png)

### sideways

| strategy | n_seeds | n_degenerate | J_bps | IS_bps | IS_std | IS_p95 | IS_cvar95 | eff_steps | L1_vs_AC | vs_AlmgrenChriss | ci_low | ci_high | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PPO | 10 | 0 | 24.31 | 14.9 | 47.6 | 93.8 | 115.7 | 14.3 | 0.145 | -5.00 | -5.57 | -4.34 | 1.75e-06 |
| SignalRule | — | — | 29.17 | 21.4 | 45.4 | 94.2 | 112.0 | 14.3 | 0.030 | -0.14 | -0.47 | 0.17 | 0.9 |
| AlmgrenChriss | — | — | 29.31 | 22.0 | 44.2 | 94.0 | 112.5 | 14.7 | 0.000 | 0.00 | — | — | — |
| VWAP | — | — | 32.82 | 19.4 | 52.7 | 106.7 | 127.4 | 19.6 | 0.479 | 3.51 | 3.51 | 3.51 | 0.0 |
| TWAP | — | — | 33.12 | 19.3 | 54.4 | 108.4 | 130.3 | 20.0 | 0.503 | 3.81 | 3.80 | 3.81 | 0.0 |
| DumpAtStart | — | — | 157.62 | 157.6 | 1.1 | 159.4 | 159.5 | 1.0 | 1.759 | 128.31 | 128.30 | 128.31 | 0.0 |
| WaitThenDump | — | — | 200.33 | 157.8 | 91.6 | 309.9 | 347.1 | 1.0 | 1.959 | 171.02 | 171.01 | 171.03 | 0.0 |

SignalRule coefficient fitted on validation seeds: beta = 1.0.

![sideways schedule](figures/sideways_schedule.png)
![sideways risk_vs_cost](figures/sideways_risk_vs_cost.png)
![sideways is_distribution](figures/sideways_is_distribution.png)

## Held out

### low_vol (outside the training family on: volatility)

| strategy | n_seeds | n_degenerate | J_bps | IS_bps | IS_std | IS_p95 | IS_cvar95 | eff_steps | L1_vs_AC | vs_AlmgrenChriss | ci_low | ci_high | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SignalRule | — | — | 20.13 | 17.2 | 49.6 | 95.0 | 112.1 | 17.5 | 0.025 | -2.29 | -2.56 | -2.03 | 4.23e-59 |
| PPO | 10 | 0 | 21.72 | 19.3 | 44.8 | 97.0 | 121.0 | 14.3 | 0.164 | -0.70 | -1.03 | -0.41 | 0.0 |
| AlmgrenChriss | — | — | 22.42 | 19.6 | 48.4 | 99.2 | 118.2 | 19.2 | 0.000 | 0.00 | — | — | — |
| TWAP | — | — | 22.74 | 19.3 | 53.7 | 107.0 | 128.6 | 20.0 | 0.177 | 0.32 | 0.32 | 0.32 | 0.0 |
| VWAP | — | — | 22.78 | 19.4 | 52.9 | 105.8 | 127.2 | 19.6 | 0.182 | 0.36 | 0.36 | 0.36 | 0.0 |
| DumpAtStart | — | — | 157.62 | 157.6 | 1.1 | 159.4 | 159.5 | 1.0 | 1.856 | 135.20 | 135.19 | 135.20 | 0.0 |
| WaitThenDump | — | — | 168.41 | 157.8 | 96.1 | 316.1 | 354.5 | 1.0 | 1.922 | 145.99 | 145.99 | 146.00 | 0.0 |

SignalRule coefficient fitted on validation seeds: beta = 1.0.

![low_vol schedule](figures/low_vol_schedule.png)
![low_vol risk_vs_cost](figures/low_vol_risk_vs_cost.png)
![low_vol is_distribution](figures/low_vol_is_distribution.png)

### high_vol (outside the training family on: volatility, vol_of_vol)

| strategy | n_seeds | n_degenerate | J_bps | IS_bps | IS_std | IS_p95 | IS_cvar95 | eff_steps | L1_vs_AC | vs_AlmgrenChriss | ci_low | ci_high | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AlmgrenChriss | — | — | 52.82 | 30.5 | 124.5 | 233.2 | 302.1 | 8.0 | 0.000 | 0.00 | — | — | — |
| SignalRule | — | — | 52.82 | 30.5 | 124.5 | 233.2 | 302.1 | 8.0 | 0.000 | 0.00 | -0.00 | 0.00 | 0.9 |
| PPO | 10 | 0 | 81.61 | 19.8 | 205.8 | 396.2 | 531.9 | 13.6 | 0.605 | 28.78 | 25.35 | 32.04 | 1.15e-06 |
| VWAP | — | — | 106.43 | 18.1 | 246.9 | 422.3 | 539.5 | 19.6 | 0.916 | 53.61 | 50.85 | 56.29 | 1.18e-255 |
| TWAP | — | — | 109.33 | 18.1 | 250.9 | 430.0 | 548.9 | 20.0 | 0.960 | 56.51 | 53.72 | 59.21 | 3.06e-271 |
| DumpAtStart | — | — | 157.62 | 157.6 | 1.1 | 159.4 | 159.5 | 1.0 | 1.554 | 104.80 | 102.65 | 106.92 | 0.0 |
| WaitThenDump | — | — | 434.72 | 153.8 | 445.7 | 873.0 | 1086.0 | 1.0 | 1.993 | 381.89 | 374.88 | 388.71 | 0.0 |

SignalRule coefficient fitted on validation seeds: beta = 0.0.

![high_vol schedule](figures/high_vol_schedule.png)
![high_vol risk_vs_cost](figures/high_vol_risk_vs_cost.png)
![high_vol is_distribution](figures/high_vol_is_distribution.png)

### thin (outside the training family on: base_spread, base_depth, permanent_impact, temporary_impact, base_market_volume)

| strategy | n_seeds | n_degenerate | J_bps | IS_bps | IS_std | IS_p95 | IS_cvar95 | eff_steps | L1_vs_AC | vs_AlmgrenChriss | ci_low | ci_high | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SignalRule | — | — | 48.98 | 39.0 | 85.5 | 175.2 | 208.7 | 16.4 | 0.026 | -0.97 | -1.51 | -0.44 | 0.00402 |
| AlmgrenChriss | — | — | 49.95 | 40.4 | 83.0 | 176.5 | 210.1 | 17.6 | 0.000 | 0.00 | — | — | — |
| VWAP | — | — | 52.11 | 38.7 | 98.5 | 201.5 | 239.8 | 19.6 | 0.300 | 2.16 | 2.16 | 2.16 | 0.0 |
| TWAP | — | — | 52.26 | 38.4 | 100.0 | 203.8 | 242.5 | 20.0 | 0.311 | 2.31 | 2.31 | 2.31 | 0.0 |
| PPO | 10 | 0 | 52.67 | 43.1 | 82.8 | 188.8 | 238.0 | 13.8 | 0.098 | 2.72 | 1.70 | 4.07 | 0.0 |
| DumpAtStart | — | — | 311.63 | 311.6 | 0.6 | 312.4 | 312.5 | 1.0 | 1.819 | 261.68 | 261.68 | 261.69 | 0.0 |
| WaitThenDump | — | — | 354.31 | 311.8 | 177.7 | 604.9 | 678.1 | 1.0 | 1.938 | 304.37 | 304.36 | 304.37 | 0.0 |

SignalRule coefficient fitted on validation seeds: beta = 1.0.

![thin schedule](figures/thin_schedule.png)
![thin risk_vs_cost](figures/thin_risk_vs_cost.png)
![thin is_distribution](figures/thin_is_distribution.png)

### deep (outside the training family on: base_depth, permanent_impact, temporary_impact, base_market_volume)

| strategy | n_seeds | n_degenerate | J_bps | IS_bps | IS_std | IS_p95 | IS_cvar95 | eff_steps | L1_vs_AC | vs_AlmgrenChriss | ci_low | ci_high | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SignalRule | — | — | 16.33 | 8.8 | 74.3 | 118.1 | 157.3 | 9.3 | 0.146 | -2.20 | -3.38 | -1.00 | 0.00342 |
| AlmgrenChriss | — | — | 18.53 | 13.4 | 61.1 | 114.8 | 138.4 | 11.1 | 0.000 | 0.00 | — | — | — |
| PPO | 10 | 0 | 19.17 | 8.7 | 87.1 | 160.4 | 206.8 | 15.7 | 0.482 | 0.64 | -0.08 | 1.44 | 0.6 |
| VWAP | — | — | 23.96 | 10.6 | 98.4 | 173.2 | 211.6 | 19.6 | 0.697 | 5.43 | 5.43 | 5.43 | 0.0 |
| TWAP | — | — | 24.33 | 10.5 | 100.0 | 175.9 | 214.5 | 20.0 | 0.732 | 5.80 | 5.80 | 5.80 | 0.0 |
| DumpAtStart | — | — | 79.39 | 79.4 | 0.7 | 80.7 | 80.9 | 1.0 | 1.672 | 60.86 | 60.85 | 60.86 | 0.0 |
| WaitThenDump | — | — | 121.95 | 79.4 | 177.7 | 372.4 | 445.7 | 1.0 | 1.980 | 103.42 | 103.41 | 103.42 | 0.0 |

SignalRule coefficient fitted on validation seeds: beta = 2.0.

![deep schedule](figures/deep_schedule.png)
![deep risk_vs_cost](figures/deep_risk_vs_cost.png)
![deep is_distribution](figures/deep_is_distribution.png)

### bull (outside the training family on: drift)

| strategy | n_seeds | n_degenerate | J_bps | IS_bps | IS_std | IS_p95 | IS_cvar95 | eff_steps | L1_vs_AC | vs_AlmgrenChriss | ci_low | ci_high | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SignalRule | — | — | 17.17 | 9.4 | 76.2 | 128.2 | 159.1 | 14.3 | 0.030 | -2.15 | -2.70 | -1.60 | 3.33e-13 |
| PPO | 10 | 0 | 17.62 | 8.2 | 80.6 | 148.9 | 195.7 | 14.0 | 0.132 | -1.70 | -2.07 | -1.40 | 0.000131 |
| VWAP | — | — | 17.62 | 4.2 | 98.5 | 166.9 | 205.3 | 19.6 | 0.479 | -1.69 | -1.69 | -1.69 | 0.0 |
| TWAP | — | — | 17.92 | 4.1 | 100.0 | 169.5 | 208.1 | 20.0 | 0.503 | -1.39 | -1.39 | -1.39 | 0.0 |
| AlmgrenChriss | — | — | 19.31 | 12.0 | 72.8 | 131.7 | 161.1 | 14.7 | 0.000 | 0.00 | — | — | — |
| DumpAtStart | — | — | 157.62 | 157.6 | 1.1 | 159.4 | 159.5 | 1.0 | 1.759 | 138.31 | 138.30 | 138.31 | 0.0 |
| WaitThenDump | — | — | 169.94 | 127.4 | 177.8 | 420.1 | 493.8 | 1.0 | 1.959 | 150.62 | 150.62 | 150.63 | 0.0 |

SignalRule coefficient fitted on validation seeds: beta = 1.0.

![bull schedule](figures/bull_schedule.png)
![bull risk_vs_cost](figures/bull_risk_vs_cost.png)
![bull is_distribution](figures/bull_is_distribution.png)

### bear (outside the training family on: drift)

| strategy | n_seeds | n_degenerate | J_bps | IS_bps | IS_std | IS_p95 | IS_cvar95 | eff_steps | L1_vs_AC | vs_AlmgrenChriss | ci_low | ci_high | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SignalRule | — | — | 37.78 | 30.0 | 75.2 | 149.8 | 179.9 | 14.3 | 0.030 | -1.53 | -2.08 | -0.98 | 5.85e-07 |
| AlmgrenChriss | — | — | 39.31 | 32.0 | 72.8 | 151.7 | 181.1 | 14.7 | 0.000 | 0.00 | — | — | — |
| PPO | 10 | 0 | 40.72 | 30.9 | 85.4 | 183.2 | 232.3 | 14.2 | 0.169 | 1.41 | 0.80 | 2.00 | 0.0 |
| VWAP | — | — | 48.02 | 34.6 | 98.5 | 197.3 | 235.7 | 19.6 | 0.479 | 8.71 | 8.71 | 8.71 | 0.0 |
| TWAP | — | — | 48.32 | 34.5 | 100.0 | 199.9 | 238.5 | 20.0 | 0.503 | 9.00 | 9.00 | 9.01 | 0.0 |
| DumpAtStart | — | — | 157.62 | 157.6 | 1.1 | 159.4 | 159.5 | 1.0 | 1.759 | 118.31 | 118.30 | 118.31 | 0.0 |
| WaitThenDump | — | — | 230.73 | 188.2 | 177.8 | 480.9 | 554.6 | 1.0 | 1.959 | 191.42 | 191.41 | 191.42 | 0.0 |

SignalRule coefficient fitted on validation seeds: beta = 1.0.

![bear schedule](figures/bear_schedule.png)
![bear risk_vs_cost](figures/bear_risk_vs_cost.png)
![bear is_distribution](figures/bear_is_distribution.png)

### gbm_medium (outside the training family on: price_process)

| strategy | n_seeds | n_degenerate | J_bps | IS_bps | IS_std | IS_p95 | IS_cvar95 | eff_steps | L1_vs_AC | vs_AlmgrenChriss | ci_low | ci_high | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SignalRule | — | — | 27.46 | 19.7 | 75.7 | 138.1 | 168.3 | 14.3 | 0.030 | -1.83 | -2.38 | -1.29 | 8.48e-10 |
| PPO | 10 | 0 | 28.87 | 19.3 | 82.8 | 164.9 | 211.8 | 14.1 | 0.150 | -0.42 | -0.67 | -0.16 | 0.1 |
| AlmgrenChriss | — | — | 29.29 | 22.0 | 72.8 | 141.1 | 169.9 | 14.7 | 0.000 | 0.00 | — | — | — |
| VWAP | — | — | 32.79 | 19.4 | 98.5 | 180.9 | 218.4 | 19.6 | 0.479 | 3.50 | 3.48 | 3.51 | 0.0 |
| TWAP | — | — | 33.09 | 19.3 | 100.0 | 183.4 | 221.3 | 20.0 | 0.503 | 3.80 | 3.78 | 3.81 | 0.0 |
| DumpAtStart | — | — | 157.62 | 157.6 | 1.1 | 159.4 | 159.5 | 1.0 | 1.759 | 128.33 | 128.30 | 128.35 | 0.0 |
| WaitThenDump | — | — | 200.28 | 157.7 | 177.8 | 447.8 | 518.9 | 1.0 | 1.959 | 170.99 | 170.92 | 171.06 | 0.0 |

SignalRule coefficient fitted on validation seeds: beta = 1.0.

![gbm_medium schedule](figures/gbm_medium_schedule.png)
![gbm_medium risk_vs_cost](figures/gbm_medium_risk_vs_cost.png)
![gbm_medium is_distribution](figures/gbm_medium_is_distribution.png)

### clustered_vol (outside the training family on: vol_of_vol, price_process, vol_process, vol_persistence)

| strategy | n_seeds | n_degenerate | J_bps | IS_bps | IS_std | IS_p95 | IS_cvar95 | eff_steps | L1_vs_AC | vs_AlmgrenChriss | ci_low | ci_high | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SignalRule | — | — | 31.95 | 19.0 | 94.6 | 169.6 | 228.3 | 14.3 | 0.030 | -1.35 | -2.00 | -0.70 | 0.000405 |
| AlmgrenChriss | — | — | 33.30 | 21.4 | 90.8 | 167.9 | 225.9 | 14.7 | 0.000 | 0.00 | — | — | — |
| PPO | 10 | 0 | 34.08 | 19.0 | 102.1 | 193.9 | 280.5 | 13.9 | 0.147 | 0.78 | -0.03 | 1.61 | 0.5 |
| VWAP | — | — | 40.39 | 18.4 | 123.4 | 216.1 | 292.8 | 19.6 | 0.479 | 7.09 | 6.37 | 7.79 | 1.32e-76 |
| TWAP | — | — | 41.01 | 18.3 | 125.3 | 219.9 | 297.4 | 20.0 | 0.503 | 7.71 | 6.99 | 8.41 | 4.53e-88 |
| DumpAtStart | — | — | 157.62 | 157.6 | 1.1 | 159.4 | 159.5 | 1.0 | 1.759 | 124.32 | 122.85 | 125.84 | 0.0 |
| WaitThenDump | — | — | 225.58 | 155.1 | 225.5 | 514.8 | 642.2 | 1.0 | 1.959 | 192.29 | 189.41 | 195.26 | 0.0 |

SignalRule coefficient fitted on validation seeds: beta = 1.0.

![clustered_vol schedule](figures/clustered_vol_schedule.png)
![clustered_vol risk_vs_cost](figures/clustered_vol_risk_vs_cost.png)
![clustered_vol is_distribution](figures/clustered_vol_is_distribution.png)

### leverage (outside the training family on: vol_of_vol, price_process, vol_process, vol_persistence, leverage_corr)

| strategy | n_seeds | n_degenerate | J_bps | IS_bps | IS_std | IS_p95 | IS_cvar95 | eff_steps | L1_vs_AC | vs_AlmgrenChriss | ci_low | ci_high | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SignalRule | — | — | 31.25 | 19.3 | 92.4 | 173.8 | 243.6 | 14.6 | 0.007 | -1.06 | -1.39 | -0.75 | 1.19e-09 |
| AlmgrenChriss | — | — | 32.32 | 20.6 | 91.3 | 172.8 | 244.1 | 14.7 | 0.000 | 0.00 | — | — | — |
| PPO | 10 | 0 | 33.96 | 18.1 | 105.4 | 204.8 | 307.9 | 13.9 | 0.138 | 1.64 | 0.82 | 2.45 | 0.0 |
| VWAP | — | — | 38.94 | 17.4 | 123.4 | 228.5 | 322.1 | 19.6 | 0.479 | 6.62 | 5.89 | 7.33 | 1.69e-63 |
| TWAP | — | — | 39.51 | 17.3 | 125.4 | 230.7 | 326.7 | 20.0 | 0.503 | 7.19 | 6.45 | 7.91 | 2.58e-72 |
| DumpAtStart | — | — | 157.62 | 157.6 | 1.1 | 159.4 | 159.5 | 1.0 | 1.759 | 125.30 | 123.78 | 126.86 | 0.0 |
| WaitThenDump | — | — | 222.71 | 154.2 | 223.1 | 535.9 | 713.0 | 1.0 | 1.959 | 190.40 | 187.23 | 193.54 | 0.0 |

SignalRule coefficient fitted on validation seeds: beta = 0.5.

![leverage schedule](figures/leverage_schedule.png)
![leverage risk_vs_cost](figures/leverage_risk_vs_cost.png)
![leverage is_distribution](figures/leverage_is_distribution.png)

## Robustness: in distribution vs held out

Mean paired gap to AlmgrenChriss in `objective_bps`, one weight per regime, over the in-distribution and the held-out regimes above; `degradation` is held out minus in distribution (positive = worse out of distribution). For multi-seed agents the 95% CI is across training seeds.

| strategy | n_seeds | vs_AlmgrenChriss_in | vs_AlmgrenChriss_held_out | degradation | ci_low | ci_high |
|---|---|---|---|---|---|---|
| SignalRule | — | -1.3 | -1.5 | -0.23 | — | — |
| DumpAtStart | — | 128.1 | 133.0 | 4.88 | — | — |
| PPO | 10 | -1.9 | 3.7 | 5.53 | 5.0 | 6.0 |
| VWAP | — | 3.6 | 9.5 | 5.90 | — | — |
| TWAP | — | 3.9 | 10.1 | 6.19 | — | — |
| WaitThenDump | — | 171.8 | 203.5 | 31.68 | — | — |

## Checkpoint selection

Mean objective (bps) on the validation seeds before training, at the checkpoint each run kept, and at the end of training. The test seeds above were never used for selection.

| run | J_untrained | J_selected | selected_step | J_final |
|---|---|---|---|---|
| PPO/s0 | 30.4 | 23.8 | 2,500,000 | 24.3 |
| PPO/s1 | 30.4 | 24.1 | 4,300,000 | 24.3 |
| PPO/s2 | 30.4 | 23.9 | 2,000,000 | 24.1 |
| PPO/s3 | 30.3 | 23.7 | 2,600,000 | 24.0 |
| PPO/s4 | 30.4 | 23.9 | 1,400,000 | 24.4 |
| PPO/s5 | 30.5 | 23.8 | 4,900,000 | 23.8 |
| PPO/s6 | 30.4 | 23.9 | 1,600,000 | 23.9 |
| PPO/s7 | 30.5 | 23.9 | 1,900,000 | 24.3 |
| PPO/s8 | 30.3 | 24.3 | 500,000 | 25.7 |
| PPO/s9 | 30.5 | 24.0 | 1,300,000 | 24.5 |

![validation curves](figures/validation_curves.png)
