# Tuning study on E1: choice of config

Rule (fixed in advance): no degenerate seed; lowest mean gap to AC on the tuning block (seeds from 1,500,000); a cheaper config within 0.05 bps of the best wins. Candidates listed cheapest first. These
numbers are on the tuning block, not the test block; the winner is scored on the test
block afterwards, once.

| label | n_seeds | n_degenerate | J_bps | gap_vs_AC | ci_low | ci_high | L1_vs_AC |
|---|---|---|---|---|---|---|---|
| A | 10 | 0 | 28.975 | 0.106 | -0.011 | 0.254 | 0.074 |
| C | 10 | 0 | 28.857 | -0.012 | -0.064 | 0.043 | 0.039 |
| D | 10 | 0 | 28.886 | 0.016 | -0.031 | 0.068 | 0.037 |
| B | 10 | 0 | 29.094 | 0.224 | 0.086 | 0.372 | 0.091 |

**Winner: C** (`scripts/lab/configs/C_n4096_50m.yaml`)
