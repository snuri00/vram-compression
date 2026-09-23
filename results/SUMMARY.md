## Qwen/Qwen2.5-1.5B

| Method | bits/weight | code entropy | Wikitext-2 PPL |
|---|---|---|---|
| BF16 baseline | 16 | n/a | 9.269 |
| RTN g128 (fixed rate) | 4.250 | 3.608 | 10.360 |
| RTN g128 (fixed rate) | 3.250 | 2.571 | 22.201 |
| RTN g128 (fixed rate) | 2.250 | 1.696 | 1056074.696 |
| GPTQ g128 (fixed rate) | 4.250 | 3.609 | 9.777 |
| GPTQ g128 (fixed rate) | 3.250 | 2.572 | 12.647 |
| GPTQ g128 (fixed rate) | 2.250 | 1.699 | 1155.480 |
| ECSQ-GPTQ (entropy coded) | 4.250 | 4.250 | 9.432 |
| ECSQ-GPTQ (entropy coded) | 3.250 | 3.250 | 10.148 |
| ECSQ-GPTQ (entropy coded) | 2.250 | 2.250 | 17.400 |
| ECSQ-GPTQ + per-column waterfilling | 4.250 | 4.245 | 9.396 |
| ECSQ-GPTQ + per-column waterfilling | 3.250 | 3.245 | 9.810 |
| ECSQ-GPTQ + per-column waterfilling | 2.250 | 2.246 | 12.544 |

### Per-layer-group statistics (mean over layers)

| group | gain_isotropic_to_limit | gain_gptq_to_wf | gain_gptq_to_wf_rot | kurtosis | kurtosis_rot | diag_max_over_median |
|---|---|---|---|---|---|---|
| qkv | 1.061 | 0.394 | 0.059 | 0.592 | -0.008 | 509.527 |
| o | 1.187 | 0.308 | 0.105 | 0.779 | -0.007 | 118.350 |
| gate_up | 0.765 | 0.097 | 0.027 | 0.395 | -0.008 | 412.143 |
| down | 0.898 | 0.271 | 0.040 | 1.119 | -0.002 | 72750.668 |

Parameter-weighted mean: gain_isotropic_to_limit = 0.845 bit, gain_gptq_to_wf = 0.179 bit, gain_gptq_to_wf_rot = 0.037 bit
