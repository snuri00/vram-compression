# vram-compression

Storing LLM weights in less GPU memory by treating quantization as a compression problem: quantize so the codes have low entropy, then entropy code them, and decode inside the matmul.

## Why

Recompressing an existing GGUF with zstd saves about 3%, because fixed-rate quantizers already use their codes near maximum entropy. The gain has to come earlier. Entropy-constrained scalar quantization (ECSQ) is within 0.254 bit of the rate-distortion bound, while fixed-rate scalar quantization loses about 0.47 bit on Gaussian and 0.94 bit on Laplacian sources.

## Results

Qwen2.5-1.5B, Wikitext-2 perplexity, context 2048. BF16 baseline: 9.269.

| bits/weight | RTN g128 | GPTQ g128 | ECSQ-GPTQ | ECSQ-GPTQ + WF |
|---|---|---|---|---|
| 4.25 | 10.36 | 9.78 | 9.43 | **9.40** |
| 3.25 | 22.20 | 12.65 | 10.15 | **9.81** |
| 2.25 | 1.06e6 | 1155 | 17.40 | **12.54** |

ECSQ with per-column waterfilling matches fixed-rate GPTQ at about 1 bit/weight less. For a 27B model that is roughly 3.4 GB.

Caveats: one small model, one metric, quality measured in simulation.

Per-layer statistics (`results/SUMMARY.md`): waterfilling over GPTQ is worth 0.18 bit in the native basis but only 0.04 bit after a random rotation, so rotation and waterfilling are substitutes.

## Methods

**Fixed rate (RTN, GPTQ):** b bits per code plus fp16 scale and zero per 128 weights, so b + 0.25 bits/weight.

**ECSQ-GPTQ:** one uniform step per tensor, unbounded levels, GPTQ error feedback. Rate is the empirical entropy of the integer codes; a static-table rANS coder reaches it within about 0.01 bit.

**ECSQ-GPTQ + WF:** step per input column proportional to the diagonal of the upper Cholesky factor of H^-1. This equalizes each column's contribution to the output error (reverse waterfilling, as in WaterSIC).

## Kernel

`kernel/rans_gemv.cu` is a fused rANS decode plus batch-1 GEMV. Weights stay entropy coded in VRAM.

A warp owns an R x C tile. Lane l decodes one independent stream holding columns l, l+32, ... for all R rows, so each activation is loaded once and reused R times. It uses a 32-bit state, 16-bit renormalization and 12-bit probabilities, with one 16 KB table per matrix in shared memory. The per-column step is folded into the activation, so dequantization is free.

v1 is bit-exact but 2 to 3x slower than bf16 cuBLAS on an RTX 3050 Laptop while storing about 5x fewer bytes. It is bound by decode latency (15 GB/s effective vs 168 GB/s). Next: interleaved multi-state rANS, wider word loads, escape codes, model integration.

## Layout

| Path | Content |
|---|---|
| `exp/` | layer-by-layer quantization, Hessians, perplexity, sweep |
| `kernel/` | CUDA kernel, numpy encoder, tests, benchmark |
| `results/` | raw JSON results and summary |
| `NOTES.md` | literature notes |

## Reproduce

```bash
pip install torch transformers datasets
exp/run_all.sh
python exp/summarize.py
python kernel/bench.py test
python kernel/bench.py bench
```
