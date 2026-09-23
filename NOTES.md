# Literature notes

Compressing LLM weights for less VRAM without pruning. Items marked [unverified] were not checked against the primary source.

## Key findings

1. Entropy coding only helps if the quantizer is designed for it. Fixed-rate codes sit near maximum entropy (GGUF + zstd gives about 3%). EntQuant (ICML 2026) reaches about 2.1 effective bits/param with entropy-constrained quantization.
2. Ordentlich and Polyanskiy give the exact distortion-rate function for quantized matrix multiplication, with a phase transition at R* ≈ 0.906 bit where the optimal scheme zeroes a fraction of coordinates.
3. Hessian-weighted distortion lowers the achievable rate by ½ log2(AM/GM) of the activation covariance eigenvalues. WaterSIC is within 0.255 bit of this limit.
4. llama.cpp Q3_K needs about 0.7 to 0.9 bpw more than EXL3/QTIP for the same KL, about 2.5 to 3 GB on a 27B model.
5. QTIP is about 0.07 bit from the Gaussian rate-distortion bound at 2 bits.
6. Weight-space symmetries are worth about 0.2% of bits for a 27B model. Their value is choosing a better representative (rotations, permutations), not saving bits directly.
7. Tensor decompositions lose to plain 4-bit rounding at matched size, and weight-space error can anticorrelate with perplexity. Objectives must live in activation space.

## Theory

| Result | Formula |
|---|---|
| Gaussian rate-distortion | D(R) = σ² 2^(-2R) |
| Shannon lower bound | R(D) ≥ h(W) - ½ log2(2πeD) |
| ECSQ gap at high rate | ½ log2(πe/6) ≈ 0.254 bit |
| Panter-Dite fixed rate | D ≈ (1/12)(∫p^(1/3))³ 2^(-2R) |
| Reverse waterfilling | D = (1/n) Σ min(λi, τ), R = (1/2n) Σ log2 max(1, λi/τ) |
| Matmul distortion-rate | Γ(R) = 2·2^(-2R) - 2^(-4R) for R ≥ R* |
| Weight-only limit | D* = σ² \|Σx\|^(1/n) 2^(-2R) |
| WaterSIC | (2πe/12) D* |

Lattice normalized second moments: scalar 0.0833, E8 0.0717, Leech 0.0658, sphere bound 0.0585.

GPTQ is Babai's nearest-plane algorithm on the Hessian lattice (arXiv 2507.18553). HIGGS linearity theorem: E[PPL] ≈ PPL* + Σ αl tl².

## Methods

| Method | Idea | Result |
|---|---|---|
| QTIP (2406.11235) | Hadamard incoherence, bitshift trellis, computed codes | Llama-2-70B 2 bit PPL 3.78 |
| QuIP# (2402.04396) | E8P lattice codebook | 0.26 bit from bound |
| WaterSIC (2603.04956) | per-column step, entropy coding | within 0.255 bit of limit |
| NestQuant (2502.09720) | nested E8 lattice | W4A4KV4 Llama-3-8B PPL 6.6 |
| EntQuant (2601.22787) | L1 entropy proxy, nvCOMP ANS | about 2.1 bits/param |
| CALDERA | W ≈ Q + LR, all quantized | Llama-2-70B 2.2 bpw PPL 3.98 |
| SeedLM (2410.10714) | LFSR seeds plus coefficients | no GPU kernel |
| YAQA (2505.22988) | end-to-end KL Hessian | 30% less KL than GPTQ |
| Basis Sharing (2410.03765) | shared cross-layer basis | helps QKV, gate, up only |

## GPU entropy decoding

| System | Rate | Note |
|---|---|---|
| DFloat11 | BF16 to about 11 bits | LUT Huffman, not fused |
| ZipServ (ASPLOS 2026) | about 11.3 bits | fixed-length codes fused into mma, 1.31x over cuBLAS on 4090 |
| ISCA 2026 tile rANS (2606.15789) | within 0.01 to 0.1 bit of entropy | decodes into shared memory |
| ECF8 (ICLR 2026) | FP8, 10 to 27% saved | lossless floor about FP4.67 |
| Unweight (Cloudflare) | 68% of MLP | 30 to 41% slower than cuBLAS on H100 |

## Open gaps

| Gap | Idea |
|---|---|
| A | ECSQ plus fused rANS GEMM (this repo) |
| B | fixed-rate waterfilling with trellis codes |
| C | shared cross-layer basis plus low-bit per-layer backbone |
| D | SeedLM with rotation, Hessian objective and trellis-chained seeds |
| E | rotation optimized for code entropy instead of kurtosis |
| F | the sub-0.906-bit regime (zero plus quantize) |
| G | ECSQ variant of llama.cpp K-quants with rANS dequant |

No proven theorem links bits/weight to loss for real LLMs; only per-layer Gaussian bounds and empirical scaling laws.
