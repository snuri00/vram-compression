// Entropy-coded weight GEMV (batch 1) for ECSQ-quantized LLM weights.
//
// Format ("tile-interleaved rANS"):
//   * W (rows x cols) holds small integer codes; W_real[r,c] = code[r,c] * step[c].
//     The per-column step is folded into the activation on the host side:
//     x'[c] = x[c] * step[c], so y = code @ x' and dequantization is free.
//   * The matrix is cut into tiles of R rows x C columns. One warp owns a tile.
//     Lane l owns columns l, l+32, l+64, ... of the tile, for all R rows, and
//     encodes them as ONE independent rANS stream in the order
//       for k in 0..C/32-1: for r in 0..R-1: code[r, c0 + l + 32k]
//     so each x' value is loaded once (coalesced across lanes) and reused R times.
//   * rANS: 32-bit state, 16-bit renormalization words, 12-bit probabilities,
//     one static table per matrix: table[slot] = sym | freq<<8 | (slot-cum)<<20.
//     At most one 16-bit word is read per decoded symbol.
//   * Per lane stream: a 32-bit offset (offs[]) + 32-bit initial state (first two words).

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define SCALE_BITS 12
#define PROB_M (1u << SCALE_BITS)
#define RANS_L (1u << 16)
#define SYM_OFFSET 128

__device__ __forceinline__ void load_table(uint32_t* tab, const uint32_t* table) {
    for (int i = threadIdx.x; i < PROB_M; i += blockDim.x) tab[i] = table[i];
    __syncthreads();
}

// Decode one symbol, return signed code.
__device__ __forceinline__ int rans_step(uint32_t& st, const uint16_t*& p, const uint32_t* tab) {
    uint32_t e = tab[st & (PROB_M - 1)];
    st = ((e >> 8) & 0xFFFu) * (st >> SCALE_BITS) + (e >> 20);
    if (st < RANS_L) st = (st << 16) | *p++;
    return (int)(e & 0xFFu) - SYM_OFFSET;
}

template <int R>
__global__ void rans_gemv_kernel(const uint16_t* __restrict__ words,
                                 const uint32_t* __restrict__ offs,
                                 const uint32_t* __restrict__ table,
                                 const float* __restrict__ x,
                                 float* __restrict__ y,
                                 int rows, int cols, int C) {
    __shared__ uint32_t tab[PROB_M];
    load_table(tab, table);
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int ncb = cols / C;
    if (warp >= (rows / R) * ncb) return;
    const int rb = warp / ncb, cb = warp % ncb;

    const uint16_t* p = words + offs[warp * 32 + lane];
    uint32_t st = ((uint32_t)p[0] << 16) | p[1];
    p += 2;

    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.f;
    const float* xp = x + cb * C + lane;
    for (int k = 0; k < C / 32; k++) {
        const float xv = xp[k * 32];
#pragma unroll
        for (int r = 0; r < R; r++) acc[r] += (float)rans_step(st, p, tab) * xv;
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
        float v = acc[r];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) {
            if (ncb == 1) y[rb * R + r] = v;
            else atomicAdd(&y[rb * R + r], v);
        }
    }
}

// Decode-only: expand to a dense bf16 matrix (DFloat11-style "decompress, then cuBLAS").
template <int R>
__global__ void rans_decode_kernel(const uint16_t* __restrict__ words,
                                   const uint32_t* __restrict__ offs,
                                   const uint32_t* __restrict__ table,
                                   const float* __restrict__ step,
                                   __nv_bfloat16* __restrict__ out,
                                   int rows, int cols, int C) {
    __shared__ uint32_t tab[PROB_M];
    load_table(tab, table);
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int ncb = cols / C;
    if (warp >= (rows / R) * ncb) return;
    const int rb = warp / ncb, cb = warp % ncb;

    const uint16_t* p = words + offs[warp * 32 + lane];
    uint32_t st = ((uint32_t)p[0] << 16) | p[1];
    p += 2;
    for (int k = 0; k < C / 32; k++) {
        const int c = cb * C + lane + 32 * k;
        const float s = step[c];
#pragma unroll
        for (int r = 0; r < R; r++)
            out[(size_t)(rb * R + r) * cols + c] = __float2bfloat16((float)rans_step(st, p, tab) * s);
    }
}

// Baseline: fixed-rate 4-bit codes, 8 per uint32, row-major, code-8 is the value.
// Same folding of the per-column step into x', so it is an apples-to-apples
// "fixed rate vs entropy coded" comparison.
__global__ void int4_gemv_kernel(const uint32_t* __restrict__ w,
                                 const float* __restrict__ x,
                                 float* __restrict__ y, int rows, int cols) {
    const int lane = threadIdx.x & 31;
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= rows) return;
    const uint4* wr = reinterpret_cast<const uint4*>(w + (size_t)row * (cols / 8));
    float acc = 0.f;
    for (int j = lane; j < cols / 32; j += 32) {
        const uint4 q = wr[j];
        const uint32_t qs[4] = {q.x, q.y, q.z, q.w};
        const float* xp = x + j * 32;
#pragma unroll
        for (int a = 0; a < 4; a++)
#pragma unroll
            for (int b = 0; b < 8; b++)
                acc += (float)((int)((qs[a] >> (4 * b)) & 15u) - 8) * xp[a * 8 + b];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = acc;
}

// ------------------------------------------------------------------ host API
static int blocks_for(int warps, int threads) { return (warps * 32 + threads - 1) / threads; }

#define DISPATCH_R(R_, KERNEL, ...)                                             \
    switch (R_) {                                                               \
        case 1: KERNEL<1><<<grid, threads, 0, stream>>>(__VA_ARGS__); break;    \
        case 2: KERNEL<2><<<grid, threads, 0, stream>>>(__VA_ARGS__); break;    \
        case 4: KERNEL<4><<<grid, threads, 0, stream>>>(__VA_ARGS__); break;    \
        case 8: KERNEL<8><<<grid, threads, 0, stream>>>(__VA_ARGS__); break;    \
        case 16: KERNEL<16><<<grid, threads, 0, stream>>>(__VA_ARGS__); break;  \
        default: return -1;                                                     \
    }

extern "C" int launch_rans_gemv(int R, const void* words, const void* offs, const void* table,
                                const void* x, void* y, int rows, int cols, int C,
                                cudaStream_t stream) {
    const int threads = 128, grid = blocks_for((rows / R) * (cols / C), threads);
    DISPATCH_R(R, rans_gemv_kernel, (const uint16_t*)words, (const uint32_t*)offs,
               (const uint32_t*)table, (const float*)x, (float*)y, rows, cols, C)
    return (int)cudaGetLastError();
}

extern "C" int launch_rans_decode(int R, const void* words, const void* offs, const void* table,
                                  const void* step, void* out, int rows, int cols, int C,
                                  cudaStream_t stream) {
    const int threads = 128, grid = blocks_for((rows / R) * (cols / C), threads);
    DISPATCH_R(R, rans_decode_kernel, (const uint16_t*)words, (const uint32_t*)offs,
               (const uint32_t*)table, (const float*)step, (__nv_bfloat16*)out, rows, cols, C)
    return (int)cudaGetLastError();
}

extern "C" int launch_int4_gemv(const void* w, const void* x, void* y, int rows, int cols,
                                cudaStream_t stream) {
    const int threads = 128, grid = blocks_for(rows, threads);
    int4_gemv_kernel<<<grid, threads, 0, stream>>>((const uint32_t*)w, (const float*)x,
                                                   (float*)y, rows, cols);
    return (int)cudaGetLastError();
}
