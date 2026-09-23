"""Host side of the tile-interleaved rANS GEMV: table building, a vectorized
numpy encoder, a pure-numpy reference decoder, and ctypes wrappers around
librans.so (built from rans_gemv.cu with nvcc; see build())."""
import ctypes
import os
import subprocess

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "librans.so")
SCALE_BITS = 12
M = 1 << SCALE_BITS
RANS_L = 1 << 16
OFFSET = 128


def build(arch="sm_86"):
    src = os.path.join(HERE, "rans_gemv.cu")
    if not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(src):
        subprocess.check_call(["nvcc", "-O3", f"-arch={arch}", "-shared", "-Xcompiler", "-fPIC",
                               "-o", LIB, src])
    lib = ctypes.CDLL(LIB)
    for f in (lib.launch_rans_gemv, lib.launch_rans_decode):
        f.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        f.restype = ctypes.c_int
    lib.launch_int4_gemv.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
    lib.launch_int4_gemv.restype = ctypes.c_int
    return lib


# --------------------------------------------------------------------- table
def build_table(sym):
    """sym: uint8 array of symbols. Returns (freq, cum, table) with 12-bit probs."""
    counts = np.bincount(sym.ravel(), minlength=256).astype(np.float64)
    present = counts > 0
    freq = np.zeros(256, np.int64)
    freq[present] = np.maximum(1, np.round(counts[present] * M / counts.sum())).astype(np.int64)
    # fix rounding so sum == M, taking/giving from the largest bins
    while freq.sum() != M:
        diff = M - freq.sum()
        i = np.argmax(freq)
        freq[i] += max(diff, 1 - freq[i]) if diff < 0 else diff
    assert freq.max() < M, "degenerate distribution (single symbol)"
    cum = np.concatenate([[0], np.cumsum(freq)[:-1]])
    table = np.zeros(M, np.uint32)
    for s in np.nonzero(freq)[0]:
        slots = np.arange(cum[s], cum[s] + freq[s])
        table[slots] = s | (freq[s] << 8) | ((slots - cum[s]) << 20)
    return freq, cum, table


# ------------------------------------------------------------------ encoding
def lane_order(sym, R, C):
    rows, cols = sym.shape
    assert rows % R == 0 and cols % C == 0 and C % 32 == 0
    a = sym.reshape(rows // R, R, cols // C, C // 32, 32)      # rb r cb k l
    return a.transpose(0, 2, 4, 3, 1).reshape(-1, (C // 32) * R)  # (tile,lane) x (k,r)


def encode(codes, R, C):
    """codes: int array (rows x cols) with values in [-128, 127].
    Returns dict with words (uint16), offs (uint32), table (uint32), bit accounting."""
    sym = (codes.astype(np.int64) + OFFSET)
    assert sym.min() >= 0 and sym.max() < 256, "code out of 8-bit alphabet (escape not implemented)"
    sym = sym.astype(np.uint8)
    freq, cum, table = build_table(sym)
    S = lane_order(sym, R, C)
    lanes, n = S.shape
    x = np.full(lanes, RANS_L, np.uint64)
    out = np.zeros((lanes, n), np.uint16)
    cnt = np.zeros(lanes, np.int64)
    idx = np.arange(lanes)
    f_all, c_all = freq.astype(np.uint64), cum.astype(np.uint64)
    for i in range(n - 1, -1, -1):
        s = S[:, i]
        f, c = f_all[s], c_all[s]
        m = x >= (np.uint64(1 << 20) * f)
        out[idx[m], cnt[m]] = (x[m] & np.uint64(0xFFFF)).astype(np.uint16)
        cnt[m] += 1
        x[m] >>= np.uint64(16)
        x = ((x // f) << np.uint64(SCALE_BITS)) + (x % f) + c
    # per-lane stream = [state_hi, state_lo] + emitted words reversed
    lens = cnt + 2
    offs = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.uint32)
    words = np.zeros(int(lens.sum()) + 8, np.uint16)
    words[offs] = (x >> np.uint64(16)).astype(np.uint16)
    words[offs + 1] = (x & np.uint64(0xFFFF)).astype(np.uint16)
    for li in range(lanes):  # reversing ragged rows; cheap relative to the encode loop
        words[offs[li] + 2: offs[li] + 2 + cnt[li]] = out[li, :cnt[li]][::-1]
    nweights = codes.size
    p = freq[sym.ravel()] / M
    return dict(words=words, offs=offs, table=table, R=R, C=C, shape=codes.shape,
                bits_payload=16 * (lens.sum() - 2 * lanes) / nweights,
                bits_state_offs=64 * lanes / nweights,
                bits_table=32 * M / nweights,
                entropy_model=float(-np.log2(p).mean()))


def decode_ref(enc):
    """Pure-numpy reference decoder (single lane at a time; for tests only)."""
    rows, cols = enc["shape"]
    R, C = enc["R"], enc["C"]
    tab = enc["table"].astype(np.int64)
    lanes = len(enc["offs"])
    n = (C // 32) * R
    S = np.zeros((lanes, n), np.int64)
    w = enc["words"].astype(np.int64)
    for li in range(lanes):
        p = int(enc["offs"][li])
        st = (w[p] << 16) | w[p + 1]
        p += 2
        for i in range(n):
            e = tab[st & (M - 1)]
            st = ((e >> 8) & 0xFFF) * (st >> SCALE_BITS) + (e >> 20)
            if st < RANS_L:
                st = (st << 16) | w[p]
                p += 1
            S[li, i] = (e & 0xFF) - OFFSET
        assert st == RANS_L, "stream did not return to initial state"
    a = S.reshape(rows // R, cols // C, 32, C // 32, R).transpose(0, 4, 1, 3, 2)
    return a.reshape(rows, cols)


# ------------------------------------------------------------------- device
class RansMatrix:
    """Entropy-coded matrix resident on GPU."""

    def __init__(self, enc, step, lib):
        self.lib = lib
        self.R, self.C = enc["R"], enc["C"]
        self.rows, self.cols = enc["shape"]
        self.words = torch.from_numpy(enc["words"].view(np.int16)).cuda()
        self.offs = torch.from_numpy(enc["offs"].view(np.int32)).cuda()
        self.table = torch.from_numpy(enc["table"].view(np.int32)).cuda()
        self.step = step.float().cuda()
        self.nbytes = sum(t.numel() * t.element_size() for t in (self.words, self.offs, self.table))

    def gemv(self, x, y=None):
        xs = (x.float() * self.step).contiguous()
        if y is None:
            y = torch.zeros(self.rows, device="cuda") if self.cols != self.C else \
                torch.empty(self.rows, device="cuda")
        elif self.cols != self.C:
            y.zero_()
        rc = self.lib.launch_rans_gemv(self.R, self.words.data_ptr(), self.offs.data_ptr(),
                                       self.table.data_ptr(), xs.data_ptr(), y.data_ptr(),
                                       self.rows, self.cols, self.C,
                                       torch.cuda.current_stream().cuda_stream)
        assert rc == 0, f"CUDA error {rc}"
        return y

    def decode(self, out=None):
        if out is None:
            out = torch.empty(self.rows, self.cols, device="cuda", dtype=torch.bfloat16)
        rc = self.lib.launch_rans_decode(self.R, self.words.data_ptr(), self.offs.data_ptr(),
                                         self.table.data_ptr(), self.step.data_ptr(),
                                         out.data_ptr(), self.rows, self.cols, self.C,
                                         torch.cuda.current_stream().cuda_stream)
        assert rc == 0, f"CUDA error {rc}"
        return out


class Int4Matrix:
    """Fixed-rate 4-bit baseline (codes in [-8, 7])."""

    def __init__(self, codes, step, lib):
        self.lib = lib
        self.rows, self.cols = codes.shape
        q = (np.clip(codes, -8, 7) + 8).astype(np.uint32).reshape(self.rows, -1, 8)
        packed = np.zeros(q.shape[:2], np.uint32)
        for b in range(8):
            packed |= q[:, :, b] << np.uint32(4 * b)
        self.w = torch.from_numpy(packed.view(np.int32)).cuda()
        self.step = step.float().cuda()
        self.nbytes = self.w.numel() * 4

    def gemv(self, x, y=None):
        xs = (x.float() * self.step).contiguous()
        y = torch.empty(self.rows, device="cuda") if y is None else y
        rc = self.lib.launch_int4_gemv(self.w.data_ptr(), xs.data_ptr(), y.data_ptr(),
                                       self.rows, self.cols, torch.cuda.current_stream().cuda_stream)
        assert rc == 0, f"CUDA error {rc}"
        return y
