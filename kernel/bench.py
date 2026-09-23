"""Usage:
  python kernel/bench.py test     # correctness (small shapes; safe while GPU is busy)
  python kernel/bench.py bench    # timing; run on an idle GPU
Synthetic ECSQ codes: round(w / step) for Gaussian or Laplacian w, with step
chosen to hit a target code entropy (bits/weight)."""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rans  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def entropy(codes):
    _, c = np.unique(codes, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def synth_codes(rows, cols, target_bits, dist="laplace", seed=0):
    rng = np.random.default_rng(seed)
    w = rng.laplace(size=(rows, cols)) if dist == "laplace" else rng.standard_normal((rows, cols))
    lo, hi = -12.0, 6.0  # bisection on log step
    for _ in range(40):
        mid = (lo + hi) / 2
        h = entropy(np.round(w[: min(rows, 256)] / np.exp(mid)))
        lo, hi = (mid, hi) if h > target_bits else (lo, mid)
    codes = np.clip(np.round(w / np.exp((lo + hi) / 2)), -127, 127).astype(np.int32)
    return codes


def test():
    lib = rans.build()
    torch.manual_seed(0)
    for (rows, cols, R, C, bits) in [(64, 256, 4, 256, 3.0), (128, 512, 8, 128, 2.0),
                                     (256, 1024, 16, 512, 4.0), (32, 64, 1, 32, 1.0)]:
        codes = synth_codes(rows, cols, bits, seed=rows)
        enc = rans.encode(codes, R, C)
        ref = rans.decode_ref(enc)
        assert np.array_equal(ref, codes), "numpy reference decode mismatch"
        step = torch.rand(cols) + 0.5
        m = rans.RansMatrix(enc, step, lib)
        dec = m.decode().float().cpu()
        want = torch.from_numpy(codes).float() * step
        assert torch.allclose(dec, want.bfloat16().float()), "GPU decode mismatch"
        x = torch.randn(cols, device="cuda")
        y = m.gemv(x)
        y_ref = torch.from_numpy(codes).float().cuda() @ (x * step.cuda())
        err = ((y - y_ref).abs().max() / y_ref.abs().max()).item()
        assert err < 1e-4, f"GEMV mismatch {err}"
        print(f"ok  {rows}x{cols} R={R} C={C}  H={entropy(codes):.2f}  "
              f"payload={enc['bits_payload']:.3f}  overhead={enc['bits_state_offs']:.3f} bit/w  gemv_err={err:.1e}")
    # int4 baseline correctness
    codes = np.random.default_rng(1).integers(-8, 8, size=(256, 1024))
    step = torch.rand(1024) + 0.5
    m4 = rans.Int4Matrix(codes, step, lib)
    x = torch.randn(1024, device="cuda")
    y_ref = torch.from_numpy(codes).float().cuda() @ (x * step.cuda())
    assert torch.allclose(m4.gemv(x), y_ref, rtol=1e-4, atol=1e-3)
    print("ok  int4 baseline")


def timeit(fn, iters=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000  # microseconds


def bench():
    lib = rans.build()
    shapes = {  # (rows, cols) = (out_features, in_features)
        "1.5B qkv": (2048, 1536), "1.5B gate_up": (17920, 1536), "1.5B down": (1536, 8960),
        "27B-like attn": (5120, 5120), "27B-like gate_up": (34816, 5120), "27B-like down": (5120, 17408),
    }
    out = []
    for name, (rows, cols) in shapes.items():
        for bits in (2.0, 3.0, 4.0):
            codes = synth_codes(rows, cols, bits, seed=rows + cols)
            step = torch.rand(cols) * 0.01 + 0.005
            x = torch.randn(cols, device="cuda")
            W = (torch.from_numpy(codes).float() * step).bfloat16().cuda()
            xb = x.bfloat16()
            t_bf16 = timeit(lambda: torch.nn.functional.linear(xb, W))
            m4 = rans.Int4Matrix(np.clip(codes, -8, 7), step, lib)
            t_int4 = timeit(lambda: m4.gemv(x))
            best = None
            for R in (4, 8, 16):
                for C in sorted({cols, 512, 1024, 2048} & set(range(32, cols + 1, 32))):
                    if cols % C or rows % R:
                        continue
                    enc = rans.encode(codes, R, C)
                    m = rans.RansMatrix(enc, step, lib)
                    yb = torch.empty(rows, device="cuda")
                    t = timeit(lambda: m.gemv(x, yb))
                    if best is None or t < best["t_rans_us"]:
                        buf = torch.empty(rows, cols, device="cuda", dtype=torch.bfloat16)
                        t_dec = timeit(lambda: torch.nn.functional.linear(xb, m.decode(buf)))
                        best = dict(R=R, C=C, t_rans_us=t, t_decode_then_cublas_us=t_dec,
                                    bits_total=8 * m.nbytes / codes.size,
                                    bits_payload=enc["bits_payload"],
                                    bits_overhead=enc["bits_state_offs"] + enc["bits_table"])
                    del m
            r = dict(shape=name, rows=rows, cols=cols, entropy=entropy(codes),
                     t_bf16_us=t_bf16, t_int4_us=t_int4, **best)
            r["speedup_vs_bf16"] = t_bf16 / r["t_rans_us"]
            r["speedup_vs_int4"] = t_int4 / r["t_rans_us"]
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}),
                  flush=True)
            out.append(r)
            del W, m4
            torch.cuda.empty_cache()
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "kernel_bench.json"), "w"), indent=1)


if __name__ == "__main__":
    {"test": test, "bench": bench}[sys.argv[1]]()
