"""Quantizers. All operate on W (rows=out, cols=in, fp32, GPU) and the input
Hessian H (cols x cols). Each returns (W_hat, info) where info carries the
bit accounting.

Fixed-rate: asymmetric min/max grid per group of 128 input columns,
            b bits per code + 32/128 bits of fp16 scale/zero per weight.
ECSQ:       unbounded uniform grid with step delta_i per column; codes are
            then entropy coded, so rate = empirical (pooled, zero-order)
            entropy of the integer codes + side info. A static-table rANS
            coder gets within ~0.01 bit of this.
WF:         delta_i = c * d_i where d_i = diag of upper Cholesky of H^-1
            (the GPTQ per-column error scale). Equalizes each column's
            contribution to the output error: reverse waterfilling at high
            rate (WaterSIC-style)."""
import math

import torch

GROUP = 128


def entropy_bits(codes):
    _, counts = torch.unique(codes, return_counts=True)
    p = counts.double() / codes.numel()
    return float(-(p * p.log2()).sum())


def hinv_chol(H, percdamp=0.01):
    """Upper Cholesky factor of (H + damp*I)^-1, computed on CPU in float64
    (robust to Qwen's massive activations, and keeps GPU memory free)."""
    dev = H.device
    H = H.detach().double().cpu()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = percdamp * torch.diag(H).mean()
    H.diagonal().add_(damp)
    while True:
        L, info = torch.linalg.cholesky_ex(H)
        if info == 0:
            break
        H.diagonal().add_(damp)  # escalate damping if still not PD
    Hinv = torch.cholesky_inverse(L)
    del L, H
    U = torch.linalg.cholesky(Hinv, upper=True)
    return U.float().to(dev), dead.to(dev)


def gptq(W, Hinv, qcol, start_group=None, blocksize=GROUP):
    """Generic GPTQ / LDLQ with error feedback. qcol(w, col) -> (deq, code).
    start_group(col, Wcur) is called at every GROUP boundary with the
    weights updated so far (aligned with blocksize so it is exact)."""
    W = W.clone()
    n = W.shape[1]
    Q = torch.zeros_like(W)
    C = torch.zeros_like(W, dtype=torch.int32)
    for i1 in range(0, n, blocksize):
        i2 = min(i1 + blocksize, n)
        if start_group is not None:
            start_group(i1, W[:, i1:i2])
        W1 = W[:, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]
            q, c = qcol(w, i1 + i)
            Q[:, i1 + i] = q
            C[:, i1 + i] = c
            err = (w - q) / Hi[i, i]
            W1[:, i:] -= err.unsqueeze(1) @ Hi[i, i:].unsqueeze(0)
            E1[:, i] = err
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q, C


# ---------------------------------------------------------------- fixed rate
class MinMax:
    def __init__(self, bits):
        self.maxq = 2 ** bits - 1

    def fit(self, Wg):
        xmin = Wg.min(1).values.clamp(max=0)
        xmax = Wg.max(1).values.clamp(min=0)
        same = xmin == xmax
        xmin[same], xmax[same] = -1, 1
        self.scale = (xmax - xmin) / self.maxq
        self.zero = torch.round(-xmin / self.scale)

    def q(self, w):
        c = torch.clamp(torch.round(w / self.scale) + self.zero, 0, self.maxq)
        return self.scale * (c - self.zero), c.int()


def rtn_fixed(W, H, bits):
    qz = MinMax(bits)
    Q = torch.empty_like(W)
    C = torch.empty_like(W, dtype=torch.int32)
    for g in range(0, W.shape[1], GROUP):
        Wg = W[:, g:g + GROUP]
        qz.fit(Wg)
        s, z = qz.scale[:, None], qz.zero[:, None]
        c = torch.clamp(torch.round(Wg / s) + z, 0, qz.maxq)
        Q[:, g:g + GROUP] = s * (c - z)
        C[:, g:g + GROUP] = c.int()
    return Q, dict(bpw=bits + 32 / GROUP, code_entropy=entropy_bits(C))


def gptq_fixed(W, H, bits):
    Hinv, dead = hinv_chol(H)
    W = W.clone()
    W[:, dead] = 0
    qz = MinMax(bits)
    Q, C = gptq(W, Hinv, lambda w, i: qz.q(w), start_group=lambda i, Wg: qz.fit(Wg))
    return Q, dict(bpw=bits + 32 / GROUP, code_entropy=entropy_bits(C))


# ---------------------------------------------------------------------- ECSQ
def _ecsq_run(W, Hinv, steps):
    return gptq(W, Hinv, lambda w, i: (torch.round(w / steps[i]) * steps[i],
                                        torch.round(w / steps[i]).int()))


def ecsq(W, H, target_bits, waterfill):
    """Entropy-constrained uniform quantization with GPTQ error feedback.
    Picks the global step multiplier c so that code entropy + side info
    hits target_bits (bisection on cheap RTN, then secant on GPTQ)."""
    Hinv, dead = hinv_chol(H)
    W = W.clone()
    W[:, dead] = 0
    d = torch.diag(Hinv)
    base = d / d.mean() if waterfill else torch.ones_like(d)
    side = (16 * W.shape[1] / W.numel()) if waterfill else 0.0  # fp16 d_i per column
    target = target_bits - side

    def rtn_H(logc):
        s = math.exp(logc) * base
        return entropy_bits(torch.round(W / s))

    # bisection on RTN entropy (H decreases with c)
    lo, hi = math.log(W.std().item()) - 12, math.log(W.std().item()) + 4
    for _ in range(40):
        mid = (lo + hi) / 2
        if rtn_H(mid) > target:
            lo = mid
        else:
            hi = mid
    logc = (lo + hi) / 2
    # two secant corrections using the real GPTQ output (dH/dlog2c ~ -1)
    for _ in range(2):
        steps = math.exp(logc) * base
        Q, C = _ecsq_run(W, Hinv, steps)
        h = entropy_bits(C)
        logc += (h - target) * math.log(2)
    steps = math.exp(logc) * base
    Q, C = _ecsq_run(W, Hinv, steps)
    h = entropy_bits(C)
    return Q, dict(bpw=h + side, code_entropy=h, levels=int(torch.unique(C).numel()))
