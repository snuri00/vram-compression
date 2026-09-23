"""Usage:
  python exp/run.py fp                     # FP16/BF16 baseline perplexity
  python exp/run.py stats                  # per-layer theory numbers, no quantization
  python exp/run.py rtn|gptq  --bits 3     # fixed-rate, g128
  python exp/run.py ecsq|ecsq_wf --bits 3.25   # entropy-coded, total bits/weight target
"""
import argparse
import json
import math
import os
import time

import torch

import common
import quant

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

_rot = {}


def rotation(n):
    if n not in _rot:
        g = torch.Generator().manual_seed(n)
        q, r = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
        _rot[n] = (q * torch.sign(torch.diag(r))).float()  # Haar orthogonal
    return _rot[n]


def am_gm_bits(x):
    x = x.double()
    return float(0.5 * math.log2(x.mean() / torch.exp(torch.log(x).mean())))


def kurtosis(W):
    z = (W - W.mean(1, keepdim=True)) / W.std(1, keepdim=True)
    return float((z ** 4).mean() - 3)


def stats_fn(li, g, W, H):
    """Theory numbers per linear group; heavy linear algebra on CPU float64."""
    n = H.shape[0]
    Hc = H.double().cpu()
    Hd = Hc.clone()
    Hd.diagonal().add_(0.01 * torch.diag(Hc).mean())
    ev = torch.linalg.eigvalsh(Hd).clamp(min=1e-30)
    del Hd
    d = torch.diag(quant.hinv_chol(H)[0]).cpu()
    R = rotation(n).cpu().double()
    Hr = R.t() @ Hc @ R
    d_r = torch.diag(quant.hinv_chol(Hr)[0]).cpu()
    Wc = W.double().cpu()
    info = dict(
        # Sigma-oblivious isotropic coding vs information-theoretic limit
        gain_isotropic_to_limit=am_gm_bits(ev),
        # GPTQ (uniform step) vs WaterSIC-style per-column step, native basis
        gain_gptq_to_wf=am_gm_bits(1 / d.double() ** 2),
        # same after random rotation (rotation equalizes the diagonal)
        gain_gptq_to_wf_rot=am_gm_bits(1 / d_r.double() ** 2),
        kurtosis=kurtosis(Wc),
        kurtosis_rot=kurtosis(Wc @ R),
        diag_max_over_median=float(torch.diag(Hc).max() / torch.diag(Hc).median()),
    )
    return W, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["fp", "stats", "rtn", "gptq", "ecsq", "ecsq_wf"])
    ap.add_argument("--bits", type=float, default=None)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--nsamples", type=int, default=128)
    a = ap.parse_args()

    t0 = time.time()
    model, tok = common.load_model(a.model)
    calib, test = common.get_data(tok, a.nsamples)
    infos = []
    if a.method != "fp":
        fn = {
            "stats": stats_fn,
            "rtn": lambda li, g, W, H: quant.rtn_fixed(W, H, int(a.bits)),
            "gptq": lambda li, g, W, H: quant.gptq_fixed(W, H, int(a.bits)),
            "ecsq": lambda li, g, W, H: quant.ecsq(W, H, a.bits, waterfill=False),
            "ecsq_wf": lambda li, g, W, H: quant.ecsq(W, H, a.bits, waterfill=True),
        }[a.method]
        infos = common.process_layers(model, calib, fn, log=lambda s: print(s, flush=True))
    out = dict(method=a.method, bits=a.bits, model=a.model, infos=infos)
    if a.method != "stats":
        out["ppl"] = common.perplexity(model, test)
    if infos and "bpw" in infos[0]:
        n = sum(i["rows"] * i["cols"] for i in infos)
        out["bpw"] = sum(i["bpw"] * i["rows"] * i["cols"] for i in infos) / n
        out["code_entropy"] = sum(i["code_entropy"] * i["rows"] * i["cols"] for i in infos) / n
    out["seconds"] = time.time() - t0
    os.makedirs(RES, exist_ok=True)
    tag = a.method + (f"_{a.bits:g}" if a.bits is not None else "")
    tag = a.model.split("/")[-1] + "__" + tag
    with open(os.path.join(RES, tag + ".json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "infos"}), flush=True)


if __name__ == "__main__":
    main()
