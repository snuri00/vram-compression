"""Shared plumbing: model loading, calibration/test data, layer-by-layer
processing (fits a 1.5B model on a 4 GB GPU), and perplexity evaluation."""
import math
import random

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = torch.device("cuda")
DTYPE = torch.bfloat16
SEQLEN = 2048


def load_model(name="Qwen/Qwen2.5-1.5B"):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=DTYPE, attn_implementation="sdpa")
    model.eval()
    model.config.use_cache = False
    return model, tok


def get_data(tok, nsamples=128, seed=0):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train = tok("\n\n".join(ds["train"]["text"]), return_tensors="pt").input_ids
    test = tok("\n\n".join(ds["test"]["text"]), return_tensors="pt").input_ids
    rng = random.Random(seed)
    calib = []
    for _ in range(nsamples):
        i = rng.randint(0, train.shape[1] - SEQLEN - 1)
        calib.append(train[:, i:i + SEQLEN])
    n = test.shape[1] // SEQLEN
    test = [test[:, i * SEQLEN:(i + 1) * SEQLEN] for i in range(n)]
    return calib, test


class _Stop(Exception):
    pass


@torch.no_grad()
def capture_inputs(model, samples):
    """Run embeddings only; return layer-0 inputs (on GPU) and the layer kwargs."""
    m = model.model
    m.embed_tokens.to(DEV)
    m.rotary_emb.to(DEV)
    inps, kw = [], {}

    def hook(_mod, args, kwargs):
        inps.append(args[0] if args else kwargs["hidden_states"])
        if not kw:
            kw.update({k: v for k, v in kwargs.items()
                       if k not in ("hidden_states", "past_key_values")})
        raise _Stop

    h = m.layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    for s in samples:
        try:
            model(s.to(DEV), use_cache=False)
        except _Stop:
            pass
    h.remove()
    m.embed_tokens.cpu()
    m.rotary_emb.cpu()
    kw["use_cache"] = False
    return torch.cat(inps, 0), kw


def linear_groups(layer):
    """Linears grouped by shared input (they share one Hessian)."""
    a, p = layer.self_attn, layer.mlp
    return {
        "qkv": [a.q_proj, a.k_proj, a.v_proj],
        "o": [a.o_proj],
        "gate_up": [p.gate_proj, p.up_proj],
        "down": [p.down_proj],
    }


@torch.no_grad()
def process_layers(model, calib, quant_fn, log=print):
    """GPTQ-style sequential pass. For each decoder layer: collect the input
    Hessian H = sum x^T x of every linear group on the (already quantized
    upstream) activations, call quant_fn(layer_idx, group_name, W, H) which
    returns (W_hat, info), write W_hat back, then propagate activations."""
    inps, kw = capture_inputs(model, calib)
    outs = torch.empty_like(inps)
    infos = []
    for li, layer in enumerate(model.model.layers):
        layer.to(DEV)
        groups = linear_groups(layer)
        Hs, cnt, hooks = {}, {}, []
        for g, mods in groups.items():
            n = mods[0].in_features
            Hs[g] = torch.zeros(n, n, device=DEV, dtype=torch.float32)
            cnt[g] = 0

            def mk(g):
                def fh(_m, args):
                    x = args[0].reshape(-1, args[0].shape[-1]).float()
                    Hs[g].addmm_(x.t(), x)
                    cnt[g] += x.shape[0]
                return fh
            hooks.append(mods[0].register_forward_pre_hook(mk(g)))
        for i in range(inps.shape[0]):
            layer(inps[i:i + 1], **kw)
        for h in hooks:
            h.remove()
        for g, mods in groups.items():
            H = Hs.pop(g).div_(cnt[g])
            W = torch.cat([m.weight.float() for m in mods], 0)
            W_hat, info = quant_fn(li, g, W, H)
            info.update(layer=li, group=g, rows=W.shape[0], cols=W.shape[1])
            infos.append(info)
            r = 0
            for m in mods:
                k = m.out_features
                m.weight.copy_(W_hat[r:r + k].to(m.weight.dtype))
                r += k
            del H, W, W_hat
            torch.cuda.empty_cache()
        for i in range(inps.shape[0]):
            outs[i:i + 1] = layer(inps[i:i + 1], **kw)
        layer.cpu()
        inps, outs = outs, inps
        torch.cuda.empty_cache()
        log(f"layer {li} done")
    return infos


@torch.no_grad()
def perplexity(model, test):
    inps, kw = capture_inputs(model, test)
    for layer in model.model.layers:
        layer.to(DEV)
        for i in range(inps.shape[0]):
            inps[i:i + 1] = layer(inps[i:i + 1], **kw)
        layer.cpu()
    model.model.norm.to(DEV)
    model.lm_head.to(DEV)
    nll, ntok = 0.0, 0
    for i in range(inps.shape[0]):
        h = model.model.norm(inps[i:i + 1])[0]
        y = test[i][0, 1:].to(DEV)
        for s in range(0, SEQLEN - 1, 512):
            e = min(s + 512, SEQLEN - 1)
            logits = model.lm_head(h[s:e]).float()
            nll += F.cross_entropy(logits, y[s:e], reduction="sum").item()
            ntok += e - s
    model.model.norm.cpu()
    model.lm_head.cpu()
    torch.cuda.empty_cache()
    return math.exp(nll / ntok)
