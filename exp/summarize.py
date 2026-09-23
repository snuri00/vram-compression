"""Collect results/*.json into results/SUMMARY.md."""
import glob
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

LABEL = {
    "fp": "BF16 baseline",
    "rtn": "RTN g128 (fixed rate)",
    "gptq": "GPTQ g128 (fixed rate)",
    "ecsq": "ECSQ-GPTQ (entropy coded)",
    "ecsq_wf": "ECSQ-GPTQ + per-column waterfilling",
}


def main():
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(RES, "*__*.json")))]
    out = []
    for model in sorted({r["model"] for r in runs}):
        rs = [r for r in runs if r["model"] == model]
        out.append(f"## {model}\n")
        out.append("| Method | bits/weight | code entropy | Wikitext-2 PPL |")
        out.append("|---|---|---|---|")
        order = list(LABEL)
        for r in sorted((r for r in rs if r["method"] != "stats"),
                        key=lambda r: (order.index(r["method"]), -(r["bits"] or 99))):
            bpw = f'{r["bpw"]:.3f}' if "bpw" in r else "16"
            ent = f'{r["code_entropy"]:.3f}' if "code_entropy" in r else "n/a"
            out.append(f'| {LABEL[r["method"]]} | {bpw} | {ent} | {r["ppl"]:.3f} |')
        st = [r for r in rs if r["method"] == "stats"]
        if st:
            agg = defaultdict(lambda: defaultdict(list))
            for i in st[0]["infos"]:
                for k, v in i.items():
                    if isinstance(v, float):
                        agg[i["group"]][k].append((v, i["rows"] * i["cols"]))
            keys = ["gain_isotropic_to_limit", "gain_gptq_to_wf", "gain_gptq_to_wf_rot",
                    "kurtosis", "kurtosis_rot", "diag_max_over_median"]
            out.append("\n### Per-layer-group statistics (mean over layers)\n")
            out.append("| group | " + " | ".join(keys) + " |")
            out.append("|---" * (len(keys) + 1) + "|")
            for g, d in agg.items():
                vals = [sum(v for v, _ in d[k]) / len(d[k]) for k in keys]
                out.append(f"| {g} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
            tot = defaultdict(float)
            wsum = 0
            for i in st[0]["infos"]:
                w = i["rows"] * i["cols"]
                wsum += w
                for k in keys[:3]:
                    tot[k] += i[k] * w
            out.append("\nParameter-weighted mean: " + ", ".join(
                f"{k} = {tot[k] / wsum:.3f} bit" for k in keys[:3]))
        out.append("")
    open(os.path.join(RES, "SUMMARY.md"), "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
