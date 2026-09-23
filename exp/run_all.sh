#!/usr/bin/env bash
# Full sweep. Each line writes results/<model>__<method>_<bits>.json
set -e
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { [ -f "results/Qwen2.5-1.5B__$1${2:+_$2}.json" ] && return; python3 exp/run.py "$1" ${2:+--bits $2} 2>&1 | grep -E '^\{|Error|error' ; }
run fp
run stats
for b in 4 3 2; do run rtn $b; run gptq $b; done
for b in 4.25 3.25 2.25; do run ecsq $b; run ecsq_wf $b; done
