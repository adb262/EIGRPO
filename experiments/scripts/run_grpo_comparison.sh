#!/usr/bin/env bash
set -euo pipefail

model="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
shift || true

python3 experiments/scripts/run_grpo_baseline.py \
  --variant regular \
  --model "$model" \
  "$@"

python3 experiments/scripts/run_grpo_baseline.py \
  --variant conditioned \
  --model "$model" \
  "$@"
