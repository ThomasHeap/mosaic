#!/bin/bash
# Repredict and score designs, populating the fields rank.py sorts on.
# Without this, rank.py orders designs arbitrarily (see rescore.py docstring).
#
#   ./submit_rescore.sh designs/ph_L75.jsonl designs/physics_L75.jsonl
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHALLENGE="$REPO/challenges/tnf_alpha"
mkdir -p "$CHALLENGE/logs"

XLA="--xla_gpu_enable_triton_gemm=false"   # see LOG.md entry 13

sbatch -A cosc020762 -p cnu -c 16 --mem=64G -t 6:00:00 --gpus=a100:1 \
  --exclude="${EXCLUDE:-bp1-gpu031}" \
  -J tnf-rescore \
  -o "$CHALLENGE/logs/%x-%j.out" \
  --wrap "cd $CHALLENGE && XLA_FLAGS='$XLA' $REPO/.venv/bin/python -u rescore.py $* \
            --out $CHALLENGE/designs/rescored.jsonl"
