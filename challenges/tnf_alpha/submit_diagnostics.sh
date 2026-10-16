#!/bin/bash
# Run one diagnostic test on a GPU node.  ./submit_diagnostics.sh monomer
set -euo pipefail
TEST="${1:-monomer}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHALLENGE="$REPO/challenges/tnf_alpha"
mkdir -p "$CHALLENGE/logs"
XLA="--xla_gpu_enable_triton_gemm=false"   # see LOG.md entry 13
sbatch -A cosc020762 -p cnu -c 16 --mem=64G -t 4:00:00 --gpus=a100:1 \
  --exclude="${EXCLUDE:-bp1-gpu031}" \
  -J "tnf-diag-$TEST" -o "$CHALLENGE/logs/%x-%j.out" \
  --wrap "cd $CHALLENGE && XLA_FLAGS='$XLA' $REPO/.venv/bin/python -u diagnostics.py --test $TEST"
