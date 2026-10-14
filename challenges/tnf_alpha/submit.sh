#!/bin/bash
# Submit a TNF-alpha design run. Never run design.py on the login node: it has no
# usable GPU (cuInit fails), so JAX silently falls back to CPU and Boltz-2 crawls.
#
#   ./submit.sh ph 8 75          # pool, n_designs, binder_length
#   ./submit.sh affinity 8 65
#
# Note: lbatch --gputype A100 cannot schedule on this cluster (it passes
# --gpus=A100:1 but the gres is lowercase a100), hence raw sbatch.
set -euo pipefail

POOL="${1:-ph}"
N_DESIGNS="${2:-8}"
LENGTH="${3:-75}"
SEED="${4:-0}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHALLENGE="$REPO/challenges/tnf_alpha"

mkdir -p "$CHALLENGE/logs" "$CHALLENGE/designs"

# EXCLUDE: comma-separated nodes to avoid. bp1-gpu031 advertises an a100 but
# CUDA fails to initialize on it ("no supported devices found for platform
# CUDA"), which kills the job minutes in -- see LOG.md entry 10.
EXCLUDE="${EXCLUDE:-bp1-gpu031}"

sbatch -A cosc020762 -p cnu -c 8 --mem=64G -t 6:00:00 --gpus=a100:1 \
  --exclude="$EXCLUDE" \
  -J "tnf-$POOL-L$LENGTH" \
  -o "$CHALLENGE/logs/%x-%j.out" \
  --wrap "cd $CHALLENGE && $REPO/.venv/bin/python design.py \
            --pool $POOL --n-designs $N_DESIGNS --binder-length $LENGTH \
            --seed $SEED --out $CHALLENGE/designs"
