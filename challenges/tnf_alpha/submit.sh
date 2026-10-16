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

# Overnight knobs. A100s are the only GPUs with enough memory for a 390-residue
# complex at 0.95 memory fraction, so the gres request is explicit; cnu and gpu
# both have some (the idle nodes are 2080s/3090s and will OOM).
PARTITION="${PARTITION:-cnu}"
WALL="${WALL:-6:00:00}"
# Hours of design work before the loop stops starting new designs. Leave ~1 h
# of headroom under WALL so a long design cannot be killed mid-flight.
BUDGET="${BUDGET:-0}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHALLENGE="$REPO/challenges/tnf_alpha"

mkdir -p "$CHALLENGE/logs" "$CHALLENGE/designs"

# EXCLUDE: comma-separated nodes to avoid. bp1-gpu031 advertises an a100 but
# CUDA fails to initialize on it ("no supported devices found for platform
# CUDA"), which kills the job minutes in -- see LOG.md entry 10.
EXCLUDE="${EXCLUDE:-bp1-gpu031}"

# XLA's Triton GEMM autotuner emits invalid kernels on this stack
# ("'tt.dot' op expected the output shape to be the concatenation...") and then
# wedges on-device while consuming no CPU -- four jobs sat idle for 35 minutes
# before this was found (LOG.md entries 12-13). Forcing cuBLAS costs the
# matmul/elementwise fusion, not GEMM performance itself. Probe 19400552 went
# from hung to 7.62 s/step with this set.
XLA="--xla_gpu_enable_triton_gemm=false"

# Cold-cache guard. Boltz-2 lazily downloads a 2.2 GB checkpoint and a 1.8 GB
# CCD tarball into one shared cache on first use. Launching several jobs at once
# against an empty cache makes them race on the same paths: one wins and the
# rest read half-written files, failing with "PytorchStreamReader failed
# reading zip archive" or "CCD component ASP not found!". That cost three of
# four jobs on the first launch -- see LOG.md entry 11. Populate the cache with
# a single job, then launch the rest.
CACHE="${MOSAIC_CACHE_DIR:-$HOME/.cache/mosaic}/boltz"
if [ ! -s "$CACHE/boltz2_conf.ckpt" ] || [ ! -s "$CACHE/mols/ASP.pkl" ]; then
  if [ "${WARMUP:-0}" != "1" ]; then
    echo "Boltz-2 cache at $CACHE is cold or incomplete." >&2
    echo "Launch ONE job first to populate it:  WARMUP=1 $0 $POOL $N_DESIGNS $LENGTH" >&2
    echo "Then submit the rest once $CACHE/mols/ASP.pkl exists." >&2
    exit 1
  fi
  echo "WARMUP: cache is cold; this job will populate it. Submit others after it loads."
fi

sbatch -A cosc020762 -p "$PARTITION" -c 8 --mem=64G -t "$WALL" --gpus=a100:1 \
  --exclude="$EXCLUDE" \
  -J "tnf-$POOL-L$LENGTH" \
  -o "$CHALLENGE/logs/%x-%j.out" \
  --wrap "cd $CHALLENGE && XLA_FLAGS='$XLA' $REPO/.venv/bin/python -u design.py \
            --pool $POOL --n-designs $N_DESIGNS --binder-length $LENGTH \
            --seed $SEED --time-budget $BUDGET \
            --monomer-weight ${MONOMER:-1.0} --out $CHALLENGE/designs"
