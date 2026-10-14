#!/bin/bash
# Fit ph_physics.PKaParams against PROPKA labels.
#
# CPU only -- no GPU needed, because the loss term consumes backbone alone and so
# can be calibrated on crystal structures rather than on predicted designs.
# PROPKA on a few hundred complexes is minutes, but it still does not belong on
# the login node.
#
# Download first, on the login node, since compute nodes may have no network:
#   ../../.venv/bin/python -I calibrate.py --fetch 600 --data-dir pdb_cache --fetch-only
# Then:
#   ./submit_calibrate.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHALLENGE="$REPO/challenges/tnf_alpha"
mkdir -p "$CHALLENGE/logs"

sbatch -A cosc020762 -p compute -c 16 --mem=32G -t 2:00:00 \
  -J tnf-calibrate \
  -o "$CHALLENGE/logs/%x-%j.out" \
  --wrap "cd $CHALLENGE && $REPO/.venv/bin/python -I -u calibrate.py \
            --data-dir $CHALLENGE/pdb_cache \
            --out $CHALLENGE/pka_params.json \
            --holdout 0.25"
