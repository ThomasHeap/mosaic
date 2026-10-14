#!/bin/bash
# Post-processing chain for the overnight designs. Run the stages in order and
# read the output of each before starting the next -- step 1 can invalidate
# everything downstream.
#
#   ./morning.sh triage     # cheap, login node: is anything worth rescoring?
#   ./morning.sh rescore    # GPU job: refold, score, save structures
#   ./morning.sh hisscan    # cheap, login node: install pH switches post hoc
#   ./morning.sh submit     # cheap: build the CSV
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO/.venv/bin/python"
cd "$(dirname "${BASH_SOURCE[0]}")"

case "${1:-triage}" in

triage)
  echo "== designs produced =="
  wc -l designs/*.jsonl 2>/dev/null | tail -20 || echo "none yet"
  echo
  echo "== the question that gates everything: did the PSSM anneal? =="
  # sharp near 1.0 means the reported sequence is a real argmax. Well below
  # means it is an argmax over a diffuse distribution and nothing downstream
  # is meaningful (LOG.md entry 14).
  $PY -I - <<'PYEOF'
import json, glob, statistics as st
rows = []
for f in glob.glob("designs/*.jsonl"):
    for line in open(f):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
if not rows:
    raise SystemExit("no designs yet")
sharp = [r["pssm_max_mean"] for r in rows if r.get("pssm_max_mean") is not None]
print(f"{len(rows)} designs, {len(sharp)} with a sharpness measurement")
if sharp:
    print(f"  sharp: min {min(sharp):.3f}  median {st.median(sharp):.3f}  "
          f"max {max(sharp):.3f}")
    good = sum(1 for s in sharp if s >= 0.8)
    print(f"  {good}/{len(sharp)} at or above 0.8")
    if good == 0:
        print("  STOP: nothing annealed. The sequences are argmaxes over noise;"
              " rescoring them would waste GPU hours. Revisit the schedule.")
print("\n  per pool:")
pools = {}
for r in rows:
    pools.setdefault(r.get("pool", "?"), []).append(r)
for pool, rs in sorted(pools.items()):
    his = [r["n_his"] / len(r["sequence"]) for r in rs]
    print(f"    {pool:10s} n={len(rs):3d}  His {min(his):.0%}-{max(his):.0%}  "
          f"median {st.median(his):.0%}")
PYEOF
  ;;

rescore)
  # Affinity designs first: they feed the histidine-scan route and carry no
  # histidine risk, so they are the designs most likely to be worth the GPU.
  shift || true
  files="${*:-designs/affinity_*.jsonl}"
  echo "rescoring: $files"
  ./submit_rescore.sh $files
  ;;

hisscan)
  $PY -I hisscan.py designs/rescored.jsonl \
      --out designs/hisscan.jsonl \
      --emit designs/hisscan_variants.jsonl
  echo
  echo "Now refold the variants before trusting them:"
  echo "  ./submit_rescore.sh designs/hisscan_variants.jsonl"
  ;;

submit)
  $PY -I rank.py designs/rescored.jsonl --out submission.csv --limit 40
  echo
  echo "== check before uploading =="
  echo " - every row's sequence is one you would defend"
  echo " - no 'config' warnings above (stale designs from a superseded run)"
  echo " - ordered best-first: Proteinbase screens the top 20 of track 1 in order"
  head -3 submission.csv
  ;;

*)
  echo "usage: $0 {triage|rescore|hisscan|submit}" >&2
  exit 1
  ;;
esac
