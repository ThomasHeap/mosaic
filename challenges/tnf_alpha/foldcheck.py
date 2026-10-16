"""Third instrument on monomer foldability, because the first two disagree.

Boltz-2 monomer pLDDT says the designs are much worse than GB1 and Top7 under
an identical protocol (0/20 reach either control). mosaic's Megascale stability
model says the opposite — 15/20 designs score above GB1. GB1 at dG +0.29, below
most of our designs, is not credible for one of the most stable small domains
known, so that model's calibration is suspect here. But the disagreement means
the foldability conclusion rests on a single instrument.

ESMFold2 is a genuinely independent read: a different architecture, and
single-sequence by construction, which is exactly the regime in question rather
than an off-label use of a co-folding model. Run on the same controls, it
breaks the tie.

    python foldcheck.py
"""
import csv, os, sys
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax, numpy as np
from mosaic.models.esmfold2 import ESMFold2Fast
from mosaic.structure_prediction import TargetChain

CONTROLS = {
    "GB1_56aa": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
    "Top7_93aa": ("MGDIQVQVNIDDNGKNFDYTYTVTTESELQKVLNELMDYIKKQGAKRVRISITARTKKEAEKFAAIL"
                  "IKVFAELGYNDINVTFDGDTVTVEGQLE"),
    "villin_35aa": "LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
}

def main():
    if jax.devices()[0].platform != "gpu":
        raise SystemExit("needs a GPU")
    model = ESMFold2Fast()
    def plddt(seq, i):
        feats, writer = model.target_only_features([TargetChain(seq, use_msa=False)])
        pred = model.predict(features=feats, writer=writer, key=jax.random.key(i))
        return float(np.asarray(pred.plddt).mean())
    print("controls (known to fold):", flush=True)
    ctrl = {}
    for i, (n, s) in enumerate(CONTROLS.items()):
        ctrl[n] = plddt(s, i)
        print(f"  {n:14s} {len(s):3d}aa  plddt {ctrl[n]:.3f}", flush=True)
    rows = list(csv.DictReader(open("submission.csv")))
    vals = []
    print("\nsubmitted designs:", flush=True)
    for i, r in enumerate(rows):
        v = plddt(r["sequence"], 100 + i); vals.append(v)
        print(f"  [{i+1}/{len(rows)}] {r['name'][:46]:46s} {v:.3f}", flush=True)
    lo = min(ctrl.values())
    print(f"\n  controls {min(ctrl.values()):.3f}-{max(ctrl.values()):.3f}")
    print(f"  designs  {min(vals):.3f}-{max(vals):.3f} median {float(np.median(vals)):.3f}")
    print(f"  designs at or above the weakest control: {sum(1 for v in vals if v>=lo)}/{len(vals)}")
    print("\nIf designs track the controls here, Boltz-2's monomer channel was "
          "pessimistic and the foldability concern is overstated. If they sit "
          "well below, two independent models agree and it is real.")

if __name__ == "__main__":
    main()
