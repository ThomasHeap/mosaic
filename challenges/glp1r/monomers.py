"""Predict each candidate alone and write a PDB, for the novelty screen.

Novelty is a hard criterion: Adaptyv's level 4 de novo needs sequence identity
at or below 30% and below-moderate structural similarity, the latter measured
on a predicted monomer structure. score.py predicts complexes and keeps no
monomer, so these are generated separately.
"""
import argparse, json, sys
from pathlib import Path
import jax, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from mosaic.structure_prediction import TargetChain

ap = argparse.ArgumentParser()
ap.add_argument("jsonl")
ap.add_argument("--out-dir", default="structures_monomer")
ap.add_argument("--recycling-steps", type=int, default=3)
ap.add_argument("--sampling-steps", type=int, default=50)
a = ap.parse_args()

from mosaic.models.protenix import ProtenixBase
model = ProtenixBase()
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in open(a.jsonl)]
print(f"predicting {len(rows)} monomers", flush=True)
for i, d in enumerate(rows):
    key = jax.random.PRNGKey(i)
    feats, writer = model.target_only_features([TargetChain(d["sequence"], use_msa=False)])
    pred = model.predict(features=feats, writer=writer,
                         recycling_steps=a.recycling_steps,
                         sampling_steps=a.sampling_steps, key=key)
    p = out / f"{d['name']}_monomer.pdb"
    pred.st.write_pdb(str(p))
    print(f"[{i+1}/{len(rows)}] {d['name'][:34]:34s} pLDDT {float(np.asarray(pred.plddt).mean()):.3f} -> {p.name}", flush=True)
print("done")
