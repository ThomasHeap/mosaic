"""Score GLP1R designs: binding, and whether they engage the activation core.

Affinity is validated -- in controls.py GLP-1 and exendin-4 score ipTM 0.900 and
0.879 against 0.556 and 0.565 for a scrambled peptide and poly-alanine. Agonism
is not predictable, so it is measured geometrically instead: how many of the 18
receptor residues contacted by GLP-1 His7-Thr11 does the design reach? That
segment's removal costs 1000-10000x cAMP potency against 100-300x affinity, so
reaching it is the structural precondition for activation. Contacting it does
not prove a design activates; failing to is strong evidence it will not, and is
the condition an ECD-only binder fails.

Lipid-belt contact is reported alongside, because the assay protein is in
detergent and the belt is exposed there. A design binding it would score on
affinity and fail activation, so a high belt count invalidates a high ipTM.

Scored with Boltz-2 and its MSA, while design.py uses ProtenixBase without one.
Boltz-2 needs the MSA here: without it epitope RMSD degrades from 3.66 to 7.80 A.
Using a different model to score than to design is deliberate -- neither
represents a membrane, so they share that blind spot but not their artefacts.
"""
import argparse, json, sys
from pathlib import Path

import gemmi, jax, jax.numpy as jnp, numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import mosaic.losses.structure_prediction as sp
from mosaic.structure_prediction import TargetChain
from target import (ACTIVATION_CORE, ACTIVATION_DEPTH_THRESHOLD,
                    glp1r_template_chain, lipid_facing_belt,
                    orthosteric_epitope, receptor_sequence)

CONTACT = 4.5


def contacts(st, binder_len, resnums, receptor_order):
    """Receptor residues in `resnums` within CONTACT of any binder heavy atom."""
    chains = list(st[0])
    binder = chains[0]
    recept = chains[1] if len(chains) > 1 else None
    if recept is None:
        return set()
    def heavy(ch, limit=None):
        out = []
        for i, r in enumerate(ch):
            if limit is not None and i >= limit:
                break
            for a in r:
                if a.element.name != "H":
                    out.append((i, np.array([a.pos.x, a.pos.y, a.pos.z])))
        return out
    bh = heavy(binder)
    rh = heavy(recept)
    if not bh or not rh:
        return set()
    B = np.array([x for _, x in bh]); R = np.array([x for _, x in rh])
    D = np.linalg.norm(R[:, None] - B[None, :], axis=-1)
    hit = {rh[i][0] for i in range(len(rh)) if (D[i] < CONTACT).any()}
    # positional index -> UniProt residue number
    nums = {receptor_order[i] for i in hit if i < len(receptor_order)}
    return nums & set(resnums)


def insertion_depth(st, receptor_order):
    """Deepest point the binder reaches along the transmembrane bundle axis.

    Contact count rewards a peptide lying across the groove; this only rewards
    entering the bundle. GLP-1's deepest heavy atom sits at -6.3 A in the 6x18
    frame, with its activating segment spanning roughly -8 to -14.

    The axis is recomputed from the *predicted* receptor in each prediction,
    not taken from the template. Predictions do not come back in the template
    frame -- using the template axis put every depth around +95 rather than -6,
    about 100 A off -- so a template axis measures from the wrong origin and
    only happens to rank correctly if every prediction lands in the same place.
    """
    chains = list(st[0])
    if len(chains) < 2:
        return None
    binder, recept = chains[0], chains[1]
    ca = {}
    for i, r in enumerate(recept):
        a = r.find_atom("CA", "*")
        if a and i < len(receptor_order):
            ca[receptor_order[i]] = np.array([a.pos.x, a.pos.y, a.pos.z])
    tm = np.array([v for k, v in ca.items() if k > 145])
    ecd = [v for k, v in ca.items() if k <= 145]
    if len(tm) < 10 or not ecd:
        return None
    cen = tm.mean(0)
    axis = np.linalg.svd(tm - cen)[2][0]
    axis = axis / np.linalg.norm(axis)
    if (np.array(ecd).mean(0) - cen) @ axis > 0:
        axis = -axis
    pts = [np.array([a.pos.x, a.pos.y, a.pos.z])
           for r in binder for a in r if a.element.name != "H"]
    if not pts:
        return None
    return float(max((p - cen) @ axis for p in pts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--out", default="designs/scored.jsonl")
    ap.add_argument("--recycling-steps", type=int, default=3)
    ap.add_argument("--sampling-steps", type=int, default=50)
    ap.add_argument("--n-seeds", type=int, default=1,
                    help="repeat every design with this many sampling seeds; one "
                         "seed cannot say whether a difference between two "
                         "designs is larger than the model's own run-to-run noise")
    ap.add_argument("--model", default="boltz2",
                    choices=["boltz2", "protenix_base"])
    args = ap.parse_args()

    seq = receptor_sequence()
    tmpl = glp1r_template_chain()
    order = [r.seqid.num for r in tmpl]
    core, belt, ep = ACTIVATION_CORE, lipid_facing_belt(), orthosteric_epitope()
    if args.model == "boltz2":
        from mosaic.models.boltz2 import Boltz2 as M
    else:
        from mosaic.models.protenix import ProtenixBase as M
    model = M()
    # Boltz-2 needs an MSA here (epitope RMSD 3.66 vs 7.80 A without);
    # ProtenixBase measured better without one.
    use_msa = args.model == "boltz2"
    key = jax.random.PRNGKey(0)

    rows = []
    designs = [json.loads(l) for f in args.jsonl for l in open(f)]
    print(f"scoring {len(designs)} designs; activation core {len(core)}, "
          f"belt {len(belt)}, epitope {len(ep)}", flush=True)
    jobs = [(d, sd) for d in designs for sd in range(args.n_seeds)]
    for i, (d, sd) in enumerate(jobs):
        b = d["sequence"]
        key = jax.random.fold_in(jax.random.PRNGKey(sd), i)
        key, k = jax.random.split(key)
        feats, writer = model.target_only_features(
            [TargetChain(b, use_msa=False),
             TargetChain(seq, use_msa=use_msa, template_chain=tmpl)])
        pred = model.predict(features=feats, writer=writer,
                             recycling_steps=args.recycling_steps,
                             sampling_steps=args.sampling_steps, key=k)
        # iPTM via the loss term with an explicit binder length: pred.iptm reads
        # 0 for target_only_features, where no chain is the designated binder.
        key, k2 = jax.random.split(key)
        val, _ = sp.IPTMLoss()(jnp.zeros((len(b), 20)), pred.model_output, k2)
        out = dict(d)
        out["seed"] = sd
        out["iptm"] = -float(val)
        out["plddt"] = float(np.asarray(pred.plddt).mean())
        hit_core = contacts(pred.st, len(b), core, order)
        hit_belt = contacts(pred.st, len(b), belt, order)
        hit_ep = contacts(pred.st, len(b), ep, order)
        out["activation_core_contacts"] = len(hit_core)
        out["activation_core_hit"] = sorted(hit_core)
        out["belt_contacts"] = len(hit_belt)
        out["epitope_contacts"] = len(hit_ep)
        depth = insertion_depth(pred.st, order)
        out["insertion_depth"] = depth
        out["reaches_activation_pocket"] = (
            depth is not None and depth > ACTIVATION_DEPTH_THRESHOLD)
        rows.append(out)
        print(f"[{i+1}/{len(jobs)}] {d['name'][:34]:34s} s{sd} iptm {out['iptm']:.3f} "
              f"core {len(hit_core):>2}/{len(core)} belt {len(hit_belt):>2} "
              f"epitope {len(hit_ep):>2}/{len(ep)} "
              f"depth {out['insertion_depth']:+6.1f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {args.out}")
    if rows:
        c = [r["activation_core_contacts"] for r in rows]
        print(f"activation-core contacts: median {np.median(c):.0f}, max {max(c)}")
        print("A design reaching none of the 18 is an antagonist candidate, not an agonist.")


if __name__ == "__main__":
    main()
