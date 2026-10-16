"""Can Boltz-2 model GLP1R at all? Run before porting the design objective.

TNF was a templated 314-residue dimer of a soluble globular protein. GLP1R is a
384-residue transmembrane receptor, modelled without a membrane, whose epitope
is partly a deep pocket. Three things could make the plan unworkable and none of
them depend on the design loss:

  1. memory -- target plus binder may not fit on one A100
  2. time -- TNF ran ~2000 s per design; several times that makes a fleet
     infeasible inside the deadline
  3. fidelity -- if the predicted receptor does not keep the template's
     transmembrane bundle, the pocket we are designing into does not exist

Reports all three. Fidelity is C-alpha RMSD of the predicted receptor against
the 6x18 template, split into ECD, TMD and the orthosteric epitope, since the
epitope is the part that has to be right.
"""
import argparse, time
import numpy as np
import jax
from mosaic.models.boltz2 import Boltz2
from mosaic.models.protenix import ProtenixMini, ProtenixBase
from mosaic.structure_prediction import TargetChain
from target import (receptor_sequence, glp1r_template_chain, orthosteric_epitope,
                    lipid_facing_belt, ECD_MAX)

def ca_by_resnum(chain):
    out = {}
    for r in chain:
        a = r.find_atom("CA", "*")
        if a:
            out[r.seqid.num] = np.array([a.pos.x, a.pos.y, a.pos.z])
    return out

def rmsd(P, Q):
    P, Q = np.asarray(P), np.asarray(Q)
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    return float(np.sqrt(((Pc @ (U @ D @ Vt) - Qc) ** 2).sum(-1).mean()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binder-length", type=int, default=40)
    ap.add_argument("--recycling-steps", type=int, default=1)
    ap.add_argument("--sampling-steps", type=int, default=25)
    # Protenix shares Boltz-2's interface (target_only_features / build_loss /
    # predict) so it is drop-in. ProtenixMini samples in 2 steps against 20 for
    # the base models, which on a 384-residue target may decide whether a fleet
    # fits the deadline. It is also an independent second opinion for ranking:
    # neither model represents a membrane, so they share that blind spot but not
    # their individual artefacts.
    ap.add_argument("--model", default="boltz2",
                    choices=["boltz2", "protenix_mini", "protenix_base"])
    args = ap.parse_args()

    seq = receptor_sequence()
    tmpl = glp1r_template_chain()
    ep, belt = orthosteric_epitope(), lipid_facing_belt()
    print(f"receptor {len(seq)} aa, binder {args.binder_length} aa, "
          f"total {len(seq)+args.binder_length}", flush=True)
    print(f"epitope {len(ep)} residues, lipid belt {len(belt)}", flush=True)

    model = {"boltz2": Boltz2, "protenix_mini": ProtenixMini,
             "protenix_base": ProtenixBase}[args.model]()
    print(f"model {args.model}", flush=True)
    chains = [TargetChain(seq, use_msa=True, template_chain=tmpl)]
    # A generic helical sequence stands in for a design; this measures the
    # target's cost, not any particular binder's quality.
    binder = "EELKKLEEELKKLEEELKKLEEELKKLEEELKKLEEELKK"[: args.binder_length]

    t0 = time.time()
    feats, writer = model.target_only_features(
        [TargetChain(binder, use_msa=False)] + chains)
    t_feat = time.time() - t0
    print(f"featurisation {t_feat:.1f}s", flush=True)

    t0 = time.time()
    pred = model.predict(feats, writer=writer,
                         recycling_steps=args.recycling_steps,
                         sampling_steps=args.sampling_steps,
                         key=jax.random.PRNGKey(0))
    t_pred = time.time() - t0
    print(f"prediction {t_pred:.1f}s  (recycling={args.recycling_steps}, "
          f"sampling={args.sampling_steps})", flush=True)

    plddt = float(np.asarray(pred.plddt).mean())
    print(f"mean pLDDT {plddt:.3f}", flush=True)

    # fidelity: predicted receptor vs template, by region
    tc = ca_by_resnum(tmpl)
    order = [r.seqid.num for r in tmpl]
    st = pred.st
    pc = {}
    for ch in st[0]:
        nums = [r.seqid.num for r in ch]
        if len(nums) < 100:      # the binder chain
            continue
        for i, r in enumerate(ch):
            a = r.find_atom("CA", "*")
            if a and i < len(order):
                pc[order[i]] = np.array([a.pos.x, a.pos.y, a.pos.z])
    def pick(sel):
        k = [n for n in sel if n in pc and n in tc]
        return ([pc[n] for n in k], [tc[n] for n in k], len(k))
    for label, sel in (("all receptor", order),
                       ("ECD", [n for n in order if n <= ECD_MAX]),
                       ("TMD", [n for n in order if n > ECD_MAX]),
                       ("orthosteric epitope", ep)):
        P, Q, n = pick(sel)
        if n >= 4:
            print(f"  CA RMSD {label:22s} {rmsd(P, Q):6.2f} A  (n={n})", flush=True)
    print("\nverdict inputs: time above vs TNF's ~2000 s per design; epitope RMSD "
          "is the number that matters, since that is the surface being designed into.")

if __name__ == "__main__":
    main()
