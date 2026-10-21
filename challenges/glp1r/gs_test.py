"""Does the Gs interface survive with a binder bound? A functional readout.

Every other metric here is a structural correlate of activation. This one tests
the functional output directly: agonism is G-protein coupling, so co-fold the
binder, the receptor and Gs-alpha and ask whether the receptor-Gs interface
still forms.

Gs-alpha (6x18 chain A, 355 aa) makes 29 receptor contacts; the beta and gamma
chains make 5 and 0, so only alpha is modelled. The readout is the count of
receptor residues contacting Gs-alpha, against 29 in the crystal, plus the
predicted interface confidence.

Calibrated the same way as everything else: pairs that both bind and differ only
in activation. If agonists support the Gs interface and their truncated
antagonists do not, the metric measures agonism. If both support it equally, it
does not, and a null result means the model is insensitive rather than that the
designs are agonists.
"""
import argparse, json, sys
from pathlib import Path

import gemmi, jax, numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mosaic.structure_prediction import TargetChain
from target import glp1r_template_chain, receptor_sequence

CONTACT = 4.5
CRYSTAL_GS_CONTACTS = 29


def gsa_template(cif=None):
    st = gemmi.read_structure(str(cif or Path(__file__).parent / "structures" / "6x18.cif"))
    st.setup_entities(); st.remove_ligands_and_waters()
    return {c.name: c for c in st[0]}["A"]


def chain_seq(ch):
    return gemmi.one_letter_code([r.name for r in ch])


def interface_contacts(st, idx_a, idx_b):
    """Residues of chain idx_a within CONTACT of any heavy atom of chain idx_b."""
    chains = list(st[0])
    if max(idx_a, idx_b) >= len(chains):
        return None
    def heavy(ch):
        return [(i, np.array([a.pos.x, a.pos.y, a.pos.z]))
                for i, r in enumerate(ch) for a in r if a.element.name != "H"]
    A, B = heavy(chains[idx_a]), heavy(chains[idx_b])
    if not A or not B:
        return None
    AX = np.array([x for _, x in A]); BX = np.array([x for _, x in B])
    D = np.linalg.norm(AX[:, None] - BX[None, :], axis=-1)
    return len({A[i][0] for i in range(len(A)) if (D[i] < CONTACT).any()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="*", help="designs; omit to run calibration only")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--recycling-steps", type=int, default=3)
    ap.add_argument("--sampling-steps", type=int, default=50)
    ap.add_argument("--out", default="probe/gs_test.jsonl")
    args = ap.parse_args()

    from mosaic.models.protenix import ProtenixBase
    model = ProtenixBase()
    rseq = receptor_sequence(); rtmpl = glp1r_template_chain()
    gtmpl = gsa_template(); gseq = chain_seq(gtmpl)
    order = [r.seqid.num for r in rtmpl]
    print(f"receptor {len(rseq)} aa + Gs-alpha {len(gseq)} aa; crystal interface "
          f"{CRYSTAL_GS_CONTACTS} receptor residues", flush=True)

    CAL = [
        ("GLP1_7_37", "agonist", "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG"),
        ("GLP1_9_37", "antagonist", "EGTFTSDVSSYLEGQAAKEFIAWLVKGRG"),
        ("exendin4", "agonist", "HGEGTFTSDLSKQMEEEAVRLFIEWLKNGGPSSGAPPPS"),
        ("exendin_9_39", "antagonist", "DLSKQMEEEAVRLFIEWLKNGGPSSGAPPPS"),
        ("no_binder", "apo_control", None),
    ]
    jobs = [(n, k, s) for n, k, s in CAL]
    for f in args.jsonl:
        for line in open(f):
            d = json.loads(line)
            jobs.append((d["name"], "design", d["sequence"]))

    rows = []
    for i, (name, kind, seq) in enumerate(jobs):
        for sd in range(args.n_seeds):
            key = jax.random.fold_in(jax.random.PRNGKey(sd), i)
            chains = []
            if seq is not None:
                chains.append(TargetChain(seq, use_msa=False))
            chains += [TargetChain(rseq, use_msa=False, template_chain=rtmpl),
                       TargetChain(gseq, use_msa=False, template_chain=gtmpl)]
            feats, writer = model.target_only_features(chains)
            pred = model.predict(features=feats, writer=writer,
                                 recycling_steps=args.recycling_steps,
                                 sampling_steps=args.sampling_steps, key=key)
            # chain order follows `chains`: receptor and Gs shift by one when a
            # binder is present.
            ri = 1 if seq is not None else 0
            gs = interface_contacts(pred.st, ri, ri + 1)
            out = dict(name=name, kind=kind, seed=sd,
                       gs_contacts=gs, plddt=float(np.asarray(pred.plddt).mean()))
            rows.append(out)
            print(f"[{len(rows)}/{len(jobs)*args.n_seeds}] {name[:28]:28s} s{sd} "
                  f"Gs-interface {gs}/{CRYSTAL_GS_CONTACTS}  pLDDT {out['plddt']:.3f}",
                  flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
