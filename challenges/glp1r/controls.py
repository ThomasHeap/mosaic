"""Does the active-minus-inactive score separate known GLP1R agonists?

Agonism cannot be predicted directly, so the plan rests on a proxy: a binder
complementary to the active receptor conformation stabilises it, which is what
agonism is thermodynamically. The proxy is worth nothing until it is shown to
rank known agonists above known non-binders, and that check comes before any
design compute.

Two details make the comparison fair.

The active and inactive structures are different constructs. 6x18 chain R is
384 residues including the extracellular domain; 5vew chain A is 263 after its
ICL3 fusion partner is stripped, transmembrane only, and carries ten
thermostabilising mutations relative to 6x18, four of them inside the
transmembrane epitope (193, 196, 225, 233). Scoring the two as deposited would
confound conformation with construct and with sequence. Both states therefore
use the *active* sequence over the 258 residues resolved in both, and differ
only in which coordinates are supplied as the template.

Affinity is scored separately on the full active receptor, because the
extracellular domain is where the affinity half of the two-domain mechanism
lives and a transmembrane-only construct would understate binding for every
ligand including the real ones.
"""
import argparse, json, sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import mosaic.losses.structure_prediction as sp
from mosaic.structure_prediction import TargetChain
from target import (chain_sequence, glp1r_template_chain,
                    inactive_template_chain, receptor_sequence,
                    restrict_chain, two_state_common_residues)

# GLP-1 7-37 and exendin-4 are the reference agonists. Nothing here is ever a
# design seed -- these exist to calibrate the score, which the rules explicitly
# permit ("you may take known binders and use them to evaluate/calibrate design
# filtering metrics").
LIGANDS = {
    "GLP1_7_37":   ("agonist",  "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG"),
    "exendin4":    ("agonist",  "HGEGTFTSDLSKQMEEEAVRLFIEWLKNGGPSSGAPPPS"),
    # N-terminally truncated GLP-1 binds the ECD but does not activate: the
    # two-domain mechanism says the N-terminus is what engages the bundle.
    # A proxy that cannot separate this from intact GLP-1 is not measuring
    # agonism, only binding.
    "GLP1_9_37":   ("binder_not_agonist", "EGTFTSDVSSYLEGQAAKEFIAWLVKGRG"),
    "scrambled":   ("negative", "GTSEAFKVLSYTGQEAWSDLIFTGAVEKRH"),
    "polyA":       ("negative", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
}


def ranking_scores(pred, binder_len, key):
    """iPTM via the loss term, as rescore.py does it.

    `pred.iptm` reads 0 for every ligand: the attribute is populated for a
    designed-binder prediction, not for one built from target_only_features
    where no chain is designated the binder. The loss term takes the binder
    length explicitly, which is what fixes the chain split.
    """
    out = pred.model_output
    seq_dummy = jnp.zeros((binder_len, 20))
    scores = {}
    for name, term in (("iptm", sp.IPTMLoss()),
                       ("ipsae_bt", sp.BinderTargetIPSAE())):
        try:
            value, _ = term(seq_dummy, out, key)
            scores[name] = -float(value)
        except Exception as exc:  # noqa: BLE001
            scores[name] = None
            scores[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
    return scores


def predict(model, chains, *, recycling_steps, sampling_steps, key):
    feats, writer = model.target_only_features(chains)
    return model.predict(features=feats, writer=writer,
                         recycling_steps=recycling_steps,
                         sampling_steps=sampling_steps, key=key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="protenix_base",
                    choices=["protenix_base", "protenix_mini", "boltz2"])
    ap.add_argument("--recycling-steps", type=int, default=3)
    ap.add_argument("--sampling-steps", type=int, default=50)
    ap.add_argument("--out", default="probe/controls.jsonl")
    args = ap.parse_args()

    if args.model == "boltz2":
        from mosaic.models.boltz2 import Boltz2 as M
    elif args.model == "protenix_mini":
        from mosaic.models.protenix import ProtenixMini as M
    else:
        from mosaic.models.protenix import ProtenixBase as M
    model = M()

    common = two_state_common_residues()
    active_full = glp1r_template_chain()
    active_cut = restrict_chain(active_full, common)
    inactive_cut = restrict_chain(inactive_template_chain(), common)
    # Same sequence for both states; only the coordinates differ.
    two_state_seq = chain_sequence(active_cut)
    full_seq = receptor_sequence()
    print(f"full receptor {len(full_seq)} aa; two-state region {len(two_state_seq)} aa",
          flush=True)

    key = jax.random.PRNGKey(0)
    rows = []
    for name, (kind, seq) in LIGANDS.items():
        out = {"name": name, "kind": kind, "length": len(seq)}
        for label, tseq, tmpl in (
            ("affinity_full", full_seq, active_full),
            ("active", two_state_seq, active_cut),
            ("inactive", two_state_seq, inactive_cut),
        ):
            key, k = jax.random.split(key)
            pred = predict(
                model,
                [TargetChain(seq, use_msa=False),
                 TargetChain(tseq, use_msa=True, template_chain=tmpl)],
                recycling_steps=args.recycling_steps,
                sampling_steps=args.sampling_steps, key=k)
            out[f"plddt_{label}"] = float(np.asarray(pred.plddt).mean())
            key, ks = jax.random.split(key)
            sc = ranking_scores(pred, len(seq), ks)
            out[f"iptm_{label}"] = sc["iptm"]
            out[f"ipsae_{label}"] = sc.get("ipsae_bt")
            if sc.get("iptm") is None:
                out[f"iptm_{label}_error"] = sc.get("iptm_error")
        if out.get("iptm_active") is None or out.get("iptm_inactive") is None:
            out["agonism_delta"] = None
        else:
            out["agonism_delta"] = out["iptm_active"] - out["iptm_inactive"]
        fmt = lambda v: "   None" if v is None else f"{v:+.3f}"
        print(f"  {name:20s} {kind:20s} "
              f"affinity {fmt(out['iptm_affinity_full'])}  "
              f"active {fmt(out['iptm_active'])}  inactive {fmt(out['iptm_inactive'])}  "
              f"delta {fmt(out['agonism_delta'])}", flush=True)
        rows.append(out)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    ag = [r["agonism_delta"] for r in rows
          if r["kind"] == "agonist" and r["agonism_delta"] is not None]
    ng = [r["agonism_delta"] for r in rows
          if r["kind"] == "negative" and r["agonism_delta"] is not None]
    if not ag or not ng:
        print("\nscores missing; see *_error fields in the output")
        return
    print(f"\nagonists delta {min(ag):+.3f}..{max(ag):+.3f}   "
          f"negatives delta {min(ng):+.3f}..{max(ng):+.3f}")
    print("The proxy is usable only if the agonists sit clearly above the "
          "negatives, and ideally above GLP1_9_37, which binds without activating.")


if __name__ == "__main__":
    main()
