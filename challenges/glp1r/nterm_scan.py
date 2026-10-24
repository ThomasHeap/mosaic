"""Second-stage refinement of the N-terminal residues on finished scaffolds.

The same shape as the TNF histidine scan, which produced 87% of that
submission's all-rounders: take a binder that already works, then vary only the
residues responsible for the functional property, on a fixed backbone.

Here the functional residues are the N-terminal ones. Design-time evidence says
they are what threads: without the N-terminal term the first five residues point
outward and the body sits deeper (depth delta -12 to -22 A), and with it all
lengths flip positive, +2.6 to +5.1 against +6.7 for GLP-1.

Variants are proposed by ProteinMPNN on the predicted complex, i.e. "given this
threaded backbone, what does a soluble-protein model want at these positions".
That is inverse folding on our own design, not grafting from a known agonist, so
it stays inside the de novo rules. Unlike the TNF scan this is not free: the
backbone is fixed but each variant still needs re-prediction to score, so the
emitted set is deliberately small.
"""
import argparse, json, sys
from pathlib import Path

import gemmi, jax, jax.numpy as jnp, numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mosaic.losses.protein_mpnn import load_chain
from mosaic.proteinmpnn.mpnn import MPNN_ALPHABET, load_mpnn_sol
from mosaic.structure_prediction import TargetChain
from target import glp1r_template_chain, receptor_sequence

AA = "ACDEFGHIKLMNPQRSTVWY"


def mpnn_logprobs(mpnn, coords, seq, key):
    n = len(seq)
    # S is (N, len(MPNN_ALPHABET)); the __call__ annotation says 23 but W_s
    # expects 21, which cost a job during the TNF work.
    S = np.zeros((n, len(MPNN_ALPHABET)))
    for i, c in enumerate(seq):
        S[i, MPNN_ALPHABET.index(c if c in MPNN_ALPHABET else "X")] = 1.0
    out = mpnn(X=jnp.asarray(coords), S=jnp.asarray(S),
               mask=jnp.ones(n, dtype=bool),
               residue_idx=jnp.arange(n) + 100,
               chain_encoding_all=jnp.zeros(n, dtype=jnp.int32),
               decoding_order=jax.random.uniform(key, (n,)), key=key)
    return np.asarray(out).reshape(-1, len(MPNN_ALPHABET))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--top-n", type=int, default=5, help="scaffolds to refine")
    ap.add_argument("--nterm-len", type=int, default=5)
    ap.add_argument("--variants", type=int, default=16, help="per scaffold")
    ap.add_argument("--out", default="designs_nterm/variants.jsonl")
    a = ap.parse_args()

    from mosaic.models.protenix import ProtenixBase
    model = ProtenixBase()
    mpnn = load_mpnn_sol(0.0)
    rseq, rtmpl = receptor_sequence(), glp1r_template_chain()
    rows = [json.loads(l) for l in open(a.jsonl)][: a.top_n]
    key = jax.random.PRNGKey(0)
    out = []

    for i, d in enumerate(rows):
        seq = d["sequence"]
        key, k = jax.random.split(key)
        feats, writer = model.target_only_features(
            [TargetChain(seq, use_msa=False),
             TargetChain(rseq, use_msa=False, template_chain=rtmpl)])
        pred = model.predict(features=feats, writer=writer, recycling_steps=3,
                             sampling_steps=50, key=k)
        tmp = Path(a.out).parent / f"_{d['name']}_cplx.pdb"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        pred.st.write_pdb(str(tmp))
        st = gemmi.read_structure(str(tmp)); st.setup_entities(); st.remove_ligands_and_waters()
        binder = st[0][0]
        native, coords = load_chain(binder)
        n = min(len(native), len(seq))
        key, k2 = jax.random.split(key)
        lp = mpnn_logprobs(mpnn, coords[:n], native[:n], k2)
        # sample the N-terminal positions from MPNN's distribution
        made = set()
        for v in range(a.variants * 4):
            if len(made) >= a.variants:
                break
            key, k3 = jax.random.split(key)
            new = list(seq)
            for p in range(a.nterm_len):
                probs = np.exp(lp[p]); probs = probs / probs.sum()
                idx = int(jax.random.choice(jax.random.fold_in(k3, p),
                                            len(probs), p=jnp.asarray(probs)))
                c = MPNN_ALPHABET[idx]
                if c in AA:
                    new[p] = c
            s2 = "".join(new)
            if s2 != seq and s2 not in made:
                made.add(s2)
        for j, s2 in enumerate(sorted(made)):
            nd = sum(1 for x, y in zip(s2[:a.nterm_len], seq[:a.nterm_len]) if x != y)
            out.append({"name": f"{d['name']}_nt{j}", "sequence": s2,
                        "parent": d["name"], "nterm_changes": nd})
        print(f"[{i+1}/{len(rows)}] {d['name'][:30]:30s} -> {len(made)} variants", flush=True)

    with open(a.out, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(out)} variants to {a.out}")


if __name__ == "__main__":
    main()
