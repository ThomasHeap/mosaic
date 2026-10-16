"""Is the binder backbone a designable fold, or only a compact blob?

Every measurement so far has been sequence-level, and all of them came back
negative: monomer pLDDT is low (0.52 median vs 0.88-0.94 for GB1/Top7) but it
is not explained by length (flat across L55-L75 within one pool), by the
pH-switch histidines (paired hisscan variants shift it by +0.001), by a lack of
compactness (1.84 long-range contacts/residue, near-native), or by the target
acting as a hydrophobic core (the binder is *more* compact alone than bound).

What is left is packing specificity: compact but not uniquely folded, i.e. a
molten globule. That is a property of the *backbone*, not the sequence, and the
standard way to test it is inverse folding. Hand ProteinMPNN the binder
backbone as predicted in the complex and ask what sequence it wants there:

  - if MPNN's preferred sequence folds into that backbone, the backbone is a
    real fold and our sequence was simply a poor fit for it -- fixable by
    redesigning sequence onto a fixed backbone.
  - if MPNN's own sequence also fails to fold, the backbone is not a designable
    fold, and no sequence-level loss term can rescue it. The fold would have to
    be specified before design, which is how Top7 (de novo, 93 aa, 0.882) was
    built and is exactly what our hallucination-based objective never does.

Native sequence recovery is the standard readout: designable backbones recover
~40-50% of the native sequence. Low recovery means MPNN disagrees with our
sequence about what belongs on that backbone.
"""
import argparse, csv, glob, json, os
import gemmi, numpy as np
import jax, jax.numpy as jnp
from mosaic.proteinmpnn.mpnn import load_mpnn_sol, MPNN_ALPHABET
from mosaic.losses.protein_mpnn import load_chain

AA = "ACDEFGHIKLMNPQRSTVWY"

def mpnn_logprobs(mpnn, coords, seq, key):
    n = len(seq)
    # The __call__ annotation says "N 23" but W_s expects len(MPNN_ALPHABET)=21.
    S = np.zeros((n, len(MPNN_ALPHABET)))
    for i, c in enumerate(seq):
        S[i, MPNN_ALPHABET.index(c)] = 1.0
    return mpnn(
        X=jnp.asarray(coords), S=jnp.asarray(S),
        mask=jnp.ones(n, dtype=bool),
        residue_idx=jnp.arange(n) + 100,
        chain_encoding_all=jnp.zeros(n, dtype=jnp.int32),
        decoding_order=jax.random.uniform(key, (n,)),
        key=key,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", default="probe/designability.csv")
    args = ap.parse_args()

    rec = {}
    for fn in sorted(glob.glob("designs/rescored*.jsonl")):
        if "merged" in fn: continue
        for line in open(fn):
            d = json.loads(line)
            if d.get("plddt_monomer") is not None:
                rec.setdefault(d["name"], d)
    # Span the pLDDT range rather than sampling the top: the question is whether
    # designability tracks it, so both tails have to be represented.
    have = [d for d in rec.values()
            if os.path.exists(f"structures/{d['name']}_complex.pdb")]
    have.sort(key=lambda d: float(d["plddt_monomer"]))
    step = max(1, len(have) // args.n)
    picked = have[::step][: args.n]

    mpnn = load_mpnn_sol(0.0)
    key = jax.random.PRNGKey(0)
    rows = []
    print(f"{'design':44s} {'plddt':>6} {'recov':>6} {'LL/res':>7}")
    for d in picked:
        st = gemmi.read_structure(f"structures/{d['name']}_complex.pdb")
        st.remove_ligands_and_waters()
        chain = st[0][0]  # binder is the first chain in the prediction
        native, coords = load_chain(chain)
        n = min(len(native), len(d["sequence"]))
        native, coords = native[:n], coords[:n]
        key, k2 = jax.random.split(key)
        lp = np.asarray(mpnn_logprobs(mpnn, coords, native, k2))
        best = "".join(MPNN_ALPHABET[int(i)] for i in np.asarray(lp.argmax(-1)).ravel())
        best = "".join(c if c in AA else "G" for c in best)
        recov = sum(a == b for a, b in zip(native, best)) / n
        ll = float(np.mean([lp[i, MPNN_ALPHABET.index(c)] for i, c in enumerate(native)]))
        print(f"{d['name'][:44]:44s} {float(d['plddt_monomer']):>6.3f} {recov:>6.1%} {ll:>7.3f}")
        rows.append({"name": f"mpnnredesign_{d['name']}"[:60], "sequence": best,
                     "molecule_class": "single_chain",
                     "parent": d["name"], "parent_plddt": d["plddt_monomer"],
                     "recovery": f"{recov:.3f}", "ll_per_res": f"{ll:.3f}"})
    os.makedirs("probe", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["name", "sequence", "molecule_class"])
        for r in rows: w.writerow([r["name"], r["sequence"], r["molecule_class"]])
    with open(args.out.replace(".csv", ".jsonl"), "w") as fh:
        for r in rows: fh.write(json.dumps(r) + "\n")
    rv = [float(r["recovery"]) for r in rows]
    print(f"\nnative sequence recovery: median {np.median(rv):.1%} "
          f"(designable backbones typically 40-50%)")
    print(f"wrote {args.out} -- fold these to test whether the backbone is realizable")

if __name__ == "__main__":
    main()
