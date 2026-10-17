"""Does ProteinMPNN actually disfavour histidine at the paratope?

The paratope_mpnn.py term rests on the claim that soluble ProteinMPNN, given a
binding interface, will not propose histidine there, and so outbids the pH
terms at weight 10.0. That claim was inferred from where histidines ended up,
not measured. This measures it: run MPNN on the predicted complex and compare
P(His) at interface positions against non-interface positions of the same
binder.
"""
import json, sys
import numpy as np
import gemmi, jax, jax.numpy as jnp
from mosaic.proteinmpnn.mpnn import load_mpnn_sol, MPNN_ALPHABET
from mosaic.losses.protein_mpnn import load_chain

HIS = MPNN_ALPHABET.index("H")

def logprobs(mpnn, coords, seq, key):
    n = len(seq)
    S = np.zeros((n, len(MPNN_ALPHABET)))
    for i, c in enumerate(seq):
        S[i, MPNN_ALPHABET.index(c) if c in MPNN_ALPHABET else MPNN_ALPHABET.index("X")] = 1.0
    return np.asarray(mpnn(
        X=jnp.asarray(coords), S=jnp.asarray(S), mask=jnp.ones(n, dtype=bool),
        residue_idx=jnp.arange(n) + 100,
        chain_encoding_all=jnp.zeros(n, dtype=jnp.int32),
        decoding_order=jax.random.uniform(key, (n,)), key=key)).reshape(-1, len(MPNN_ALPHABET))

def main():
    scan = {}
    for fn in ("designs/hisscan_fleet2.jsonl", "designs/hisscan_f4.jsonl"):
        try:
            for l in open(fn):
                d = json.loads(l); scan[d["name"]] = d
        except FileNotFoundError:
            pass
    mpnn = load_mpnn_sol(0.0)
    key = jax.random.PRNGKey(0)
    inside, outside, n_done = [], [], 0
    for name, s in scan.items():
        import os
        pdb = f"structures/{name}_complex.pdb"
        if not os.path.exists(pdb) or not s["interface_positions"]:
            continue
        st = gemmi.read_structure(pdb); st.setup_entities(); st.remove_ligands_and_waters()
        # Binder is the first chain; score it in isolation so the comparison is
        # between positions of one chain, not between chains.
        seq, coords = load_chain(st[0][0])
        n = min(len(seq), len(s["parent_sequence"]))
        if n < 10:
            continue
        key, k2 = jax.random.split(key)
        lp = logprobs(mpnn, coords[:n], seq[:n], k2)
        p_his = np.exp(lp[:, HIS])
        iface = {i for i in s["interface_positions"] if i < n}
        if not iface or len(iface) == n:
            continue
        inside.extend(p_his[i] for i in iface)
        outside.extend(p_his[i] for i in range(n) if i not in iface)
        n_done += 1
        if n_done >= 40:
            break
    print(f"designs measured: {n_done}")
    print(f"  P(His) at interface positions:     mean {np.mean(inside):.4f}  median {np.median(inside):.4f}  n={len(inside)}")
    print(f"  P(His) at non-interface positions: mean {np.mean(outside):.4f}  median {np.median(outside):.4f}  n={len(outside)}")
    r = np.mean(inside) / max(np.mean(outside), 1e-9)
    print(f"  ratio interface/non-interface: {r:.2f}")
    print(f"  natural histidine frequency is about 0.023 for reference")

if __name__ == "__main__":
    main()
