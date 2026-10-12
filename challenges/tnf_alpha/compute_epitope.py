"""TNF-alpha epitope on the 157-aa challenge construct, split by protomer.

Human complexes (3WD5 adalimumab, 4G3Y infliximab, 3ALQ TNFR2) all number their
TNF chains 1..157 on the mature soluble sequence, so contacts map directly.
Mouse conservation comes from a sequence alignment of P01375(77-233) vs P06804 mature.
"""

import sys
from collections import defaultdict
from pathlib import Path

import gemmi

DIR = Path(sys.argv[1])
CUT = 4.5

HUMAN = (
    "VRSSSRTPSDKPVAHVVANPQAEGQLQWLNRRANALLANGVELRDNQLVVPSEGLYLIYSQVLFKGQGCPSTHVLLTHTI"
    "SRIAVSYQTKVNLLSAIKSPCQRETPEGAEAKPWYEPIYLGGVFQLEKGDRLSAEINRPDYLDFAESGQVYFGIIAL"
)
assert len(HUMAN) == 157

# human TNF chains in each complex -> partner description
HUMAN_COMPLEXES = {
    "3WD5": ("adalimumab", {"A"}),
    "4G3Y": ("infliximab", {"C"}),
    "3ALQ": ("TNFR2", {"A", "B", "C", "D", "E", "F"}),
}


def aa1(res):
    info = gemmi.find_tabulated_residue(res.name)
    return info.one_letter_code.upper() if info else "X"


def mouse_mature():
    txt = (DIR / "p06804.fasta").read_text().splitlines()
    full = "".join(l.strip() for l in txt if not l.startswith(">"))
    return full[79:]  # mature soluble mouse TNF, residues 80..end


def conservation_map():
    """construct pos (1..157) -> (human aa, mouse aa, conserved?)"""
    mouse = mouse_mature()
    sc = gemmi.AlignmentScoring()
    aln = gemmi.align_string_sequences(list(HUMAN), list(mouse), [], sc)
    out = {}
    i = j = 0
    for c in aln.cigar_str():
        pass
    # walk the alignment via match string
    h_al, m_al = [], []
    # gemmi exposes add_gaps to render each sequence with gaps
    h_str = aln.add_gaps("".join(HUMAN), 1)
    m_str = aln.add_gaps("".join(mouse), 2)
    for a, b in zip(h_str, m_str):
        if a != "-" and b != "-":
            out[i + 1] = (a, b, a == b)
        if a != "-":
            i += 1
        if b != "-":
            j += 1
    ident = sum(1 for v in out.values() if v[2])
    print(f"human vs mouse mature TNF: {ident}/{len(out)} aligned identities "
          f"({100 * ident / len(out):.0f}%), mouse len={len(mouse)}")
    return out


def main():
    cons = conservation_map()

    # (pdb, partner) -> {tnf_chain: set(construct positions)}
    per_complex = {}
    # construct pos -> set of pdb ids
    seen = defaultdict(set)
    # for 3ALQ: how the footprint splits across the two protomers of a groove
    groove_split = None

    for pdb_id, (partner, tnf_chain_names) in HUMAN_COMPLEXES.items():
        st = gemmi.read_structure(str(DIR / f"{pdb_id}.cif"))
        st.setup_entities()
        st.remove_ligands_and_waters()
        model = st[0]
        ns = gemmi.NeighborSearch(st, 5.0).populate()

        contacts = defaultdict(set)  # tnf chain -> positions
        # also per (tnf chain, partner chain) for the groove analysis
        pairwise = defaultdict(set)

        for chain in model:
            if chain.name not in tnf_chain_names:
                continue
            for res in chain.get_polymer():
                n = res.seqid.num
                if not (1 <= n <= 157):
                    continue
                if aa1(res) != HUMAN[n - 1]:
                    continue  # engineered/mutant position, skip
                for atom in res:
                    for mark in ns.find_atoms(atom.pos, "\0", radius=CUT):
                        cra = mark.to_cra(model)
                        if cra.chain.name in tnf_chain_names:
                            continue
                        if cra.atom.pos.dist(atom.pos) <= CUT:
                            contacts[chain.name].add(n)
                            pairwise[(chain.name, cra.chain.name)].add(n)

        per_complex[pdb_id] = (partner, contacts)
        for positions in contacts.values():
            for n in positions:
                seen[n].add(pdb_id)

        print(f"\n### {pdb_id} ({partner})")
        for ch in sorted(contacts):
            ps = sorted(contacts[ch])
            print(f"  chain {ch}: {len(ps)} residues {ps}")

        if pdb_id == "3ALQ":
            # pick the receptor chain with the largest two-protomer footprint
            by_receptor = defaultdict(dict)
            for (tnf_c, rec_c), ps in pairwise.items():
                by_receptor[rec_c][tnf_c] = ps
            best = max(by_receptor.items(), key=lambda kv: sum(len(v) for v in kv[1].values()))
            rec_c, sides = best
            ordered = sorted(sides.items(), key=lambda kv: -len(kv[1]))
            print(f"  -> groove for receptor chain {rec_c}:")
            for tnf_c, ps in ordered:
                print(f"       TNF {tnf_c}: {len(ps)} res {sorted(ps)}")
            groove_split = [(c, sorted(p)) for c, p in ordered]

    print("\n\n=== Consensus epitope (human construct numbering) ===")
    print(f"{'pos':>4} {'aa':>2} {'n':>2}  {'mouse':>5}  complexes")
    core, core_cons = [], []
    for n in sorted(seen):
        h, m, same = cons.get(n, ("?", "?", False))
        tag = "same" if same else f"->{m}"
        marker = ""
        if len(seen[n]) >= 2:
            core.append(n)
            marker = " *"
            if same:
                core_cons.append(n)
        print(f"{n:>4} {HUMAN[n-1]:>2} {len(seen[n]):>2}  {tag:>5}  {','.join(sorted(seen[n]))}{marker}")

    print(f"\nCore epitope (>=2 complexes), {len(core)} residues:")
    print(f"  1-indexed: {core}")
    print(f"  sequence : {''.join(HUMAN[n-1] for n in core)}")
    print(f"\nCore AND conserved in mouse, {len(core_cons)} residues:")
    print(f"  1-indexed: {core_cons}")
    print(f"  sequence : {''.join(HUMAN[n-1] for n in core_cons)}")

    if groove_split:
        print("\n=== Two-protomer groove (use this for epitope_idx) ===")
        print("Target modelled as TNF trimer, chains concatenated in order P1,P2,P3.")
        print("0-indexed into a 157*3 concatenated target:")
        for k, (ch, ps) in enumerate(groove_split[:2]):
            offs = k * 157
            idx = [offs + (n - 1) for n in ps]
            print(f"  protomer {k+1} (xtal chain {ch}): {idx}")
        both = []
        for k, (ch, ps) in enumerate(groove_split[:2]):
            both += [k * 157 + (n - 1) for n in ps]
        print(f"\nepitope_idx = {sorted(both)}")

    his = [i + 1 for i, c in enumerate(HUMAN) if c == "H"]
    his_epi = [n for n in his if n in seen]
    print(f"\nHis in construct: {his}; of those in the epitope: {his_epi}")


if __name__ == "__main__":
    main()
