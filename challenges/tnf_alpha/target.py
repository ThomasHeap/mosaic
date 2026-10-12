"""TNF-alpha target definition for the Anthropic x Adaptyv 2026 Challenge 02.

Target: human TNF-alpha soluble trimer, UniProt P01375 residues 77-233 (157 aa),
tag-free. Reference structure 1TNF chains A-C.

All residue numbers in this module are 1-indexed on the 157-aa construct below,
which coincides with the mature-protein numbering used by every TNF chain in
1TNF / 3WD5 / 4G3Y / 3ALQ. Regenerate the epitope sets with `compute_epitope.py`.
"""

from pathlib import Path

import gemmi

HUMAN_TNF = (
    "VRSSSRTPSDKPVAHVVANPQAEGQLQWLNRRANALLANGVELRDNQLVVPSEGLYLIYSQVLFKGQGCPSTHVLLTHTI"
    "SRIAVSYQTKVNLLSAIKSPCQRETPEGAEAKPWYEPIYLGGVFQLEKGDRLSAEINRPDYLDFAESGQVYFGIIAL"
)
assert len(HUMAN_TNF) == 157

# Mouse TNF (P06804, mature soluble, 156 aa). 79% identical to human over the
# 156 aligned positions, matching the challenge page's "124 of 156".
MOUSE_TNF = (
    "LRSSSQNSSDKPVAHVVANHQVEEQLEWLSQRANALLANGMDLKDNQLVVPADGLYLVYSQVLFKGQGCPDYVLLTHTVS"
    "RFAISYQEKVNLLSAVKSPCPKDTPEGAELKPWYEPIYLGGVFQLEKGDQLSAEVNLPKYLDFAESGQVYFGVIAL"
)
assert len(MOUSE_TNF) == 156

# --- Epitope ----------------------------------------------------------------
# Computed as TNF residues within 4.5 A of a partner chain in the receptor-site
# complexes 3WD5 (adalimumab Fab), 4G3Y (infliximab Fab) and 3ALQ (TNFR2), then
# intersected. Residues seen in only one complex are dropped as antibody-specific
# or crystal-packing artefacts.

# Contacted in >= 2 of the 3 complexes.
CORE_EPITOPE = [20, 21, 23, 67, 71, 73, 75, 77, 97, 110, 111,
                113, 115, 137, 140, 141, 143, 144, 145, 146]

# Core epitope positions where human and mouse differ. Contacting these buys
# human affinity at the cost of challenge criterion #2 (mouse cross-reactivity).
# P20H, S71D, H73(indel), I97V, A111L, D140K.
MOUSE_DIVERGENT = [20, 71, 73, 97, 111, 140]

# Core epitope positions conserved human -> mouse. Bias contacts here.
CONSERVED_EPITOPE = [p for p in CORE_EPITOPE if p not in MOUSE_DIVERGENT]

# The receptor groove spans two adjacent protomers: in 3ALQ, TNFR2 chain T
# contacts 15 residues on one protomer and 13 on its neighbour. A binder
# designed against a TNF *monomer* is being designed against a surface that does
# not exist in the assay construct, so always co-fold at least a dimer.
GROOVE_PROTOMER_1 = [20, 21, 23, 31, 32, 33, 63, 67, 113, 115, 143, 144, 145, 146, 149]
GROOVE_PROTOMER_2 = [71, 72, 73, 75, 77, 85, 86, 87, 88, 89, 91, 92, 97]

# The only histidine anywhere in the epitope. Sits in the groove, and is lost to
# an indel in mouse -- so do NOT build the pH switch on top of it if you also
# want mouse cross-reactivity. Put the switch histidines on the binder instead.
EPITOPE_HIS = 73


def groove_epitope_idx(n_protomers: int = 2, conserved_only: bool = True) -> list[int]:
    """0-indexed epitope positions into the concatenated target token array.

    `mosaic`'s `binder_features` lays the tokens out as [binder, chain_1, ...,
    chain_n], and `BinderTargetContact.epitope_idx` indexes into the *target*
    region only -- so protomer k contributes offsets k * 157 + (resnum - 1).
    """
    if n_protomers < 2:
        raise ValueError("the receptor groove spans two protomers; use >= 2")
    sides = [GROOVE_PROTOMER_1, GROOVE_PROTOMER_2]
    idx = []
    for k, side in enumerate(sides):
        for resnum in side:
            if conserved_only and resnum in MOUSE_DIVERGENT:
                continue
            idx.append(k * len(HUMAN_TNF) + (resnum - 1))
    return sorted(idx)


def tnf_template_chains(n: int = 2, cif: str | Path | None = None) -> list[gemmi.Chain]:
    """First `n` TNF chains of 1TNF, to pass as `TargetChain.template_chain`.

    The trimer groove geometry is the whole point of the target, so template the
    target chains rather than hoping a co-folding model rebuilds the interface.
    """
    path = Path(cif) if cif is not None else Path(__file__).with_name("1tnf.cif")
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_ligands_and_waters()
    chains = [ch for ch in st[0] if len(ch.get_polymer()) > 100]
    if len(chains) < n:
        raise ValueError(f"{path} has {len(chains)} TNF chains, need {n}")
    return chains[:n]
