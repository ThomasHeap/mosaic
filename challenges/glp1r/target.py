"""GLP1R target definition: sequence, template, orthosteric epitope, lipid belt.

Residue numbers are UniProt P43220-1 throughout, which is also the assay
construct numbering (24-432). 6x18 chain R resolves 29-423 with internal gaps at
130-135 and 339-343, so indices from it need no mapping.
"""
from pathlib import Path
import gemmi
import numpy as np

CIF = Path(__file__).with_name("structures") / "6x18.cif"
ECD_MAX = 145          # ECD is approx 24-145, TMD 146-432
CONTACT_CUTOFF = 4.5   # heavy-atom, as used for the TNF epitope


def _receptor_and_peptide(cif: Path | None = None):
    st = gemmi.read_structure(str(cif or CIF))
    st.setup_entities()
    st.remove_ligands_and_waters()
    chains = {c.name: c for c in st[0]}
    return chains["R"], chains["P"]   # R = GLP1R, P = GLP-1


def receptor_sequence(cif: Path | None = None) -> str:
    """Resolved receptor sequence, 29-423.

    The assay construct is 24-432; the unresolved termini and the two internal
    gaps are disordered and carry no epitope, so the resolved span is used and
    the difference is noted rather than modelled.
    """
    R, _ = _receptor_and_peptide(cif)
    return gemmi.one_letter_code(R.get_polymer().extract_sequence())


def glp1r_template_chain(cif: Path | None = None) -> gemmi.Chain:
    """Chain R of 6x18, the active-state receptor, for `template_chain`.

    The active conformation is the point: a binder complementary to it
    stabilises it, which is what agonism requires. Co-folding without a template
    would also have to invent a transmembrane bundle with no membrane present.
    """
    R, _ = _receptor_and_peptide(cif)
    return R


def _heavy(chain):
    out = []
    for r in chain:
        for a in r:
            if a.element.name != "H":
                out.append((r.seqid.num, np.array([a.pos.x, a.pos.y, a.pos.z])))
    return out


def orthosteric_epitope(cif: Path | None = None, domain: str = "all") -> list[int]:
    """Receptor residues within 4.5 A of GLP-1 in 6x18.

    `domain` selects "ecd", "tmd" or "all". The split matters: the ECD half is
    the affinity trap, while the TMD half is what activation requires, so a
    design restricted to the ECD is an antagonist.
    """
    R, P = _receptor_and_peptide(cif)
    ra, pa = _heavy(R), _heavy(P)
    RX = np.array([x for _, x in ra])
    PX = np.array([x for _, x in pa])
    D = np.linalg.norm(RX[:, None] - PX[None, :], axis=-1)
    hit = sorted({ra[i][0] for i in range(len(ra)) if (D[i] < CONTACT_CUTOFF).any()})
    if domain == "ecd":
        return [n for n in hit if n <= ECD_MAX]
    if domain == "tmd":
        return [n for n in hit if n > ECD_MAX]
    return hit


def lipid_facing_belt(cif: Path | None = None) -> list[int]:
    """Transmembrane residues whose side chains point out into the lipid.

    These must be excluded, not merely left unrewarded. The belt is 65 residues
    at 72% hydrophobic against the epitope's 52 at 52%, so it is the larger and
    stickier surface, and Boltz-2 models no membrane. A binder here would score
    on affinity, measured on the detergent-solubilised construct where the belt
    is exposed, and fail activation, since in a cell it is buried in lipid.

    Identified geometrically: the central 60% of the transmembrane span along the
    bundle axis, keeping residues more than 8 A off-axis whose CA->CB vector
    points away from it.
    """
    R, _ = _receptor_and_peptide(cif)
    ca = {}
    for r in R:
        a = r.find_atom("CA", "*")
        if a:
            ca[r.seqid.num] = np.array([a.pos.x, a.pos.y, a.pos.z])
    tm = np.array([v for k, v in ca.items() if k > ECD_MAX])
    cen = tm.mean(0)
    axis = np.linalg.svd(tm - cen)[2][0]
    axis = axis / np.linalg.norm(axis)
    proj = {k: float((v - cen) @ axis) for k, v in ca.items()}
    span = [proj[k] for k in ca if k > ECD_MAX]
    lo, hi = np.percentile(span, 20), np.percentile(span, 80)
    byname = {r.seqid.num: r for r in R}
    belt = []
    for k, v in ca.items():
        if k <= ECD_MAX or not (lo <= proj[k] <= hi):
            continue
        cb = byname[k].find_atom("CB", "*")
        if cb is None:
            continue
        cbv = np.array([cb.pos.x, cb.pos.y, cb.pos.z])
        radial = (v - cen) - ((v - cen) @ axis) * axis
        nr = np.linalg.norm(radial)
        if nr <= 8.0:
            continue
        if (cbv - v) @ (radial / nr) > 0.3:
            belt.append(k)
    return sorted(belt)


def to_target_index(resnums, seq_start: int | None = None, cif: Path | None = None):
    """Map UniProt residue numbers to 0-based offsets into `receptor_sequence()`.

    The sequence is the resolved span with gaps collapsed, so offsets are
    positional and cannot be computed as resnum - start.
    """
    R, _ = _receptor_and_peptide(cif)
    order = [r.seqid.num for r in R]
    pos = {n: i for i, n in enumerate(order)}
    missing = [n for n in resnums if n not in pos]
    if missing:
        raise ValueError(f"residues not resolved in 6x18: {missing}")
    return sorted(pos[n] for n in resnums)


if __name__ == "__main__":
    seq = receptor_sequence()
    ep, ecd, tmd = orthosteric_epitope(), orthosteric_epitope(domain="ecd"), orthosteric_epitope(domain="tmd")
    belt = lipid_facing_belt()
    print(f"receptor sequence   {len(seq)} aa (resolved 29-423 of construct 24-432)")
    print(f"orthosteric epitope {len(ep)} residues: ECD {len(ecd)}, TMD {len(tmd)}")
    print(f"lipid-facing belt   {len(belt)} residues")
    print(f"overlap             {len(set(ep) & set(belt))}")
    print(f"epitope offsets     {to_target_index(ep)[:8]} ...")
    print(f"belt offsets        {to_target_index(belt)[:8]} ...")


INACTIVE_CIF = Path(__file__).with_name("structures") / "5vew.cif"
# 5vew chain A is a crystallisation construct: residues 1002-1161 are a fusion
# partner inserted in ICL3, not receptor. Templating the chain as deposited
# would put 160 residues of lysozyme in the model.
FUSION_MIN = 1000


def inactive_template_chain(cif: Path | None = None) -> gemmi.Chain:
    """Chain A of 5vew with the ICL3 fusion partner removed.

    The inactive state is the other half of the agonism proxy: a design that
    binds the active conformation better than this one stabilises the active
    state, which is what agonism is thermodynamically.
    """
    st = gemmi.read_structure(str(cif or INACTIVE_CIF))
    st.setup_entities()
    st.remove_ligands_and_waters()
    chain = {c.name: c for c in st[0]}["A"]
    keep = gemmi.Chain("A")
    for r in chain:
        if r.seqid.num < FUSION_MIN:
            keep.add_residue(r)
    return keep


def two_state_common_residues(cif_active=None, cif_inactive=None) -> list[int]:
    """Residue numbers resolved in both the active and inactive structures.

    The two come from different constructs -- 5vew is transmembrane-only and
    384 against 263 residues -- so comparing binding between them without
    restricting to the shared set confounds conformation with construct.
    """
    R, _ = _receptor_and_peptide(cif_active)
    act = {r.seqid.num for r in R if r.find_atom("CA", "*")}
    ina = {r.seqid.num for r in inactive_template_chain(cif_inactive)
           if r.find_atom("CA", "*")}
    return sorted(act & ina)


def restrict_chain(chain: gemmi.Chain, resnums) -> gemmi.Chain:
    """A copy of `chain` holding only `resnums`."""
    keep = set(resnums)
    out = gemmi.Chain(chain.name)
    for r in chain:
        if r.seqid.num in keep:
            out.add_residue(r)
    return out


def chain_sequence(chain: gemmi.Chain) -> str:
    return gemmi.one_letter_code([r.name for r in chain])


# Receptor residues within 4.5 A of GLP-1 His7-Thr11, the segment whose removal
# costs 1000-10000x cAMP potency against 100-300x affinity. Spread across TM1,
# TM2, TM3, TM5, TM6 and TM7, i.e. the helices whose rearrangement is activation.
ACTIVATION_CORE = [152, 190, 194, 233, 234, 237, 241, 300, 306, 309, 310, 313,
                   372, 380, 384, 387, 388, 391]


def activation_core(cif: Path | None = None, recompute: bool = False) -> list[int]:
    """Receptor residues contacted by GLP-1's activating N-terminal segment.

    Used instead of an active-minus-inactive binding differential, which was
    measured and does not work: over GLP-1, exendin-4, GLP-1 9-37, a scrambled
    peptide and poly-alanine the delta ran -0.005 to +0.030 for the agonists and
    -0.131 to +0.058 for the negatives, with the scrambled peptide scoring the
    highest delta of all. Two reasons it cannot work here. The inactive
    structure 5vew is transmembrane-only, so every ligand binds it and the
    active construct badly -- ipTM 0.25 to 0.51 against 0.87 to 0.90 on the full
    receptor -- and the differential is noise on top of a weak signal. And
    agonism is not a binding preference, it is whether a specific pocket is
    engaged, which a scalar affinity difference does not express.

    Contacting these residues is a geometric condition, checkable on one
    prediction with nothing to calibrate. It does not prove a design activates;
    it establishes that the design engages the machinery that activation
    requires, which is the strongest statement available without a cell assay.
    """
    if not recompute:
        return list(ACTIVATION_CORE)
    R, P = _receptor_and_peptide(cif)
    ra = _heavy(R)
    seg = [(n, x) for n, x in _heavy(P) if 7 <= n <= 11]
    RX = np.array([x for _, x in ra])
    SX = np.array([x for _, x in seg])
    D = np.linalg.norm(RX[:, None] - SX[None, :], axis=-1)
    return sorted({ra[i][0] for i in range(len(ra)) if (D[i] < CONTACT_CUTOFF).any()})


def bundle_axis(cif: Path | None = None):
    """Centroid and unit axis of the transmembrane bundle, oriented so that
    larger projections mean deeper into the bundle.

    GLP-1's residues lie on a monotone gradient along this axis: His7 deepest at
    -8.4 A, falling to -29.3 A at Gly22 up in the extracellular domain, with the
    activating segment His7-Thr11 spanning -8.4 to -13.8. Insertion depth is
    therefore a scalar measure of threading, where contact count also rewards a
    peptide lying across the groove without entering it.
    """
    R, _ = _receptor_and_peptide(cif)
    ca = {}
    for r in R:
        a = r.find_atom("CA", "*")
        if a:
            ca[r.seqid.num] = np.array([a.pos.x, a.pos.y, a.pos.z])
    tm = np.array([v for k, v in ca.items() if k > ECD_MAX])
    cen = tm.mean(0)
    axis = np.linalg.svd(tm - cen)[2][0]
    axis = axis / np.linalg.norm(axis)
    ecd = np.array([v for k, v in ca.items() if k <= ECD_MAX]).mean(0)
    if (ecd - cen) @ axis > 0:
        axis = -axis
    return cen, axis


# Depth reached by GLP-1 His7, the deepest point of the activating segment, and
# the shallow end of that segment at Thr11. A binder reaching past about -14 A
# has entered the activation pocket rather than resting in the groove.
GLP1_HIS7_DEPTH = -8.4
ACTIVATION_DEPTH_THRESHOLD = -14.0
