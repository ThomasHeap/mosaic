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
