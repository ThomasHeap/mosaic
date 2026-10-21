"""TfR1 target: templates, the allowed design region, and exclusions.

Residue numbers are UniProt P02786 throughout, which is also the assay construct
numbering (human 121-760). 11qc resolves 122-760 on each protomer.

The design region is allowed rather than dictated. BinderTargetContact with
epitope_idx restricted to a hand-picked patch forces the binder onto that patch;
with epitope_idx=None it scores against the whole target and the optimiser picks
the easiest surface, which on GLP1R meant the lipid belt. Here the term is given
a wide allowed region, and where each design actually landed is classified
afterwards by mouse conservation, which is the quantity criterion 2 ranks on.
"""
import sys
from pathlib import Path

import gemmi
import numpy as np

# python -I strips the script directory from sys.path
sys.path.insert(0, str(Path(__file__).parent))
from conservation import conserved_fraction

HERE = Path(__file__).parent
DIMER_CIF = HERE / "structures" / "11qc.cif"
APICAL_COMPLEX_CIF = HERE / "structures" / "11rn.cif"

CONTACT_CUTOFF = 4.5

# Apical domain. 189-380 makes 85 cross-domain CA contacts against 924 internal
# (ratio 0.09) and contains every residue the 11rn binder touches, so it can be
# templated alone instead of the 639-residue protomer or the 1278-residue dimer.
APICAL_DOMAIN = (189, 380)

# Residues the 62-aa binder of 11rn chain X contacts on protomer A. The
# challenge recommends this site; only 2 of the 9 are conserved in mouse, so a
# binder here is likely human-specific and forfeits criterion 2.
APICAL_EPITOPE = [202, 204, 208, 209, 210, 211, 212, 348, 371]

# Most-conserved surface patch inside the apical domain, 89% conserved.
CONSERVED_PATCH = list(range(317, 326))

# Excluded from the allowed region:
#  - the truncation boundary, since cutting 189-380 out of the protomer exposes
#    a surface that does not exist in the assay construct, and a design could
#    bind the scar
#  - the C-terminal end, where the Twin-Strep capture tag attaches; the brief
#    says the tag must not be part of the binding
BOUNDARY_MARGIN = 6
TAG_PROXIMAL = list(range(745, 761))


def _protomer(cif=None, chain="A"):
    st = gemmi.read_structure(str(cif or DIMER_CIF))
    st.setup_entities()
    st.remove_ligands_and_waters()
    return {c.name: c for c in st[0]}[chain]


def apical_template_chain(cif=None) -> gemmi.Chain:
    """Protomer A truncated to the apical domain, for `template_chain`."""
    lo, hi = APICAL_DOMAIN
    src = _protomer(cif)
    keep = gemmi.Chain(src.name)
    for r in src:
        if lo <= r.seqid.num <= hi:
            keep.add_residue(r)
    return keep


def protomer_template_chain(cif=None) -> gemmi.Chain:
    """Full protomer A, for checking that truncation did not change binding."""
    return _protomer(cif)


def chain_sequence(ch) -> str:
    return gemmi.one_letter_code([r.name for r in ch])


def resnums(ch) -> list[int]:
    return [r.seqid.num for r in ch]


def allowed_region(chain) -> list[int]:
    """Residue numbers a design may contact: everything in `chain` except the
    truncation boundary and the tag-proximal C-terminus."""
    nums = resnums(chain)
    lo, hi = min(nums), max(nums)
    out = []
    for n in nums:
        if n - lo < BOUNDARY_MARGIN or hi - n < BOUNDARY_MARGIN:
            continue
        if n in TAG_PROXIMAL:
            continue
        out.append(n)
    return out


def excluded_region(chain) -> list[int]:
    allowed = set(allowed_region(chain))
    return [n for n in resnums(chain) if n not in allowed]


def to_target_index(chain, wanted) -> list[int]:
    """Residue numbers to 0-based offsets into `chain_sequence(chain)`.

    Positional, not resnum arithmetic: the resolved span may collapse gaps.
    """
    pos = {n: i for i, n in enumerate(resnums(chain))}
    missing = [n for n in wanted if n not in pos]
    if missing:
        raise ValueError(f"not resolved in this chain: {missing}")
    return sorted(pos[n] for n in wanted)


def apical_epitope_from_structure(cif=None) -> list[int]:
    """Recompute the 11rn chain X footprint rather than trusting the constant."""
    st = gemmi.read_structure(str(cif or APICAL_COMPLEX_CIF))
    st.setup_entities()
    st.remove_ligands_and_waters()
    ch = {c.name: c for c in st[0]}
    def heavy(c):
        return [(r.seqid.num, np.array([a.pos.x, a.pos.y, a.pos.z]))
                for r in c for a in r if a.element.name != "H"]
    R, X = heavy(ch["A"]), heavy(ch["X"])
    RX = np.array([x for _, x in R]); XX = np.array([x for _, x in X])
    D = np.linalg.norm(RX[:, None] - XX[None, :], axis=-1)
    return sorted({R[i][0] for i in range(len(R)) if (D[i] < CONTACT_CUTOFF).any()})


if __name__ == "__main__":
    ap = apical_template_chain()
    pr = protomer_template_chain()
    print(f"apical template   {len(ap)} residues {min(resnums(ap))}-{max(resnums(ap))}")
    print(f"full protomer     {len(pr)} residues {min(resnums(pr))}-{max(resnums(pr))}")
    print(f"allowed (apical)  {len(allowed_region(ap))} residues")
    print(f"excluded (apical) {len(excluded_region(ap))}: boundary + tag-proximal")
    rec = apical_epitope_from_structure()
    print(f"\n11rn footprint recomputed: {rec}")
    print(f"  matches constant: {rec == APICAL_EPITOPE}")
    print(f"  conserved: {conserved_fraction(rec):.0%}")
    print(f"conserved patch 317-325: {conserved_fraction(CONSERVED_PATCH):.0%}")
    print(f"allowed region overall:  {conserved_fraction(allowed_region(ap)):.0%}")
