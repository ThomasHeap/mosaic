"""Quantitative pH-selectivity scoring via proton linkage + predicted pKa values.

There is no model that predicts "pH-dependent binding" end to end, and no dataset
to train one on -- but the question decomposes. pH-dependent affinity is a
*thermodynamic consequence* of ionizable groups whose pKa changes on binding, and
that relation is exact, not learned. So the only thing needing prediction is pKa,
for which open-source tools and experimental training data both exist.

Linkage relation. For ionizable groups i with pKa_free,i in the unbound binder
and pKa_bound,i in the complex, the association constant varies with pH as

    log10 Ka(pH) = log10 Ka_deprot
                   + sum_i log10 (1 + 10^(pKa_bound,i - pH))
                   - sum_i log10 (1 + 10^(pKa_free,i - pH))

A histidine whose pKa is *suppressed* on binding (pKa_bound < pKa_free) therefore
binds more weakly as pH falls: at low pH the group wants a proton, and the complex
will not accommodate it. That is exactly the challenge's criterion #1.

What this buys you over counting histidines: the per-histidine selectivity
available across a 7.4 -> 6.0 window is bounded, and the bound is tight enough to
dictate the design. See `report_ceiling()`:

  - pKa_free 6.5 (an ordinary surface histidine), pKa_bound fully suppressed
      -> 0.57 log10 (3.7-fold) per histidine
  - pKa_free 7.0  -> 0.90 log10 (7.9-fold) per histidine
  - pKa_free 7.5  -> 1.16 log10 (14-fold) per histidine

So "no detectable binding at pH 6.0" (call it >= 2 log10, ~100-fold) is simply not
reachable with one or two ordinary histidines. It needs either ~4 of them, or
~2-3 whose *free*-state pKa has been deliberately raised. That second lever is the
one usually missed: selectivity rises monotonically with pKa_free, so you want the
histidine to be comfortably protonatable when the binder is free (neighbouring
Asp/Glu raise its pKa) and strongly resistant to protonation in the complex
(buried against a hydrophobic or cationic patch of the epitope).

Training your own model, if you want one: the right target is pKa, not pH-dependent
binding. PKAD-2 holds ~1,500 experimentally measured protein pKa values with
structures, which is a real supervised set; a graph net over the local environment
is the standard architecture and gets within ~0.7 pKa-unit MAE, about level with
PROPKA3. That accuracy is the catch -- 0.7 units of pKa error is ~0.7 log10 of
selectivity error, i.e. the size of the whole per-histidine effect. Use these
scores to *rank and triage* designs, never as a pass/fail gate.
"""

from __future__ import annotations

import math
import tempfile
from collections import defaultdict
from pathlib import Path

import gemmi

# pKa of free histidine in water; the reference point for "ordinary".
HIS_MODEL_PKA = 6.5


def _log10_1p10(x: float) -> float:
    """log10(1 + 10^x), overflow-safe for large |x|."""
    if x > 12:
        return x
    if x < -12:
        return 0.0
    return math.log10(1.0 + 10.0**x)


def log_ka_shift(pka_bound: float, pka_free: float, ph: float) -> float:
    """Contribution of one ionizable group to log10 Ka at a given pH."""
    return _log10_1p10(pka_bound - ph) - _log10_1p10(pka_free - ph)


def selectivity_log10(
    pkas: list[tuple[float, float]],
    ph_high: float = 7.4,
    ph_low: float = 6.0,
) -> float:
    """log10 of the affinity ratio Ka(pH 7.4) / Ka(pH 6.0).

    `pkas` is a list of (pKa_bound, pKa_free) pairs, one per ionizable group.
    Positive means tighter binding at 7.4 than at 6.0 -- the desired direction.
    A value >= 2 is the rough threshold for "no detectable binding" at pH 6.0
    given a low-nanomolar baseline.
    """
    total = 0.0
    for pka_bound, pka_free in pkas:
        total += log_ka_shift(pka_bound, pka_free, ph_high)
        total -= log_ka_shift(pka_bound, pka_free, ph_low)
    return total


def report_ceiling(
    ph_high: float = 7.4, ph_low: float = 6.0, suppressed: float = 2.0
) -> None:
    """Print per-histidine selectivity as a function of free-state pKa."""
    print(f"per-His log10 selectivity, pH {ph_high} vs {ph_low}, "
          f"pKa_bound = {suppressed}")
    for pka_free in (6.0, 6.5, 7.0, 7.5, 8.0):
        s = selectivity_log10([(suppressed, pka_free)], ph_high, ph_low)
        n_needed = math.ceil(2.0 / s) if s > 0 else float("inf")
        print(f"  pKa_free {pka_free:>4.1f} -> {s:>5.2f} log10 "
              f"({10**s:>6.1f}-fold); need {n_needed} for 2 log10")


# --- pKa prediction ---------------------------------------------------------


def propka_pkas(
    structure: gemmi.Structure | str | Path,
    residue_types: tuple[str, ...] = ("HIS",),
) -> dict[tuple[str, int], float]:
    """Predicted pKa per residue, keyed by (chain, residue number).

    Uses PROPKA3 (LGPL, `uv pip install propka`). Heuristic and fast -- seconds
    per structure -- which is what makes it usable for triaging hundreds of
    designs. Batch runs belong in a Slurm job, not on the login node.
    """
    import propka.run

    with tempfile.TemporaryDirectory() as tmp:
        if isinstance(structure, gemmi.Structure):
            st = structure.clone()
            st.setup_entities()
            st.remove_ligands_and_waters()
            path = Path(tmp) / "input.pdb"
            st.write_pdb(str(path))
        else:
            # PROPKA writes its output next to the input, so copy in.
            src = Path(structure)
            path = Path(tmp) / src.name
            path.write_bytes(src.read_bytes())

        mol = propka.run.single(str(path), optargs=["--quiet"], write_pka=False)

        out: dict[tuple[str, int], float] = {}
        conformation = mol.conformations[mol.conformation_names[0]]
        for group in conformation.groups:
            if group.residue_type not in residue_types:
                continue
            if group.pka_value is None:
                continue
            atom = group.atom
            out[(atom.chain_id, int(atom.res_num))] = float(group.pka_value)
        return out


def score_pair(
    complex_structure: gemmi.Structure | str | Path,
    binder_structure: gemmi.Structure | str | Path,
    binder_chain: str = "A",
    ph_high: float = 7.4,
    ph_low: float = 6.0,
) -> dict:
    """Score one design: predicted pKas on both end states -> selectivity.

    `complex_structure` is the binder-target complex, `binder_structure` the
    binder alone (both as predicted). Only histidines on `binder_chain` count,
    since those are the ones the design controls.
    """
    bound = propka_pkas(complex_structure)
    free = propka_pkas(binder_structure)

    pairs, per_residue = [], []
    for (chain, resnum), pka_free in sorted(free.items()):
        if chain != binder_chain:
            continue
        pka_bound = bound.get((chain, resnum))
        if pka_bound is None:
            continue
        pairs.append((pka_bound, pka_free))
        per_residue.append({
            "resnum": resnum,
            "pka_free": round(pka_free, 2),
            "pka_bound": round(pka_bound, 2),
            "shift": round(pka_bound - pka_free, 2),
            "selectivity": round(
                selectivity_log10([(pka_bound, pka_free)], ph_high, ph_low), 3
            ),
        })

    return {
        "n_his": len(pairs),
        "selectivity_log10": round(selectivity_log10(pairs, ph_high, ph_low), 3),
        "fold": round(10 ** selectivity_log10(pairs, ph_high, ph_low), 1),
        "histidines": per_residue,
    }


if __name__ == "__main__":
    report_ceiling()
