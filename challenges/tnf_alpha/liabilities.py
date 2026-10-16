"""Sequence liability screen for the submitted designs.

The submission form asks for liability scores alongside the sequences. These are
the standard developability flags applied to a protein therapeutic candidate,
computed from sequence alone. They are reported rather than used to reject:
Adaptyv expresses and assays in a defined system, not in a patient, so most of
these are informational for this competition. Two of them are not, and are worth
acting on:

  - **Net charge and pI.** A protein whose isoelectric point sits near the assay
    pH has minimal net charge there and is prone to aggregation. Since this
    assay runs at pH 7.4 *and* pH 6.0, a pI between those values is a genuine
    risk, and the pH-selectivity objective pushes designs acidic, so this needs
    checking rather than assuming.
  - **Long homopolymer runs.** These indicate the optimizer collapsed rather
    than converged, and the high ProteinMPNN weight is known to encourage them.

The rest (glycosylation sequons, deamidation and isomerization motifs, oxidation
sites) matter for a developability assessment of a real candidate and are
included so the submission carries them, but a sequon is only a liability if the
expression host glycosylates and the site is exposed.

    python -I liabilities.py submission.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# Henderson-Hasselbalch pKa values for isoelectric point estimation
# (Bjellqvist scale, the usual choice for sequence-only pI).
PKA_SIDECHAIN = {"D": 3.65, "E": 4.25, "C": 8.37, "Y": 10.46,
                 "H": 6.00, "K": 10.53, "R": 12.48}
PKA_NTERM, PKA_CTERM = 8.6, 3.55
POSITIVE = set("HKR")


def net_charge_at(seq: str, ph: float) -> float:
    """Net charge from Henderson-Hasselbalch over all ionisable groups."""
    charge = 1.0 / (1.0 + 10 ** (ph - PKA_NTERM))
    charge -= 1.0 / (1.0 + 10 ** (PKA_CTERM - ph))
    for aa, pka in PKA_SIDECHAIN.items():
        n = seq.count(aa)
        if not n:
            continue
        if aa in POSITIVE:
            charge += n / (1.0 + 10 ** (ph - pka))
        else:
            charge -= n / (1.0 + 10 ** (pka - ph))
    return charge


def isoelectric_point(seq: str) -> float:
    """pI by bisection on net charge."""
    lo, hi = 2.0, 13.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if net_charge_at(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def longest_run(seq: str) -> tuple[str, int]:
    best_aa, best = seq[0] if seq else "", 0
    cur_aa, cur = None, 0
    for aa in seq:
        cur = cur + 1 if aa == cur_aa else 1
        cur_aa = aa
        if cur > best:
            best_aa, best = aa, cur
    return best_aa, best


def screen(seq: str) -> dict:
    n = len(seq)
    pi = isoelectric_point(seq)
    run_aa, run_len = longest_run(seq)
    # N-X-S/T with X != P is the canonical N-glycosylation sequon.
    sequons = [m.start() + 1 for m in re.finditer(r"N[^P][ST]", seq)]
    deamidation = [m.start() + 1 for m in re.finditer(r"N[GSTN]", seq)]
    isomerization = [m.start() + 1 for m in re.finditer(r"D[GSTP]", seq)]

    flags = []
    # The assay spans pH 6.0-7.4; a pI inside that window means near-zero net
    # charge at an assay condition, which is where aggregation happens.
    if 6.0 <= pi <= 7.4:
        flags.append(f"pI {pi:.1f} inside the assay pH window")
    if abs(net_charge_at(seq, 7.4)) > 12:
        flags.append(f"net charge {net_charge_at(seq, 7.4):+.0f} at pH 7.4")
    if run_len >= 4:
        flags.append(f"{run_aa}x{run_len} homopolymer run")
    if sequons:
        flags.append(f"{len(sequons)} N-glyc sequon(s)")

    return {
        "length": n,
        "pI": round(pi, 2),
        "net_charge_pH7.4": round(net_charge_at(seq, 7.4), 2),
        "net_charge_pH6.0": round(net_charge_at(seq, 6.0), 2),
        "n_glyc_sequons": len(sequons),
        "n_glyc_positions": sequons,
        "deamidation_motifs": len(deamidation),
        "isomerization_motifs": len(isomerization),
        "n_met_trp_oxidisable": seq.count("M") + seq.count("W"),
        "free_cysteines": seq.count("C"),
        "longest_homopolymer": f"{run_aa}x{run_len}",
        "flags": flags,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="submission.csv")
    ap.add_argument("--out", type=Path, default=Path("liabilities.jsonl"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    results = []
    print(f"{'name':44s} {'pI':>5} {'Q7.4':>6} {'Q6.0':>6} {'glyc':>4} "
          f"{'run':>5}  flags")
    for r in rows:
        s = screen(r["sequence"])
        s["name"] = r["name"]
        results.append(s)
        print(f"{r['name'][:44]:44s} {s['pI']:>5.1f} "
              f"{s['net_charge_pH7.4']:>+6.1f} {s['net_charge_pH6.0']:>+6.1f} "
              f"{s['n_glyc_sequons']:>4} {s['longest_homopolymer']:>5}  "
              f"{'; '.join(s['flags']) if s['flags'] else '-'}")

    args.out.write_text("".join(json.dumps(r) + "\n" for r in results))

    n_flagged = sum(1 for r in results if r["flags"])
    print(f"\n{n_flagged}/{len(results)} designs carry at least one flag")
    pis = [r["pI"] for r in results]
    inside = sum(1 for p in pis if 6.0 <= p <= 7.4)
    print(f"pI range {min(pis):.1f}-{max(pis):.1f}; "
          f"{inside}/{len(pis)} inside the assay pH window (6.0-7.4)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
