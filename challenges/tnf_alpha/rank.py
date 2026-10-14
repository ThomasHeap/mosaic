"""Triage designs and emit the submission CSV.

Applies the challenge's hard constraints, then orders by whatever scores are
available. Cheap enough for the login node -- the expensive part (repredicting
each design and running Tier 2 pKa scoring) happens in a GPU job upstream and
arrives here as fields on the JSONL records.

    python rank.py designs/*.jsonl --out submission.csv --limit 40
"""

import argparse
import csv
import json
from pathlib import Path

VALID_CLASSES = {"protein", "nanobody", "scfv", "fab_kappa", "fab_lambda"}
MIN_LEN, MAX_LEN = 10, 250

# C-alpha RMSD between the binder predicted alone and in complex. The Nipah
# authors discarded designs that "adopted wildly different conformations" in
# isolation; a binder that only folds when bound is unlikely to express. Applied
# only when rescore.py has measured it -- an unmeasured design is not rejected,
# it is merely unranked.
MAX_MONOMER_COMPLEX_RMSD = 5.0
# How close the annealed PSSM got to one-hot. Well below 1.0 means the reported
# sequence is an argmax over a still-fuzzy distribution.
MIN_PSSM_SHARPNESS = 0.8
# Any single amino acid above this fraction is a homopolymer, not a protein.
# Probe 19400994 produced a 32%-histidine 75-mer before the global histidine cap
# was added, and the high ProteinMPNN weight is itself known to encourage
# homopolymers (the Nipah authors note this). Catch it at the gate regardless of
# which term caused it.
MAX_SINGLE_AA_FRACTION = 0.25


def load(paths):
    seen_seq = {}
    records = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            seq = r["sequence"]
            # The challenge requires unique designs; identical sequences from
            # different seeds are a real occurrence with a shared objective.
            if seq in seen_seq:
                continue
            seen_seq[seq] = r["name"]
            records.append(r)
    return records


def check(r):
    """Reasons this design would be rejected, if any."""
    problems = []
    seq = r["sequence"]
    if not (MIN_LEN <= len(seq) <= MAX_LEN):
        problems.append(f"length {len(seq)} outside {MIN_LEN}-{MAX_LEN}")
    if r.get("molecule_class") not in VALID_CLASSES:
        problems.append(f"bad molecule_class {r.get('molecule_class')!r}")
    if "C" in seq:
        problems.append("contains cysteine")
    if set(seq) - set("ARNDCQEGHILKMFPSTWYV"):
        problems.append("non-standard residue")
    rmsd = r.get("monomer_complex_rmsd")
    if rmsd is not None and float(rmsd) > MAX_MONOMER_COMPLEX_RMSD:
        problems.append(
            f"monomer/complex RMSD {float(rmsd):.1f} A > {MAX_MONOMER_COMPLEX_RMSD}"
        )
    from collections import Counter
    if seq:
        aa, count = Counter(seq).most_common(1)[0]
        frac = count / len(seq)
        if frac > MAX_SINGLE_AA_FRACTION:
            problems.append(
                f"{aa} is {frac:.0%} of the sequence "
                f"(> {MAX_SINGLE_AA_FRACTION:.0%})"
            )
    sharp = r.get("pssm_max_mean")
    if sharp is not None and float(sharp) < MIN_PSSM_SHARPNESS:
        problems.append(f"PSSM not sharpened ({float(sharp):.2f})")
    return problems


def sort_key(r):
    """Challenge criteria in their stated order of importance.

    pH selectivity first, then mouse cross-reactivity, then affinity -- each
    falling back to a neutral value when the upstream scoring has not run, so
    partially scored pools still order sensibly.
    """
    return (
        -float(r.get("selectivity_log10", 0.0)),
        -float(r.get("mouse_iptm", 0.0)),
        -float(r.get("iptm", 0.0)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--out", type=Path, default=Path("submission.csv"))
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    records = load(args.jsonl)
    kept, rejected = [], []
    for r in records:
        problems = check(r)
        (rejected if problems else kept).append((r, problems))

    kept.sort(key=lambda rp: sort_key(rp[0]))

    print(f"loaded {len(records)} unique designs: "
          f"{len(kept)} pass, {len(rejected)} rejected")
    for r, problems in rejected:
        print(f"  reject {r['name']}: {'; '.join(problems)}")

    unscored = sum(1 for r, _ in kept if "selectivity_log10" not in r)
    if unscored:
        print(f"WARNING: {unscored}/{len(kept)} designs have no Tier 2 pH score; "
              f"their order is not meaningful for criterion #1")

    selected = [r for r, _ in kept[: args.limit]]
    by_pool = {}
    for r in selected:
        by_pool[r.get("pool", "?")] = by_pool.get(r.get("pool", "?"), 0) + 1
    print(f"selected {len(selected)} by pool: {by_pool}")

    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "sequence", "molecule_class"])
        for r in selected:
            w.writerow([r["name"], r["sequence"], r["molecule_class"]])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
