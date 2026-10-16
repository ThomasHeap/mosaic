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

# The downloadable template (proteinbase.com/templates/competition-submission-
# template.csv) labels a plain protein "single_chain". The challenge FAQ text
# says "protein". The template is the file that gets filled in, so it is treated
# as canonical and "protein" is accepted on input then normalised on output.
VALID_CLASSES = {"protein", "single_chain", "nanobody", "scfv",
                 "fab_kappa", "fab_lambda"}
OUTPUT_CLASS = {"protein": "single_chain"}
MIN_LEN, MAX_LEN = 10, 250

# C-alpha RMSD between the binder predicted alone and in complex. The Nipah
# authors discarded designs that "adopted wildly different conformations" in
# isolation; a binder that only folds when bound is unlikely to express. Applied
# only when rescore.py has measured it -- an unmeasured design is not rejected,
# it is merely unranked.
MAX_MONOMER_COMPLEX_RMSD = 5.0
# The RMSD above is only interpretable when the monomer prediction is itself
# confident. The first rescored design had rmsd 12.5 A with plddt_monomer 0.49 --
# at that confidence the monomer structure is unreliable, so a large RMSD cannot
# distinguish "folds differently alone" from "has no confident fold alone". Below
# this confidence the RMSD is ignored and the low confidence is reported instead,
# which is the more honest statement of the same concern.
MIN_PLDDT_FOR_RMSD = 0.7
# How close the annealed PSSM got to one-hot. Well below 1.0 means the reported
# sequence is an argmax over a still-fuzzy distribution.
MIN_PSSM_SHARPNESS = 0.8
# Any single amino acid above this fraction is a homopolymer, not a protein.
# Probe 19400994 produced a 32%-histidine 75-mer before the global histidine cap
# was added, and the high ProteinMPNN weight is itself known to encourage
# homopolymers (the Nipah authors note this). Catch it at the gate regardless of
# which term caused it.
MAX_SINGLE_AA_FRACTION = 0.25

# Binding confidence below which pH selectivity is not meaningful. Selectivity is
# a *ratio* of affinities, so a design that binds weakly at both pH values can
# score a large ratio while being useless -- the worst offender in the first
# ranked list had selectivity 2.04 with an ipTM of 0.163. The challenge ranks
# pH-selectivity first among designs that bind; a non-binder scores nothing on
# criterion #1 either.
#
# This is a *tier boundary, not a reject*. The Nipah de novo writeup reports a
# 90% hit rate with filtering and consensus ranking removed entirely, and notes
# none of their binders would have passed BindCraft's default filters -- most
# failing an unsaturated-hydrogen-bond check that was evidently not predictive.
# So weak binders are ordered last rather than discarded.
MIN_IPTM_FOR_SELECTIVITY = 0.70

# Selectivity is bucketed to this width before ordering, because differences
# finer than our measurement error are not real. Calibration put PROPKA's error
# on the pKa *shift* at 0.46 log10 per site (LOG.md entry 8), so treating a
# 0.19 difference as decisive is false precision: it ranked a design with
# selectivity 2.30 and mouse ipTM 0.436 above one with 2.11 and 0.917, trading
# a large real difference in criterion #2 for a difference in criterion #1 that
# is inside the noise. Within a bucket, order by the challenge's next criterion.
SELECTIVITY_BUCKET = 0.5


def portfolio(ordered, limit):
    """Build a deliberately hedged selection rather than a single ranking.

    Only 8 designs clear selectivity 2.0 *and* mouse ipTM 0.80 *and* ipTM 0.75
    simultaneously, so a 20-slot submission cannot be uniformly strong. A strict
    sort resolves that trade-off implicitly and badly: it fills the list with the
    highest selectivity, some of which barely bind mouse TNF at all, and
    criterion #2 is mouse cross-reactivity.

    So the slots are split explicitly:
      - a core of all-rounders, decent on all three criteria
      - a bet on criterion #1: highest selectivity, whatever the mouse number
      - a bet on criterion #2: highest mouse, whatever the selectivity
    If one criterion dominates in the lab, one of the bets pays; the core is
    what pays if none does. `ordered` must already be ranked and deduped.
    """
    core = [r for r in ordered
            if (r.get("selectivity_log10") or 0) >= 1.7
            and (r.get("mouse_iptm") or 0) >= 0.80
            and (r.get("iptm") or 0) >= 0.75]
    picked, names = list(core[:limit]), {r["name"] for r in core[:limit]}

    # Every hedge pick must still clear the binding floor. Without this the
    # mouse hedge pulled in a design at ipTM 0.352 with selectivity 0.01 -- a
    # near-certain non-binder, which scores nothing on *any* criterion and so
    # hedges nothing. A bet on one criterion is only a bet if the design binds.
    eligible = [r for r in ordered
                if (r.get("iptm") or 0) >= MIN_IPTM_FOR_SELECTIVITY]

    def fill(key, n):
        for r in sorted(eligible, key=key):
            if len(picked) >= limit or n <= 0:
                return
            if r["name"] in names:
                continue
            picked.append(r)
            names.add(r["name"])
            n -= 1

    remaining = limit - len(picked)
    # Split the leftover slots evenly between the two bets.
    fill(lambda r: -(r.get("selectivity_log10") or 0), remaining - remaining // 2)
    fill(lambda r: -(r.get("mouse_iptm") or 0), limit - len(picked))
    # Any slots still open go to the plain ranking.
    fill(lambda r: eligible.index(r), limit - len(picked))

    print(f"portfolio: {len(core[:limit])} all-rounders "
          f"(sel>=1.7, mouse>=0.80, ipTM>=0.75) + "
          f"{len(picked) - len(core[:limit])} hedge picks")
    return picked[:limit]


def dedupe_lineage(kept):
    """Keep only the best member of each design lineage.

    A histidine-scan variant differs from its parent by two to four residues,
    and several variants of one parent differ from each other by one. Those are
    not distinct designs: the challenge requires submissions to be unique and to
    show adequate sequence and structural diversity, and spending two of forty
    slots on near-identical sequences wastes one of them.

    `kept` must already be sorted best-first, so the first member of a lineage
    seen is the one to keep.
    """
    out, claimed = [], set()
    for r in kept:
        lineage = r.get("parent") or r["name"]
        if lineage in claimed:
            continue
        claimed.add(lineage)
        out.append(r)
    return out


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
    plddt_mono = r.get("plddt_monomer")
    if (rmsd is not None and float(rmsd) > MAX_MONOMER_COMPLEX_RMSD
            and plddt_mono is not None
            and float(plddt_mono) >= MIN_PLDDT_FOR_RMSD):
        problems.append(
            f"monomer/complex RMSD {float(rmsd):.1f} A > "
            f"{MAX_MONOMER_COMPLEX_RMSD} at monomer pLDDT "
            f"{float(plddt_mono):.2f}"
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
    """Challenge criteria in their stated order, gated on credible binding.

    Tier 0 is designs whose ipTM clears `MIN_IPTM_FOR_SELECTIVITY`, ordered by
    pH selectivity, then mouse cross-reactivity, then affinity -- the
    challenge's own order. Tier 1 is everything else, ordered by affinity
    first, since for a design that may not bind the useful question is whether
    it binds at all.

    Every term falls back to a neutral value when the upstream scoring has not
    run, so a partially rescored pool still orders sensibly.
    """
    iptm = float(r.get("iptm") or 0.0)
    sel = float(r.get("selectivity_log10") or 0.0)
    tier = 0 if iptm >= MIN_IPTM_FOR_SELECTIVITY else 1
    if tier == 0:
        # Bucket down, so higher selectivity still sorts first between buckets
        # while ties inside one are broken by mouse cross-reactivity.
        bucket = -int(sel / SELECTIVITY_BUCKET)
        return (tier, bucket, -float(r.get("mouse_iptm") or 0.0), -iptm, -sel)
    return (tier, -iptm, -sel, 0.0, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--out", type=Path, default=Path("submission.csv"))
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--portfolio", action="store_true",
                    help="hedge across the three criteria instead of a strict sort")
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

    # Designs with no provenance stamp came from a config predating it, and
    # cannot be compared against the rest. Loud, because the failure mode is a
    # stale file silently entering a submission.
    unstamped = [r["name"] for r, _ in kept if not r.get("config")
                 and r.get("pool") != "hisscan"]
    if unstamped:
        print(f"WARNING: {len(unstamped)} designs carry no `config` stamp and "
              f"predate the current objective: {unstamped[:5]}"
              f"{' ...' if len(unstamped) > 5 else ''}")
        print("  Check they are not from a superseded run before submitting.")

    unscored = sum(1 for r, _ in kept if "selectivity_log10" not in r)
    if unscored:
        print(f"WARNING: {unscored}/{len(kept)} designs have no Tier 2 pH score; "
              f"their order is not meaningful for criterion #1")

    ordered = dedupe_lineage([r for r, _ in kept])
    dropped = len(kept) - len(ordered)
    if dropped:
        print(f"collapsed {dropped} near-duplicate designs "
              f"(a scan variant and its parent, or siblings of one parent)")
    selected = (portfolio(ordered, args.limit) if args.portfolio
                else ordered[: args.limit])
    by_pool = {}
    for r in selected:
        by_pool[r.get("pool", "?")] = by_pool.get(r.get("pool", "?"), 0) + 1
    print(f"selected {len(selected)} by pool: {by_pool}")

    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "sequence", "molecule_class"])
        for r in selected:
            cls = r["molecule_class"]
            w.writerow([r["name"], r["sequence"], OUTPUT_CLASS.get(cls, cls)])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
