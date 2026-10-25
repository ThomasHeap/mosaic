"""Check every figure in METHODS.md against the data, and the submission format.

On the TNF submission three separate figures drifted into the methods document
unnoticed across edits, including a paragraph describing a selection that no
longer existed. Reading it by eye missed them twice. So the check is mechanical
and exits non-zero on any drift.
"""
import csv
import glob
import itertools
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
FAIL = []


def check(label, claim, actual):
    ok = str(claim) == str(actual)
    if not ok:
        FAIL.append(label)
    print(f"  {'ok ' if ok else 'BAD'} {label:38s} doc {claim!s:22s} data {actual}")


def agg(files, key):
    g = defaultdict(list)
    for f in files:
        try:
            for line in open(f):
                r = json.loads(line)
                g[r["name"]].append(r)
        except FileNotFoundError:
            pass
    return {n: st.mean(r[key] for r in x if r.get(key) is not None)
            for n, x in g.items()
            if any(r.get(key) is not None for r in x)}


def main():
    doc = (HERE / "METHODS.md").read_text()
    sub = list(csv.DictReader(open(HERE / "submission.csv")))
    met = list(csv.DictReader(open(HERE / "submission_metrics.csv")))
    tier = {json.loads(l)["name"]: json.loads(l)["structural_tier"]
            for l in open(HERE / "probe" / "novelty2.jsonl")}

    # Header claims
    L = [len(r["sequence"]) for r in sub]
    check("length range", "40–55", f"{min(L)}–{max(L)}")
    ident = lambda a, b: (lambda n: sum(x == y for x, y in zip(a[:n], b[:n])) / n)(
        min(len(a), len(b)))
    check("max pairwise identity", "27%",
          f"{max(ident(a['sequence'], b['sequence']) for a, b in itertools.combinations(sub, 2)):.0%}")
    check("no cysteines", 0, sum(1 for r in sub if "C" in r["sequence"]))
    check("submission size", 20, len(sub))

    # Novelty
    levels = Counter(int(m["novelty_level"]) for m in met)
    check("submitted level 4", 13, levels[4])
    check("submitted level 3", 7, levels[3])
    pool = Counter(tier.values())
    check("pool level 4 (below_moderate)", 21, pool["below_moderate"])
    check("pool level 3 (moderate)", 19, pool["moderate"])
    check("pool level 2 (high), excluded", 8, pool["high"])
    check("candidates screened", 48, sum(pool.values()))

    # Scored ranges
    core = [float(m["activation_core_contacts"]) for m in met]
    iptm = [float(m["iptm"]) for m in met]
    check("submitted core range", "16.2–17.6", f"{min(core):.1f}–{max(core):.1f}")
    check("submitted ipTM range", "0.51–0.86", f"{min(iptm):.2f}–{max(iptm):.2f}")
    check("median ipTM, activation-first", "0.63", f"{st.median(iptm):.2f}")
    chg = [sum(r["sequence"].count(c) for c in "DEKR") / len(r["sequence"])
           for r in sub]
    check("charged fraction", "33–58%",
          f"{min(chg)*100:.0f}–{max(chg)*100:.0f}%")

    # Design counts
    names = set()
    # The design runs only: the N-terminal refinement variants are a separate
    # stage reported as "0 of 80", and smoke tests are not designs.
    skip = ("scored", "designs_nterm", "smoke")
    for f in glob.glob(str(HERE / "designs_*/*.jsonl")):
        if any(k in f for k in skip):
            continue
        for line in open(f):
            names.add(json.loads(line)["name"])
    check("designs generated", 138, len(names))

    # Target definition
    sys.path.insert(0, str(HERE))
    from target import (ACTIVATION_CORE, DEEP_CORE, lipid_facing_belt,
                        orthosteric_epitope)
    ep = orthosteric_epitope()
    check("orthosteric epitope size", 52, len(ep))
    check("activation core size", 18, len(ACTIVATION_CORE))
    check("deep core size", 15, len(DEEP_CORE))
    belt = lipid_facing_belt()
    check("lipid belt size", 65, len(belt))
    tmd = [r for r in ep if r > 145]
    check("epitope fraction transmembrane", "58%", f"{len(tmd)/len(ep):.0%}")

    # Figures that must appear in the document
    for fig in ("17–18 of 18", "11–12", "15–18", "0–6", "0.87", "0.42",
                "r = +0.45", "0 of 80", "27.3 of 29", "138 designs",
                "2.46 Å", "+6.7 Å"):
        if fig not in doc:
            FAIL.append(f"{fig} absent")
            print(f"  BAD figure {fig!r} is not in METHODS.md")

    # Format
    ALLOWED = {"protein", "nanobody", "scfv", "fab_kappa", "fab_lambda"}
    check("format: columns", "name,sequence,molecule_class",
          ",".join(sub[0].keys()))
    check("format: molecule_class allowed", True,
          {r["molecule_class"] for r in sub} <= ALLOWED)
    check("format: unique sequences", 20, len({r["sequence"] for r in sub}))
    check("format: length within 10-250", True,
          all(10 <= len(r["sequence"]) <= 250 for r in sub))
    check("format: standard residues only", 0,
          sum(1 for r in sub if set(r["sequence"]) - set("ACDEFGHIKLMNPQRSTVWY")))
    check("format: no VH:VL colon", 0, sum(1 for r in sub if ":" in r["sequence"]))
    check("metrics rows match submission", True,
          [m["name"] for m in met] == [r["name"] for r in sub])
    check("metrics has no empty cells", 0,
          sum(1 for m in met for v in m.values() if v in ("", "None")))
    check("no level-2 design submitted", 0,
          sum(1 for m in met if tier.get(m["name"]) == "high"))

    print(f"\n  {'FAILURES: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
