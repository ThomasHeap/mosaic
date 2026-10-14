"""Negative controls for the Tier 2 pH-selectivity score.

Adalimumab and infliximab bind the TNF receptor groove and neither is
pH-dependent -- they are conventional IgGs, not engineered recycling antibodies.
So the Tier 2 score must place them near zero. If it does not, it is measuring
something other than a pH switch and any ranking built on it is worthless.

This is the cheapest meaningful validation available: both have crystal
structures in complex with TNF (3WD5, 4G3Y), so it needs no prediction and no
GPU. The challenge rules explicitly permit known binders for calibrating filters.

A useful positive reference is harder to come by -- there is no TNF binder with a
published pH switch -- so the comparison is against TNF's own histidines
(`ph_score` run on 1TNF) and against whatever the design pools produce. The claim
being tested is therefore one-sided: a score that cannot distinguish a known
non-switching binder from a designed one is broken, but passing this does not
prove the score is sensitive.

    python -I controls.py --data-dir pdb_cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gemmi

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ph_score import propka_pkas, selectivity_log10  # noqa: E402

# Binder chains and the TNF chain each Fab sits on, from the deposited entries.
CONTROLS = {
    "3WD5": {"name": "adalimumab Fab", "binder": ("H", "L"), "target": ("A",)},
    "4G3Y": {"name": "infliximab Fab", "binder": ("H", "L"), "target": ("C",)},
}


def load(path: Path) -> gemmi.Structure:
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_ligands_and_waters()
    st.remove_alternative_conformations()
    return st


def isolate(st: gemmi.Structure, keep: tuple[str, ...]) -> gemmi.Structure:
    out = st.clone()
    for name in [c.name for c in out[0]]:
        if name not in keep:
            out[0].remove_chain(name)
    return out


def score(st: gemmi.Structure, binder: tuple[str, ...]) -> dict:
    """pH selectivity contributed by histidines on the binder chains."""
    bound = propka_pkas(st)
    free = propka_pkas(isolate(st, binder))

    pairs, detail = [], []
    for (chain, resnum), pka_free in sorted(free.items()):
        if chain not in binder:
            continue
        pka_bound = bound.get((chain, resnum))
        if pka_bound is None:
            continue
        pairs.append((pka_bound, pka_free))
        site_sel = selectivity_log10([(pka_bound, pka_free)])
        detail.append({
            "chain": chain,
            "resnum": resnum,
            "pka_free": round(pka_free, 2),
            "pka_bound": round(pka_bound, 2),
            "shift": round(pka_bound - pka_free, 2),
            "selectivity": round(site_sel, 3),
        })

    total = selectivity_log10(pairs)
    return {
        "n_his": len(pairs),
        "selectivity_log10": round(total, 3),
        "fold": round(10**total, 2),
        # The single worst offender matters: one histidine with a large shift is
        # what an accidental pH switch looks like.
        "max_site_selectivity": round(
            max((d["selectivity"] for d in detail), default=0.0), 3
        ),
        "histidines": detail,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("pdb_cache"))
    ap.add_argument("--out", type=Path, default=Path("controls.json"))
    args = ap.parse_args()

    results = {}
    for pdb_id, spec in CONTROLS.items():
        path = args.data_dir / f"{pdb_id}.cif"
        if not path.exists():
            path = Path(f"{pdb_id}.cif")
        if not path.exists():
            print(f"{pdb_id}: not found, skipping "
                  f"(download from files.rcsb.org/download/{pdb_id}.cif)")
            continue
        st = load(path)
        present = {c.name for c in st[0]}
        binder = tuple(c for c in spec["binder"] if c in present)
        if not binder:
            print(f"{pdb_id}: no binder chains among {sorted(present)}")
            continue
        r = score(st, binder)
        results[pdb_id] = {"name": spec["name"], **r}
        print(f"\n{pdb_id} ({spec['name']}), binder chains {binder}")
        print(f"  {r['n_his']} histidines, total selectivity "
              f"{r['selectivity_log10']:+.3f} log10 ({r['fold']}x), "
              f"worst single site {r['max_site_selectivity']:+.3f}")
        for d in sorted(r["histidines"],
                        key=lambda x: -abs(x["selectivity"]))[:5]:
            print(f"    {d['chain']}{d['resnum']:<4} "
                  f"pKa {d['pka_free']:>5.2f} -> {d['pka_bound']:>5.2f} "
                  f"(shift {d['shift']:+.2f})  sel {d['selectivity']:+.3f}")

    if results:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")

        print("\nverdict:")
        worst = max(r["selectivity_log10"] for r in results.values())
        # A conventional antibody should sit far below the ~2 log10 the
        # challenge is asking for. Anything approaching it means the score is
        # reporting a switch where none exists.
        if worst < 0.5:
            print(f"  PASS: known non-pH-dependent binders score at most "
                  f"{worst:+.3f} log10, well below the ~2 target. The score is "
                  f"not manufacturing selectivity out of ordinary interfaces.")
        else:
            print(f"  FAIL: a conventional antibody scores {worst:+.3f} log10. "
                  f"The score is picking up ordinary interface burial as a pH "
                  f"switch -- do not rank on it until this is understood.")


if __name__ == "__main__":
    main()
