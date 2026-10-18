"""Check every figure in METHODS.md against the data that produced it.

Two passes of reading the document by eye missed a stale epitope list, a pair of
transposed control values and a misattributed error, so the check is mechanical.
Figures that require a GPU or PROPKA run (the antibody controls) are listed as
externally verified with the command that reproduces them, rather than re-run
here.
"""
import csv, glob, itertools, json, math, statistics as st, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

FAIL = []


def check(label, claim, actual):
    ok = str(claim) == str(actual)
    if not ok:
        FAIL.append(label)
    print(f"  {'ok ' if ok else 'BAD'} {label:34s} doc {claim!s:22s} data {actual}")


def main():
    doc = Path("METHODS.md").read_text()
    rec = {}
    for fn in glob.glob("designs/rescored*.jsonl"):
        for line in open(fn):
            d = json.loads(line)
            rec.setdefault(d["sequence"], d)
    sub = list(csv.DictReader(open("submission.csv")))
    s = [rec[r["sequence"]] for r in sub if r["sequence"] in rec]
    col = lambda k: sorted(float(d[k]) for d in s if d.get(k) is not None)
    ident = lambda a, b: (lambda n: sum(x == y for x, y in zip(a[:n], b[:n])) / n)(
        min(len(a), len(b)))

    L = [len(r["sequence"]) for r in sub]
    check("length range", "55–75", f"{min(L)}–{max(L)}")
    check("max pairwise identity", "18%",
          f"{max(ident(a['sequence'], b['sequence']) for a, b in itertools.combinations(sub, 2)):.0%}")
    check("pool size", "1,228", f"{len(rec):,}")
    check("hisscan variants", "735",
          sum(1 for d in rec.values() if "_his" in d.get("name", "")))
    check("histidines per design", "5–11",
          f"{min(r['sequence'].count('H') for r in sub)}–{max(r['sequence'].count('H') for r in sub)}")
    check("submitted selectivity", "1.74–3.22",
          f"{col('selectivity_log10')[0]:.2f}–{col('selectivity_log10')[-1]:.2f}")
    check("submitted mouse ipTM", "0.811–0.964",
          f"{col('mouse_iptm')[0]:.3f}–{col('mouse_iptm')[-1]:.3f}")
    check("submitted ipTM", "0.765–0.910",
          f"{col('iptm')[0]:.3f}–{col('iptm')[-1]:.3f}")
    check("distinct lineages", 20,
          len({r["name"].split("_his")[0] for r in sub}))
    charge = [sum(r["sequence"].count(c) for c in "KR")
              - sum(r["sequence"].count(c) for c in "DE") for r in sub]
    check("net charge", "\u221212 to +1, median \u22123",
          f"{min(charge)} to {max(charge):+d}, median {st.median(charge):+.0f}"
          .replace("-", "\u2212").replace("\u2212", "\u2212", 1))

    from target import GROOVE_PROTOMER_1, GROOVE_PROTOMER_2, MOUSE_DIVERGENT
    g = set(GROOVE_PROTOMER_1) | set(GROOVE_PROTOMER_2)
    check("groove footprint", 28, len(g))
    check("conserved groove positions", 24, len(g) - len(set(MOUSE_DIVERGENT) & g))

    rows = [(float(d["iptm"]), float(d["plddt_monomer"])) for d in rec.values()
            if d.get("iptm") is not None and d.get("plddt_monomer") is not None]
    mx = st.mean([a for a, _ in rows]); my = st.mean([b for _, b in rows])
    r = (sum((a - mx) * (b - my) for a, b in rows)
         / (sum((a - mx) ** 2 for a, _ in rows) * sum((b - my) ** 2 for _, b in rows)) ** 0.5)
    check("r(ipTM, monomer pLDDT)", "+0.043", f"{r:+.3f}")

    cal = json.load(open("pka_params.json"))
    check("calibration complexes", 316, cal["n_complexes"])
    check("calibration histidines", "1,272", f"{cal['n_histidines']:,}")
    m, n = cal["test_metrics"], cal["test_null_baselines"]
    check("null improvement, pKa", "11%", f"{100*(1-m['mue_pka']/n['mue_pka']):.0f}%")
    check("null improvement, shift", "20%", f"{100*(1-m['mue_shift']/n['mue_shift']):.0f}%")
    check("selectivity rank r", "0.38", f"{m['r_selectivity']:.2f}")

    # pH table, recomputed from the linkage relation itself.
    delta = lambda b, f, pH: math.log10(1 + 10 ** (b - pH)) - math.log10(1 + 10 ** (f - pH))
    sel = lambda b, f: delta(b, f, 7.4) - delta(b, f, 6.0)
    for free, per, need in ((6.0, "0.28", 8), (6.5, "0.57", 4),
                            (7.0, "0.90", 3), (7.5, "1.16", 2)):
        best = max(sel(b / 100, free) for b in range(200, int(free * 100)))
        check(f"pH table, pKa_free {free}", f"{per}/{need}",
              f"{best:.2f}/{math.ceil(2.0/best)}")

    # Upload requirements: name and sequence columns, optional molecule_class
    # from a fixed set, scFv/Fab written {VH}:{VL}.
    ALLOWED = {"single_chain", "nanobody", "scfv", "fab_kappa", "fab_lambda"}
    cols = list(sub[0].keys())
    mc = {r["molecule_class"] for r in sub}
    check("format: columns", "name,sequence,molecule_class", ",".join(cols))
    check("format: molecule_class allowed", True, mc <= ALLOWED)
    check("format: unique names", 20, len({r["name"] for r in sub}))
    check("format: unique sequences", 20, len({r["sequence"] for r in sub}))
    check("format: standard residues only", 0,
          sum(1 for r in sub if set(r["sequence"]) - set("ACDEFGHIKLMNPQRSTVWY")))
    check("format: no stray whitespace", 0,
          sum(1 for r in sub if r["sequence"] != r["sequence"].strip()))
    # A colon would mean a VH:VL pair, which single_chain designs must not have.
    check("format: no VH:VL colon", 0, sum(1 for r in sub if ":" in r["sequence"]))
    check("format: length within 10-250", True,
          all(10 <= len(r["sequence"]) <= 250 for r in sub))

    print("\n  externally verified (needs PROPKA, not re-run here):")
    print("    adalimumab -0.312, infliximab +0.385, infliximab L92 -2.35/+0.33/6.09")
    print("    reproduce with: python -I controls.py --out controls.json")

    # The document uses a Unicode minus (U+2212), not an ASCII hyphen.
    norm = doc.replace("\u2212", "-")
    for figure in ("55\u201375", "18%", "1,228", "735", "5\u201311", "1.74\u20133.22",
                   "0.811\u20130.964", "0.765\u20130.910", "+0.043", "316", "1,272",
                   "0.38", "-0.312", "+0.385"):
        if figure.replace("\u2212", "-") not in norm:
            FAIL.append(f"{figure} absent from METHODS.md")
            print(f"  BAD figure {figure} is not in METHODS.md")

    print(f"\n  {'FAILURES: ' + ', '.join(FAIL) if FAIL else 'all figures verified'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
