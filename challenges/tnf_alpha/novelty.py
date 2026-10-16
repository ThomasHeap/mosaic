"""Screen designs against Adaptyv's novelty criteria.

Novelty is a *hard* filter in this competition, not a preference, and nothing in
this pipeline measured it: rank.py dedupes lineages against each other, which
says nothing about similarity to known proteins. Adaptyv's ProteinTyper scores
1-4 and only level 4 counts as de novo:

  level 4 (de novo)  sequence identity <= 30%  AND  below moderate structural
  structural moderate   > 70% coverage with TM-score >= 0.5
  structural high       >= 70% coverage with TM-score >= 0.8

Sequence novelty is near-certain for hallucinated designs, since there is no
homology to anything. The exposure is structural: these are small helical
bundles and the PDB is full of helical bundles, which is the failure mode
Adaptyv call out directly -- de novo status gets harder as design methods reuse
common structural elements.

Takes the monomer structures rescore.py already wrote, so no refolding.
"""
import argparse, csv, json, os, subprocess, sys
from pathlib import Path

FOLDSEEK = "/software/local/apps/foldseek/bin/foldseek"
# qcov is the fraction of the query covered; alntmscore is foldseek's TM-score.
FIELDS = "query,target,fident,alnlen,qcov,tcov,alntmscore,evalue,prob"

def classify(hits):
    """Adaptyv's structural tiers, applied to the best hit by TM-score."""
    high = [h for h in hits if h["qcov"] >= 0.70 and h["tm"] >= 0.80]
    mod  = [h for h in hits if h["qcov"] >  0.70 and h["tm"] >= 0.50]
    if high: return "high", high[0]
    if mod:  return "moderate", mod[0]
    return "below_moderate", (max(hits, key=lambda h: h["tm"]) if hits else None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="submission.csv")
    ap.add_argument("--struct-dir", default="structures")
    ap.add_argument("--db", default="/user/work/cp20141/foldseek_db/pdb")
    ap.add_argument("--out", default="probe/novelty.jsonl")
    ap.add_argument("--tmp", default="/user/work/cp20141/foldseek_db/tmp_search")
    args = ap.parse_args()

    seqs = {r["name"]: r["sequence"] for r in csv.DictReader(open(args.submission))}
    # rescore.py names structures by design name; submission.csv carries the same.
    pdbs = {n: f"{args.struct_dir}/{n}_monomer.pdb" for n in seqs}
    missing = [n for n, p in pdbs.items() if not os.path.exists(p)]
    if missing:
        print(f"WARNING: {len(missing)} designs have no monomer structure, skipped:")
        for n in missing[:5]: print(f"  {n}")
    present = {n: p for n, p in pdbs.items() if os.path.exists(p)}
    if not present:
        sys.exit("no structures found; nothing to screen")

    qdir = Path(args.tmp) / "query"
    qdir.mkdir(parents=True, exist_ok=True)
    for n, p in present.items():
        (qdir / f"{n}.pdb").write_bytes(Path(p).read_bytes())

    res = Path(args.tmp) / "hits.tsv"
    cmd = [FOLDSEEK, "easy-search", str(qdir), args.db, str(res),
           str(Path(args.tmp) / "work"), "--format-output", FIELDS,
           "--exhaustive-search", "0", "-e", "10", "--max-seqs", "1000",
           "--threads", "8"]
    print("running foldseek easy-search ...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        sys.exit(f"foldseek failed ({r.returncode})")

    byq = {}
    for line in res.read_text().splitlines():
        f = line.split("\t")
        if len(f) < 9: continue
        q = f[0].replace(".pdb", "")
        byq.setdefault(q, []).append({
            "target": f[1], "fident": float(f[2]), "qcov": float(f[4]),
            "tcov": float(f[5]), "tm": float(f[6]), "evalue": float(f[7])})

    rows, flagged = [], 0
    print(f"\n{'design':44s} {'tier':>15} {'bestTM':>7} {'qcov':>6}  top hit")
    for n in present:
        hits = sorted(byq.get(n, []), key=lambda h: -h["tm"])
        tier, best = classify(hits)
        if tier != "below_moderate": flagged += 1
        bt = f"{best['tm']:.3f}" if best else "-"
        bc = f"{best['qcov']:.2f}" if best else "-"
        tgt = best["target"][:22] if best else "(no hits)"
        print(f"{n[:44]:44s} {tier:>15} {bt:>7} {bc:>6}  {tgt}")
        rows.append({"name": n, "structural_tier": tier,
                     "best_tm": best["tm"] if best else None,
                     "best_qcov": best["qcov"] if best else None,
                     "best_target": best["target"] if best else None,
                     "n_hits": len(hits)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for r_ in rows: fh.write(json.dumps(r_) + "\n")
    print(f"\n{flagged}/{len(present)} designs at moderate or high structural "
          f"similarity -- those are NOT level 4 de novo")
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
