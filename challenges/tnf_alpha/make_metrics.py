"""Regenerate submission_metrics.csv from submission.csv and the scored pool.

METHODS.md points reviewers at this file, and Track 3 pools submitted
information for selection, so a stale copy misdescribes the designs being
judged. The version dated before the last two selection changes shared only 11
of 20 names with the submission it claimed to describe. Generating it from the
submission rather than maintaining it by hand removes that failure mode.
"""
import csv, glob, json
from pathlib import Path

RATIONALE = {
    "hisscan": "affinity-optimised backbone, interface histidines installed post hoc by scan",
    "ph": "in-loop histidine budget 3 with the differentiable linkage objective",
    "ph_hot": "in-loop histidine budget 5 with the differentiable linkage objective",
    "affinity": "affinity-only objective, no in-loop pH term",
    "physics": "differentiable proton-linkage objective in the design loss",
}

COLUMNS = ["name", "sequence", "molecule_class", "length", "n_histidine",
           "ph_selectivity_log10", "iptm_human", "iptm_mouse",
           "ipsae_binder_target", "ipsae_target_binder", "plddt_complex",
           "plddt_monomer", "monomer_complex_rmsd_A", "net_charge",
           "novelty_structural_tier", "design_pool", "rationale"]


def main():
    rec = {}
    for fn in glob.glob("designs/rescored*.jsonl"):
        for line in open(fn):
            d = json.loads(line)
            rec.setdefault(d["sequence"], d)
    tiers = {}
    for fn in glob.glob("probe/novelty*.jsonl"):
        for line in open(fn):
            d = json.loads(line)
            if d.get("structural_tier"):
                tiers[d["name"]] = d["structural_tier"]

    rows, missing = [], []
    for r in csv.DictReader(open("submission.csv")):
        d = rec.get(r["sequence"])
        if d is None:
            missing.append(r["name"])
            continue
        seq = r["sequence"]
        pool = d.get("pool") or "unknown"
        if "_his" in r["name"]:
            pool = "hisscan"
        num = lambda k, nd=3: (None if d.get(k) is None else round(float(d[k]), nd))
        rows.append({
            "name": r["name"], "sequence": seq,
            "molecule_class": r.get("molecule_class", "single_chain"),
            "length": len(seq), "n_histidine": seq.count("H"),
            "ph_selectivity_log10": num("selectivity_log10"),
            "iptm_human": num("iptm"), "iptm_mouse": num("mouse_iptm"),
            "ipsae_binder_target": num("ipsae_bt"),
            "ipsae_target_binder": num("ipsae_tb"),
            "plddt_complex": num("plddt_complex"),
            "plddt_monomer": num("plddt_monomer"),
            "monomer_complex_rmsd_A": num("monomer_complex_rmsd", 1),
            "net_charge": sum(seq.count(c) for c in "KR") - sum(seq.count(c) for c in "DE"),
            "novelty_structural_tier": tiers.get(r["name"], "unscreened"),
            "design_pool": pool,
            "rationale": RATIONALE.get(pool, ""),
        })
    if missing:
        raise SystemExit(f"no scored record for: {missing}")

    with open("submission_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote submission_metrics.csv: {len(rows)} rows")
    from collections import Counter
    print(f"  pools: {dict(Counter(r['design_pool'] for r in rows))}")
    print(f"  novelty: {dict(Counter(r['novelty_structural_tier'] for r in rows))}")


if __name__ == "__main__":
    main()
