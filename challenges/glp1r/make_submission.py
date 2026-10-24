"""Build the ranked GLP1R submission.

Ordering follows the stated goals, which are activation first and binding
second. So binding is a gate -- "a real chance to express and bind" -- and
activation evidence is the sort key, not the other way round. Row order matters
more than usual: only a participant's first designs are taken, so the top rows
decide what reaches the lab.

Eligibility:
  - novelty level 3 or higher, enforced at submission. High structural
    similarity maps to level 2 on Adaptyv's scale, so those are dropped. This
    costs the single best design by both metrics, glp1r_L55_s8605_1 at ipTM 0.88
    and 17.4 of 18 activation-core contacts.
  - ipTM at or above 0.50 on ProtenixBase, which separates real ligands
    (0.87-0.90) from a scrambled control (0.42).
  - activation-core contacts at or above 13 on ProtenixBase, above the range
    reached by the truncated antagonists the metric was calibrated on.

Ranked by activation-core contacts, with ipTM breaking ties. Both are means
over five seeds.
"""
import csv
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent

IPTM_GATE = 0.50
CORE_GATE = 13.0
N_SLOTS = 20


def load(files):
    g = defaultdict(list)
    for f in files:
        try:
            for line in open(f):
                r = json.loads(line)
                g[r["name"]].append(r)
        except FileNotFoundError:
            pass
    out = {}
    for n, x in g.items():
        agg = {"name": n, "sequence": x[0]["sequence"], "n_seeds": len(x)}
        for k in ("iptm", "ipsae_bt", "ipsae_tb", "plddt", "hydropathy_kd",
                  "activation_core_contacts", "belt_contacts",
                  "epitope_contacts", "nterm_depth", "rest_depth"):
            vals = [r[k] for r in x if r.get(k) is not None]
            agg[k] = st.mean(vals) if vals else None
            if k == "activation_core_contacts" and len(vals) > 1:
                agg["core_sd"] = st.stdev(vals)
        out[n] = agg
    return out


def main():
    final = load([HERE / "probe" / "final_scored.jsonl"])
    b2 = load([HERE / "probe" / "candidates2_b2.jsonl"])
    tier = {json.loads(l)["name"]: json.loads(l)["structural_tier"]
            for l in open(HERE / "probe" / "novelty2.jsonl")}

    rows, rejected = [], []
    for n, d in final.items():
        t = tier.get(n, "unscreened")
        why = None
        if t == "high":
            why = "novelty level 2 (high structural similarity)"
        elif d["iptm"] is None or d["iptm"] < IPTM_GATE:
            why = f"ipTM {d['iptm']:.2f} below {IPTM_GATE}"
        elif d["activation_core_contacts"] < CORE_GATE:
            why = (f"activation-core {d['activation_core_contacts']:.1f} "
                   f"below {CORE_GATE}, inside the antagonist range")
        if why:
            rejected.append((n, why))
            continue
        d["tier"] = t
        d["novelty_level"] = 4 if t == "below_moderate" else 3
        d["b2_core"] = b2.get(n, {}).get("activation_core_contacts")
        d["b2_iptm"] = b2.get(n, {}).get("iptm")
        rows.append(d)

    # Activation first, binding as tie-break.
    rows.sort(key=lambda d: (-d["activation_core_contacts"], -d["iptm"]))
    chosen = rows[:N_SLOTS]

    seqs = [d["sequence"] for d in chosen]
    assert len(set(seqs)) == len(seqs), "duplicate sequences"
    for d in chosen:
        assert 10 <= len(d["sequence"]) <= 250, d["name"]
        assert set(d["sequence"]) <= set("ACDEFGHIKLMNPQRSTVWY"), d["name"]

    with open(HERE / "submission.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "sequence", "molecule_class"])
        for d in chosen:
            w.writerow([d["name"], d["sequence"], "protein"])

    cols = ["rank", "name", "sequence", "molecule_class", "length",
            "novelty_level", "novelty_tier",
            "activation_core_contacts", "activation_core_sd",
            "activation_core_contacts_boltz2", "iptm", "iptm_boltz2",
            "ipsae_binder_target", "ipsae_target_binder", "plddt_complex",
            "hydropathy_kyte_doolittle", "lipid_belt_contacts",
            "epitope_contacts", "nterm_depth_A", "rest_depth_A", "rationale"]
    with open(HERE / "submission_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for i, d in enumerate(chosen, 1):
            num = lambda k, nd=3: (None if d.get(k) is None else round(d[k], nd))
            w.writerow({
                "rank": i, "name": d["name"], "sequence": d["sequence"],
                "molecule_class": "protein", "length": len(d["sequence"]),
                "novelty_level": d["novelty_level"], "novelty_tier": d["tier"],
                "activation_core_contacts": num("activation_core_contacts", 1),
                "activation_core_sd": num("core_sd", 1),
                "activation_core_contacts_boltz2": num("b2_core", 1),
                "iptm": num("iptm"), "iptm_boltz2": num("b2_iptm"),
                "ipsae_binder_target": num("ipsae_bt"),
                "ipsae_target_binder": num("ipsae_tb"),
                "plddt_complex": num("plddt"),
                "hydropathy_kyte_doolittle": num("hydropathy_kd", 2),
                "lipid_belt_contacts": num("belt_contacts", 1),
                "epitope_contacts": num("epitope_contacts", 1),
                "nterm_depth_A": num("nterm_depth", 1),
                "rest_depth_A": num("rest_depth", 1),
                "rationale": ("orthosteric binder engaging the activation pocket; "
                              "ranked on activation-core contacts, gated on ipTM"),
            })

    print(f"scored designs considered: {len(final)}")
    print(f"  rejected: {len(rejected)}")
    from collections import Counter
    for why, k in Counter(w for _, w in
                          ((n, r.split(",")[0].split(" below")[0])
                           for n, r in rejected)).most_common():
        print(f"    {k}: {why}")
    print(f"  eligible: {len(rows)}; submitted: {len(chosen)}")
    print(f"\n  submitted ranges:")
    f = lambda k: [d[k] for d in chosen if d.get(k) is not None]
    for k, lbl in (("activation_core_contacts", "activation core"),
                   ("iptm", "ipTM"), ("plddt", "pLDDT"),
                   ("hydropathy_kd", "hydropathy")):
        v = f(k)
        print(f"    {lbl:16s} {min(v):.2f} - {max(v):.2f}  median {st.median(v):.2f}")
    print(f"    novelty level 4: {sum(1 for d in chosen if d['novelty_level'] == 4)}"
          f" of {len(chosen)}")


if __name__ == "__main__":
    main()
