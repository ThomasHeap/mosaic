"""Tests for two of the stated limitations, run on the submitted designs.

Both limitations are stated in METHODS.md as unresolved. They are testable, and
the tests are cheap, so leaving them unresolved was a matter of time rather than
of principle.

**Test 1 -- is the low monomer pLDDT real, or an artefact?**

Every submitted design has monomer pLDDT near 0.5, which would normally be read
as "does not fold on its own" and hence an expression risk. But Boltz-2 is being
asked to fold a 55-90 residue de novo sequence with no MSA, which is a regime
where co-folding models are known to be unreliable. The reading is only
meaningful relative to a control.

So the same prediction is run on proteins of similar size that are *known* to
fold: GB1 (56 aa, one of the best-characterised small domains) and Top7 (93 aa,
the original de novo design, experimentally confirmed). If the controls also come
out near 0.5, the metric is uninformative here and the concern is an artefact. If
they come out near 0.85, the designs really are predicted to be floppy.

**Test 2 -- does the interface survive without a template?**

Designs were optimised against a TNF dimer templated on 1TNF. If evaluation
co-folds TNF from sequence alone, an interface that only appears with the
template would not reproduce. Re-predicting the submitted complexes without a
template and comparing ipTM measures how much of the predicted interface quality
depends on the template.

    python diagnostics.py --test monomer
    python diagnostics.py --test template
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import jax
import numpy as np

import mosaic.losses.structure_prediction as sp
from mosaic.models.boltz2 import Boltz2
from mosaic.structure_prediction import TargetChain

from target import HUMAN_TNF, tnf_template_chains

# Positive controls for the monomer test: similar length to the designs, and
# both known to fold. GB1 is a natural 56-residue domain; Top7 is the original
# de novo design and was confirmed crystallographically.
CONTROLS = {
    "GB1_1PGB_56aa": (
        "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
    ),
    "Top7_1QYS_93aa": (
        "MGDIQVQVNIDDNGKNFDYTYTVTTESELQKVLNELMDYIKKQGAKRVRISITARTKKEAEKFAAIL"
        "IKVFAELGYNDINVTFDGDTVTVEGQLE"
    ),
}


def predict_monomer(model, seq, key, recycling_steps=3, sampling_steps=50):
    features, writer = model.target_only_features(
        [TargetChain(seq, use_msa=False)]
    )
    return model.predict(
        features=features, writer=writer, recycling_steps=recycling_steps,
        sampling_steps=sampling_steps, key=key,
    )


def predict_complex(model, seq, key, templated, recycling_steps=3,
                    sampling_steps=50):
    if templated:
        tnf = [
            TargetChain(HUMAN_TNF, use_msa=True, template_chain=ch)
            for ch in tnf_template_chains(2)
        ]
    else:
        tnf = [TargetChain(HUMAN_TNF, use_msa=True) for _ in range(2)]
    features, writer = model.target_only_features(
        [TargetChain(seq, use_msa=False)] + tnf
    )
    return model.predict(
        features=features, writer=writer, recycling_steps=recycling_steps,
        sampling_steps=sampling_steps, key=key,
    )


def iptm_of(pred, binder_len, key):
    value, _ = sp.IPTMLoss()(np.zeros((binder_len, 20)), pred.model_output, key)
    return -float(value)


def load_submission(path="submission.csv"):
    import csv
    return list(csv.DictReader(open(path)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["monomer", "template", "repro"],
                    required=True)
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--csv", default="submission.csv")
    args = ap.parse_args()

    if jax.devices()[0].platform != "gpu":
        raise SystemExit("needs a GPU; use submit_diagnostics.sh")

    rows = load_submission(args.csv)
    if args.limit:
        rows = rows[: args.limit]
    model = Boltz2()
    out_path = args.out or Path(f"diagnostics_{args.test}.jsonl")

    if args.test == "monomer":
        print("controls first -- these calibrate what a 'good' value looks like",
              flush=True)
        results = []
        for name, seq in CONTROLS.items():
            pred = predict_monomer(model, seq, jax.random.key(0))
            plddt = float(np.asarray(pred.plddt).mean())
            results.append({"name": name, "kind": "control",
                            "length": len(seq), "plddt_monomer": plddt})
            print(f"  CONTROL {name}: plddt_monomer = {plddt:.3f}", flush=True)
        for i, row in enumerate(rows):
            seq = row["sequence"]
            pred = predict_monomer(model, seq, jax.random.key(100 + i))
            plddt = float(np.asarray(pred.plddt).mean())
            results.append({"name": row["name"], "kind": "design",
                            "length": len(seq), "plddt_monomer": plddt})
            print(f"  [{i + 1}/{len(rows)}] {row['name']}: {plddt:.3f}",
                  flush=True)
        with out_path.open("w") as fh:
            for r in results:
                fh.write(json.dumps(r) + "\n")
        ctrl = [r["plddt_monomer"] for r in results if r["kind"] == "control"]
        des = [r["plddt_monomer"] for r in results if r["kind"] == "design"]
        print(f"\ncontrols: {min(ctrl):.3f}-{max(ctrl):.3f}   "
              f"designs: {min(des):.3f}-{max(des):.3f} "
              f"(median {float(np.median(des)):.3f})")
        print("If the controls sit near the designs, this metric says nothing "
              "about these designs. If the controls are much higher, the "
              "designs really are predicted not to fold alone.")

    elif args.test == "repro":
        # Precision, not accuracy. There is no public dataset of pH-dependent
        # binding affinities paired with structures, so the selectivity score
        # cannot be checked against experiment. What can be checked is whether
        # it is even reproducible: if independent predictions of the same
        # sequence disagree by more than the differences we rank on, the
        # ranking is noise regardless of whether the physics is right.
        from ph_score import score_pair
        results = []
        for i, row in enumerate(rows):
            seq = row["sequence"]
            vals = []
            for rep in range(args.replicates):
                k = jax.random.key(1000 * (rep + 1) + i)
                cx = predict_complex(model, seq, k, templated=True)
                mono = predict_monomer(model, seq, jax.random.fold_in(k, 1))
                try:
                    ph = score_pair(cx.st, mono.st,
                                    binder_chain=cx.st[0][0].name)
                    vals.append(ph["selectivity_log10"])
                except Exception as exc:  # noqa: BLE001
                    print(f"    rep {rep} failed: {type(exc).__name__}: {exc}")
            if not vals:
                continue
            rec = {"name": row["name"], "selectivity_replicates": vals,
                   "mean": float(np.mean(vals)), "sd": float(np.std(vals)),
                   "range": float(max(vals) - min(vals))}
            results.append(rec)
            print(f"  [{i + 1}/{len(rows)}] {row['name']}: "
                  f"{[round(v, 2) for v in vals]}  sd={rec['sd']:.3f}  "
                  f"range={rec['range']:.3f}", flush=True)
        with out_path.open("w") as fh:
            for r in results:
                fh.write(json.dumps(r) + "\n")
        sds = [r["sd"] for r in results]
        rngs = [r["range"] for r in results]
        print(f"\nwithin-design sd of selectivity: median "
              f"{float(np.median(sds)):.3f}, max {max(sds):.3f}")
        print(f"within-design range: median {float(np.median(rngs)):.3f}")
        print("Ranking buckets selectivity at 0.5 log10. If the median "
              "within-design sd approaches that, the bucket width is doing all "
              "the work and finer ordering is meaningless.")

    else:
        results = []
        for i, row in enumerate(rows):
            seq = row["sequence"]
            k = jax.random.key(200 + i)
            t = predict_complex(model, seq, k, templated=True)
            u = predict_complex(model, seq, k, templated=False)
            rec = {
                "name": row["name"],
                "iptm_templated": iptm_of(t, len(seq), k),
                "iptm_untemplated": iptm_of(u, len(seq), k),
                "plddt_templated": float(np.asarray(t.plddt).mean()),
                "plddt_untemplated": float(np.asarray(u.plddt).mean()),
            }
            rec["iptm_delta"] = rec["iptm_untemplated"] - rec["iptm_templated"]
            results.append(rec)
            print(f"  [{i + 1}/{len(rows)}] {row['name']}: "
                  f"templated {rec['iptm_templated']:.3f} -> untemplated "
                  f"{rec['iptm_untemplated']:.3f} "
                  f"({rec['iptm_delta']:+.3f})", flush=True)
        with out_path.open("w") as fh:
            for r in results:
                fh.write(json.dumps(r) + "\n")
        d = [r["iptm_delta"] for r in results]
        kept = sum(1 for r in results if r["iptm_untemplated"] >= 0.70)
        print(f"\nmedian ipTM change without the template: "
              f"{float(np.median(d)):+.3f}")
        print(f"{kept}/{len(results)} still clear ipTM 0.70 untemplated")

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
