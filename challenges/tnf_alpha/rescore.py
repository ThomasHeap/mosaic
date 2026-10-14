"""Repredict and score finished designs, populating the fields rank.py reads.

Without this step `rank.py` is inert: it sorts on `selectivity_log10`, `iptm`
and `mouse_iptm`, all of which default to neutral when absent, so an unrescored
pool comes out in arbitrary order. Generating designs and not rescoring them is
the one shortcut the Nipah writeup explicitly calls out as "much, much more
risky".

Four things are measured per design, in the order the challenge ranks them:

  1. **pH selectivity** (criterion #1) -- PROPKA on the predicted complex and on
     the predicted binder alone, combined through the proton-linkage relation in
     `ph_score`. This is the Tier 2 number; the in-loop physics term is only a
     gradient direction (LOG.md entry 8).
  2. **Mouse cross-reactivity** (criterion #2) -- the same complex refolded
     against mouse TNF. A design that only binds human loses this criterion, and
     the conserved-epitope restriction is supposed to prevent that, so this is
     the direct test of whether it worked.
  3. **Affinity proxies** (criterion #3) -- iPTM plus both ipSAE directions, the
     ranking loss from the Nipah recipe, at `recycling_steps=3` with several
     samples to damp seed variance.
  4. **Monomer/complex consistency** -- C-alpha RMSD of the binder predicted
     alone versus in the complex. The Nipah authors discarded designs that
     "adopted wildly different conformations" in isolation; a binder that only
     folds when bound is unlikely to express or behave.

Reprediction uses *target-only* features for every chain, so the binder carries
real reference atoms and sidechains. Design-time features substitute UNK/G and
produce no sidechains, which is fine for a gradient but wrong for scoring.

    python rescore.py designs/*.jsonl --out designs/rescored.jsonl
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import gemmi
import jax
import jax.numpy as jnp
import numpy as np

import mosaic.losses.structure_prediction as sp
from mosaic.common import TOKENS
from mosaic.models.boltz2 import Boltz2
from mosaic.structure_prediction import TargetChain

from ph_score import score_pair
from target import HUMAN_TNF, MOUSE_TNF, tnf_template_chains


def one_hot(seq: str) -> jnp.ndarray:
    return jax.nn.one_hot(jnp.asarray([TOKENS.index(c) for c in seq]), 20)


def ca_rmsd(st_a: gemmi.Structure, chain_a: str,
            st_b: gemmi.Structure, chain_b: str) -> float | None:
    """C-alpha RMSD between the same chain in two predictions, superposed."""
    try:
        poly_a = st_a[0][chain_a].get_polymer()
        poly_b = st_b[0][chain_b].get_polymer()
        sup = gemmi.calculate_superposition(
            poly_a, poly_b, gemmi.PolymerType.PeptideL, gemmi.SupSelect.CaP
        )
        return float(sup.rmsd)
    except Exception:  # noqa: BLE001 - a missing chain must not stop the sweep
        return None


def predict(model, chains, pssm, *, recycling_steps, sampling_steps, key):
    features, writer = model.target_only_features(chains)
    return model.predict(
        PSSM=pssm, features=features, writer=writer,
        recycling_steps=recycling_steps, sampling_steps=sampling_steps, key=key,
    )


def ranking_scores(pred, binder_len: int, key) -> dict:
    """iPTM and both ipSAE directions -- the Nipah ranking loss."""
    seq_dummy = jnp.zeros((binder_len, 20))
    out = pred.model_output
    scores = {}
    for name, term in (
        ("iptm", sp.IPTMLoss()),
        ("ipsae_bt", sp.BinderTargetIPSAE()),
        ("ipsae_tb", sp.TargetBinderIPSAE()),
    ):
        try:
            value, _ = term(seq_dummy, out, key)
            # These terms return a loss (negated score); report the score.
            scores[name] = -float(value)
        except Exception as exc:  # noqa: BLE001
            scores[name] = None
            scores[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
    return scores


def rescore_one(model, record, *, recycling_steps, sampling_steps, key, mouse,
                struct_dir=None):
    seq = record["sequence"]
    pssm_complex = None  # full-sequence features carry identity already
    binder = TargetChain(seq, use_msa=False)
    tnf_templates = tnf_template_chains(2)

    human_chains = [binder] + [
        TargetChain(HUMAN_TNF, use_msa=True, template_chain=ch)
        for ch in tnf_templates
    ]
    k1, k2, k3 = jax.random.split(key, 3)

    pred_complex = predict(
        model, human_chains, pssm_complex,
        recycling_steps=recycling_steps, sampling_steps=sampling_steps, key=k1,
    )
    pred_mono = predict(
        model, [binder], pssm_complex,
        recycling_steps=recycling_steps, sampling_steps=sampling_steps, key=k2,
    )

    out = dict(record)
    out.update(ranking_scores(pred_complex, len(seq), k1))
    out["plddt_complex"] = float(np.asarray(pred_complex.plddt).mean())
    out["plddt_monomer"] = float(np.asarray(pred_mono.plddt).mean())
    # Keep the structures: the histidine scan (hisscan.py) needs the predicted
    # complex backbone, and refolding every design again on a GPU just to
    # recover it would be wasteful when the scan itself is backbone-only.
    if struct_dir is not None:
        struct_dir = Path(struct_dir)
        struct_dir.mkdir(parents=True, exist_ok=True)
        for tag, st in (("complex", pred_complex.st), ("monomer", pred_mono.st)):
            path = struct_dir / f"{record['name']}_{tag}.pdb"
            try:
                st.write_pdb(str(path))
                out[f"{tag}_pdb"] = str(path)
            except Exception as exc:  # noqa: BLE001
                out[f"{tag}_pdb_error"] = f"{type(exc).__name__}: {exc}"

    out["monomer_complex_rmsd"] = ca_rmsd(
        pred_complex.st, pred_complex.st[0][0].name,
        pred_mono.st, pred_mono.st[0][0].name,
    )

    # Criterion #1: PROPKA on both predicted end states, through the linkage
    # relation. Binder is chain A in both predictions.
    try:
        ph = score_pair(
            pred_complex.st, pred_mono.st,
            binder_chain=pred_complex.st[0][0].name,
        )
        out["selectivity_log10"] = ph["selectivity_log10"]
        out["ph_n_his"] = ph["n_his"]
        out["ph_histidines"] = ph["histidines"]
    except Exception as exc:  # noqa: BLE001
        out["selectivity_log10"] = None
        out["ph_error"] = f"{type(exc).__name__}: {exc}"

    # Criterion #2: the same complex against mouse TNF. No template -- the mouse
    # trimer has no deposited structure here, so it is co-folded from sequence.
    if mouse:
        try:
            mouse_chains = [binder] + [
                TargetChain(MOUSE_TNF, use_msa=True) for _ in range(2)
            ]
            pred_mouse = predict(
                model, mouse_chains, pssm_complex,
                recycling_steps=recycling_steps,
                sampling_steps=sampling_steps, key=k3,
            )
            ms = ranking_scores(pred_mouse, len(seq), k3)
            out["mouse_iptm"] = ms.get("iptm")
            out["mouse_ipsae_bt"] = ms.get("ipsae_bt")
        except Exception as exc:  # noqa: BLE001
            out["mouse_iptm"] = None
            out["mouse_error"] = f"{type(exc).__name__}: {exc}"

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--out", type=Path, default=Path("designs/rescored.jsonl"))
    ap.add_argument("--recycling-steps", type=int, default=3)
    ap.add_argument("--sampling-steps", type=int, default=50)
    ap.add_argument("--no-mouse", action="store_true",
                    help="skip the mouse cross-reactivity refold (saves ~1/3)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--struct-dir", type=Path, default=Path("structures"),
                    help="where to write predicted complex/monomer PDBs")
    args = ap.parse_args()

    if not jax.devices()[0].platform == "gpu":
        raise SystemExit("rescoring needs a GPU; use submit_rescore.sh")

    records, seen = [], set()
    for path in args.jsonl:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["sequence"] in seen:
                continue
            seen.add(r["sequence"])
            records.append(r)
    if args.limit:
        records = records[: args.limit]
    print(f"rescoring {len(records)} unique designs", flush=True)

    model = Boltz2()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as fh:
        for i, record in enumerate(records):
            key = jax.random.key(i)
            try:
                out = rescore_one(
                    model, record,
                    recycling_steps=args.recycling_steps,
                    sampling_steps=args.sampling_steps,
                    key=key, mouse=not args.no_mouse,
                    struct_dir=args.struct_dir,
                )
            except Exception as exc:  # noqa: BLE001 - keep the sweep going
                print(f"  [{i + 1}/{len(records)}] {record['name']} FAILED: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                continue
            fh.write(json.dumps(out) + "\n")
            fh.flush()
            print(f"  [{i + 1}/{len(records)}] {out['name']} "
                  f"iptm={out.get('iptm')} "
                  f"sel={out.get('selectivity_log10')} "
                  f"mouse_iptm={out.get('mouse_iptm')} "
                  f"rmsd={out.get('monomer_complex_rmsd')}", flush=True)

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
