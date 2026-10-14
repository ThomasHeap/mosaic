"""Fit `ph_physics.PKaParams` against PROPKA labels on real protein complexes.

Why this needs neither designs nor a GPU. The loss term consumes *backbone only*
-- a virtual C-beta built from N/CA/C, because design-time features in `mosaic`
carry no sidechains. So it can be evaluated on any crystal structure by throwing
that structure's sidechains away, while PROPKA supplies labels computed from the
full structure. Features are restricted to what design will actually see;
labels come from the best source available. That is the right setup, and it
also removes the distribution-shift worry that makes distilling PROPKA into a
learned surrogate awkward.

Both end states come from one complex: PROPKA on the whole thing gives
pKa_bound for the chosen chain's histidines, PROPKA on that chain alone gives
pKa_free. Exactly the decomposition `predict_pka` makes.

The fit reports two errors, and the second is the one that matters. A model can
track absolute pKa well while getting the *shift* between end states wrong, and
selectivity depends only on the shift.

    python calibrate.py --fetch 150 --data-dir pdb_cache --out params.json

Runs on CPU. Use `submit_calibrate.sh` for the real run; PROPKA on a few hundred
complexes is minutes of compute but it does not belong on the login node.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gemmi
import jax.numpy as jnp
import numpy as np

# This script is meant to be run with `python -I` because it parses downloaded
# structure files. That drops the script's own directory from sys.path, so the
# sibling modules need putting back explicitly -- the cached structures live in
# a subdirectory and are not importable, so this adds no exposure.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gravity import pseudo_cb  # noqa: E402
from ph_physics import (  # noqa: E402
    HIS_MODEL_PKA,
    SIDECHAIN_CHARGE,
    PKaParams,
    linkage_selectivity,
    predict_pka,
)
from ph_score import propka_pkas  # noqa: E402

RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_FILE = "https://files.rcsb.org/download/{}.cif"

# Parameters the fit is free to move, with bounds that keep them physical.
# burial_cutoff / burial_width / burial_reference stay fixed: they define the
# burial coordinate itself, and letting them float trades off against
# alpha_desolv without changing the model's expressiveness.
FIT_SPEC = [
    ("epsilon_solvent", 40.0, 120.0),
    ("epsilon_interior", 2.0, 40.0),
    ("screening_length", 3.0, 20.0),
    ("alpha_desolv", 0.0, 10.0),
    ("offset", -3.0, 3.0),
]


def fetch_ids(n: int) -> list[str]:
    """Protein-protein complexes: two protein entities, decent resolution."""
    import urllib.request

    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "equals",
                        "value": 2,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less",
                        "value": 2.0,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                        "operator": "range",
                        "value": {"from": 120, "to": 600},
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": n},
            "sort": [{"sort_by": "score", "direction": "desc"}],
        },
    }
    req = urllib.request.Request(
        RCSB_SEARCH,
        data=json.dumps(query).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as fh:
        result = json.load(fh)
    return [r["identifier"] for r in result.get("result_set", [])]


def download(ids: list[str], data_dir: Path) -> list[Path]:
    """Download cif files into their own directory, skipping ones already there."""
    import urllib.error
    import urllib.request

    data_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for pdb_id in ids:
        path = data_dir / f"{pdb_id}.cif"
        if not path.exists():
            try:
                with urllib.request.urlopen(
                    RCSB_FILE.format(pdb_id), timeout=60
                ) as fh:
                    path.write_bytes(fh.read())
            except (urllib.error.URLError, TimeoutError) as exc:
                print(f"  skip {pdb_id}: {exc}")
                continue
        paths.append(path)
    return paths


def backbone_and_charge(chain) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """(backbone [n,4,3], charge [n], residue numbers) for one chain.

    Residues missing any of N/CA/C are dropped -- a virtual C-beta needs all
    three, and crystal structures have gaps.
    """
    bb, charges, resnums = [], [], []
    charge_table = np.asarray(SIDECHAIN_CHARGE)
    from mosaic.common import TOKENS

    for res in chain.get_polymer():
        atoms = {a.name: a for a in res}
        if not {"N", "CA", "C"} <= atoms.keys():
            continue
        o = atoms.get("O", atoms["C"])
        bb.append([
            [atoms["N"].pos.x, atoms["N"].pos.y, atoms["N"].pos.z],
            [atoms["CA"].pos.x, atoms["CA"].pos.y, atoms["CA"].pos.z],
            [atoms["C"].pos.x, atoms["C"].pos.y, atoms["C"].pos.z],
            [o.pos.x, o.pos.y, o.pos.z],
        ])
        info = gemmi.find_tabulated_residue(res.name)
        aa = info.one_letter_code.upper() if info else "X"
        charges.append(charge_table[TOKENS.index(aa)] if aa in TOKENS else 0.0)
        resnums.append(int(res.seqid.num))
    return (
        np.asarray(bb, dtype=np.float32),
        np.asarray(charges, dtype=np.float32),
        resnums,
    )


def build_examples(paths: list[Path]) -> list[dict]:
    """One example per usable complex: backbone features plus PROPKA labels."""
    examples = []
    for path in paths:
        try:
            st = gemmi.read_structure(str(path))
            st.setup_entities()
            st.remove_ligands_and_waters()
            st.remove_alternative_conformations()
            chains = [c for c in st[0] if len(c.get_polymer()) >= 40]
            if len(chains) < 2:
                continue
            # Smallest chain plays the binder; it is the closest analogue to a
            # minibinder sitting on a larger target.
            binder = min(chains, key=lambda c: len(c.get_polymer()))
            his = [
                r.seqid.num
                for r in binder.get_polymer()
                if r.name == "HIS"
            ]
            if not his:
                continue

            mono = st.clone()
            for name in [c.name for c in mono[0]]:
                if name != binder.name:
                    mono[0].remove_chain(name)

            bound = propka_pkas(st)
            free = propka_pkas(mono)

            b_bb, b_q, b_res = backbone_and_charge(binder)
            t_bb, t_q = [], []
            for c in st[0]:
                if c.name == binder.name:
                    continue
                bb, q, _ = backbone_and_charge(c)
                if len(bb):
                    t_bb.append(bb)
                    t_q.append(q)
            if not t_bb or not len(b_bb):
                continue
            t_bb = np.concatenate(t_bb)
            t_q = np.concatenate(t_q)

            index = {r: i for i, r in enumerate(b_res)}
            sites, y_free, y_bound = [], [], []
            for resnum in his:
                key = (binder.name, resnum)
                if key not in bound or key not in free or resnum not in index:
                    continue
                sites.append(index[resnum])
                y_free.append(free[key])
                y_bound.append(bound[key])
            if not sites:
                continue

            examples.append({
                "id": path.stem,
                "binder_bb": b_bb,
                "binder_q": b_q,
                "target_bb": t_bb,
                "target_q": t_q,
                "sites": np.asarray(sites, dtype=np.int32),
                "y_free": np.asarray(y_free, dtype=np.float32),
                "y_bound": np.asarray(y_bound, dtype=np.float32),
            })
            print(f"  {path.stem}: chain {binder.name}, {len(sites)} His")
        except Exception as exc:  # noqa: BLE001 - one bad entry must not stop the fit
            print(f"  skip {path.stem}: {type(exc).__name__}: {exc}")
    return examples


ARRAY_KEYS = ("binder_bb", "binder_q", "target_bb", "target_q",
              "sites", "y_free", "y_bound")


def save_examples(examples: list[dict], path: Path) -> None:
    """Flatten the examples into one npz so a refit does not re-run PROPKA."""
    flat = {"ids": np.array([e["id"] for e in examples])}
    for i, ex in enumerate(examples):
        for key in ARRAY_KEYS:
            flat[f"{i}_{key}"] = ex[key]
    np.savez_compressed(path, n=np.array(len(examples)), **flat)


def load_examples(path: Path) -> list[dict]:
    data = np.load(path, allow_pickle=False)
    n = int(data["n"])
    ids = data["ids"]
    return [
        {"id": str(ids[i]), **{k: data[f"{i}_{k}"] for k in ARRAY_KEYS}}
        for i in range(n)
    ]


def null_baselines(data: list[dict]) -> dict:
    """What a model that has learned nothing would score.

    Two nulls matter. Constant pKa at the model value is the trivial predictor
    for absolute pKa, and zero shift is the trivial predictor for the quantity
    selectivity actually depends on. A published benchmark of fast pKa methods
    found none of them beat these with confidence, so the fitted model has to be
    checked against them rather than praised for a plausible-looking MUE.
    """
    yf = jnp.concatenate([jnp.asarray(e["y_free"]) for e in data])
    yb = jnp.concatenate([jnp.asarray(e["y_bound"]) for e in data])
    const = jnp.full_like(yf, HIS_MODEL_PKA)
    mue_pka = float(jnp.mean(jnp.abs(jnp.concatenate([const - yf, const - yb]))))
    mue_shift = float(jnp.mean(jnp.abs(yb - yf)))
    sel_true = linkage_selectivity(yb, yf)
    # Zero shift means pka_bound == pka_free, hence exactly zero selectivity.
    mue_sel = float(jnp.mean(jnp.abs(sel_true)))
    return {"mue_pka": mue_pka, "mue_shift": mue_shift,
            "mue_selectivity": mue_sel}


def predict_all(examples: list[dict], params: PKaParams):
    """Model pKa at every labelled site, concatenated across examples."""
    free_out, bound_out = [], []
    for ex in examples:
        b_cb = pseudo_cb(jnp.asarray(ex["binder_bb"]))
        t_cb = pseudo_cb(jnp.asarray(ex["target_bb"]))
        pka_free, pka_bound = predict_pka(
            b_cb,
            t_cb,
            jnp.asarray(ex["binder_q"]),
            jnp.asarray(ex["target_q"]),
            params,
        )
        sites = jnp.asarray(ex["sites"])
        free_out.append(pka_free[sites])
        bound_out.append(pka_bound[sites])
    return jnp.concatenate(free_out), jnp.concatenate(bound_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", type=int, default=0,
                    help="query RCSB for this many complexes before fitting")
    ap.add_argument("--data-dir", type=Path, default=Path("pdb_cache"))
    ap.add_argument("--out", type=Path, default=Path("pka_params.json"))
    ap.add_argument("--holdout", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--relabel", action="store_true",
                    help="re-run PROPKA even if cached labels exist")
    ap.add_argument("--fetch-only", action="store_true",
                    help="download and exit; compute nodes may have no network")
    args = ap.parse_args()

    if args.fetch:
        print(f"querying RCSB for {args.fetch} complexes")
        ids = fetch_ids(args.fetch)
        print(f"  got {len(ids)} ids; downloading")
        paths = download(ids, args.data_dir)
    else:
        paths = sorted(args.data_dir.glob("*.cif"))
    print(f"{len(paths)} structures available")
    if args.fetch_only:
        print("fetch-only: stopping before the fit")
        return

    # PROPKA labelling is ~25 min for a few hundred complexes, while a refit is
    # seconds. Caching the labels is what makes iterating on the functional form
    # practical rather than an overnight loop.
    cache = args.data_dir / "labels.npz"
    if cache.exists() and not args.relabel:
        examples = load_examples(cache)
        print(f"loaded {len(examples)} labelled complexes from {cache}")
    else:
        print("labelling with PROPKA (both end states)")
        examples = build_examples(paths)
        save_examples(examples, cache)
        print(f"cached labels to {cache}")
    n_sites = sum(len(e["sites"]) for e in examples)
    print(f"{len(examples)} complexes, {n_sites} labelled histidines")
    if n_sites < 20:
        raise SystemExit("too few labelled histidines to fit; fetch more")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(examples))
    n_test = max(1, int(args.holdout * len(examples)))
    test = [examples[i] for i in order[:n_test]]
    train = [examples[i] for i in order[n_test:]]
    print(f"split: {len(train)} train / {len(test)} test complexes")

    names = [n for n, _, _ in FIT_SPEC]
    lo = np.array([l for _, l, _ in FIT_SPEC])
    hi = np.array([h for _, _, h in FIT_SPEC])
    x0 = np.array([getattr(PKaParams(), n) for n in names], dtype=float)

    def to_params(x):
        clipped = np.clip(x, lo, hi)
        return PKaParams(**{n: float(v) for n, v in zip(names, clipped)})

    def objective(x, data):
        params = to_params(x)
        pf, pb = predict_all(data, params)
        yf = jnp.concatenate([jnp.asarray(e["y_free"]) for e in data])
        yb = jnp.concatenate([jnp.asarray(e["y_bound"]) for e in data])
        # Fit both end states and the shift. Without the shift term the fit can
        # satisfy absolute pKa while getting the difference -- the only thing
        # selectivity depends on -- systematically wrong.
        mse = jnp.mean((pf - yf) ** 2) + jnp.mean((pb - yb) ** 2)
        shift_mse = jnp.mean(((pb - pf) - (yb - yf)) ** 2)
        return float(mse + 2.0 * shift_mse)

    from scipy.optimize import minimize

    print("fitting by Nelder-Mead (as JustHISpKa fits its two scalars)")
    result = minimize(
        objective, x0, args=(train,), method="Nelder-Mead",
        options={"maxiter": 2000, "fatol": 1e-4, "xatol": 1e-3},
    )
    fitted = to_params(result.x)
    print(f"  converged={result.success} after {result.nit} iterations")

    def report(tag, data, params):
        pf, pb = predict_all(data, params)
        yf = jnp.concatenate([jnp.asarray(e["y_free"]) for e in data])
        yb = jnp.concatenate([jnp.asarray(e["y_bound"]) for e in data])
        mue_abs = float(
            jnp.mean(jnp.abs(jnp.concatenate([pf - yf, pb - yb])))
        )
        mue_shift = float(jnp.mean(jnp.abs((pb - pf) - (yb - yf))))
        sel_pred = linkage_selectivity(pb, pf)
        sel_true = linkage_selectivity(yb, yf)
        mue_sel = float(jnp.mean(jnp.abs(sel_pred - sel_true)))
        corr = float(jnp.corrcoef(sel_pred, sel_true)[0, 1]) if len(
            sel_pred
        ) > 2 else float("nan")
        print(f"  {tag:18s} MUE(pKa)={mue_abs:.3f}  MUE(shift)={mue_shift:.3f}  "
              f"MUE(sel)={mue_sel:.3f}  r(sel)={corr:+.3f}")
        return {"mue_pka": mue_abs, "mue_shift": mue_shift,
                "mue_selectivity": mue_sel, "r_selectivity": corr}

    null = null_baselines(test)
    print("\nnull baselines on test (a model that learned nothing):")
    print(f"  constant pKa={HIS_MODEL_PKA}  MUE(pKa)={null['mue_pka']:.3f}")
    print(f"  zero shift              MUE(shift)={null['mue_shift']:.3f}  "
          f"MUE(sel)={null['mue_selectivity']:.3f}")

    print("\nbefore fit (physically scaled guesses):")
    report("train", train, PKaParams())
    base_test = report("test", test, PKaParams())
    print("after fit:")
    report("train", train, fitted)
    metrics = report("test", test, fitted)

    payload = {
        "params": {n: getattr(fitted, n) for n in names},
        "fixed": {
            n: getattr(fitted, n)
            for n in ("burial_cutoff", "burial_width", "burial_reference")
        },
        "n_complexes": len(examples),
        "n_histidines": n_sites,
        "test_metrics": metrics,
        "test_metrics_before_fit": base_test,
        "test_null_baselines": null,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")

    # The decision this run exists to inform.
    n_test_sites = sum(len(e["sites"]) for e in test)
    print("\nverdict:")
    if len(test) < 10 or n_test_sites < 40:
        # A correlation over a couple of complexes is noise, and it will look
        # spectacular roughly half the time. Refuse to call it.
        print(f"  INCONCLUSIVE: only {len(test)} test complexes / "
              f"{n_test_sites} histidines. Need >=10 and >=40 to judge; "
              f"the selectivity correlation above is not meaningful at this size.")
    elif metrics["mue_shift"] >= null["mue_shift"]:
        print(f"  REJECT: MUE(shift)={metrics['mue_shift']:.3f} is no better "
              f"than predicting zero shift ({null['mue_shift']:.3f}). The model "
              f"adds nothing on the quantity selectivity depends on.")
    elif metrics["mue_shift"] < 0.5 and metrics["r_selectivity"] > 0.5:
        print("  shift error and selectivity correlation are usable -- promote"
              " ph_physics from gradient-direction to a weighted objective.")
    else:
        print("  shift error too large or selectivity uncorrelated. Keep"
              " ph_physics as a gradient direction only and rank on ph_score.")
        print("  PROPKA's own MUE is ~0.7-1.0 pKa units, so part of this floor"
              " is label noise rather than model error.")


if __name__ == "__main__":
    main()
