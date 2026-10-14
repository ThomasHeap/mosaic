"""Histidine-scan a finished binder to install a pH switch after the fact.

This is the route the lab actually uses. pH-dependent ("recycling") antibodies
are not designed with histidines in from the start -- a working binder is found
first, then its interface is histidine-scanned and the variants are screened.
Doing it in that order avoids the failure the in-loop objective hit: penalizing
a contact-weighted histidine *count* is cheapest to satisfy by raising
histidine everywhere, and probe 19400994 produced a 32%-histidine 75-mer
(LOG.md entry 14).

Why restricting histidines to the epitope cannot be done during design: which
binder positions contact the epitope is an *output* of optimization, not an
input. At initialization every position is a candidate, and `mosaic`'s masks are
static. After the fact the contacts are known, so the restriction becomes
trivial.

Why this is cheap. `ph_physics.predict_pka` is backbone-only by construction --
virtual C-beta, no sidechains, because design-time features have none. So a
substitution changes only the charge vector, not the geometry, and a whole scan
is a few hundred evaluations of a small JAX function. No refolding, no GPU. The
rigid-backbone assumption is the same one JustHISpKa makes by working from the
complex structure.

What it does NOT do: it will not tell you a histidine is tolerated
*structurally*. A position can be a fine switch electrostatically and still be
one the fold needs for packing. Top variants should be refolded before
submission -- `--emit` writes them as design records for `rescore.py`.

One thing to keep in mind from the linkage analysis: selectivity rises with
pKa_free, which needs acidic neighbours in the *free* binder. Those neighbours
need not be at contact positions, so this scan restricts histidines to the
interface but never restricts anything else.

    python -I hisscan.py designs/rescored.jsonl --out designs/hisscan.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import gemmi
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gravity import pseudo_cb  # noqa: E402
from mosaic.common import TOKENS  # noqa: E402
from ph_physics import (  # noqa: E402
    SIDECHAIN_CHARGE,
    linkage_selectivity,
    load_params,
    predict_pka,
)
from target import HUMAN_TNF, groove_epitope_idx  # noqa: E402


def backbone_of(chain) -> tuple[np.ndarray, list[int], str]:
    """(backbone [n,4,3], residue numbers, one-letter sequence) for a chain."""
    bb, nums, seq = [], [], []
    for res in chain.get_polymer():
        atoms = {a.name: a for a in res}
        if not {"N", "CA", "C"} <= atoms.keys():
            continue
        o = atoms.get("O", atoms["C"])
        bb.append([
            [atoms[n].pos.x, atoms[n].pos.y, atoms[n].pos.z] for n in ("N", "CA", "C")
        ] + [[o.pos.x, o.pos.y, o.pos.z]])
        nums.append(int(res.seqid.num))
        info = gemmi.find_tabulated_residue(res.name)
        seq.append(info.one_letter_code.upper() if info else "X")
    return np.asarray(bb, dtype=np.float32), nums, "".join(seq)


def load_complex(path: Path):
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_ligands_and_waters()
    st.remove_alternative_conformations()
    chains = [c for c in st[0] if len(c.get_polymer()) > 0]
    if len(chains) < 2:
        raise ValueError(f"{path}: need a binder and at least one target chain")
    # rescore.py writes the binder first.
    return chains[0], chains[1:]


def charge_vector(seq: str) -> np.ndarray:
    table = np.asarray(SIDECHAIN_CHARGE)
    return np.asarray(
        [table[TOKENS.index(c)] if c in TOKENS else 0.0 for c in seq],
        dtype=np.float32,
    )


def interface_positions(
    binder_cb: np.ndarray, target_cb: np.ndarray, cutoff: float
) -> list[int]:
    """Binder indices within `cutoff` of any epitope C-beta."""
    d = np.linalg.norm(
        binder_cb[:, None, :] - target_cb[None, :, :], axis=-1
    )
    return [int(i) for i in np.flatnonzero(d.min(-1) <= cutoff)]


def score_variant(
    binder_cb, target_cb, binder_seq, target_charge, his_at, params
) -> float:
    """Total log10 selectivity if `his_at` positions were histidine."""
    seq = list(binder_seq)
    for i in his_at:
        seq[i] = "H"
    q = jnp.asarray(charge_vector("".join(seq)))
    pka_free, pka_bound = predict_pka(
        jnp.asarray(binder_cb), jnp.asarray(target_cb), q,
        jnp.asarray(target_charge), params,
    )
    idx = jnp.asarray(list(his_at))
    return float(
        linkage_selectivity(pka_bound[idx], pka_free[idx]).sum()
    )


def scan_one(record, *, max_his, cutoff, top_singles, params):
    pdb = record.get("complex_pdb")
    if not pdb or not Path(pdb).exists():
        return None
    binder_chain, target_chains = load_complex(Path(pdb))
    b_bb, _, b_seq = backbone_of(binder_chain)
    if not len(b_bb):
        return None
    binder_cb = np.asarray(pseudo_cb(jnp.asarray(b_bb)))

    t_bb, t_seq = [], []
    for ch in target_chains:
        bb, _, seq = backbone_of(ch)
        if len(bb):
            t_bb.append(bb)
            t_seq.append(seq)
    if not t_bb:
        return None
    target_cb_all = np.asarray(pseudo_cb(jnp.asarray(np.concatenate(t_bb))))
    target_seq_all = "".join(t_seq)

    # Restrict to the conserved groove epitope where the indices line up with a
    # 2 x 157 target; otherwise fall back to the whole target surface.
    epitope = groove_epitope_idx(n_protomers=2, conserved_only=True)
    if len(target_cb_all) == 2 * len(HUMAN_TNF):
        target_cb = target_cb_all[epitope]
        target_charge = charge_vector("".join(target_seq_all[i] for i in epitope))
    else:
        target_cb = target_cb_all
        target_charge = charge_vector(target_seq_all)

    positions = interface_positions(binder_cb, target_cb, cutoff)
    # Never propose replacing a residue that is already histidine, and leave
    # glycine and proline alone -- both are usually there for backbone reasons.
    positions = [
        i for i in positions if b_seq[i] not in ("H", "G", "P")
    ]
    if not positions:
        return None

    singles = sorted(
        (
            (score_variant(binder_cb, target_cb, b_seq, target_charge, (i,), params), i)
            for i in positions
        ),
        reverse=True,
    )
    best = [i for _, i in singles[:top_singles]]

    variants = []
    for k in range(1, max_his + 1):
        for combo in itertools.combinations(best, k):
            sel = score_variant(
                binder_cb, target_cb, b_seq, target_charge, combo, params
            )
            seq = list(b_seq)
            for i in combo:
                seq[i] = "H"
            variants.append({
                "his_positions": [int(i) for i in combo],
                "selectivity_log10": round(sel, 3),
                "sequence": "".join(seq),
                "n_his_added": len(combo),
            })
    variants.sort(key=lambda v: -v["selectivity_log10"])

    return {
        "name": record["name"],
        "parent_sequence": b_seq,
        "parent_iptm": record.get("iptm"),
        "interface_positions": positions,
        "singles": [
            {"position": int(i), "selectivity_log10": round(s, 3)}
            for s, i in singles[:top_singles]
        ],
        "variants": variants,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+", help="rescored designs (need complex_pdb)")
    ap.add_argument("--out", type=Path, default=Path("designs/hisscan.jsonl"))
    ap.add_argument("--emit", type=Path, default=None,
                    help="also write top variants as design records for rescore.py")
    ap.add_argument("--max-his", type=int, default=4,
                    help="largest combination to enumerate; the linkage table "
                         "says 2 log10 needs ~4 ordinary histidines")
    ap.add_argument("--cutoff", type=float, default=10.0,
                    help="C-beta distance to the epitope counted as interface")
    ap.add_argument("--top-singles", type=int, default=8,
                    help="best single positions kept before enumerating combos")
    ap.add_argument("--emit-per-design", type=int, default=3)
    args = ap.parse_args()

    params = load_params()
    records = []
    for path in args.jsonl:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"scanning {len(records)} designs", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    emitted = 0
    with args.out.open("w") as fh:
        emit_fh = args.emit.open("w") if args.emit else None
        try:
            for record in records:
                try:
                    result = scan_one(
                        record, max_his=args.max_his, cutoff=args.cutoff,
                        top_singles=args.top_singles, params=params,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  {record.get('name')}: {type(exc).__name__}: {exc}")
                    continue
                if result is None:
                    print(f"  {record.get('name')}: no scannable interface")
                    continue
                fh.write(json.dumps(result) + "\n")
                best = result["variants"][0]
                print(f"  {result['name']}: {len(result['interface_positions'])} "
                      f"interface positions, best {best['n_his_added']} His -> "
                      f"{best['selectivity_log10']:+.2f} log10 "
                      f"at {best['his_positions']}", flush=True)
                if emit_fh:
                    for v in result["variants"][: args.emit_per_design]:
                        emit_fh.write(json.dumps({
                            "name": f"{result['name']}_his{len(v['his_positions'])}"
                                    f"_{'_'.join(str(i) for i in v['his_positions'])}",
                            "sequence": v["sequence"],
                            "molecule_class": "protein",
                            "pool": "hisscan",
                            "parent": result["name"],
                            "predicted_selectivity_log10": v["selectivity_log10"],
                            "n_his": v["sequence"].count("H"),
                        }) + "\n")
                        emitted += 1
        finally:
            if emit_fh:
                emit_fh.close()

    print(f"wrote {args.out}")
    if args.emit:
        print(f"wrote {emitted} variants to {args.emit} -- refold these with "
              f"rescore.py before trusting them")


if __name__ == "__main__":
    main()
