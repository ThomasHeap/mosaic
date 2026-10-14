"""De novo minibinder design against the TNF-alpha receptor groove.

Submit via `submit.sh` -- this loads Boltz-2 and must run on a GPU node.

Three sub-pools, because the challenge's ranked criteria pull against each other
and 20-40 slots is enough to hedge rather than bet:

  affinity  -- no histidine constraint, contacts biased to mouse-conserved
               epitope residues. Baseline; maximizes the chance of *any* hit.
  ph        -- interface histidine budget on top of the affinity objective.
               Targets criterion #1 (pH-selective binding, ranked first).
  ph_hot    -- a larger histidine budget. Deliberately over-pushed: if the
               moderate budget turns out not to switch, this brackets it.

Usage:
    python design.py --pool ph --n-designs 8 --binder-length 75 --out designs/
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import jax
import jax.numpy as jnp
import numpy as np

import mosaic.losses.structure_prediction as sp
from mosaic.common import TOKENS
from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
from mosaic.losses.transformations import NoCys
from mosaic.models.boltz2 import Boltz2
from mosaic.optimizers import simplex_APGM
from mosaic.proteinmpnn.mpnn import load_mpnn_sol
from mosaic.structure_prediction import TargetChain

from gravity import tnf_gravitational_loss
from ph_physics import ElectrostaticPKaSelectivity, load_params
from ph_switch import ph_switch_loss
from target import HUMAN_TNF, groove_epitope_idx, tnf_template_chains

POOLS = {
    "affinity": dict(his_target=None, physics=False),
    "ph": dict(his_target=3.0, physics=False),
    "ph_hot": dict(his_target=5.0, physics=False),
    # Optimizes the scored quantity (log10 selectivity) through a differentiable
    # pKa model rather than a histidine-count proxy. Its parameters are not yet
    # calibrated -- see ph_physics.calibrate -- so it runs as its own pool and
    # is compared against the proxy pools rather than replacing them.
    "physics": dict(his_target=2.0, physics=True),
}


def build_objective(
    model, binder_length, epitope_idx, his_target, mpnn, gravity_weight=2.0,
    physics=False,
):
    """Complex + monomer objective for one sub-pool."""
    target_chains = [
        TargetChain(HUMAN_TNF, use_msa=True, template_chain=ch)
        for ch in tnf_template_chains(2)
    ]

    complex_features, _ = model.binder_features(binder_length, target_chains)
    monomer_features, _ = model.binder_features(binder_length, [])

    # Contacts are restricted to the groove epitope, conserved positions only --
    # this is what buys mouse cross-reactivity (criterion #2) rather than an
    # adalimumab-like human-only footprint.
    interface = (
        4.0 * sp.BinderTargetContact(epitope_idx=epitope_idx, contact_distance=20.0)
        + 1.0 * sp.WithinBinderContact()
        + 1.0 * sp.DistogramRadiusOfGyration(target_radius=15.0)
        + 0.3 * sp.HelixLoss()
        + 5.0 * InverseFoldingSequenceRecovery(mpnn, temp=jnp.array(0.01))
    )
    # Signed coordinate-space field on top of the distogram contact term. The
    # distogram term says "touch the groove"; this one says which residues of it
    # matter and which to actively avoid (the six that diverge in mouse).
    if gravity_weight:
        interface = interface + gravity_weight * tnf_gravitational_loss()

    if physics:
        # A mild count proxy stays on as a prior: the physics term rewards
        # selectivity wherever it can find it, and without any pressure to put
        # histidines in the interface at all it has little to work with early on.
        interface = interface + 3.0 * ElectrostaticPKaSelectivity(
            epitope_idx=tuple(epitope_idx), params=load_params()
        ) + 0.5 * ph_switch_loss(epitope_idx=epitope_idx, target_count=his_target)
    elif his_target is not None:
        interface = interface + 2.0 * ph_switch_loss(
            epitope_idx=epitope_idx, target_count=his_target
        )

    complex_loss = model.build_loss(
        loss=interface, features=complex_features, recycling_steps=1
    )
    # The monomer term is what keeps the binder a real, foldable protein rather
    # than a surface-complementary ribbon that only exists in the complex.
    monomer_loss = model.build_loss(
        loss=0.2 * sp.PLDDTLoss()
        + sp.DistogramRadiusOfGyration(target_radius=15.0)
        + 0.3 * sp.HelixLoss(),
        features=monomer_features,
        recycling_steps=1,
    )

    # Cysteine is excluded: free cysteines in a secreted minibinder invite
    # disulfide scrambling, and the assay construct is tag-free.
    return NoCys(complex_loss + 0.5 * monomer_loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=sorted(POOLS), default="ph")
    ap.add_argument("--n-designs", type=int, default=8)
    ap.add_argument("--binder-length", type=int, default=75)
    ap.add_argument("--n-steps", type=int, default=150)
    # Probe 19400552 diverged with (0.1, 0.9): best loss at step 8, then the
    # binder walked off the epitope (d_attract 9.3 -> 18.1 A) and monomer pLDDT
    # fell 0.87 -> 0.55. Lower momentum is the standard fix for an accelerated
    # method overshooting; simplex_APGM returns the best iterate so divergence
    # does not corrupt the output, it just wastes most of the budget.
    ap.add_argument("--stepsize", type=float, default=0.08)
    ap.add_argument("--momentum", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    # Was 2.0, but grav swung 2.7 -> 10.8 in the probe and dominated the loss
    # changes. Scaled back so it steers without driving the trajectory.
    ap.add_argument("--gravity-weight", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=Path("designs"))
    args = ap.parse_args()

    if not jax.devices()[0].platform == "gpu":
        raise SystemExit(
            f"refusing to run on {jax.devices()[0].platform}: submit with submit.sh"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    epitope_idx = groove_epitope_idx(n_protomers=2, conserved_only=True)
    his_target = POOLS[args.pool]["his_target"]
    physics = POOLS[args.pool]["physics"]

    print(f"pool={args.pool} his_target={his_target} physics={physics} "
          f"binder_length={args.binder_length} epitope={len(epitope_idx)} positions")

    model = Boltz2()
    mpnn = load_mpnn_sol(0.05)
    objective = build_objective(
        model, args.binder_length, epitope_idx, his_target, mpnn,
        gravity_weight=args.gravity_weight, physics=physics,
    )

    out_path = args.out / f"{args.pool}_L{args.binder_length}.jsonl"
    with out_path.open("a") as fh:
        for i in range(args.n_designs):
            key = jax.random.key(args.seed * 10_000 + i)
            # The objective is wrapped in NoCys, which takes a 19-dim simplex
            # and reinserts a zero column at the cysteine index. So the
            # optimizer works in 19 dimensions, and the result has to be mapped
            # back to the 20-token alphabet before decoding -- as NoCys's own
            # docstring warns. Initialising at 20 instead trips an assertion
            # ~7 minutes in, after the model has loaded.
            x0 = jax.nn.softmax(
                0.5 * jax.random.gumbel(key, shape=(args.binder_length, 19))
            )
            _, pssm = simplex_APGM(
                loss_function=objective,
                x=x0,
                n_steps=args.n_steps,
                stepsize=args.stepsize,
                momentum=args.momentum,
                key=jax.random.fold_in(key, 1),
            )
            full_pssm = NoCys.sequence(pssm)
            seq = "".join(TOKENS[j] for j in np.asarray(full_pssm).argmax(-1))
            assert "C" not in seq, "NoCys decode leaked a cysteine"
            record = {
                "name": f"tnf_{args.pool}_L{args.binder_length}_s{args.seed}_{i}",
                "sequence": seq,
                "molecule_class": "protein",
                "pool": args.pool,
                "his_target": his_target,
                "physics": physics,
                "n_his": seq.count("H"),
            }
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            print(f"[{i + 1}/{args.n_designs}] {record['name']} "
                  f"nHis={record['n_his']} {seq}")

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
