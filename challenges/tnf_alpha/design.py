"""De novo minibinder design against the TNF-alpha receptor groove.

Submit via `submit.sh` -- this loads Boltz-2 and must run on a GPU node.

The loss and optimizer schedule follow the recipe that won the in-silico Nipah
competition (blog.escalante.bio, "180 lines of code to win..."), because reading
that post showed this pipeline was weaker than it on every piece of core design
machinery. Deviations from it are limited to what this challenge actually needs:
the epitope restriction, the gravitational term, and the pH objective.

What was adopted, and why each mattered:

  - **Confidence terms** (PAE / iPTM / pTM-energy / pLDDT). Their stated main
    advantage was "optimizing directly on a Boltz-2-based loss that correlates
    extremely well with the contest metric". This file previously had none of
    them in the complex branch, so it optimized proxies for binding while
    ignoring the quantities that track the metric.
  - **ProteinMPNN weight 10.0 at temp 0.001**, up from 5.0 at 0.01. The first
    runs showed sequence_recovery stuck at 0.04-0.17, i.e. designs ProteinMPNN
    did not consider plausible for their own backbone. They call 10.0 "probably
    a bit high" (it encourages homopolymers), so this is the upper end.
  - **A three-stage annealed optimizer** with gradients clipped to norm 1.0 and
    a large stepsize, rather than one stage with an unclipped gradient and a
    small stepsize. Stages 2 and 3 use `scale > 1` in log space, which is
    negative entropic regularization and drives the PSSM toward one-hot. This
    fixes both earlier failures at once: divergence (clip first, then step big)
    and a PSSM that never sharpened (nnz stuck near 7.7).
  - **num_samples=4 during design.** Four samples through one forward pass are
    far cheaper than four runs, and cut gradient variance.
  - **No monomer branch during design.** They optimize the complex only and
    check monomer-vs-complex consistency afterwards, discarding designs that
    change conformation in isolation. That is both cheaper per step and a
    better-targeted check, so the refold now lives in `rescore.py`.
  - **Final PSSM, not the best iterate.** With sharpening stages the last PSSM
    is the answer; taking an early "best" iterate would undo the annealing.

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
    # pKa model rather than a histidine-count proxy. Its parameters beat every
    # null but only by 11-20% (LOG.md entry 8), so it runs as its own pool and
    # is compared against the proxy pools rather than replacing them.
    "physics": dict(his_target=2.0, physics=True),
}

# (n_steps, stepsize factor on sqrt(L), momentum, proximal scale, logspace).
# Stage 1 explores; stages 2-3 sharpen via negative entropic regularization.
STAGES = (
    (100, 0.2, 0.3, 1.00, False),
    (50, 0.5, 0.0, 1.25, True),
    (15, 0.5, 0.0, 1.40, True),
)
# Gradients are clipped to this norm before the step, so a large stepsize is
# safe. The default here is sqrt(L), which with momentum is what diverged.
MAX_GRAD_NORM = 1.0


def build_objective(
    model, binder_length, epitope_idx, his_target, mpnn, gravity_weight=1.0,
    physics=False, num_samples=4,
):
    """Complex-only objective. Monomer consistency is checked in rescore.py."""
    target_chains = [
        TargetChain(HUMAN_TNF, use_msa=True, template_chain=ch)
        for ch in tnf_template_chains(2)
    ]
    features, _ = model.binder_features(binder_length, target_chains)

    # Contacts are restricted to the groove epitope, conserved positions only --
    # this is what buys mouse cross-reactivity (criterion #2) rather than an
    # adalimumab-like human-only footprint.
    loss = (
        1.0 * sp.BinderTargetContact(epitope_idx=epitope_idx, contact_distance=20.0)
        + 1.0 * sp.WithinBinderContact()
        + 10.0 * InverseFoldingSequenceRecovery(mpnn, temp=jnp.array(0.001))
        # Confidence terms, weights from the Nipah recipe.
        + 0.05 * sp.BinderTargetPAE()
        + 0.05 * sp.TargetBinderPAE()
        + 0.4 * sp.WithinBinderPAE()
        + 0.025 * sp.IPTMLoss()
        + 0.025 * sp.pTMEnergy()
        + 0.1 * sp.PLDDTLoss()
    )

    # Signed coordinate-space field: which epitope residues matter, and which to
    # actively avoid (the mouse-divergent ones).
    if gravity_weight:
        loss = loss + gravity_weight * tnf_gravitational_loss()

    if physics:
        # A weak count proxy stays on as a prior: without any pressure to put
        # histidines in the interface at all, the selectivity term has little to
        # act on early in a trajectory.
        loss = loss + 3.0 * ElectrostaticPKaSelectivity(
            epitope_idx=tuple(epitope_idx), params=load_params()
        ) + 0.5 * ph_switch_loss(epitope_idx=epitope_idx, target_count=his_target)
    elif his_target is not None:
        loss = loss + 2.0 * ph_switch_loss(
            epitope_idx=epitope_idx, target_count=his_target
        )

    objective = model.build_multisample_loss(
        loss=loss, features=features, recycling_steps=1, num_samples=num_samples
    )
    # Cysteine is excluded: free cysteines in a secreted minibinder invite
    # disulfide scrambling, and the assay construct is tag-free. NoCys takes a
    # 19-wide simplex, so the optimizer runs in 19 dimensions.
    return NoCys(objective)


def anneal(objective, x, binder_length, key, verbose=True, stage_scale=1.0):
    """Run the three-stage schedule, carrying the PSSM forward between stages."""
    scale_len = float(np.sqrt(binder_length))
    for stage, (n_steps, step_factor, momentum, scale, logspace) in enumerate(STAGES):
        n_steps = max(1, int(round(n_steps * stage_scale)))
        if verbose:
            print(f"  stage {stage + 1}/{len(STAGES)}: {n_steps} steps, "
                  f"stepsize={step_factor * scale_len:.2f}, momentum={momentum}, "
                  f"scale={scale}, logspace={logspace}", flush=True)
        x, _ = simplex_APGM(
            loss_function=objective,
            x=x,
            n_steps=n_steps,
            stepsize=step_factor * scale_len,
            momentum=momentum,
            scale=scale,
            logspace=logspace,
            max_gradient_norm=MAX_GRAD_NORM,
            key=key,
        )
        key = jax.random.fold_in(key, stage)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=sorted(POOLS), default="ph")
    ap.add_argument("--n-designs", type=int, default=8)
    ap.add_argument("--binder-length", type=int, default=75)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--gravity-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    # Scales every stage's step count. Use a small value to verify a config end
    # to end before committing GPU hours -- three launches were burned on
    # configs that failed minutes in (LOG.md entries 11-13).
    ap.add_argument("--stage-scale", type=float, default=1.0)
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

    total_steps = sum(max(1, int(round(s[0] * args.stage_scale))) for s in STAGES)
    print(f"pool={args.pool} his_target={his_target} physics={physics} "
          f"binder_length={args.binder_length} epitope={len(epitope_idx)} positions "
          f"samples={args.num_samples} steps={total_steps}", flush=True)

    model = Boltz2()
    mpnn = load_mpnn_sol(0.05)
    objective = build_objective(
        model, args.binder_length, epitope_idx, his_target, mpnn,
        gravity_weight=args.gravity_weight, physics=physics,
        num_samples=args.num_samples,
    )

    out_path = args.out / f"{args.pool}_L{args.binder_length}.jsonl"
    with out_path.open("a") as fh:
        for i in range(args.n_designs):
            key = jax.random.key(args.seed * 10_000 + i)
            # Gumbel init scaled by a random temperature, so the designs in a
            # pool start from genuinely different places rather than the same
            # near-uniform simplex.
            k_temp, k_gumbel, k_opt = jax.random.split(key, 3)
            temperature = jax.random.uniform(k_temp, minval=0.25, maxval=0.75)
            x = jax.nn.softmax(
                temperature
                * jax.random.gumbel(k_gumbel, shape=(args.binder_length, 19))
            )

            print(f"[design {i + 1}/{args.n_designs}]", flush=True)
            x = anneal(objective, x, args.binder_length, k_opt,
                       stage_scale=args.stage_scale)

            # With sharpening stages the final PSSM is the answer; an earlier
            # "best" iterate would discard the annealing.
            full_pssm = NoCys.sequence(x)
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
                # Sharpness of the final PSSM: how close the annealing got to
                # one-hot. Near 1.0 is converged; well below means the reported
                # sequence is an argmax over a still-fuzzy distribution.
                "pssm_max_mean": float(np.asarray(full_pssm).max(-1).mean()),
            }
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            print(f"[{i + 1}/{args.n_designs}] {record['name']} "
                  f"nHis={record['n_his']} "
                  f"sharp={record['pssm_max_mean']:.3f} {seq}", flush=True)

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
