"""Design TfR1 binders.

Criteria, hardest first: pH-selective binding (measurable KD at 7.4, none at
6.0), then mouse cross-reactivity, then human affinity. The same three as TNF
Track 3, so the objective is largely that one with two substitutions.

**The epitope is not hand-picked.** The challenge hints at the apical domain
around Tyr211, but Tyr211 differs in mouse (Y to D), and only 2 of the 9
residues the 11rn apical binder touches are conserved. Since cross-reactivity
outranks affinity, following the hint optimises the third criterion at the cost
of the second. Instead the gravitational term weights every allowed residue by
mouse conservation -- conserved pulls, divergent pushes -- and the optimiser
chooses where to bind. `BinderTargetContact` is given the whole allowed region
rather than a patch, so it supplies general interface pressure while the signed
field decides placement.

**No in-loop pH term.** On TNF the pH switch came from post-hoc interface
histidine scanning, not the design loss: raw design output cleared all three
criteria at 0.4% against 3.7% for scanned variants, and 19 of 20 submitted
designs were variants. Placing histidines is a substitution on a fixed backbone,
scored by a backbone-only pKa model with no refolding, so it installs after the
fact for no GPU cost. Designing with a histidine budget in the loop produced
designs with a median of *zero* histidines at the interface. So this objective
optimises affinity and placement only; hisscan.py installs the switch.

The target is a 192-residue templated apical domain, not the 1278-residue
homodimer: the apical site sits on one protomer with the other 39 A away, and
189-380 makes 85 cross-domain CA contacts against 924 internal.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import mosaic.losses.structure_prediction as sp
from gravity import tfr1_gravitational_loss
from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
from mosaic.losses.transformations import NoCys
from mosaic.optimizers import simplex_APGM
from mosaic.proteinmpnn.mpnn import load_mpnn_sol
from mosaic.structure_prediction import TargetChain
from target import (allowed_region, apical_template_chain, chain_sequence,
                    protomer_template_chain, to_target_index)

CONFIG = "tfr1-conservation-field-v1"

# Same three-stage schedule as TNF: explore, then sharpen twice under negative
# entropic regularisation.
STAGES = (
    (100, 0.2, 0.3, 1.00, False),
    (50, 0.5, 0.0, 1.25, True),
    (15, 0.5, 0.0, 1.40, True),
)
MAX_GRAD_NORM = 1.0


def build_objective(model, binder_length, mpnn, *, gravity_weight=1.0,
                    num_samples=4, full_protomer=False, esm_weight=0.0,
                    esmc=None):
    chain = (protomer_template_chain() if full_protomer
             else apical_template_chain())
    seq = chain_sequence(chain)
    target = [TargetChain(seq, use_msa=True, template_chain=chain)]
    features, _ = model.binder_features(binder_length, target)

    # The whole allowed region, loosely: general interface pressure. Placement
    # within it is the signed field's job, not this term's.
    allowed = to_target_index(chain, allowed_region(chain))

    loss = (
        1.0 * sp.BinderTargetContact(epitope_idx=allowed, contact_distance=20.0)
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

    # Signed coordinate-space field: conserved surface attracts, divergent
    # surface repels, boundary and tag-proximal residues repel hard. This is
    # what buys criterion 2, and it is why no epitope is hand-picked.
    if gravity_weight:
        loss = loss + gravity_weight * tfr1_gravitational_loss(chain)

    objective = model.build_multisample_loss(
        loss=loss, features=features, recycling_steps=1, num_samples=num_samples
    )

    # Composition guard. Off by default: on GLP1R the designs ran 42-50% charged
    # against 23% for the real ligands, which no structure-based score detects.
    if esm_weight and esmc is not None:
        from mosaic.losses.esmc import ESMCPseudoPerplexity
        objective = objective + esm_weight * ESMCPseudoPerplexity(esm=esmc)

    # Cysteine is excluded: a free cysteine in a small secreted binder invites
    # disulfide scrambling, and TfR1's own ectodomain is disulfide-rich. The
    # optimiser runs in 19 dimensions and NoCys.sequence() reinserts the column.
    return NoCys(objective)


def anneal(objective, x, binder_length, key, stage_scale=1.0, verbose=True):
    scale_len = float(np.sqrt(binder_length))
    for stage, (n_steps, step_factor, momentum, scale, logspace) in enumerate(STAGES):
        n_steps = max(1, int(round(n_steps * stage_scale)))
        if verbose:
            print(f"  stage {stage + 1}/{len(STAGES)}: {n_steps} steps", flush=True)
        x, _ = simplex_APGM(
            loss_function=objective, x=x, n_steps=n_steps,
            stepsize=step_factor * scale_len, momentum=momentum, scale=scale,
            logspace=logspace, max_gradient_norm=MAX_GRAD_NORM, key=key,
        )
        key = jax.random.fold_in(key, stage)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-designs", type=int, default=4)
    # TNF's submitted binders were 55-75 aa and the 11rn apical binder is 62,
    # so this range is where comparable binders to this surface sit.
    ap.add_argument("--binder-length", type=int, default=65)
    ap.add_argument("--gravity-weight", type=float, default=1.0)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--esm-weight", type=float, default=0.0)
    ap.add_argument("--full-protomer", action="store_true",
                    help="template the whole 639-residue protomer instead of "
                         "the apical domain; for checking that truncation did "
                         "not change predicted binding")
    ap.add_argument("--stage-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("designs"))
    args = ap.parse_args()

    chain = (protomer_template_chain() if args.full_protomer
             else apical_template_chain())
    print(f"config={CONFIG} length={args.binder_length} "
          f"gravity={args.gravity_weight} esm={args.esm_weight} "
          f"target={'protomer' if args.full_protomer else 'apical'} "
          f"({len(chain)} residues, {len(allowed_region(chain))} allowed)",
          flush=True)

    from mosaic.models.boltz2 import Boltz2
    model = Boltz2()
    mpnn = load_mpnn_sol(0.05)
    esmc = None
    if args.esm_weight:
        from mosaic.losses.esmc import load_esmc
        esmc = load_esmc(model_name="esmc_300m")

    objective = build_objective(
        model, args.binder_length, mpnn, gravity_weight=args.gravity_weight,
        num_samples=args.num_samples, full_protomer=args.full_protomer,
        esm_weight=args.esm_weight, esmc=esmc,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    tag = "prot" if args.full_protomer else "api"
    path = args.out / f"{tag}_L{args.binder_length}_s{args.seed}.jsonl"
    from mosaic.common import TOKENS
    with open(path, "w") as fh:
        for i in range(args.n_designs):
            key = jax.random.PRNGKey(args.seed * 1000 + i)
            x = jax.random.uniform(key, (args.binder_length, 19)) * 0.1
            t0 = time.time()
            x = anneal(objective, x, args.binder_length, key,
                       stage_scale=args.stage_scale)
            elapsed = time.time() - t0
            full = np.asarray(NoCys.sequence(jax.nn.softmax(x)))
            seq = "".join(TOKENS[j] for j in full.argmax(-1))
            assert "C" not in seq, seq
            rec = {
                "name": f"tfr1_{tag}_L{args.binder_length}_s{args.seed}_{i}",
                "sequence": seq, "molecule_class": "protein", "config": CONFIG,
                "gravity_weight": args.gravity_weight,
                "esm_weight": args.esm_weight,
                "binder_length": args.binder_length, "seed": args.seed,
                "full_protomer": args.full_protomer, "seconds": elapsed,
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{i+1}/{args.n_designs}] {rec['name']} {elapsed:.0f}s\n  {seq}",
                  flush=True)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
