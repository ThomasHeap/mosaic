"""Design GLP1R agonist candidates.

Three target-side constraints, all derived from 6x18 and all measured rather
than chosen:

  attract  the orthosteric epitope, 52 residues of the GLP-1 footprint
  require  ACTIVATION_CORE, the 18 residues GLP-1 His7-Thr11 contacts
  repel    the lipid-facing belt, 65 residues

The activation requirement is the point of the whole objective. The challenge
recommends targeting the extracellular peptide-binding groove, but that is the
affinity half of a two-domain mechanism: of the 52 footprint residues, 30 are
transmembrane, and GLP-1's N-terminal half makes about 40 transmembrane contacts
against 5 to the ECD while its C-terminal half makes about 30 ECD contacts
against 11. A design confined to the groove is a competitive antagonist, and
objective 02 discounts anything that does not activate. Truncating two residues
from GLP-1's N-terminus costs 100-300x affinity but 1000-10000x cAMP potency.

The repulsion exists because the assay protein sits in DDM/CHS detergent, so the
lipid-facing belt is solvent-exposed in the construct and no structure model
here represents a membrane. The belt is both larger and more hydrophobic than
the epitope (65 residues at 72% against 52 at 52%), so it is the easier surface
to bind. A design that bound it would score on objective 01, where affinity is
measured on the solubilised construct, and fail objective 02, since in a cell
that surface is buried in lipid.

Activation is not scored by an active-minus-inactive binding differential. That
was measured on known ligands and does not separate agonists from negatives
(see controls.py and PLAN.md); contacting the activation core is a geometric
condition instead, with nothing to calibrate.

Model is ProtenixBase with a template and no MSA, which measured better than
every alternative on the epitope -- 2.46 A against 2.54 with an MSA and 3.66 for
Boltz-2 -- and removes a remote MSA service that queued for 73 minutes on a
novel sequence. Boltz-2 with its MSA is kept for scoring, so the model that
ranks a design is not the model that made it.
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
from mosaic.losses.esmc import ESMCPseudoPerplexity, load_esmc
from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
from mosaic.losses.transformations import NoCys
from mosaic.models.protenix import ProtenixBase
from mosaic.optimizers import simplex_APGM
from mosaic.proteinmpnn.mpnn import load_mpnn_sol
from mosaic.structure_prediction import TargetChain
from target import (ACTIVATION_CORE, DEEP_CORE, activation_core,
                    glp1r_template_chain, lipid_facing_belt,
                    orthosteric_epitope, receptor_sequence, to_target_index)

CONFIG = "glp1r-agonist-v1"

# Same three-stage schedule as the TNF work: explore, then sharpen twice with
# negative entropic regularisation.
STAGES = (
    (100, 0.2, 0.3, 1.00, False),
    (50, 0.5, 0.0, 1.25, True),
    (15, 0.5, 0.0, 1.40, True),
)
MAX_GRAD_NORM = 1.0


def build_objective(model, binder_length, mpnn, *, activation_weight=2.0,
                    belt_weight=0.5, nterm_weight=0.0, nterm_len=5,
                    esm_weight=0.0, esmc=None, num_samples=2):
    seq = receptor_sequence()
    target = [TargetChain(seq, use_msa=False, template_chain=glp1r_template_chain())]
    features, _ = model.binder_features(binder_length, target)

    epitope = to_target_index(orthosteric_epitope())
    core = to_target_index(activation_core())
    belt = to_target_index(lipid_facing_belt())

    deep = to_target_index(DEEP_CORE)
    # The activation term above is satisfied by any binder position, so a design
    # can lie across the pocket and collect contacts without entering it: the
    # first sweep reached 11-14 of 18 core residues while binding worse than
    # GLP-1. paratope_idx restricts this term to the binder's N-terminal
    # residues, which is the part that has to thread, and targets only the
    # deepest subset of the core. This mirrors how GLP-1 works -- its C-terminal
    # half holds the ECD while His7-Glu9 reaches the bundle.
    # paratope_idx must be a jnp array, not a list: the term indexes
    # binder_target_max_p[paratope_idx] directly, and JAX rejects a bare list
    # there. epitope_idx gets away with a list because it is used as
    # arr[:, idx], where the index is already a tuple.
    nterm_term = sp.BinderTargetContact(
        epitope_idx=deep, paratope_idx=jnp.arange(nterm_len),
        contact_distance=8.0)

    loss = (
        # The whole epitope, loosely: this is the surface to sit on.
        1.0 * sp.BinderTargetContact(epitope_idx=epitope, contact_distance=20.0)
        # The activation core, tightly and at higher weight. 8 A rather than 20
        # because reaching these residues means entering the bundle, not merely
        # pointing at it.
        + activation_weight * sp.BinderTargetContact(
            epitope_idx=core, contact_distance=8.0)
        # Repulsion. BinderTargetContact returns -log P(contact), so a negative
        # weight penalises contact. Unbounded below in principle; the epitope
        # terms above are what keeps it in check, and belt contact is reported
        # per design so a runaway is visible.
        - belt_weight * sp.BinderTargetContact(
            epitope_idx=belt, contact_distance=8.0)
        + 1.0 * sp.WithinBinderContact()
        + (nterm_weight * nterm_term if nterm_weight else 0.0 * nterm_term)
        + 10.0 * InverseFoldingSequenceRecovery(mpnn, temp=jnp.array(0.001))
        + 0.05 * sp.BinderTargetPAE()
        + 0.05 * sp.TargetBinderPAE()
        + 0.4 * sp.WithinBinderPAE()
        + 0.025 * sp.IPTMLoss()
        + 0.1 * sp.PLDDTLoss()
    )
    objective = model.build_multisample_loss(
        loss=loss, features=features, recycling_steps=1, num_samples=num_samples
    )

    # Designs come out 42-50% charged against 23% for GLP-1 and exendin-4. A
    # half-charged peptide threading a hydrophobic bundle is implausible whatever
    # the structure model says, and no amount of scoring against that model will
    # surface it -- charge showed no relation to predicted core contacts
    # (r = -0.175). ESMC pseudo-perplexity scores how protein-like a sequence is
    # independent of structure, which is the constraint the objective lacks.
    # Added at the top level, where sequence-only terms compose.
    if esm_weight and esmc is not None:
        objective = objective + esm_weight * ESMCPseudoPerplexity(esm=esmc)
    # Cysteine is excluded: GLP-1 and exendin-4 both lack it, and a free
    # cysteine in a short secreted peptide invites disulfide scrambling. The
    # optimiser therefore runs in 19 dimensions and NoCys.sequence() reinserts
    # the zeroed column before decoding.
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
    # GLP-1 is 31 residues and exendin-4 is 39. The architecture has to reach
    # into the bundle from the groove, which is what those lengths do; a
    # globular miniprotein cannot.
    ap.add_argument("--binder-length", type=int, default=40)
    ap.add_argument("--activation-weight", type=float, default=2.0)
    ap.add_argument("--belt-weight", type=float, default=0.5)
    ap.add_argument("--nterm-weight", type=float, default=0.0,
                    help="weight on an N-terminal-only term toward the deepest "
                         "core residues; 0 disables")
    ap.add_argument("--esm-weight", type=float, default=0.0,
                    help="weight on ESMC pseudo-perplexity, which penalises "
                         "unnatural composition; 0 disables")
    ap.add_argument("--nterm-len", type=int, default=5,
                    help="how many N-terminal binder residues that term covers")
    ap.add_argument("--num-samples", type=int, default=2)
    ap.add_argument("--stage-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("designs"))
    args = ap.parse_args()

    print(f"config={CONFIG} length={args.binder_length} "
          f"activation_weight={args.activation_weight} belt_weight={args.belt_weight} "
          f"nterm_weight={args.nterm_weight} nterm_len={args.nterm_len} "
          f"esm_weight={args.esm_weight}",
          flush=True)
    print(f"epitope {len(orthosteric_epitope())} residues, "
          f"activation core {len(ACTIVATION_CORE)}, "
          f"lipid belt {len(lipid_facing_belt())}", flush=True)

    model = ProtenixBase()
    mpnn = load_mpnn_sol(0.05)
    esmc = load_esmc(model_name="esmc_300m") if args.esm_weight else None
    objective = build_objective(
        model, args.binder_length, mpnn,
        activation_weight=args.activation_weight,
        belt_weight=args.belt_weight, nterm_weight=args.nterm_weight,
        nterm_len=args.nterm_len, esm_weight=args.esm_weight, esmc=esmc,
        num_samples=args.num_samples,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"L{args.binder_length}_s{args.seed}.jsonl"
    with open(path, "w") as fh:
        for i in range(args.n_designs):
            key = jax.random.PRNGKey(args.seed * 1000 + i)
            # 19 columns: NoCys inserts the zeroed cysteine channel.
            x = jax.random.uniform(key, (args.binder_length, 19)) * 0.1
            t0 = time.time()
            x = anneal(objective, x, args.binder_length, key,
                       stage_scale=args.stage_scale)
            elapsed = time.time() - t0
            # NoCys.sequence() reinserts the zeroed cysteine column, so decode
            # through it rather than indexing a 19-token alphabet by hand.
            from mosaic.common import TOKENS
            full = np.asarray(NoCys.sequence(jax.nn.softmax(x)))
            seq = "".join(TOKENS[j] for j in full.argmax(-1))
            assert "C" not in seq, seq
            rec = {
                "name": f"glp1r_L{args.binder_length}_s{args.seed}_{i}",
                "sequence": seq, "molecule_class": "protein",
                "config": CONFIG, "activation_weight": args.activation_weight,
                "belt_weight": args.belt_weight,
                "nterm_weight": args.nterm_weight, "nterm_len": args.nterm_len,
                "esm_weight": args.esm_weight,
                "binder_length": args.binder_length, "seed": args.seed,
                "seconds": elapsed,
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{i+1}/{args.n_designs}] {rec['name']} {elapsed:.0f}s\n  {seq}",
                  flush=True)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
