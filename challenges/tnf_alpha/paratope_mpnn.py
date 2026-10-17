"""Inverse-folding recovery that does not dictate the paratope.

Designed parents carry about 6 histidines and a median of *zero* at the
interface, against 1.29 expected from random placement on the same chains, with
2.41 per design sitting exactly one residue from a contacting position against
1.04 expected. The displacement is real and not an artefact of interface
positions occurring in runs.

The motivating hypothesis for this term -- that soluble ProteinMPNN refuses
histidine in a paratope and outbids the pH objective at weight 10.0 against 0.5
-- is false. Measured over 40 complexes, MPNN assigns P(His) 0.0295 at interface
positions and 0.0243 elsewhere, slightly favouring the interface, both near the
0.023 natural frequency.

What survives of the argument is weaker but still worth testing. MPNN does not
single out the paratope, but at weight 10.0 and temp 0.001 it is a strong
uniform anchor holding histidine near 2.5% everywhere, interface included, and
the pH terms at weight 0.5 have to move it from there. Releasing that anchor
over contacting positions may let them. It may equally fail, because the
affinity terms price histidine at a buried contact through predicted iPTM and
this term does nothing about that.

Raising the pH weight instead is the move that produced a 32% histidine 75-mer
(probe 19400994), which is why TotalHistidineBudget exists. The alternative is
to stop asking MPNN about the positions it should not be deciding: let it
govern the scaffold, and leave paratope composition to the binding and pH
terms.

The paratope cannot be masked statically, since which positions contact the
epitope is an output of optimisation. It is therefore down-weighted softly by
the same contact probability ph_switch.py uses, with a floor so MPNN retains
some say at the interface rather than none -- a completely unconstrained
paratope invites the composition artefacts MPNN is there to prevent.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from mosaic.common import LossTerm
from mosaic.losses.protein_mpnn import inverse_fold
from mosaic.losses.structure_prediction import (
    StructureModelOutput,
    contact_log_probability,
)
from mosaic.proteinmpnn.mpnn import ProteinMPNN


class ScaffoldInverseFoldingRecovery(LossTerm):
    """`InverseFoldingSequenceRecovery` weighted away from the paratope.

    Identical to the stock term except that the per-position recovery is
    averaged with weight `clip(1 - p_contact, paratope_floor, 1)` instead of
    uniformly. At `paratope_floor=1.0` it reduces exactly to the stock term.

    Args:
        mpnn: ProteinMPNN instance.
        temp: sampling temperature, as for the stock term.
        epitope_idx: 0-indexed positions in the target region; None uses the
            whole target.
        contact_distance: angstroms defining "in the interface". Matches
            ph_switch.py's default so the two terms agree on what the paratope
            is.
        paratope_floor: residual MPNN weight at a certain-contact position.
        num_samples, jacobi_iterations, bias: as for the stock term.
    """

    mpnn: ProteinMPNN
    temp: Float
    epitope_idx: list[int] | None = None
    contact_distance: float = 8.0
    paratope_floor: float = 0.1
    num_samples: int = 16
    jacobi_iterations: int = 10
    bias: Float[Array, "N 20"] = None

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        binder_len = sequence.shape[0]

        sequences = jax.vmap(
            lambda k: jax.nn.one_hot(
                inverse_fold(
                    self.mpnn,
                    binder_length=binder_len,
                    output=output,
                    temp=self.temp,
                    key=k,
                    jacobi_iterations=self.jacobi_iterations,
                    bias=self.bias,
                ),
                20,
            )
        )(jax.random.split(key, self.num_samples))
        average_sequence = jax.lax.stop_gradient(sequences.mean(0))

        # Per-position recovery, before averaging. The stock term takes .mean()
        # here, which is what gives MPNN equal authority over the paratope.
        per_position = (average_sequence * sequence).sum(-1)

        log_p = contact_log_probability(
            output.distogram_logits[:binder_len, binder_len:],
            self.contact_distance,
            bins=output.distogram_bins,
        )
        if self.epitope_idx is not None:
            log_p = log_p[:, self.epitope_idx]
        log_p_none = jnp.log1p(-jnp.exp(log_p).clip(0.0, 1.0 - 1e-6)).sum(-1)
        p_contact = 1.0 - jnp.exp(log_p_none)

        w = jnp.clip(1.0 - p_contact, self.paratope_floor, 1.0)
        ip = (w * per_position).sum() / w.sum()

        return -ip, {
            "sequence_recovery": ip,
            "scaffold_weight": w.mean(),
            "paratope_size": p_contact.sum(),
        }
