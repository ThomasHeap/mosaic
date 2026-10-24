"""Coordinate-space epitope targeting, after ORBIT's "gravitational binding loss".

`mosaic`'s `BinderTargetContact` works on the distogram: it rewards contact
probability with a set of epitope columns, treating every column alike. ORBIT
(research.mandrake.bio, winner of the GEM x Adaptyv RBX1 challenge, itself built
on `mosaic`) instead puts an attractive field in *coordinate* space, with pull
strength scaled by per-residue importance, and notes it extends to repulsive and
exclusion zones.

For TfR1 that extension carries the whole epitope decision. Criterion 2 ranks
designs by how close the mouse/human KD ratio is to 1, and mouse conservation
over this surface is extremely uneven: the challenge's own epitope hint points
at Tyr211, which is one of the residues that *differs* in mouse (Y to D), and
only 2 of the 9 residues the 11rn apical binder touches are conserved. A patch
centred on 322, in the same domain, is 89% conserved.

So rather than a hand-picked epitope, this term weights every allowed residue by
mouse conservation: conserved surface pulls, divergent surface pushes, and the
optimiser chooses where to bind within that field. Two regions are excluded
outright, with strong repulsion -- the truncation boundary, which is a surface
that does not exist in the assay construct, and the C-terminal residues where
the designs' own capture tag attaches.

That is a graded version of the allowed-region decision in PLAN.md. A binary
in/out split would throw away the information that 322 is better than 211 by a
long way, not merely different.

Distances use a virtual C-beta built from backbone N/CA/C. Design-time features in
`mosaic` carry no sidechains (the reference-atom channel is UNK/G), so real CB is
unavailable in the loop -- and a virtual CB is the standard substitute, smooth in
the backbone coordinates and defined even for glycine.
"""


import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from mosaic.common import LossTerm
from mosaic.losses.structure_prediction import StructureModelOutput

from conservation import conservation
from target import allowed_region, excluded_region, resnums

# Contacted in all three reference complexes (3WD5 / 4G3Y / 3ALQ) and conserved
# in mouse. See the consensus table in compute_epitope.py output.
TRIPLE_CONSERVED = [23, 67]


def pseudo_cb(backbone: Float[Array, "N 4 3"]) -> Float[Array, "N 3"]:
    """Virtual C-beta from backbone N, CA, C (trRosetta/AF2 convention).

    Smooth in the inputs and well-defined for glycine, unlike real CB.
    """
    n, ca, c = backbone[:, 0], backbone[:, 1], backbone[:, 2]
    b = ca - n
    cc = c - ca
    a = jnp.cross(b, cc)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * cc + ca


def tfr1_epitope_weights(
    chain,
    conserved_weight: float = 2.0,
    divergent_weight: float = -1.0,
    excluded_weight: float = -4.0,
) -> dict[int, float]:
    """Signed per-residue weights, keyed by offset into the target chain.

    Positive pulls the binder in, negative pushes it away. Weight is linear in
    mouse conservation over the allowed region: a residue identical in mouse
    gets `conserved_weight`, one that differs gets `divergent_weight`, so the
    optimiser is free to choose where to bind but pays for divergent surface.

    The truncation boundary and the tag-proximal C-terminus get
    `excluded_weight`, a harder push, because binding there is not merely worse
    for criterion 2 but invalid: one is a surface the assay construct does not
    have, the other is where the capture tag sits.
    """
    cons = conservation()
    pos = {n: i for i, n in enumerate(resnums(chain))}
    allowed = set(allowed_region(chain))
    weights: dict[int, float] = {}
    for n in resnums(chain):
        if n not in allowed:
            weights[pos[n]] = excluded_weight
            continue
        rec = cons.get(n)
        same = bool(rec and rec["same"])
        weights[pos[n]] = conserved_weight if same else divergent_weight
    return weights


class GravitationalEpitope(LossTerm):
    """Signed attractive/repulsive field from weighted target residues.

    For each weighted target residue j, the distance that matters is the distance
    to the *nearest* binder residue, taken as a soft minimum so the gradient
    reaches more than one binder position early in optimization. Using a soft min
    rather than a mean is what stops the term collapsing the whole binder onto a
    single hotspot.

    Attractive residues use a flat-bottomed well: no further reward once within
    `contact_distance`, so the term pulls the binder into place and then stops
    pulling instead of driving a clash. Repulsive residues get the mirror image,
    pushing until `exclusion_distance`.

    Args:
        idx: target-region indices of the weighted residues.
        weights: signed weight per entry of `idx`; positive attracts.
        contact_distance: CB-CB distance counted as "in contact" (A).
        exclusion_distance: how far to push repulsive residues (A).
        temperature: soft-min sharpness (A). Larger reaches further but blurs.
    """

    idx: tuple[int, ...]
    weights: Float[Array, " K"]
    contact_distance: float = 8.0
    exclusion_distance: float = 12.0
    temperature: float = 2.0

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        cb = pseudo_cb(output.backbone_coordinates)
        binder_cb = cb[:binder_len]
        target_cb = cb[binder_len:][jnp.asarray(self.idx)]

        # [K, binder_len] pairwise distances, K weighted target residues.
        d = jnp.linalg.norm(
            target_cb[:, None, :] - binder_cb[None, :, :], axis=-1
        )
        # Soft minimum over binder positions, per target residue.
        soft_min = -self.temperature * jax.nn.logsumexp(
            -d / self.temperature, axis=-1
        )

        w = jnp.asarray(self.weights)
        attract = jnp.clip(w, 0.0, None)
        repel = jnp.clip(-w, 0.0, None)

        # Flat-bottomed well: softplus is ~0 inside the target distance and
        # grows linearly outside, so there is no reward for over-tightening.
        pull = attract * jax.nn.softplus(soft_min - self.contact_distance)
        push = repel * jax.nn.softplus(self.exclusion_distance - soft_min)
        loss = (pull + push).sum() / jnp.maximum(jnp.abs(w).sum(), 1e-6)

        # Report the two sides separately: a single number hides whether the
        # binder is failing to engage or failing to avoid.
        n_attract = jnp.maximum(attract.sum(), 1e-6)
        n_repel = jnp.maximum(repel.sum(), 1e-6)
        return loss, {
            "grav": loss,
            "d_attract": (attract * soft_min).sum() / n_attract,
            "d_repel": (repel * soft_min).sum() / n_repel,
        }


def tfr1_gravitational_loss(chain, **kwargs) -> GravitationalEpitope:
    """`GravitationalEpitope` preconfigured with TfR1 conservation weighting."""
    weights = tfr1_epitope_weights(chain)
    idx = tuple(sorted(weights))
    return GravitationalEpitope(
        idx=idx,
        weights=jnp.array([weights[i] for i in idx], dtype=jnp.float32),
        **kwargs,
    )
