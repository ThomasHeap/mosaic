"""Coordinate-space epitope targeting, after ORBIT's "gravitational binding loss".

`mosaic`'s `BinderTargetContact` works on the distogram: it rewards contact
probability with a set of epitope columns, treating every column alike. ORBIT
(research.mandrake.bio, winner of the GEM x Adaptyv RBX1 challenge, itself built
on `mosaic`) instead puts an attractive field in *coordinate* space, with pull
strength scaled by per-residue importance, and notes it extends to repulsive and
exclusion zones.

That extension is what makes it worth adopting here, because the TNF epitope is
not uniform. The conservation analysis in `target.py` splits it cleanly:

  - two residues (E23, Q67) are contacted in all three reference complexes and
    conserved in mouse -- the residues to build the interface around
  - twelve more are contacted in two complexes and conserved -- ordinary pull
  - six (P20, S71, H73, I97, A111, D140) are core epitope but *diverge* in mouse,
    and are exactly why adalimumab is human-specific

A binder that leans on those last six wins criterion #3 (human affinity) and
loses criterion #2 (mouse cross-reactivity). A distogram contact term cannot say
that; a signed coordinate-space field can, by making them repulsive. So this term
encodes "bind here, specifically not there" in one objective.

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

from target import CORE_EPITOPE, GROOVE_PROTOMER_1, GROOVE_PROTOMER_2, MOUSE_DIVERGENT

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


def tnf_epitope_weights(
    n_protomers: int = 2,
    triple_weight: float = 2.0,
    conserved_weight: float = 1.0,
    divergent_weight: float = -1.0,
) -> dict[int, float]:
    """Signed per-residue weights, keyed by index into the target token array.

    Positive pulls the binder in, negative pushes it away. Indices are offsets
    into the concatenated target region, matching `target.groove_epitope_idx`.
    """
    sides = [GROOVE_PROTOMER_1, GROOVE_PROTOMER_2][:n_protomers]
    weights: dict[int, float] = {}
    for k, side in enumerate(sides):
        for resnum in side:
            if resnum in MOUSE_DIVERGENT:
                w = divergent_weight
            elif resnum in TRIPLE_CONSERVED:
                w = triple_weight
            elif resnum in CORE_EPITOPE:
                w = conserved_weight
            else:
                # In the groove but not in the >=2-complex core: mild pull, since
                # it is still the right face of the protein.
                w = 0.3 * conserved_weight
            weights[k * 157 + (resnum - 1)] = w
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


def tnf_gravitational_loss(n_protomers: int = 2, **kwargs) -> GravitationalEpitope:
    """`GravitationalEpitope` preconfigured with the TNF epitope weighting."""
    weights = tnf_epitope_weights(n_protomers=n_protomers)
    idx = tuple(sorted(weights))
    return GravitationalEpitope(
        idx=idx,
        weights=jnp.array([weights[i] for i in idx], dtype=jnp.float32),
        **kwargs,
    )
