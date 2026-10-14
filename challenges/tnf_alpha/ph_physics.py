"""Differentiable pH-selectivity loss, after the JustHISpKa electrostatic model.

`ph_switch.py` optimizes a *proxy* (how many histidines sit in the interface).
This module optimizes the quantity the challenge is actually scored on -- the
log10 ratio of affinity at pH 7.4 versus pH 6.0 -- by making the pKa model itself
differentiable and pushing it through the exact proton-linkage relation.

The route comes from Hogues, Wei & Sulea, "Improved Structure-Based Histidine pKa
Prediction for pH-Responsive Protein Design" (JCIM 2025; the JustHISpKa method,
0.4 pH-unit MUE on PKAD histidines). Two features of that method make it the
right blueprint rather than PROPKA:

  1. It assumes **electrostatics dominates** the protonated/unprotonated free
     energy difference, and drops everything else. No hydrogen-bond geometry
     term -- which matters because design-time features in `mosaic` carry no
     sidechains, so an H-bond term is not computable in the loop anyway.
  2. The map from electrostatic energy to pKa is **linear**, with two fitted
     scalars. A linear readout means clean gradients and a 2-3 parameter
     calibration, which is cheap to fit against PROPKA labels exactly as the
     paper fits its own against experiment.

Both energy terms survive the soft relaxation:

  **Coulomb.** Protonation puts +1 on the imidazole. Its interaction with the
  surrounding charges is `sum_j q_j / (eps * r_ij)`, where `q_j` is the
  *expected* charge of residue j under the PSSM -- Asp/Glu contribute -1,
  Lys/Arg +1, weighted by their probability. Differentiable in coordinates
  (through r) and in sequence (through q). Nearby negative charge stabilizes the
  protonated form and *raises* pKa; nearby positive charge *lowers* it.

  **Reaction field / desolvation.** Protonating a buried group is costly because
  the charge loses solvent. Proxied by a smooth neighbour count (a Born-radius
  stand-in), which *lowers* pKa with increasing burial.

The end-state trick. The linkage relation needs pKa in the free binder *and* in
the complex, which are two different `mosaic` loss branches. Instead of running
both, this term takes the complex prediction and splits the environment: summing
over binder neighbours only gives pKa_free, adding the target's contribution
gives pKa_bound. The binder backbone is assumed unchanged on binding -- the same
rigid approximation JustHISpKa makes by working from the complex structure, and a
fair one for a small rigid minibinder. It also means the *shift* is isolated
exactly: it is the target's electrostatic contribution and nothing else.

That gives both design levers from the linkage analysis in one objective, with
the right signs falling out of the physics rather than being hand-imposed:
acidic residues the binder places near its own histidine raise pKa_free, while
burial against the epitope suppresses pKa_bound.

**The parameters below are physically scaled but NOT calibrated.** Fit them
against PROPKA (or JustHISpKa) on predicted complexes before trusting the
magnitude -- `calibrate()` documents the procedure. Until then use this term for
its gradient direction and keep `ph_score.py` as the number you rank on.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from mosaic.common import TOKENS, LossTerm
from mosaic.losses.structure_prediction import StructureModelOutput

from gravity import pseudo_cb

IDX_HIS = TOKENS.index("H")

# kcal/mol per pKa unit at 298 K (2.303 * R * T).
RT_LN10 = 1.364
# Coulomb constant in kcal/mol with charges in e and distance in angstrom.
COULOMB_K = 332.06
# Intrinsic pKa of histidine in water.
HIS_MODEL_PKA = 6.5

# Formal sidechain charge near neutral pH, in mosaic-20 token order
# ("ARNDCQEGHILKMFPSTWYV"). Histidine is given its small average charge at
# pH 7.4 rather than 0 or 1.
_CHARGES = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}
SIDECHAIN_CHARGE = jnp.array(
    [_CHARGES.get(aa, 0.0) for aa in TOKENS], dtype=jnp.float32
)


def _log10_1p10(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """log10(1 + 10^x), via softplus for numerical stability at large |x|."""
    return jax.nn.softplus(x * jnp.log(10.0)) / jnp.log(10.0)


def linkage_selectivity(
    pka_bound: Float[Array, "..."],
    pka_free: Float[Array, "..."],
    ph_high: float = 7.4,
    ph_low: float = 6.0,
) -> Float[Array, "..."]:
    """log10 Ka(ph_high) / Ka(ph_low) for one ionizable group. Positive is wanted.

    The exact proton-linkage relation, differentiable. Identical to
    `ph_score.selectivity_log10` for a single group, so the in-loop objective and
    the ranking score agree by construction rather than by coincidence.
    """
    def shift(ph):
        return _log10_1p10(pka_bound - ph) - _log10_1p10(pka_free - ph)

    return shift(ph_high) - shift(ph_low)


class PKaParams(eqx.Module):
    """Fitted scalars of the pKa model. Defaults are physically scaled guesses.

    `calibrate.py` fits these against PROPKA labels. Kept as a pytree so the fit
    and the loss term share one implementation of the physics
    """

    # Dielectric is interpolated from solvent to interior by local burial. A
    # fixed low epsilon is wrong by a wide margin for surface pairs: the
    # reaction field in SIE largely cancels Coulomb once a pair is solvated, and
    # ignoring that put an Asp-His pair at a pKa shift of +6 units instead of
    # the ~1 unit such pairs actually show.
    epsilon_solvent: float = 80.0
    epsilon_interior: float = 10.0
    # Debye-like range cutoff, so a sum over many distant charges cannot
    # accumulate into a spurious large shift.
    screening_length: float = 8.0
    alpha_desolv: float = 3.0
    offset: float = 0.0
    burial_cutoff: float = 10.0
    burial_width: float = 2.0
    burial_reference: float = 18.0


def _burial(
    d: Float[Array, "N M"], mask: Float[Array, "N M"], p: PKaParams
) -> Float[Array, " N"]:
    """Smooth neighbour count -- a differentiable Born-radius stand-in."""
    occupancy = jax.nn.sigmoid((p.burial_cutoff - d) / p.burial_width)
    return (occupancy * mask).sum(-1) / p.burial_reference


def _coulomb(
    d: Float[Array, "N M"],
    charges: Float[Array, " M"],
    mask: Float[Array, "N M"],
    burial: Float[Array, " N"],
    p: PKaParams,
) -> Float[Array, " N"]:
    """Interaction of a +1 charge at each site with its environment, kcal/mol.

    Screening interpolates between solvent and interior dielectric by the burial
    of the titrating site, and decays exponentially beyond `screening_length`.
    """
    # Floor the distance: 2 A is below any real CB-CB separation, and this also
    # keeps the 1/d gradient bounded.
    safe_d = jnp.maximum(d, 2.0)
    eps = p.epsilon_solvent - (
        p.epsilon_solvent - p.epsilon_interior
    ) * jnp.clip(burial, 0.0, 1.0)
    decay = jnp.exp(-safe_d / p.screening_length)
    pair = charges[None, :] * decay * mask / (eps[:, None] * safe_d)
    return COULOMB_K * pair.sum(-1)


def _pairwise_dist(a, b):
    """Distances with an epsilon before the sqrt.

    `jnp.linalg.norm` of a zero vector -- which the diagonal of a self-distance
    matrix always contains -- has a NaN derivative, and masking the diagonal to
    `jnp.inf` afterwards does not help, because inf poisons the backward pass
    too. Self-interaction is removed multiplicatively by the caller's mask.
    """
    delta = a[:, None, :] - b[None, :, :]
    return jnp.sqrt((delta * delta).sum(-1) + 1e-6)


def predict_pka(
    binder_cb: Float[Array, "N 3"],
    target_cb: Float[Array, "M 3"],
    binder_charge: Float[Array, " N"],
    target_charge: Float[Array, " M"],
    params: PKaParams = PKaParams(),
) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
    """(pka_free, pka_bound) for every binder position, as if each were histidine.

    Both end states come from one structure: summing over binder neighbours
    alone gives the free state, adding the target gives the bound state. The
    binder backbone is assumed unchanged on binding -- the same rigid
    approximation JustHISpKa makes by working from the complex -- which isolates
    the shift as exactly the target's electrostatic contribution.

    Pass an empty `target_cb` to score a monomer; the bound state then equals
    the free state.
    """
    n = binder_cb.shape[0]
    d_bb = _pairwise_dist(binder_cb, binder_cb)
    off_diag = 1.0 - jnp.eye(n)

    # Free state: binder environment only. Burial comes first, since it sets
    # the dielectric that screens the Coulomb sum.
    burial_free = _burial(d_bb, off_diag, params)
    e_free = _coulomb(d_bb, binder_charge, off_diag, burial_free, params)

    if target_cb.shape[0] == 0:
        burial_bound, e_bound = burial_free, e_free
    else:
        d_bt = _pairwise_dist(binder_cb, target_cb)
        all_bt = jnp.ones_like(d_bt)
        burial_bound = burial_free + _burial(d_bt, all_bt, params)
        e_bound = e_free + _coulomb(
            d_bt, target_charge, all_bt, burial_bound, params
        )

    # Linear readout, as in JustHISpKa. Favourable (negative) electrostatic
    # energy for the +1 form raises pKa; burial lowers it.
    def to_pka(energy, burial):
        return (
            HIS_MODEL_PKA
            + params.offset
            - energy / RT_LN10
            - params.alpha_desolv * burial
        )

    return to_pka(e_free, burial_free), to_pka(e_bound, burial_bound)


class ElectrostaticPKaSelectivity(LossTerm):
    """Maximize expected pH selectivity through a differentiable pKa model.

    Args:
        epitope_idx: target-region indices forming the epitope. Burial and
            Coulomb contributions from the target are restricted to these, so
            the term rewards a switch at the intended site rather than anywhere
            on the target surface.
        ph_high / ph_low: the assay pair, 7.4 and 6.0 for this challenge.
        target_selectivity: selectivity (log10) beyond which there is no further
            reward. ~2 is "no detectable binding"; chasing more buys nothing and
            costs affinity at pH 7.4.
        epsilon: effective dielectric screening the Coulomb sum. 20 is a
            middling protein-interior value; a calibration target.
        alpha_desolv: pKa units lost at full burial. A calibration target.
        burial_cutoff / burial_width: smooth neighbour-count geometry (angstrom).
        burial_reference: neighbour count treated as "fully buried".
    """

    epitope_idx: tuple[int, ...] | None = None
    ph_high: float = 7.4
    ph_low: float = 6.0
    target_selectivity: float = 2.0
    params: PKaParams = PKaParams()

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        cb = pseudo_cb(output.backbone_coordinates)
        binder_cb = cb[:binder_len]
        target_cb = cb[binder_len:]
        if self.epitope_idx is not None:
            target_cb = target_cb[jnp.asarray(self.epitope_idx)]

        # Expected sidechain charge per position, from the soft sequence. The
        # target is at its known identity via the model's own token
        # distribution, the binder at whatever the optimizer currently holds.
        binder_charge = sequence @ SIDECHAIN_CHARGE
        target_full = output.full_sequence[binder_len:]
        if self.epitope_idx is not None:
            target_full = target_full[jnp.asarray(self.epitope_idx)]
        target_charge = target_full @ SIDECHAIN_CHARGE

        pka_free, pka_bound = predict_pka(
            binder_cb, target_cb, binder_charge, target_charge, self.params
        )

        per_site = linkage_selectivity(
            pka_bound, pka_free, self.ph_high, self.ph_low
        )

        # Expected selectivity under the current sequence distribution: only
        # positions that are actually histidine can switch. Summed, because
        # several histidines compose additively in log space.
        p_his = sequence[:, IDX_HIS]
        expected = (p_his * per_site).sum()

        # One-sided hinge: reward up to the target, nothing beyond it.
        loss = jax.nn.relu(self.target_selectivity - expected)

        return loss, {
            "selectivity": expected,
            "mean_pka_free": (p_his * pka_free).sum() / jnp.maximum(p_his.sum(), 1e-6),
            "mean_pka_shift": (p_his * (pka_bound - pka_free)).sum()
            / jnp.maximum(p_his.sum(), 1e-6),
        }


def calibrate() -> None:
    """How to fit `epsilon`, `alpha_desolv` and the offset. Not yet run.

    JustHISpKa fits two scalars by Nelder-Mead against experimental pKa. The same
    procedure applies here with PROPKA as the label source, which is the right
    choice for three reasons: labels are free (~0.6 s per complex, see
    `ph_score.propka_pkas`), they can be generated on *predicted design-like*
    structures rather than crystal structures, and agreement with the ranking
    score is what actually matters for the pipeline to be self-consistent.

    Procedure:
      1. Predict ~200 binder-target complexes (any pool, GPU job) and keep both
         the complex and binder-only structures.
      2. Label every binder histidine with `propka_pkas` on each end state.
      3. Fit (epsilon, alpha_desolv, offset) by least squares on pKa, using
         `scipy.optimize.minimize(method="Nelder-Mead")` -- three parameters, so
         derivative-free is fine and robust to the floored distances.
      4. Report MUE against held-out PROPKA values, and separately the error on
         the *shift* pka_bound - pka_free, which is what selectivity depends on.
         A model can track absolute pKa well and the shift badly; the shift is
         the one that matters.

    Until step 4 shows a usable shift error, treat this term as a gradient
    direction only and rank on `ph_score.py`.
    """
    raise NotImplementedError(calibrate.__doc__)
