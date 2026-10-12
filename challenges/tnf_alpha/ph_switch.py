"""Loss terms for pH-selective binding (challenge criterion #1).

The challenge ranks first a binder that engages human TNF-alpha at pH 7.4 with no
detectable binding at pH 6.0. No structure-prediction model in `mosaic` is
pH-aware, so this cannot be optimized directly. What *can* be optimized is the
standard mechanism behind pH-dependent protein interfaces: a histidine buried in
the interface is neutral at 7.4 and protonates as the pH drops through its pKa
(~6.0-6.5), and the resulting buried charge plus desolvation penalty breaks the
interface. This is how pH-dependent ("recycling") antibodies are engineered.

So these terms shape the *paratope composition* -- put a controlled number of
histidines exactly where the binder contacts the target -- and leave the actual
pH discrimination to be checked post hoc. Treat them as hypothesis generators,
not as a pH oracle: the honest validation is to rank designs by interface His
burial afterwards and to hedge your submission slots (see design.py).
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from mosaic.common import TOKENS, LossTerm
from mosaic.losses.structure_prediction import (
    StructureModelOutput,
    contact_log_probability,
)

IDX_HIS = TOKENS.index("H")


class InterfaceHistidineCount(LossTerm):
    """Drive the expected number of *interface* histidines toward a target.

    For each binder position i we take p_contact(i) -- the probability under the
    distogram that i contacts the epitope -- and p_His(i) from the soft sequence.
    Their product summed over i is the expected number of histidines sitting in
    the interface. A squared penalty on that count is what makes this usable:
    a plain `-log p_His` weighted by contact would drive *every* interface
    position to histidine, which destroys the interface instead of switching it.

    Args:
        epitope_idx: 0-indexed positions in the target region (see
            `target.groove_epitope_idx`). None uses the whole target.
        target_count: how many interface histidines to aim for. 2-4 is the
            useful range; 1 rarely gives a clean switch and >4 tends to cost
            too much affinity at pH 7.4.
        contact_distance: angstroms defining "in the interface".
    """

    epitope_idx: list[int] | None = None
    target_count: float = 3.0
    contact_distance: float = 8.0

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        log_p = contact_log_probability(
            output.distogram_logits[:binder_len, binder_len:],
            self.contact_distance,
            bins=output.distogram_bins,
        )
        if self.epitope_idx is not None:
            log_p = log_p[:, self.epitope_idx]

        # P(position i touches at least one epitope residue), via the
        # complement of "touches none". Clipped because log1p of the
        # complement is numerically touchy once a contact is near-certain.
        log_p_none = jnp.log1p(-jnp.exp(log_p).clip(0.0, 1.0 - 1e-6)).sum(-1)
        p_contact = 1.0 - jnp.exp(log_p_none)

        p_his = sequence[:, IDX_HIS]
        expected = (p_contact * p_his).sum()
        loss = (expected - self.target_count) ** 2

        return loss, {
            "interface_his": expected,
            "interface_size": p_contact.sum(),
        }


class BuriedHistidinePacking(LossTerm):
    """Reward histidines that are *buried* at the interface, not just nearby.

    A surface histidine titrates with the pKa of free histidine and changes
    nothing on binding; the switch needs the imidazole desolvated by the
    partner. Burial is proxied here by how many target residues a position
    contacts at close range -- a histidine wedged against several epitope
    residues is the one whose pKa shifts and whose protonation costs us
    energy.

    Use this alongside `InterfaceHistidineCount`, which sets the count of histidines
    this term decides *where* those histidines go.
    """

    epitope_idx: list[int] | None = None
    contact_distance: float = 6.0
    num_contacts: int = 4

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        log_p = contact_log_probability(
            output.distogram_logits[:binder_len, binder_len:],
            self.contact_distance,
            bins=output.distogram_bins,
        )
        if self.epitope_idx is not None:
            log_p = log_p[:, self.epitope_idx]

        # Burial score: mean log-probability over the `num_contacts` closest
        # epitope residues. Summing all of them would let one very tight
        # contact stand in for genuine burial.
        top = jnp.sort(log_p, descending=True, axis=-1)[:, : self.num_contacts]
        burial = top.mean(-1)

        p_his = sequence[:, IDX_HIS]
        # Weighted mean burial over the histidine mass in the binder.
        total_his = p_his.sum() + 1e-6
        mean_burial = (p_his * burial).sum() / total_his

        return -mean_burial, {"his_burial": mean_burial}


class NoHistidineOutsideInterface(LossTerm):
    """Penalize histidine where it cannot contribute to the switch.

    Histidines far from the epitope add pH-dependent *folding* behaviour and
    aggregation risk without buying any binding selectivity, so push the
    histidine budget into the paratope rather than over the whole chain.
    """

    epitope_idx: list[int] | None = None
    contact_distance: float = 8.0
    weight_floor: float = 0.05

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        log_p = contact_log_probability(
            output.distogram_logits[:binder_len, binder_len:],
            self.contact_distance,
            bins=output.distogram_bins,
        )
        if self.epitope_idx is not None:
            log_p = log_p[:, self.epitope_idx]

        log_p_none = jnp.log1p(-jnp.exp(log_p).clip(0.0, 1.0 - 1e-6)).sum(-1)
        p_contact = 1.0 - jnp.exp(log_p_none)

        outside = jnp.clip(1.0 - p_contact, self.weight_floor, 1.0)
        stray = (outside * sequence[:, IDX_HIS]).sum()

        return stray, {"stray_his": stray}


def ph_switch_loss(
    epitope_idx: list[int],
    target_count: float = 3.0,
    count_weight: float = 1.0,
    burial_weight: float = 0.5,
    stray_weight: float = 0.3,
):
    """The three terms combined, with weights that worked as a starting point.

    Tune `target_count` per sub-pool rather than trusting one value -- the whole
    point of hedging submission slots is that the right histidine budget is not
    known in advance.
    """
    return (
        count_weight * InterfaceHistidineCount(
            epitope_idx=epitope_idx, target_count=target_count
        )
        + burial_weight * BuriedHistidinePacking(epitope_idx=epitope_idx)
        + stray_weight * NoHistidineOutsideInterface(epitope_idx=epitope_idx)
    )
