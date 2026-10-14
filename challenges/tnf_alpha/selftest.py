"""Sign and consistency tests for the design losses. Run before any design job.

These started as throwaway inline checks and are a file because they earned it:
they caught three bugs in `ph_physics` that would not have raised an error --
a pKa shift six times too large, a selectivity that was identically zero, and
NaN coordinate gradients. Two of those would have produced plausible-looking
trajectories optimizing nonsense.

Each check states the expected magnitude, not just the expected sign. "Asp
raises pKa" passes for a model that is wrong by a factor of six; "Asp raises
pKa by 0.5-2.0 units" does not.

    python -I selftest.py

CPU only, a few seconds. No GPU and no model weights required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ph_score  # noqa: E402
from gravity import pseudo_cb, tnf_epitope_weights, tnf_gravitational_loss  # noqa: E402
from mosaic.common import TOKENS  # noqa: E402
from ph_physics import (  # noqa: E402
    ElectrostaticPKaSelectivity,
    linkage_selectivity,
    predict_pka,
)
from target import (  # noqa: E402
    CONSERVED_EPITOPE,
    CORE_EPITOPE,
    HUMAN_TNF,
    MOUSE_TNF,
    groove_epitope_idx,
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


# --- fixtures ---------------------------------------------------------------

B, T = 10, 12


def fake_output(target_offset: float):
    """Binder strung along x; target slab offset in y by `target_offset`."""
    bb = np.zeros((B + T, 4, 3), dtype=np.float32)
    for i in range(B):
        x = i * 3.8
        bb[i] = [[x - 1.2, 0, 0.5], [x, 0, 0], [x + 1.2, 0, -0.5], [x + 1.0, 0, -1.5]]
    for j in range(T):
        x, y = j * 3.8, target_offset
        bb[B + j] = [
            [x - 1.2, y, 0.5], [x, y, 0], [x + 1.2, y, -0.5], [x + 1.0, y, -1.5]
        ]
    full = np.zeros((B + T, 20), dtype=np.float32)
    full[:, TOKENS.index("A")] = 1.0

    class Output:
        backbone_coordinates = jnp.asarray(bb)
        full_sequence = jnp.asarray(full)

    return Output()


def sequence_with(his_at: int, neighbour: str):
    s = np.zeros((B, 20), dtype=np.float32)
    s[:, TOKENS.index("A")] = 1.0
    s[his_at] = 0
    s[his_at, TOKENS.index("H")] = 1.0
    for k in (his_at - 1, his_at + 1):
        if 0 <= k < B:
            s[k] = 0
            s[k, TOKENS.index(neighbour)] = 1.0
    return jnp.asarray(s)


# --- target definition ------------------------------------------------------


def test_target():
    print("target definition")
    check("human construct length", len(HUMAN_TNF) == 157, f"{len(HUMAN_TNF)} aa")
    check("mouse construct length", len(MOUSE_TNF) == 156, f"{len(MOUSE_TNF)} aa")
    check(
        "conserved epitope is a strict subset",
        set(CONSERVED_EPITOPE) < set(CORE_EPITOPE)
        and len(CONSERVED_EPITOPE) == 14,
        f"{len(CONSERVED_EPITOPE)} of {len(CORE_EPITOPE)} core residues",
    )
    idx = groove_epitope_idx(n_protomers=2, conserved_only=True)
    spans_both = any(i < 157 for i in idx) and any(i >= 157 for i in idx)
    check(
        "groove spans two protomers",
        spans_both,
        f"{sum(i < 157 for i in idx)} on P1, {sum(i >= 157 for i in idx)} on P2",
    )


# --- gravitational loss -----------------------------------------------------


def test_gravity():
    print("gravitational epitope loss")
    weights = tnf_epitope_weights()
    n_repel = sum(1 for w in weights.values() if w < 0)
    check(
        "mouse-divergent residues are repulsive",
        n_repel == 4,
        f"{n_repel} repulsive of {len(weights)} weighted",
    )

    loss = tnf_gravitational_loss()
    key = jax.random.key(0)
    seq = jnp.ones((B, 20)) / 20

    # Approaching the target must reduce the loss monotonically.
    values = []
    for d in (40.0, 20.0, 12.0, 8.0):
        out = fake_output(d)
        cb = pseudo_cb(out.backbone_coordinates)
        # The fixture's target is only T residues, so score against a loss
        # restricted to indices that exist.
        small = loss.__class__(
            idx=tuple(range(T)),
            weights=jnp.ones(T),
            contact_distance=loss.contact_distance,
            exclusion_distance=loss.exclusion_distance,
            temperature=loss.temperature,
        )
        v, _ = small(seq, out, key)
        values.append(float(v))
        assert cb.shape[0] == B + T
    check(
        "loss decreases as binder approaches",
        all(a > b for a, b in zip(values, values[1:])),
        " -> ".join(f"{v:.2f}" for v in values),
    )

    out = fake_output(8.0)
    small = loss.__class__(idx=tuple(range(T)), weights=jnp.ones(T))
    g = jax.grad(
        lambda b: small(
            seq,
            type("O", (), {"backbone_coordinates": b,
                           "full_sequence": out.full_sequence})(),
            key,
        )[0]
    )(out.backbone_coordinates)
    check(
        "coordinate gradient is finite",
        bool(jnp.isfinite(g).all()) and float(jnp.linalg.norm(g)) > 0,
        f"norm {float(jnp.linalg.norm(g)):.4f}",
    )


# --- pH physics -------------------------------------------------------------


def test_ph_physics():
    print("electrostatic pKa model")
    loss = ElectrostaticPKaSelectivity()
    key = jax.random.key(0)
    far = fake_output(40.0)

    base = float(loss(sequence_with(5, "A"), far, key)[1]["mean_pka_free"])
    acidic = float(loss(sequence_with(5, "D"), far, key)[1]["mean_pka_free"])
    basic = float(loss(sequence_with(5, "K"), far, key)[1]["mean_pka_free"])

    # Magnitude matters: the first version of this model got the signs right and
    # the size wrong by 6x, which is what made it useless.
    check(
        "Asp neighbours raise pKa_free by 0.5-2.0",
        0.5 <= acidic - base <= 2.0,
        f"{base:.2f} -> {acidic:.2f} (+{acidic - base:.2f})",
    )
    check(
        "Lys neighbours lower pKa_free by 0.5-2.0",
        0.5 <= base - basic <= 2.0,
        f"{base:.2f} -> {basic:.2f} ({basic - base:+.2f})",
    )

    shifts, sels = [], []
    for d in (40.0, 12.0, 9.0, 7.0, 5.0):
        _, aux = loss(sequence_with(5, "D"), fake_output(d), key)
        shifts.append(float(aux["mean_pka_shift"]))
        sels.append(float(aux["selectivity"]))
    check(
        "target burial lowers pKa_bound",
        all(a >= b for a, b in zip(shifts, shifts[1:])) and shifts[-1] < -0.3,
        " -> ".join(f"{v:+.2f}" for v in shifts),
    )
    check(
        "selectivity rises as target approaches",
        all(a <= b for a, b in zip(sels, sels[1:])) and sels[-1] > 0.1,
        " -> ".join(f"{v:.3f}" for v in sels),
    )
    check(
        "far-apart state has ~zero selectivity",
        abs(sels[0]) < 1e-3,
        f"{sels[0]:.5f}",
    )

    out = fake_output(7.0)
    seq = sequence_with(5, "D")
    g_seq = jax.grad(lambda x: loss(x, out, key)[0])(seq)
    g_crd = jax.grad(
        lambda b: loss(
            seq,
            type("O", (), {"backbone_coordinates": b,
                           "full_sequence": out.full_sequence})(),
            key,
        )[0]
    )(out.backbone_coordinates)
    check(
        "sequence gradient is finite",
        bool(jnp.isfinite(g_seq).all()),
        f"norm {float(jnp.linalg.norm(g_seq)):.4f}",
    )
    # This is the one that was NaN: inf-masked self-distances poison the
    # backward pass, and norm() of a zero vector has no derivative.
    check(
        "coordinate gradient is finite (was NaN)",
        bool(jnp.isfinite(g_crd).all()),
        f"norm {float(jnp.linalg.norm(g_crd)):.4f}",
    )
    check(
        "gradient pushes toward histidine",
        float(g_seq[5, TOKENS.index("H")]) < 0,
        f"dL/dp_His = {float(g_seq[5, TOKENS.index('H')]):.4f}",
    )

    pf, pb = predict_pka(
        jnp.zeros((3, 3)), jnp.zeros((0, 3)), jnp.zeros(3), jnp.zeros(0)
    )
    check(
        "monomer mode leaves the two states equal",
        bool(jnp.allclose(pf, pb)),
        f"pKa {float(pf[0]):.3f} both states",
    )


# --- agreement between the in-loop and ranking scores -----------------------


def test_linkage_agreement():
    print("linkage relation: jax term vs numpy ranking score")
    worst = 0.0
    for pka_bound, pka_free in [(4.0, 6.5), (5.5, 7.0), (3.0, 7.5), (7.0, 6.0)]:
        a = float(linkage_selectivity(jnp.asarray(pka_bound), jnp.asarray(pka_free)))
        b = ph_score.selectivity_log10([(pka_bound, pka_free)])
        worst = max(worst, abs(a - b))
    check(
        "agree to 1e-4 across the useful range",
        worst < 1e-4,
        f"max |difference| = {worst:.2e}",
    )


def main():
    for fn in (test_target, test_gravity, test_ph_physics, test_linkage_agreement):
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
