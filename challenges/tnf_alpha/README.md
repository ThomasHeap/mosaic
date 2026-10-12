# Challenge 02 — TNF-alpha (Anthropic x Adaptyv 2026)

Target: human TNF-alpha soluble trimer, P01375 residues 77–233 (157 aa), tag-free.
Deadline 2026-10-18. Up to 40 designs (Track 1) / 20 (Tracks 2–3), CSV with
`name,sequence,molecule_class`, ordered by preference.

Ranked evaluation criteria:

1. **pH-selective binding** — binds human TNF at pH 7.4, no detectable binding at pH 6.0
2. **Mouse cross-reactivity** — binds mouse TNF as well as human
3. **Affinity** against human TNF

## What the target analysis settled

`compute_epitope.py` derives the epitope from contacts (4.5 Å heavy-atom) in three
receptor-site complexes — 3WD5 (adalimumab Fab), 4G3Y (infliximab Fab), 3ALQ (TNFR2) —
and keeps residues seen in ≥2 of them. All TNF chains in those structures number on
the mature sequence, so the mapping is exact. Results live in `target.py`.

Three things that should shape any attempt:

- **The groove spans two protomers.** TNFR2 chain T contacts 15 residues on one
  protomer and 13 on its neighbour. Design against a TNF *monomer* and you are
  designing for a surface that does not exist in the assay construct. Co-fold at
  least a dimer, templated on 1TNF.
- **Only 14 of the 20 core epitope residues are conserved in mouse.** The six that
  differ (P20H, S71D, H73 indel, I97V, A111L, D140K) are why adalimumab is
  human-specific. Criterion #2 says bias contacts onto the conserved 14.
- **His73 is the only histidine in the epitope** — and it is lost to an indel in
  mouse. So do not build the pH switch on the target's histidine; it trades
  criterion #1 against #2. Put the switch histidines on the binder.

## How criterion #1 is handled

pH-dependent affinity is a thermodynamic consequence of ionizable groups whose pKa
shifts on binding, via the proton-linkage relation in `ph_score.py`. That relation is
exact; only pKa needs predicting. The per-histidine selectivity available across a
7.4 → 6.0 window is bounded and the bound is what dictates the design:

| free-state pKa | log10 selectivity per His | histidines needed for 2 log10 |
|---|---|---|
| 6.0 | 0.28 | 8 |
| 6.5 (ordinary surface His) | 0.57 | 4 |
| 7.0 | 0.90 | 3 |
| 7.5 | 1.16 | 2 |

So "no detectable binding at pH 6.0" is unreachable with one or two ordinary
histidines. Two levers, both needed:

- **raise pKa_free** — the histidine must be readily protonatable in the *free*
  binder (neighbouring Asp/Glu do this). Selectivity rises monotonically with
  pKa_free, and this is the lever most often missed.
- **suppress pKa_bound** — bury the imidazole against a hydrophobic or cationic
  patch of the epitope so protonation is costly in the complex.

Sanity check on TNF itself: His73 scores pKa_free 6.82 (elevated, in the groove),
while His15 and His78 sit at 5.06 and 4.01 — already buried when free, and they
contribute 0.044 and 0.003 log units. Low free-state pKa means a dead switch.

## Three tiers, because the good score is not differentiable

Gradient-based design needs a differentiable objective, and PROPKA is a heuristic
over a discrete structure — it has no gradient w.r.t. a soft sequence. Hence:

- **Tier 1 — differentiable, in the optimization loop** (`ph_switch.py`).
  Hand-crafted proxies: an expected *count* of interface histidines driven to a
  target (a squared penalty, not `-log p`, or every interface position goes to
  histidine), a burial term deciding where they go, and a penalty on histidine
  outside the interface. Crude, but it steers.
- **Tier 2 — non-differentiable, for ranking** (`ph_score.py`). PROPKA on the
  predicted complex and free binder, combined through the linkage relation into a
  log10 selectivity. This is the number to trust, used to triage and order the
  submission. PROPKA's MAE is ~0.7 pKa units ≈ 0.7 log10 of selectivity — the size
  of the whole per-histidine effect — so rank with it, never gate on it.
- **Tier 3 — stretch goal: make Tier 2 differentiable.** Distil PROPKA into a JAX
  surrogate. PROPKA is a fast deterministic function, so label a large PDB subset
  with it and regress from features `mosaic` already exposes differentiably
  (`atom37_coords` distances × neighbour identity from the soft sequence). Fitting
  a deterministic function, not noisy experiment, so the surrogate can track
  PROPKA closely and then drops straight in as a `LossTerm`. Training against
  *experimental* pKa instead (PKAD-2, ~1,500 measured values) is the better science
  but the worse deadline bet.

## Running it

Design runs need a GPU node — `design.py` refuses to start on CPU:

```bash
./submit.sh ph        8 75      # pool, n_designs, binder_length
./submit.sh ph_hot    8 75
./submit.sh affinity  8 65
```

Pools hedge the histidine budget rather than betting on one value: `affinity` (none,
baseline), `ph` (3), `ph_hot` (5). The table above says 3 only suffices with elevated
pKa_free, so the bracket is deliberate.

Then rank and build the submission:

```bash
python rank.py designs/*.jsonl --out submission.csv
```

## Status / next steps

- [x] epitope derived and mouse conservation mapped
- [x] pH-selectivity scoring validated against TNF's own histidines
- [ ] repredict designs and populate Tier 2 scores (needs a GPU job)
- [ ] calibrate: run Tier 2 on adalimumab/infliximab as negative controls for
      pH selectivity — the challenge permits known binders for filter calibration
- [ ] decide submission split across the three pools
