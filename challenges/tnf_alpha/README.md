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
- **Tier 3 — differentiable physics, no training** (`ph_physics.py`). Rather than
  distil PROPKA into a learned surrogate, port the *model* that JustHISpKa uses:
  electrostatics-only, with a linear map to pKa. Coulomb over the environment's
  expected charge under the PSSM is differentiable in sequence and coordinates
  alike; burial gives the reaction-field term. Both end states come out of the
  single complex prediction — binder neighbours alone give pKa_free, adding the
  target gives pKa_bound — so the shift is exactly the target's contribution, and
  the result goes through the same linkage relation as Tier 2 (they agree to
  1e-4). No labels, no training, no distribution shift, and it optimizes the
  scored quantity instead of a proxy. `calibrate.py` fits its scalars against
  PROPKA labels.

  **Calibration result (316 complexes, 1272 histidines, 25% held out).** The
  model beats the null consistently but not by much:

  | metric (test) | null | fitted |
  |---|---|---|
  | MUE(pKa) | 0.821 (constant 6.5) | 0.727 |
  | MUE(shift) | 0.570 (zero shift) | 0.457 |
  | MUE(selectivity) | 0.125 | 0.101 |
  | r(selectivity) | — | +0.379 |

  Read the null column first. Predicting 6.5 for every histidine gets MUE 0.821,
  so the model's apparent absolute accuracy is mostly "histidine pKa is usually
  near 6.5" — it adds 11%. It adds most (20%) on the *shift*, which is at least
  the quantity selectivity depends on, but 0.457 log units of error per site
  against a per-histidine effect of 0.5–1.2 means error is comparable to signal
  for one site, and only averages down over 3–4 of them.

  So this stays a gradient direction and `ph_score.py` stays the ranking score.
  Two things to fix if it is revisited. `screening_length` fits to 3.38 against a
  lower bound of 3.0 — the fit wants electrostatics to act almost only on
  immediate neighbours, which together with `alpha_desolv` falling 3.0 → 0.82
  means it has reduced the SIE continuum it was meant to approximate to
  something close to a nearest-neighbour model. And `r(sel)` *fell* with fitting
  (+0.416 → +0.379) while MUE improved, so the least-squares objective is
  misaligned with the rank correlation the score is actually used for — fit
  Spearman instead.

  `ph_physics.load_params()` reads the fit, and `design.py`'s `physics` pool uses
  it. Fitted values: `epsilon_solvent` 91.4, `epsilon_interior` 17.6,
  `screening_length` 3.38, `alpha_desolv` 0.82, `offset` ≈ 0.

  Labels are cached in `pdb_cache/labels.npz`, so a refit is seconds rather than
  the ~25 minutes PROPKA labelling costs.

  A learned surrogate is still the fallback if the analytic form proves too crude:
  PROPKA labels are free (~0.6 s/complex, so 20k structures is ~90 s on one
  128-core node) and the features `mosaic` exposes differentiably are enough to
  regress on. But it inherits a distribution-shift problem the analytic form does
  not have — PROPKA labels come from full-sidechain structures, and design-time
  predictions have no sidechains.

### Pools

| pool | pH objective | purpose |
|---|---|---|
| `affinity` | none | baseline; maximizes chance of any hit |
| `ph` | count proxy, 3 histidines | needs elevated pKa_free to reach 2 log10 |
| `ph_hot` | count proxy, 5 histidines | brackets the budget from above |
| `physics` | differentiable linkage selectivity | optimizes the scored quantity directly |

The `physics` pool keeps a weak count proxy alongside the physics term: without
any pressure to put histidines in the interface at all, the selectivity term has
little to act on early in a trajectory.

## Running it

Design runs need a GPU node — `design.py` refuses to start on CPU:

```bash
./submit.sh affinity 8 65      # pool, n_designs, binder_length
./submit.sh ph       8 75
./submit.sh ph_hot   8 75
./submit.sh physics  8 75
```

Then rank and build the submission:

```bash
python rank.py designs/*.jsonl --out submission.csv
```

## Status / next steps

- [x] epitope derived and mouse conservation mapped
- [x] pH-selectivity scoring validated against TNF's own histidines
- [x] coordinate-space epitope weighting with mouse-divergent repulsion
- [x] differentiable linkage-selectivity loss, sign-tested
- [x] calibrate `ph_physics` against PROPKA (316 complexes, 1272 histidines)
- [x] negative controls: adalimumab −0.312, infliximab +0.385 log10 (PASS)
- [ ] **generate designs — nothing has been run yet, this is the critical path**
- [ ] repredict designs and populate Tier 2 scores (needs a GPU job)

See [LOG.md](LOG.md) for what has been tried, including what did not work.
- [ ] decide submission split across the three pools
