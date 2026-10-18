# Methods — TNF-α Challenge 02 (Track 3)

20 de novo single-chain binders, 55–75 aa, no cysteines, maximum pairwise
sequence identity 18%. Per-design metrics are in `submission_metrics.csv`.

Tools: `mosaic` (Escalante Bio) for differentiable design, Boltz-2 for
co-folding and scoring, soluble ProteinMPNN for inverse folding, PROPKA 3.5.1
for pKa prediction. All open source. No commercial tools were used.

## Epitope definition

The epitope was computed from contacts rather than selected by hand. Heavy-atom
contacts within 4.5 Å were taken from three complexes that occupy the receptor
site — 3WD5 (adalimumab Fab), 4G3Y (infliximab Fab), 3ALQ (TNFR2) — and
residues present in at least two of the three were kept. TNF chains in these
entries are numbered on the mature 157-residue sequence, so no alignment was
required.

Core epitope, residues present in at least two of the three complexes
(construct numbering): 20, 21, 23, 67, 71, 73, 75, 77, 97, 110, 111, 113, 115,
137, 140, 141, 143, 144, 145, 146.

Design targeted a wider groove footprint taken from 3ALQ and split by protomer,
28 residues in total: 20, 21, 23, 31, 32, 33, 63, 67, 113, 115, 143, 144, 145,
146, 149 on one protomer and 71, 72, 73, 75, 77, 85, 86, 87, 88, 89, 91, 92, 97
on its neighbour. Excluding the four mouse-divergent residues inside it leaves
the 24 positions the contact term was restricted to.

The receptor groove spans two protomers. In 3ALQ, TNFR2 chain T contacts 15
residues on one protomer and 13 on its neighbour. All designs were therefore
co-folded against a TNF dimer templated on 1TNF. A binder optimised against a
TNF monomer targets a surface that does not exist in the assay construct.

Fourteen of the 20 core residues are conserved in mouse. The six that differ are
P20H, S71D, H73 (indel), I97V, A111L and D140K. This accounts for the human
specificity of adalimumab and makes criterion 2 a constraint on binder
placement.

His73 is the only histidine in the epitope and is lost to an indel in mouse. The
pH switch was therefore built on the binder, not on the target.

## pH selectivity

No structure-prediction model used here is pH-aware. pH-dependent affinity is a
consequence of pKa shifts on binding, related exactly by proton linkage:

    log10[Ka(7.4) / Ka(6.0)] = Σ_i [ δ(pKa_bound,i, pKa_free,i, 7.4)
                                   − δ(pKa_bound,i, pKa_free,i, 6.0) ]

    δ(b, f, pH) = log10(1 + 10^(b − pH)) − log10(1 + 10^(f − pH))

Only pKa requires prediction. Evaluating the relation before designing gave two
constraints.

Per-histidine selectivity across a 7.4 → 6.0 window is bounded:

| free-state pKa | log10 selectivity per His | His required for 100× |
|---|---|---|
| 6.0 | 0.28 | 8 |
| 6.5 (typical surface His) | 0.57 | 4 |
| 7.0 | 0.90 | 3 |
| 7.5 | 1.16 | 2 |

"No detectable binding at pH 6.0" is therefore not reachable with one or two
ordinary histidines. Submitted designs carry 5–11 histidines.

Selectivity increases monotonically with the free-state pKa, which neighbouring
Asp and Glu raise. Several submitted designs are consequently acidic (net charge
−12 to +1, median −3).

This was checked against measured antibodies before designing. Infliximab's
CDR-L3 His92 shows a −2.35 pKa shift on binding but contributes only +0.33
log10, because its free-state pKa is 6.09.

## Design

The loss function and optimiser follow the published recipe that won the
in-silico portion of the Adaptyv Nipah competition: Boltz-2 contact terms,
confidence terms (BinderTargetPAE and TargetBinderPAE 0.05, WithinBinderPAE 0.4,
IPTMLoss 0.025, pTMEnergy 0.025, PLDDTLoss 0.1), soluble ProteinMPNN sequence
recovery at weight 10 and temperature 0.001, a three-stage annealed
`simplex_APGM` schedule (100 steps at momentum 0.3, then 50 and 15 steps in log
space at proximal scale 1.25 and 1.40) with gradients clipped to norm 1.0, and
`num_samples=4`. Cysteine is excluded throughout.

Two terms were added for this target.

A signed coordinate-space epitope field assigns per-residue weights that attract
the binder to conserved epitope residues and repel it from the four
mouse-divergent residues inside the groove (P20, S71, H73, I97). Distances use a
virtual Cβ built from backbone N, CA and C, since design-time features in
`mosaic` carry no sidechains. A distogram contact term cannot distinguish residues to
contact from residues to avoid.

A differentiable proton-linkage objective makes the scored quantity itself the
target. An electrostatics-only pKa model — Coulomb over the environment's
expected charge under the PSSM, plus a burial desolvation term, mapped linearly
to pKa following JustHISpKa (Hogues, Wei & Sulea, JCIM 2025) — is evaluated on
both end states from a single complex prediction: summing over binder neighbours
alone gives pKa_free, adding the target gives pKa_bound. The result is passed
through the linkage relation above.

Four pools were run to separate the criteria: affinity only, two in-loop
histidine budgets (3 and 5), and the differentiable linkage objective. 1,228
unique designs were generated and scored, 735 of them histidine-scan variants.

## Interface histidine scanning

Which binder positions contact the epitope is an output of optimisation, so
histidines cannot be restricted to the interface during design. After design it
is straightforward, and it matches established practice for pH-dependent
antibodies: identify a binder, then scan its interface.

For each design with ipTM ≥ 0.70, interface positions were enumerated (Cβ within
10 Å of a conserved epitope residue, excluding existing His, Gly and Pro),
substitutions scored through the linkage relation, and combinations of up to four
positions evaluated. The scan is backbone-only, so a substitution changes the
charge vector and not the geometry; no refolding is needed to score it.

Variants were refolded and rescored, since the scan does not model structural
tolerance. Across 735 variants with a matched parent the median selectivity gain
is +0.454 log10 and the maximum +5.262, 66% improve on their parent, and the
median ipTM cost is −0.054. Nineteen of the 20 submitted designs came from this
route.

## Scoring and ranking

Designs were refolded with full-sidechain features at `recycling_steps=3` and
scored on four axes: pH selectivity (PROPKA on the predicted complex and the
predicted free binder, combined through the linkage relation), mouse
cross-reactivity (refolded against mouse TNF, P06804 mature sequence), affinity
(ipTM and both ipSAE directions), and monomer/complex Cα RMSD.

Negative controls: adalimumab and infliximab bind this groove and neither is
pH-dependent. Scored through the same pipeline they give −0.312 and +0.385
log10, against a target of approximately 2.0.

All 20 designs clear selectivity 1.7, mouse ipTM 0.80 and ipTM 0.75
simultaneously. Submitted ranges are selectivity 1.74–3.22 (median 2.25), mouse
ipTM 0.811–0.964 (median 0.908) and ipTM 0.765–0.910 (median 0.864); 18 designs
in the pool also clear the stricter selectivity 2.0. Each is the sole
representative of its design lineage.

Earlier revisions of this submission were hedged across the three criteria,
filling 7 of 20 slots with designs strong on one and weak on another, because
only 15 pool designs then cleared all three. Histidine scanning the later design
batches raised that to 29 and the hedging became unnecessary.

Selectivity is bucketed at 0.5 log10 before ordering, so that designs within
roughly one PROPKA site-error of each other are ranked by the next criterion
rather than by a difference the predictor cannot resolve.

## Novelty screen

Foldseek (PDB 05.2024) was run on the monomer structures from rescore.py against
Adaptyv's published thresholds, where level 4 de novo requires sequence identity
<= 30% and below-moderate structural similarity, moderate being > 70% coverage at
TM-score >= 0.5. Tiers enter ranking as a tie-break beneath the three scored
criteria, with mouse ipTM bucketed at 0.1, PROPKA-style, so that differences
below predictor error do not decide a slot.

15 of the twenty are below moderate and 5 are moderate; none reach the high
tier. The 5 are kept:

  tnf_ph_L55_s505_17_his4_29_40_30_23        TM 0.656  qcov 0.89  ipTM 0.902
  tnf_ph_hot_L75_s406_8_his4_66_67_70_73     TM 0.751  qcov 0.85  ipTM 0.872
  tnf_ph_hot_L75_s8106_9_his4_24_16_66_73    TM 0.669  qcov 0.80  ipTM 0.858
  tnf_ph_hot_L65_s8105_10_his4_30_28_1_64    TM 0.652  qcov 0.83  ipTM 0.769
  tnf_ph_hot_L75_s406_11_his4_72_65_68_30    TM 0.651  qcov 0.95  ipTM 0.765

These scores are approximate: they rest on Boltz-2 monomer structures at pLDDT
~0.5, while Adaptyv predict with ESMFold and segment domains before TM-align.

No design resembles a known TNF binder. Across 13,514 hits from the pool there
are no matches to TNFR1, TNFR2, adalimumab, infliximab, certolizumab or TNF
itself, and the nearest neighbours are bacteriophage capsids, bacterial proteins
and other designed proteins rather than human ones. Excluding cysteine also
rules out the cysteine-rich-domain fold both natural receptors use, so the
architecture proven for this site was not searched.

## Limitations

Monomer pLDDT is approximately 0.5 across the submitted set, so these binders
are not predicted to fold confidently in isolation. ESMFold2, single-sequence by
construction, reproduces the gap at median 0.46 against 0.83 for GB1 and 0.77
for Top7, so it is not an artefact of co-folding. Whether it is an expression
liability is unresolved: nothing here links monomer confidence to expression
outcome, and the Nipah work this recipe follows reports a 90% hit rate without
reporting the quantity. A monomer-confidence branch in the design loss raised it
by 0.08 pLDDT at a cost of 0.25 ipTM and was not used; monomer pLDDT and ipTM
are near-independent across the 1,228 scored designs (r = +0.043).

The differentiable pKa model is weak. Calibrated against PROPKA on 316 protein
complexes (1,272 histidines, 25% held out by complex), it improves on a
constant-predictor null by 11% on absolute pKa and 20% on the pKa shift, with a
selectivity rank correlation of r = 0.38. It was used only as a gradient
direction; all ranking used PROPKA.

The selectivity validation is one-sided. Two known non-pH-dependent binders score
near zero, which shows the score does not report selectivity where none exists.
It does not establish that the score finds selectivity reliably where it does. No
TNF binder with a published pH switch was available as a positive control.

Designs were optimised against a templated TNF dimer. If evaluation co-folds
without a template, predicted interface quality may not reproduce.
