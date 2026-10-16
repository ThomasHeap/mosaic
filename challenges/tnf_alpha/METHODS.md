# Methods — TNF-α Challenge 02 (Track 3)

20 de novo single-chain binders, 55–90 aa, no cysteines, maximum pairwise
sequence identity 22%. Per-design metrics are in `submission_metrics.csv`.

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

Core epitope (construct numbering): 20, 21, 23, 67, 71, 73, 75, 77, 97, 110,
111, 113, 115, 137, 140, 141, 143, 144, 145, 146.

Three results from this analysis determined the design setup.

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
ordinary histidines. Submitted designs carry 4–11 histidines.

Selectivity increases monotonically with the free-state pKa, which neighbouring
Asp and Glu raise. Several submitted designs are consequently acidic (net charge
−12 to +3, median −3).

This was checked against measured antibodies before designing. Infliximab's
CDR-L3 His92 shows a −2.35 pKa shift on binding but contributes only +0.33
log10, because its free-state pKa is 6.09. TNF's own His15 and His78, at
pKa_free 5.06 and 4.01, contribute 0.044 and 0.003.

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
`mosaic` carry no sidechains. A distogram contact term cannot distinguish
residues to contact from residues to avoid; this term can, which is how
criterion 2 was optimised rather than measured afterwards.

A differentiable proton-linkage objective makes the scored quantity itself the
target. An electrostatics-only pKa model — Coulomb over the environment's
expected charge under the PSSM, plus a burial desolvation term, mapped linearly
to pKa following JustHISpKa (Hogues, Wei & Sulea, JCIM 2025) — is evaluated on
both end states from a single complex prediction: summing over binder neighbours
alone gives pKa_free, adding the target gives pKa_bound. The result is passed
through the linkage relation above.

Four pools were run to separate the criteria: affinity only, two in-loop
histidine budgets (3 and 5), and the differentiable linkage objective. 455
designs were generated in total.

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

Variants were then refolded and rescored, which is required: the scan does not
model structural tolerance, and the median ipTM cost of a four-histidine variant
was −0.117. Measured outcome across 223 variants: median selectivity gain +0.510
log10, maximum +2.744, 61 of 85 in the first batch improving. Sixteen of the 20
submitted designs came from this route.

## Scoring and ranking

Designs were refolded with full-sidechain features at `recycling_steps=3` and
scored on four axes: pH selectivity (PROPKA on the predicted complex and the
predicted free binder, combined through the linkage relation), mouse
cross-reactivity (refolded against mouse TNF, P06804 mature sequence), affinity
(ipTM and both ipSAE directions), and monomer/complex Cα RMSD.

Negative controls: adalimumab and infliximab bind this groove and neither is
pH-dependent. Scored through the same pipeline they give −0.312 and +0.385
log10, against a target of approximately 2.0.

The submission is a portfolio rather than a strict ranking. Eight designs clear
selectivity 2.0, mouse ipTM 0.80 and ipTM 0.75 simultaneously, so 20 slots
cannot be uniformly strong on all three criteria. Rows 1–13 meet all three at
reduced thresholds (selectivity ≥ 1.7, mouse ipTM ≥ 0.80, ipTM ≥ 0.75). Rows
14–17 favour criterion 1 (selectivity 2.30–2.83, lower mouse ipTM). Rows 18–20
favour criterion 2 (mouse ipTM ≥ 0.96, selectivity 1.20–1.38). All 20 clear ipTM
0.70, since selectivity is an affinity ratio and is not meaningful for a design
that does not bind at either pH.

Selectivity is bucketed at 0.5 log10 before ordering, because PROPKA's error on
the pKa shift is 0.46 log10 per site and finer distinctions are not measurable.

## Limitations

Monomer pLDDT is approximately 0.5 across the submitted set, so these binders are
not predicted to fold confidently in isolation. Whether this reflects an
expression liability or an artefact of co-folding short de novo monomers is
unresolved.

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

### Monomer foldability: a measured, unresolved liability

The single clearest weakness in this submission is that the designs are not
predicted to fold confidently on their own. Two structure predictors agree,
each against the same two natural controls:

| instrument | GB1 / Top7 controls | these designs | designs reaching control |
|---|---|---|---|
| Boltz-2 monomer pLDDT | 0.935 / 0.882 | median 0.541 | 0 / 20 |
| ESMFold2 monomer pLDDT | 0.829 / 0.725 | median 0.459 | 0 / 20 |

ESMFold2 is more conservative in absolute terms but reproduces the separation,
and it is single-sequence by construction rather than a co-folding model used
off-label. mosaic's Megascale stability model disagrees, placing 15 of 20
designs above GB1, but it also scores GB1 itself at dG +0.29 — below most of
our designs — which is not credible for one of the best-characterised small
domains known, so it is treated as the outlier. Its sign convention was checked
separately against scrambled sequences and is correct; the issue is calibration.

A binder that only folds when bound is a plausible expression and aggregation
risk, which is why this is reported rather than omitted. Two caveats belong
with it. First, the inference from low monomer pLDDT to experimental failure is
**not validated here**: these designs differ from well-folded controls, but no
measured relationship links that difference to expression outcome, and the
Nipah de novo work that this pipeline's recipe follows reports a 90% hit rate
without reporting monomer pLDDT at all. Second, a disordered-alone binder that
folds on binding is a real and functional class of molecule; the costs are
entropic and proteolytic rather than automatically fatal.

Two routes to fixing it were tried and both measured as not worth taking.

**Adding monomer confidence to the design loss costs binding.** Four arms at
n=2, 65-residue binders, pH pool:

| arm | monomer pLDDT | composition | ipTM (unselected) |
|---|---|---|---|
| baseline | 0.555 | ok | 0.599 |
| monomer 2.0 + PLL 0.5 | 0.694, 0.638 | ok | 0.367 |
| monomer 3.0 + PLL 1.0 | 0.806, 0.811 | 40% / 34% Ala | 0.326 |
| monomer 2.0 + PLL 0.5 + MPNN | 0.643, 0.572 | ok | 0.371 |
| monomer 3.0 + PLL 1.0 + MPNN | 0.531, 0.697 | ok | 0.343 |

The highest pLDDT occurs only in the arm that collapsed to 40% alanine, past
rank.py's 25% homopolymer guard: poly-alanine helices read as foldable to any
confidence head, so most of the apparent gain was a composition artifact rather
than folding. ESMC pseudo-perplexity did not prevent that collapse; ProteinMPNN
on the monomer branch did, being structure-conditioned, but it removes the
pLDDT gain along with the artifact. Within-arm spread (0.531 vs 0.697) exceeds
every between-arm difference, so n=2 cannot rank these configurations.
Runtimes matched baseline (2278s vs 2069s per design), ruling out undertraining.

Notably, soluble-weight ProteinMPNN was already applied throughout at weight
10.0 (`soluble_v_48_020.pt`), and did not prevent the problem, because in the
complex branch it answers a different question: given this backbone *as it
appears bound*, what sequence would a soluble-protein model choose? Solubility
conditional on a fold is not the same property as having a self-supporting fold.

**Selecting on it is free but inert.** Over 515 scored designs, ipTM and monomer
pLDDT are near-independent (r = +0.095; an earlier estimate of +0.006 was
attenuated by restricting the sample to ipTM >= 0.75), so preferring
better-folded designs costs essentially no binding. But inserted as a
tie-break beneath the challenge's three stated criteria it changed one design
of twenty, for +0.007 pLDDT and -0.016 ipTM. After lineage dedup and the hedge
floors only 13 all-rounders remain, so those constraints are already nearly
binding and no slack remains for a fourth axis.

The tail explains the ceiling. The five designs with the highest monomer pLDDT
(0.71-0.81) are all rejected for monomer/complex RMSD of 5.3-18.0 A: they fold
confidently alone into a conformation that is not the bound one. That is argu-
ably worse than low confidence, because a stable alternative fold competes with
the binding-competent state. Within this pool, high monomer confidence and
bound-state agreement are close to mutually exclusive.

The submission therefore retains the selection optimised for the three scored
criteria, with this liability stated rather than papered over. The route not
taken, given more compute budget, is to generate a substantially larger pool
under the baseline objective and select on foldability only where it is free —
the pool, not the loss function, is the binding constraint.
