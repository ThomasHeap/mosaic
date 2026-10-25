# Methods — GLP1R Challenge 03

20 de novo single-chain binders, 40–55 aa, no cysteines, maximum pairwise
sequence identity 27%. Per-design metrics are in `submission_metrics.csv`.

Tools: `mosaic` (Escalante Bio) for differentiable design, ProtenixBase for
co-folding and scoring, Boltz-2 as an independent check, soluble ProteinMPNN for
inverse folding, Foldseek for the novelty screen. All open source. No commercial
tools were used.

## Approach

Four stages, in order, each covered by a section below.

1. **Epitope.** Taken from the GLP-1 footprint in 6X18, split by domain. 58% of
   it is transmembrane, and the extracellular half alone is the affinity site,
   so the recommended epitope on its own gives an antagonist.
2. **Design.** Differentiable optimisation against the templated active-state
   receptor, with three target-side constraints: attract the orthosteric
   epitope, require the activation core, repel the lipid-facing belt.
3. **Scoring.** ProtenixBase for binding, calibrated so that real ligands and a
   scrambled control separate; Boltz-2 as an independent check on the
   activation metric.
4. **Novelty and ranking.** Foldseek against the PDB, then ranking on activation
   with binding as a gate.

The activation analysis in the next section is background to stages 2 and 4: it
sets which residues a design must reach, and it is why the epitope is larger
than the one the challenge page recommends.

## Epitope definition

**Stated explicitly**: designs target the GLP-1 orthosteric site on human GLP1R,
spanning both the extracellular domain and the transmembrane bundle.

The epitope was computed from contacts, not chosen. Heavy-atom contacts within
4.5 Å were taken between GLP-1 (chain P) and the receptor (chain R) in 6X18, the
2.1 Å active-state structure with Gs bound. That gives 52 receptor residues.

Two results from this analysis determined the design setup.

**The site is 58% transmembrane**, and the split by peptide half is clean.
GLP-1's N-terminal half (His7–Ser18) makes about 40 transmembrane contacts and
5 to the ECD; its C-terminal half (Tyr19–Arg36) makes about 30 ECD contacts and
11 to the transmembrane domain. This is the two-domain mechanism: the ECD is an
affinity trap for the ligand C-terminus while the N-terminus engages the bundle,
and engagement of the bundle is what activates. Truncating two residues from
GLP-1's N-terminus costs 100–300× affinity but 1,000–10,000× cAMP potency.

So the challenge page's recommended epitope, the ECD peptide-binding groove, is
the affinity half alone. A design confined to it is a competitive antagonist,
which objective 2 discounts.

**An activation core of 18 residues** was defined as those within 4.5 Å of
GLP-1 His7–Thr11, the activating segment: 152, 190, 194, 233, 234, 237, 241,
300, 306, 309, 310, 313, 372, 380, 384, 387, 388, 391. These span TM1, TM2, TM3,
TM5, TM6 and TM7 — the helices whose rearrangement is activation.

## Design

Differentiable optimisation in `mosaic` against ProtenixBase with the active
receptor (6X18 chain R) as a template and no MSA. Templated and without an MSA
measured better than every alternative on the epitope: Cα RMSD 2.46 Å against
2.54 Å with an MSA and 3.66 Å for Boltz-2, at mean pLDDT 0.853.

Three target-side contact terms, all derived from the structure:

- the 52-residue orthosteric epitope at 20 Å, weight 1.0, as general placement
  pressure;
- the 18-residue activation core at 8 Å, weight 2.0. The tighter distance is
  deliberate: reaching these residues means entering the bundle rather than
  pointing at it;
- the 65-residue lipid-facing belt at 8 Å, weight −2.0. The assay protein is in
  DDM/CHS detergent, so the belt is solvent-exposed in the construct, and no
  structure model here represents a membrane. The belt is both larger and more
  hydrophobic than the epitope (65 residues at 72% against 52 at 52%), so it is
  the easier surface to bind. A design binding it would score on objective 1,
  where affinity is measured on the solubilised construct, and fail objective 2,
  since in a cell that surface is buried in lipid.

Plus the published Nipah recipe as the affinity objective: soluble ProteinMPNN
sequence recovery at weight 10 and temperature 0.001, confidence terms
(BinderTargetPAE and TargetBinderPAE 0.05, WithinBinderPAE 0.4, IPTMLoss 0.025,
PLDDTLoss 0.1), and a three-stage annealed `simplex_APGM` schedule with
gradients clipped to norm 1.0. Cysteine is excluded throughout.

**A fourth term, restricted to the binder's N-terminus, is what made the
difference.** The activation term above accepts contact from any binder
position, so a design can lie across the pocket and collect contacts without
entering it: an initial sweep reached 11–14 of 18 core residues while binding
worse than GLP-1. A second contact term restricted via `paratope_idx` to the
first five binder residues, targeting only the 15 receptor residues GLP-1
His7–Glu9 touches, fixes this. It mirrors the real mechanism, where the
C-terminal half holds the ECD while His7–Glu9 reaches the bundle.

It works by the intended mechanism, not incidentally. Measuring the depth of the
binder's first five residues against the rest along the transmembrane axis:
without the term the N-terminus points outward and the body sits deeper, by
−11.8 to −22.5 Å across three binder lengths. With it, all three flip positive
(+2.6, +5.1, +4.6 Å), against +6.7 Å for GLP-1 and +6.6 Å for exendin-4.

138 designs were generated at 40, 55 and 65 residues. 55 was best: the
N-terminal term raised activation-core contacts by +3.1 there on ProtenixBase
and +4.7 on Boltz-2, against +0.6 and +0.2 at 65 residues.

## Scoring

Designs were refolded at `recycling_steps=3` over five seeds and scored on
binding (ipTM, ipSAE both directions), activation-core contacts, lipid-belt
contacts, mean Kyte-Doolittle hydropathy and pLDDT.

**Binding is calibrated.** Through the same pipeline, GLP-1 7-37 scores ipTM
0.87 and exendin-4 0.90, against 0.42 for a scrambled peptide of GLP-1's
composition and 0.25 for an early design. The 0.50 gate used for selection sits
above the scramble.

**Activation is screened, not predicted.** There is no validated prediction of
receptor activation here, and the methods should not imply one. What is measured
is whether a design engages the residues GLP-1's activating segment engages.
That metric was calibrated on pairs that both bind and differ only in
activation — each agonist against its own N-terminally truncated antagonist —
over five seeds each:

| ligand | activation-core contacts |
|---|---|
| GLP-1 7-37 (agonist) | 17–18 of 18 |
| GLP-1 9-37 (binds, no activation) | 11–12 |
| exendin-4 (agonist) | 15–18 |
| exendin(9-39) (antagonist) | 0–6 |

The ranges do not overlap within either pair, and ipTM barely moves inside a
pair (0.87 against 0.85; 0.90 against 0.88), as it should when both members
bind. Two limits belong with that result. The calibration is partly circular:
the core is defined by the residues GLP-1 His7–Thr11 contacts, and each
antagonist is that agonist with those residues removed, so fewer contacts
follows by construction. And the count means nothing on its own — a scrambled
peptide reaches 10–18 across seeds and is rejected only by ipTM.

**Boltz-2 provides an independent check on the activation metric**, which is the
number most exposed to being scored by the model that did the designing. It
reproduces the calibration on both pairs with no overlap (17–18 against 11–13;
15–18 against 1–5), and agrees that 16 of 18 earlier candidates sit above the
antagonist range. Its ipTM is not usable here: a scrambled peptide scores
0.84 ± 0.18 against 0.91–0.94 for the real ligands, so ProtenixBase remains the
binding gate. Per-design agreement on core counts is moderate (r = +0.45), so
the two models agree on which designs pass rather than on their exact order.

## Novelty and ranking

Foldseek (PDB 05.2024) was run on predicted monomer structures against Adaptyv's
published thresholds, where level 4 requires sequence identity ≤ 30% and
below-moderate structural similarity, moderate being > 70% coverage at
TM-score ≥ 0.5. Of 48 candidates, 21 were level 4, 19 level 3 and 8 level 2. The
eight at level 2 were excluded, since level 3 is enforced at submission. That
cost the best design by both metrics, at ipTM 0.88 with 17.4 core contacts.

The submitted 20 are 13 level 4 and 7 level 3. The structural matches are
unrelated proteins that happen to contain similar helical segments — an ATP
synthase c-ring, a Get3 ATPase, an integrin-binding protein — rather than
GLP1R binders or peptide hormones. Short helical designs match many PDB entries.

Ranking follows the stated goal order, activation first and binding second: ipTM
gates at 0.50 and activation-core contacts sort, with ipTM breaking ties. Of 40
eligible designs only one failed the gate, so the cut is set by the 20 slots
rather than by quality. Ranking by binding instead would raise median ipTM from
0.63 to 0.72 but drop the weakest activation score from 16.2 to 13.2 and level-4
designs from 13 to 9. Since the mean seed standard deviation on core counts is
1.2, the ordering was also checked with counts bucketed to integers; that moves
one design of twenty.

## Limitations

**No functional evidence of activation.** Every number here comes from two
structure-prediction models. The claim supported is that these designs engage
the activation pocket with the geometry GLP-1 uses, not that they activate the
receptor.

**Attempts to measure activation functionally failed, and one failure was
instructive.** An active-minus-inactive binding differential does not separate
the calibration pairs: agonists span −0.005 to +0.030 and negatives −0.131 to
+0.058, with the scrambled peptide scoring highest. Insertion depth separates
the exendin pair but not the GLP-1 pair. A Gs ternary-complex test, co-folding
binder, receptor and Gs-α and counting whether the receptor–Gs interface forms,
gives 27.3 of 29 contacts with **no binder at all**, against 27.3–28.3 for
agonists and antagonists alike: the template is an active-state Gs-bound
structure, so it fixes the answer before the binder is considered. That closes
the whole family of readouts taken from an active-state template, including the
TM6 position.

**A second-stage refinement of the N-terminus did not help.** Taking the five
best scaffolds and resampling the first five residues from ProteinMPNN's
distribution on the threaded backbone gave 0 of 80 variants better than their
parent on both axes; medians fell from ipTM 0.79 to 0.56 and from 16.2 to 14.2
core contacts. The analogous stage worked for a pH switch on another target
because the proposing model scored the property being installed; ProteinMPNN
does not score pocket engagement.

**Composition is unusual.** The submitted designs are 33–58% charged against 23%
for both GLP-1 and exendin-4. Charge does not track predicted core contacts
(r = −0.175), so no structure-based score here would surface it, and a
half-charged peptide threading a hydrophobic bundle is chemically questionable
whatever the models report.

**The scoring model is also the design model.** Boltz-2 checks the activation
metric independently, but ProtenixBase both designs and gates on binding,
because Boltz-2's binding score does not discriminate on this target and
AlphaFold2 multimer was tried and gave results that tracked the parameter set
rather than the sequence.
