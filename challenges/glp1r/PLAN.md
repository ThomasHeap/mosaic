# Challenge 03 — GLP1R (prep notes)

Target: human GLP1R, UniProt P43220-1 residues 24-432, in DDM/CHS detergent,
N-terminal FLAG and C-terminal His, plus unlisted thermostabilising mutations.
Up to 20 designs, de novo only, 10-250 aa. Deadline 2026-10-25 23:59 AoE.

Two criteria, ranked: (1) binding affinity, (2) receptor activation in a
cell-based functional assay. A binder that does not activate does not count.

## The central constraint

Activation is not a filter on top of binding, it dictates where to bind. Class B
GPCRs follow a two-domain mechanism: the extracellular domain is an affinity
trap for the ligand C-terminus, while the ligand N-terminus engages the
transmembrane bundle, and that engagement is what activates. The potency data is
asymmetric -- truncating two N-terminal residues of GLP-1 costs 100-300x
affinity but 1000-10000x cAMP potency.

6x18.cif (2.1 A, active state, GLP-1 bound) confirms it directly. Of the 52
receptor residues within 4.5 A of GLP-1, 30 (58%) are transmembrane, and the
split is clean:

  GLP-1 His7-Ser18 (N-terminal half)  ~5 ECD contacts,  ~40 TMD contacts
  GLP-1 Tyr19-Arg36 (C-terminal half) ~30 ECD contacts, ~11 TMD contacts

So the challenge's *recommended* epitope, the ECD peptide-binding groove, is the
affinity half only. A design confined to it is a competitive antagonist, which
objective 02 discounts. Agonism requires reaching TM2/TM3 (190-241), ECL2/TM5
(298-313) and TM7 (372-391).

Caveat: one study reports GLP-1's first five residues are dispensable when the
peptide is fused to the receptor, and argues the ECD can activate the TMD in
trans. That is the minority view and the only one under which an ECD-only
agonist is plausible. Worth reading before committing.

## Consequences for design

Architecture is forced. A globular miniprotein cannot reach the TMD pocket; the
geometry wants roughly what GLP-1 and exendin-4 already are, a ~30-45 residue
helix with an extended N-terminal segment that threads into the bundle while the
rest lies along the ECD groove. This is unlike the TNF binders (globular, 50-90
aa). It also makes the monomer-foldability question from that challenge moot: a
long single helix has little tertiary structure by construction.

Agonism has a computable proxy. With an active structure (6x18) and an inactive
one (5vew) in hand, a design that prefers the active conformation stabilises it,
which is agonism thermodynamically. That is the same shape of score as TNF pH
selectivity -- predict under two conditions, take the log ratio -- so ph_score.py
adapts rather than being discarded. Note 5vew is TMD-only (no ECD) and a
different construct length, so the comparison has to be restricted to the common
TMD region or it confounds conformation with construct.

## Agonism: a geometric criterion, not a binding differential

The planned proxy, active-minus-inactive binding, was measured on known ligands
and does not work:

  ligand          kind                affinity   active  inactive    delta
  GLP-1 7-37      agonist                0.900    0.355     0.360   -0.005
  exendin-4       agonist                0.879    0.304     0.275   +0.030
  GLP-1 9-37      binds, no activation   0.866    0.324     0.481   -0.157
  scrambled       negative               0.556    0.512     0.454   +0.058
  poly-alanine    negative               0.565    0.247     0.378   -0.131

Agonists span -0.005 to +0.030 and negatives -0.131 to +0.058, overlapping
completely, with the scrambled peptide scoring the highest delta of anything.
Two reasons. 5vew is transmembrane-only, so every ligand binds both constructs
badly (0.25-0.51 against 0.87-0.90 on the full receptor) and the difference is
noise on a weak signal. And agonism is not a binding preference between two
conformations; it is whether one specific pocket is engaged, which a scalar
affinity difference cannot express.

The affinity channel is validated by the same run: real ligands score 0.866 to
0.900 and negatives 0.556 to 0.565, a clean 0.3 separation on the full active
receptor. Objective 01 has a trustworthy score. Note also that GLP-1 9-37
scores 0.866, essentially tied with intact GLP-1 at 0.900 -- the affinity score
correctly reports that it binds and correctly says nothing about whether it
activates.

So activation is handled geometrically. target.ACTIVATION_CORE is the 18
receptor residues within 4.5 A of GLP-1 His7-Thr11, the segment whose removal
costs 1000-10000x cAMP potency against 100-300x affinity: 152, 190, 194, 233,
234, 237, 241, 300, 306, 309, 310, 313, 372, 380, 384, 387, 388, 391, spread
over TM1, TM2, TM3, TM5, TM6 and TM7. All 18 lie inside the orthosteric
epitope; two also fall in the lipid belt and are excluded there.

Contacting them is checkable on a single prediction with nothing to calibrate.
It does not prove activation. It establishes that a design engages the
machinery activation requires, which is the strongest claim available without a
cell assay, and it is a condition the ECD-only binders the challenge page
recommends would fail.

## Epitope must be specified, and so must an exclusion zone

The epitope cannot be left to the optimiser, for a reason TNF did not have. In
6x18 the GLP-1 orthosteric footprint is 52 residues at 52% hydrophobic, while
the lipid-facing transmembrane belt is 65 residues at 72% hydrophobic. The belt
is the larger and stickier surface, Boltz-2 models no membrane, and the two sets
overlap by only 2 residues.

A design that binds the belt would score on objective 01, since affinity is
measured on the detergent-solubilised construct where that surface is exposed,
and fail objective 02, since in a cell it is buried in lipid. That is the worst
available failure: passing the first criterion for a biologically meaningless
reason.

BinderTargetContact takes epitope_idx but is attractive only, so it rewards the
footprint without penalising the belt. A second copy at negative weight gives
the repulsion, since the term returns -log P(contact). Negative-weight terms are
unbounded below and may need clipping.

Epitope: the 52 footprint residues, keeping the TMD subset (190-241, 298-313,
372-391) as required contacts rather than optional, since those drive activation.
Truncating the receptor to its extracellular part would remove the belt by
construction but also the pocket, so template the full 24-432 instead.

## Model choice: ProtenixBase

Measured on the templated 384-residue receptor plus a 40-residue binder, CA RMSD
against the 6x18 template:

  model           time   pLDDT    all    ECD    TMD   epitope
  boltz2         19.9s   0.810   4.30   2.70   1.86   3.66
  protenix_mini  11.9s   0.774   5.70   4.03   2.35   3.85
  protenix_base  14.7s   0.814   4.12   3.14   1.14   2.54

ProtenixBase for design, Boltz-2 to score. The orthosteric epitope is the
surface being designed into and ProtenixBase reproduces it at 2.54 A against
3.66, while also being faster. Boltz-2 is better on the ECD alone. Scoring with
the model that did the designing is how a binder to a pocket that does not exist
gets a high score, and neither model represents a membrane, so they share that
blind spot but not their individual artefacts.

An earlier note here said Protenix could not be used without patching mosaic.
That was wrong. Both CCD files had been truncated by a download timeout -- 63 MB
of 408 MB, stopping at 6548 of 43677 component blocks -- so every residue lookup
returned None, including ALA and GLY.

## What the TNF work implies for this challenge

Two transfers and one warning.

Novelty is a hard criterion and enters ranking from the start here, not after
the fact. novelty.py and the Foldseek PDB database at
/user/work/cp20141/foldseek_db are already built. In TNF it moved the submission
from 13 clean of 20 to 15 with no cost to the scored criteria, but only once
mouse ipTM was bucketed so it had slack to bind on.

Check the conjunction rate early. TNF Track 3's difficulty was that three
independent criteria at 16.7%, 44.9% and 35.3% intersect at 2.9%. Here there are
two, affinity and activation, and the equivalent number should be measured on
the first fleet rather than assumed.

The warning: 87% of TNF's all-rounders came from post-hoc histidine scanning,
not from design, because installing a pH switch is a sequence substitution on a
fixed backbone and can be done after the fact for free. Agonism is not like
that. It is geometric -- the binder has to reach TM2/TM3, ECL2/TM5 and TM7 -- so
there is no cheap second stage to fall back on and it has to be designed in.
Expect the in-loop objective to carry the whole burden here.

## Activation: what has been tried and what is left

Two approaches have been measured and neither works as a selection signal.

The active-minus-inactive binding differential fails on known ligands. Over
GLP-1 7-37, exendin-4, GLP-1 9-37, a scrambled peptide and poly-alanine the
delta runs -0.005 to +0.030 for the agonists and -0.131 to +0.058 for the
negatives, with the scrambled peptide scoring highest of all. 5vew is
transmembrane-only so every ligand binds both constructs badly, and agonism is
not a binding preference between conformations anyway.

Activation-core contact count discriminates only among designs that bind. A
scrambled peptide reaches 10 to 18 of the 18 core residues across five seeds, so
on its own the count is noise; the scramble is rejected by ipTM, 0.42 against
0.85 to 0.90 for the real ligands.

Scoring model settled in passing: Boltz-2 cannot score this target. It ranks the
scrambled peptide at ipTM 0.954, above GLP-1 at 0.934, while ProtenixBase gives
0.422 against 0.869. Affinity is therefore scored with ProtenixBase, which means
the scoring model is also the design model -- not ideal, but a non-discriminating
independent scorer is worse than none.

Four ideas remain, in the order worth trying:

1. **Agonist/antagonist calibration pairs: done.** Pairs that both bind and
   differ only in activation, five seeds each, ProtenixBase. Core contacts
   separate both with no overlap between agonist and antagonist ranges:

     GLP-1 7-37      17-18 of 18     GLP-1 9-37      11-12
     exendin-4       15-18           exendin(9-39)    0-6

   ipTM barely moves within a pair (0.87 against 0.85, 0.90 against 0.88), as it
   should when both members bind. Limit: both antagonists are the agonist with
   its N-terminal residues removed, and the core is defined as the receptor
   residues contacted by GLP-1 His7-Thr11, so a truncated peptide contacts fewer
   of them by construction. The result shows the metric tracks the activating
   segment. It does not show that a de novo binder reaching those residues with
   different chemistry would activate.

2. **Insertion depth: tested, dropped.** Along the transmembrane axis GLP-1
   reaches -6.4 A, matching the crystal's -6.3. It separates the exendin pair
   (-5.8 against -14.2) but not the GLP-1 pair (-5.9 against -6.6), and
   exendin(9-39) varies by 6 A between seeds. An earlier run appeared to separate
   both pairs, but it measured from the template's origin while predictions come
   back about 100 A away; the axis has to be recomputed from each prediction.

3. **Gs ternary-complex compatibility.** 6x18 contains the Gs heterotrimer.
   Agonism is G-protein coupling, so co-folding binder, receptor and Gs and
   asking whether the Gs interface survives measures the functional output
   rather than a structural correlate. Unknown whether the model is sensitive
   enough to report a real difference.

4. **TM6 position.** The outward swing of TM6's intracellular half is what class
   B activation physically is. Unusable as set up: templating 6x18 puts TM6 in
   the active position before the binder is considered, so the prediction is
   biased toward the wanted answer. Templating the inactive state would test
   induced fit, which structure predictors do not really do. Parked unless 1-3
   fail.

Independent scoring: Boltz-2 confirms the core-contact metric, ProtenixBase
gates binding. The earlier claim that Boltz-2 "cannot score this target" rested
on one prediction per ligand, where the scrambled peptide happened to score ipTM
0.954 against GLP-1's 0.934. Over five seeds the scramble is 0.84 +/- 0.18,
range 0.53 to 0.96, so that single draw was the top of a wide range; Protenix
gives 0.42 +/- 0.02 on the same peptide and remains the better binding gate.
But Boltz-2's core-contact counts separate both agonist/antagonist pairs with no
overlap -- GLP-1 17-18 against 11-13, exendin-4 15-18 against 1-5 -- reaching
the same conclusion as Protenix independently. Since the core count is the
number most at risk from scoring with the design model, that is the useful
independence.

AlphaFold2 multimer was tried and is not usable. Template-only, since mosaic's
wrapper takes no MSA, results depend on which parameter set rather than on
sampling: exendin-4 scores ipTM 0.70-0.73 with 15-16 core contacts under models
1 and 5, and 0.12-0.21 with 0 contacts under models 2, 3 and 4. The scramble
reaches 0.46 under model 1 and overlaps the agonists. OpenFold3 is untried and
no longer needed.

Selection rule that follows: gate on ProtenixBase ipTM first, then rank by
activation-core contacts, confirmed on Boltz-2. The core count is a screen for engaging the activating
pocket, not a prediction of activation, and the methods should say so.

## De novo compliance: settled

Adaptyv's novelty scale scores only global sequence identity (MMseqs2) and
domain-level structural similarity (Foldseek, TM-align). It does not look at
contact residues; the only residue-level test is CDRH3 edit distance for
antibodies. A design at <= 30% identity with below-moderate structural
similarity is level 4 whether or not it reuses a known binding mode.

Their definition of de novo nonetheless excludes "grafting a motif onto a
natural scaffold", while describing current practice approvingly as conforming
to a functional motif "rather than retrofitting an existing protein to host
one". That separates the two things cleanly:

  - Grafting GLP-1's N-terminal segment onto a scaffold is out. It is motif
    grafting by their definition and relies on an existing binder as a starting
    point, which the hard criteria forbid outright.
  - Constraining the design to contact TM2/TM3, ECL2/TM5 and TM7 is in. That is
    a property of the target, not a borrowed piece of the ligand, and the binder
    sequence is found by optimisation.

So: never seed from GLP-1 or exendin-4, constrain the epitope, and screen the
result. Designs are checked for identity against the known GLP1R ligands --
GLP-1 7-37, exendin-4, liraglutide, semaglutide, tirzepatide -- and anything at
or above 30% to any of them is dropped regardless of how well it scores. The
TNF pool gave the equivalent check a clean result, with no match to any known
TNF binder across 13,514 Foldseek hits.

## Reuse from tnf_alpha

Direct: design.py, rescore.py, rank.py, selftest.py, diagnostics.py, target.py
(structure-file driven via gemmi), liabilities.py, controls.py, gravity.py.
Drop: hisscan.py, ph_score.py, ph_physics.py, ph_switch.py, calibrate.py --
about 1500 lines of pH machinery with no analogue here, except ph_score.py's
two-state pattern noted above.
Rework: compute_epitope.py for the GLP-1 footprint.

Two criteria instead of three should also help. TNF Track 3's binding constraint
was the conjunction: 16.7% x 44.9% x 35.3% gave 2.9% all-rounders in 515 designs.

## Structures

  6x18.cif  2.1 A  active GLP1R (chain R) + GLP-1 (chain P) + Gs + Nb35
  5vai.cif  4.1 A  same composition, superseded by 6x18
  3iol.cif  2.1 A  ECD only + exendin-4, for groove detail
  5vew.cif  2.7 A  inactive GLP1R TMD + antagonist, two copies
