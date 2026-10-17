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

## Model choice: Boltz-2, measured

Boltz-2 handles the target. A templated 384-residue receptor plus a 40-residue
binder predicts in 19.9 s at mean pLDDT 0.810, and reproduces the template:
CA RMSD 1.86 A over the transmembrane domain, 2.70 A over the ECD, 3.66 A over
the orthosteric epitope. The bundle survives, so the pocket being designed into
exists in the prediction. The epitope is the loosest region, which is expected
for a deep pocket and is the number to watch.

Protenix is unusable here without patching mosaic. Its target_only_features
passes template=c.template_chain straight to protenix's featurize, which
dereferences .res_id, and binder_features builds the binder as
TargetChain("X" * n) with no template. A designed binder has no template by
definition, so every design call fails on AttributeError. It remains an option
as an independent scorer if protenix.py is patched to tolerate a null template.

## Open question to settle first

Whether reproducing the activation chemistry counts as de novo. The N-terminal
contacts are chemically constrained (His7 and Glu9 make specific polar contacts),
so a working agonist will likely recapitulate them. The rules forbid starting
from an existing binder or modifying one, and require sequence and structural
diversity from known proteins; they do not obviously forbid matching functional
contact chemistry with a divergent sequence. This decides the whole approach and
should be resolved before any compute is spent.

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
