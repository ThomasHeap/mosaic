# Challenge 04 — TfR1 (prep notes)

Target: human TfR1 ectodomain, UniProt P02786-1 residues 121-760 (640 aa),
tag-free, supplied as a homodimer so each target presents two copies of the
epitope. Mouse construct 123-763. Designs are captured through a C-terminal
Twin-Strep tag which must not be part of the binding. Up to 20 designs,
de novo only, 10-250 aa. Deadline 2026-11-01 23:59 AoE.

Criteria, in this order:
1. pH-selective binding -- measurable KD at pH 7.4, no detectable binding at
   pH 6.0; designs binding at both rank after, by the 6.0/7.4 KD ratio.
2. Mouse cross-reactivity -- among designs with human KD under 1 uM, ranked by
   how close the mouse/human KD ratio is to 1.
3. Affinity -- lowest human KD at pH 7.4.

## This is TNF Track 3 again

The three criteria are the same, in the same order. The TNF pipeline therefore
transfers almost intact: the proton-linkage relation, the differentiable pKa
model and its PROPKA calibration, ph_score.py, hisscan.py, rank.py's
three-criteria portfolio, novelty.py and the Foldseek PDB database.

The single most useful thing carried over is that the pH switch comes from
post-hoc interface histidine scanning, not from the design loss: on TNF, raw
design output cleared all three criteria at 0.4% against 3.7% for scanned
variants, and 19 of 20 submitted designs were variants. Histidine placement is a
substitution on a fixed backbone, so it installs after the fact for no GPU cost.
Expect the same here and budget for the scan rather than for in-loop pH terms.

pH-dependent binding is also native to this receptor rather than engineered onto
it: transferrin binds TfR1 at 7.4 and releases iron in the acidic endosome, and
TfR1 is the standard blood-brain-barrier shuttle target. Known pH-dependent TfR1
binders exist and may be used to calibrate filters, which the rules permit, but
not as design starting points.

## Structures

  11qc  2.4 A  ectodomain homodimer, chains A/B, residues 122-760
  11rn  2.4 A  homodimer plus a 62-residue binder on the apical domain (X/Y)
  1cx8  3.2 A  eight copies of the ectodomain
  1suv  7.5 A  ectodomain with transferrin (N lobe 3-331, C lobe 332-676)

11rn gives the recommended epitope directly, the way 3ALQ and the anti-TNF Fabs
gave the TNF groove. 1suv shows the transferrin site, which is a different
surface -- worth knowing, since a binder there would compete with the natural
ligand.

## Two findings that should drive the approach

**The apical site sits on one protomer, so the dimer is avoidable.** Chain X of
11rn contacts 9 residues of protomer A (202, 204, 208-212, 348, 371) and none of
protomer B, whose closest approach to the epitope is 39 A. TNF forced a
templated dimer because its groove spanned two protomers; this does not. Better
still, the apical domain is cleanly separable: residues 189-380 make 85
cross-domain CA contacts against 924 internal, a ratio of 0.09, and every
epitope residue lies inside it. So design can run against a 192-residue
templated domain rather than a 639-residue protomer or a 1278-residue dimer.
Truncation does create an artificial boundary surface, so boundary residues
should be excluded from the epitope and probably repelled, as the lipid belt is
for GLP1R.

**The recommended epitope is the worst available choice for criterion 2.** Only
2 of its 9 residues are conserved in mouse: 204 D->Q, 208 R->N, 211 Y->D,
212 L->P, 348 N->K, 371 K->Q all differ and 210 has no mouse equivalent. Overall
human/mouse identity is 77.7%, so the apical site is far more divergent than the
receptor as a whole. A binder there will almost certainly be human-specific,
which forfeits criterion 2. For comparison, TNF's epitope was 14 of 20 conserved
and that was already the limiting constraint.

Conserved surface patches do exist elsewhere. Scanning protomer A for surface
residues (12 or fewer CA neighbours within 10 A) and scoring each 12 A patch by
mouse conservation gives, best first:

  centre 322  8 residues  88% conserved  (317-325)
  centre 664  8 residues  75%            (661-669)
  centre 757  8 residues  75%            (752-760)
  centre 633  9 residues  67%            (627-640)

Residue 322 sits in the apical domain, so a patch centred there keeps the
truncation advantage while being 88% conserved rather than 22%. The 664 and 757
patches are in the helical and protease-like domains. The 757 patch abuts the
C-terminus, which is where the Twin-Strep capture tag attaches, so binding there
risks interfering with capture and should be avoided.

## Epitope: allowed, not dictated

Decided. BinderTargetContact with epitope_idx set to a hand-picked patch forces
the binder there; with epitope_idx=None it scores against the whole target and
the optimiser takes the easiest surface, which on GLP1R was the lipid belt. So
the term gets a wide allowed region and where each design landed is classified
afterwards by mouse conservation, which is what criterion 2 ranks on.

The allowed region is the apical domain minus two exclusions: a 6-residue margin
at each truncation boundary, since cutting 189-380 out of the protomer exposes a
surface absent from the assay construct and a design could bind the scar, and
residues 745-760, where the Twin-Strep capture tag attaches. That leaves 180 of
192 residues allowed, 74% conserved overall -- against 22% for the recommended
apical footprint and 89% for the 317-325 patch inside the same domain.

Leaving it allowed rather than forced also measures something a hand-picked
patch cannot: how often free designs land on conserved versus divergent surface.
If most land on the divergent apical footprint, the choice has to be forced
after all, and that is worth knowing from one run.

## Open questions, in the order worth settling
2. **Does the 317-325 patch look bindable?** Conservation says nothing about
   whether a surface is concave enough. Needs a look at the local geometry,
   and ideally a check that something binds nearby in a known structure.
3. **Whether truncating to 189-380 changes predicted binding** against the full
   protomer, on a handful of designs. Cheap to test and it decides the compute
   budget for everything after.
4. **Histidine availability at each candidate epitope.** TNF's His73 was lost to
   a mouse indel, which forced the switch onto the binder. Worth checking
   whether either TfR1 epitope has its own histidines and whether they are
   conserved.

## Reuse from tnf_alpha

Direct: ph_physics.py, ph_score.py, ph_switch.py, hisscan.py, calibrate.py and
pka_params.json (the PROPKA-calibrated pKa model, 316 complexes), rank.py,
novelty.py, gravity.py, liabilities.py, selftest.py, verify_methods.py.
Rework: target.py for the new epitope and templates, compute_epitope.py for the
11rn contacts, rescore.py for a single-chain templated target.
