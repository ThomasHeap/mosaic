# TfR1: commands to finish the submission

Deadline 2026-11-01 23:59 AoE, which is 2026-11-02 12:00 UTC.

Criteria, hardest first: pH-selective binding (measurable KD at 7.4, none at
6.0), then mouse cross-reactivity, then human affinity. The same three in the
same order as TNF Track 3, so most of that pipeline transfers. Up to 20 designs,
de novo only, 10-250 aa, captured through a C-terminal Twin-Strep tag that must
not be part of the binding.

All GPU work goes through `sbatch --gpus=a100:1 -p cnu -A cosc020762`; the login
node has no GPU and falls back to CPU silently. Run from `challenges/tfr1`.
Python is `../../.venv/bin/python -I`, and `-I` strips the script directory from
`sys.path`, which is why each script inserts it.

## Done

- `structures/` 11qc (ectodomain homodimer), 11rn (with a 62-aa apical binder),
  1cx8, 1suv (with transferrin)
- `sequences/` human P02786, mouse Q62351
- `conservation.py` human/mouse alignment, cached to `conservation.json`
- `target.py` apical template, full protomer, allowed and excluded regions;
  recomputes the 11rn footprint rather than trusting a constant
- `gravity.py` conservation-weighted signed field
- `design.py` smoke tested at 132 s per design on a 5% schedule, so about
  45 min at full schedule
- copied unchanged from tnf_alpha: `ph_physics.py`, `ph_score.py`,
  `ph_switch.py`, `rank.py`, `novelty.py`, `liabilities.py`, `pka_params.json`

## Running

45 designs: lengths 55, 65 and 75, three seeds each, five designs per job.

    sbatch ... design.py --n-designs 5 --binder-length <L> \
      --gravity-weight 1.0 --seed <n> --out $PWD/designs

## Left to do, in order

**1. Rescore.** Refold and score human binding, mouse binding and pH
selectivity. `rescore.py` needs porting: it imports `HUMAN_TNF`, `MOUSE_TNF` and
`tnf_template_chains`, so swap in the TfR1 equivalents. Mouse is a separate
co-fold against Q62351 residues 123-763, as TNF did against P06804.

    sbatch ... rescore.py designs/*.jsonl --out designs/rescored.jsonl

**2. Classify where each design landed.** This is the point of the
allowed-region approach: read the predicted contacts and compute the
mouse-conserved fraction of the surface actually bound. Not written yet; the
contact code in `glp1r/score.py` is the closest model.

    ../../.venv/bin/python -I classify_epitope.py designs/rescored.jsonl

**3. Histidine scan.** Where the pH switch comes from. Backbone-only pKa, no
refolding, so the scan itself needs no GPU. `hisscan.py` needs two import
changes (`HUMAN_TNF`, `groove_epitope_idx`).

    ../../.venv/bin/python -I hisscan.py designs/rescored.jsonl \
      --out designs/hisscan.jsonl --emit designs_his/variants.jsonl

**4. Rescore the variants.** Required, because the scan does not model
structural tolerance: on TNF the median ipTM cost of a four-histidine variant
was -0.054.

    sbatch ... rescore.py designs_his/variants.jsonl --out designs/rescored_his.jsonl

**5. Novelty.** A hard criterion: level 3 or higher is enforced at submission,
and high structural similarity maps to level 2. Needs monomer structures first;
`glp1r/monomers.py` does that in about three minutes for sixteen designs.

    sbatch ... monomers.py <candidates>.jsonl --out-dir structures_monomer
    module load apps/foldseek/05.2024
    ../../.venv/bin/python -I novelty.py --submission <candidates>.csv \
      --struct-dir structures_monomer --out probe/novelty.jsonl \
      --tmp /user/work/cp20141/foldseek_db/tmp_tfr1

**6. Rank and build the submission.** `rank.py` already implements the
three-criteria portfolio and takes novelty tiers as a tie-break.

    ../../.venv/bin/python -I rank.py designs/rescored*.jsonl --portfolio \
      --limit 20 --out submission.csv

**7. METHODS.md**, then a `verify_submission.py` in the style of the GLP1R one,
checking every figure against the data and exiting non-zero on drift.

## Decisions already settled, with the evidence

**No in-loop pH term.** On TNF the switch came from post-hoc scanning: raw
design output cleared all three criteria at 0.4% against 3.7% for scanned
variants, and 19 of 20 submitted designs were variants. Designing with a
histidine budget produced a median of zero histidines at the interface against
1.29 expected from random placement. ProteinMPNN was measured and exonerated as
the cause (P(His) 0.0295 at interface positions against 0.0243 elsewhere). So
design optimises affinity and placement; the scan installs the switch.

**The epitope is not hand-picked.** The challenge hints at the apical domain
around Tyr211, but Tyr211 differs in mouse (Y to D) and only 2 of the 9 residues
the 11rn binder touches are conserved, against 77.7% identity for the receptor
overall. Since cross-reactivity outranks affinity, following the hint trades a
higher-ranked criterion for a lower one. `gravity.py` instead weights every
allowed residue by mouse conservation: 134 conserved attract at +2.0, 46
divergent repel at -1.0, 12 boundary or tag-proximal residues repel at -4.0.
Confirmed live in the loss trace as term 9.

**Single chain, truncated domain.** The apical site sits on one protomer with
the other 39 A away, and residues 189-380 make 85 cross-domain CA contacts
against 924 internal, so design runs against a 192-residue templated domain
rather than the 639-residue protomer or the 1278-residue dimer. TNF forced a
templated dimer because its groove spanned two protomers; this does not.

## If time runs short

Steps 3 to 7 need little or no GPU and take a few hours. The scan is free. Skip
extra design runs before skipping the verifier: on TNF three figures drifted
into the methods document unnoticed and two readings by eye missed them, and on
GLP1R the verifier caught two more on its first run.

## Open questions

- Whether truncating to 189-380 changes predicted binding against the full
  protomer. `design.py --full-protomer` runs the comparison. Cheap, and it
  decides whether the truncation is safe.
- Whether the 317-325 patch, at 89% mouse conservation, is geometrically
  bindable. Conservation says nothing about concavity.
- Whether either candidate surface has its own conserved histidines. TNF's
  His73 was lost to a mouse indel, which forced the switch onto the binder; if
  TfR1 has a conserved interface histidine, the target could carry part of it.
