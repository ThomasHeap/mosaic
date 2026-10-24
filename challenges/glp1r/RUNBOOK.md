# GLP1R: commands to finish the submission

Deadline 2026-10-25 23:59 AoE, which is 2026-10-26 12:00 UTC.

Every GPU step goes through `sbatch --gpus=a100:1 -p cnu -A cosc020762`; the
login node has no GPU and falls back to CPU silently. Run everything from
`challenges/glp1r`. The venv is `../../.venv/bin/python -I`, and `-I` strips the
script directory from `sys.path`, which is why the scripts insert it themselves.

## State at the time of writing

138 designs generated, all scored on ProtenixBase. 48 clear the candidate gate
(ipTM >= 0.50 and activation-core contacts >= 13, the latter above the range
their truncated antagonists reach). Novelty screened on all 48: 21 level 4,
19 level 3, 8 level 2. The 8 at level 2 are excluded, since the selection
prompt enforces level 3 or higher at submission. 40 remain eligible.

## Done

```bash
# design, repeated across seeds; 4 designs per job, about 2h each at L55
sbatch ... design.py --n-designs 4 --binder-length 55 \
  --activation-weight 2.0 --belt-weight 2.0 --nterm-weight 2.0 --nterm-len 5 \
  --seed <n> --out $PWD/designs_l55c

# score on the design model: ipTM, activation-core contacts, belt contacts, depth
sbatch ... score.py <design files> --model protenix_base --n-seeds 5 \
  --out probe/scored_new_<chunk>.jsonl

# cross-check on an independent model; Boltz-2's ipTM is unusable here
# (its scrambled control scores 0.84 +/- 0.18) but its core counts agree
sbatch ... score.py probe/candidates2.jsonl --model boltz2 --n-seeds 3 \
  --out probe/candidates2_b2.jsonl

# monomer structures, needed by the novelty screen
sbatch ... monomers.py probe/cand2_<chunk>.jsonl --out-dir structures_monomer

# novelty: Foldseek against the PDB, Adaptyv's thresholds
module load apps/foldseek/05.2024
../../.venv/bin/python -I ../tnf_alpha/novelty.py \
  --submission probe/candidates2.csv --struct-dir structures_monomer \
  --out probe/novelty2.jsonl --tmp /user/work/cp20141/foldseek_db/tmp_glp1r2
```

## Running

```bash
# final metrics on the 40 eligible designs: adds ipSAE both directions and
# mean Kyte-Doolittle hydropathy, which the selection prompt asks for
sbatch ... score.py probe/eligible.jsonl --model protenix_base --n-seeds 5 \
  --out probe/final_scored.jsonl          # job 19691048, about 40 min
```

## Left to do

```bash
# 1. build the ranked submission (no GPU)
#    gate on binding, then rank by activation evidence, because the stated
#    goals are activation first and binding second. Enforces 20 rows, unique
#    sequences, 10-250 aa, and drops anything at novelty level 2.
../../.venv/bin/python -I make_submission.py          # writes submission.csv
                                                      # and submission_metrics.csv

# 2. verify every figure against its source, as for TNF (no GPU)
../../.venv/bin/python -I verify_submission.py        # non-zero exit on any drift

# 3. METHODS.md, then re-run the verifier over it
```

`make_submission.py` and `verify_submission.py` are not written yet.

## If time runs short

The submission is buildable from what is already on disk: `probe/final_scored.jsonl`
once the running job lands, plus `probe/novelty2.jsonl` and
`probe/candidates2_b2.jsonl`. Steps 1 to 3 need no GPU and take under an hour.
Skip extra design runs rather than the verifier: on TNF, three separate figures
drifted into the methods document unnoticed, and a reviewer reads those numbers.

## Decisions that are already settled, with the evidence

- **L55 with the N-terminal term** is the design configuration. The term raises
  activation-core contacts (Protenix +3.1, Boltz-2 +4.7 at this length) and does
  so by the intended mechanism: without it the first five residues point away
  from the pocket (depth delta -11.8 A), with it they lead (+5.1 A), against
  +6.7 A for GLP-1 itself.
- **ProtenixBase scores binding, Boltz-2 checks activation-core contacts.**
  Boltz-2's ipTM cannot separate a scrambled peptide from GLP-1 here; its core
  counts agree with Protenix on both agonist/antagonist calibration pairs.
- **Rank on activation, gate on binding.** The selection prompt orders the goals
  activation first, binding second, and asks first for a real chance to express
  and bind.
- **State the epitope.** Where a submission does not, the organisers predict the
  complex and read the epitope off the model.

## Known limits to state in METHODS.md

- There is no validated prediction of activation. Activation-core contact count
  is a screen calibrated on agonist/antagonist pairs, and the calibration is
  partly circular: the core is defined by the residues GLP-1's activating
  segment touches, and each antagonist is that agonist with those residues
  removed.
- Every number comes from the two models that also did the designing.
- Attempts that failed and why: the active-minus-inactive binding differential
  (does not separate the calibration pairs), insertion depth as a ranking metric
  (separates one pair, not the other), the Gs ternary-complex test (the apo
  control scores the same as agonists, because the template fixes the answer),
  and MPNN-proposed N-terminal variants (0 of 80 beat their parent).
