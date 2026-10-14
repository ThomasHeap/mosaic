# Experiment log — TNF-alpha Challenge 02

Append-only. Newest last. Every entry records the exact command, when it ran, why,
and what came out. Negative results stay in: several things here looked right and
were not, and the reason they were caught is written down.

All timestamps are from `git log` / `sacct`, not from memory. All paths are
relative to `challenges/tnf_alpha/`; `$PY` means `../../.venv/bin/python`.

Deadline **2026-10-18**. Adaptyv's experimental data is expected December 2026,
so nothing below is wet-lab validated — every verdict is in-silico and
provisional.

---

## 1 — Epitope definition

**When** 2026-10-12, committed 10:52 as `270b8c2`.

**Why** The challenge recommends the receptor-binding site at the protomer
interface but gives no residue numbers. Guessing the epitope from memory would
set the target for everything downstream, so it had to be computed.

**Ran**
```bash
# downloads: 1TNF, 3WD5, 4G3Y, 3ALQ, 7KP7 from files.rcsb.org
#            P01375 (human), P06804 (mouse) from rest.uniprot.org
$PY -I compute_epitope.py <dir-with-cifs>
```

**Result** 20 core residues (contacted within 4.5 Å of a partner chain in ≥2 of
3WD5 / 4G3Y / 3ALQ):
`20, 21, 23, 67, 71, 73, 75, 77, 97, 110, 111, 113, 115, 137, 140, 141, 143, 144, 145, 146`

Of those, 14 are conserved in mouse. The mouse alignment reproduced the challenge
page's "124 of 156, 79%" exactly — the check that the numbering was right. All TNF
chains in those entries number on the mature 1–157 sequence, verified residue by
residue, so no alignment was needed for the human structures.

**Changed the plan three ways**
- **The groove spans two protomers.** TNFR2 chain T contacts 15 residues on one
  protomer and 13 on its neighbour. The target must be co-folded as at least a
  dimer; a binder designed against a TNF monomer targets a surface that does not
  exist in the assay construct.
- **Six core residues diverge in mouse** (P20H, S71D, H73 indel, I97V, A111L,
  D140K) — exactly why adalimumab is human-specific. Criterion #2 is mouse
  cross-reactivity, so contacts get biased onto the conserved 14.
- **His73 is the epitope's only histidine** and is lost to an indel in mouse, so
  building the pH switch on the target's histidine trades criterion #1 against
  #2. The switch goes on the binder.

**Mistake caught** The mouse sequence written from memory dropped the N-terminal
`LRSS`. Found by diffing against the UniProt download and corrected. The analysis
read the FASTA directly, so its numbers were never affected — only the hardcoded
constant. *Verify every sequence against a primary download, even a familiar
one.*

---

## 2 — pH selectivity: linkage analysis and PROPKA

**When** 2026-10-12, committed 10:52 as `270b8c2` (same commit as above).

**Why** Criterion #1 is ranked first and no structure-prediction model in
`mosaic` is pH-aware. Needed to know whether the criterion was even reachable
before building an objective for it.

**Ran**
```bash
uv pip install --python .venv/bin/python propka      # PROPKA 3.5.1, LGPL
$PY -I ph_score.py                                   # the selectivity ceiling table
# plus a one-off score_pair() on 1TNF chain A vs the ABC trimer
```

**Result** Per-histidine selectivity across a 7.4 → 6.0 window is bounded, and
tightly enough to dictate the design:

| free-state pKa | log10 per His | His needed for 2 log10 |
|---|---|---|
| 6.0 | 0.28 | 8 |
| 6.5 (ordinary surface) | 0.57 | 4 |
| 7.0 | 0.90 | 3 |
| 7.5 | 1.16 | 2 |

So "no detectable binding at pH 6.0" is unreachable with one or two ordinary
histidines, and **selectivity rises monotonically with the free-state pKa** — the
lever a histidine-count objective cannot see.

Validated on TNF itself: His73 scores pKa_free 6.82 (elevated, in the groove),
while His15 and His78 sit at 5.06 and 4.01, already buried when free, and
contribute 0.044 and 0.003 log units. **Low free-state pKa means a dead switch.**

**Decision** Pools bracket the histidine budget instead of betting on one value:
`ph` targets 3, `ph_hot` targets 5.

---

## 3 — PROPKA timing, to cost the options

**When** 2026-10-12, before `54aa18b`.

**Why** Needed real numbers to choose between ranking-only use, distilling PROPKA
into a learned surrogate, and an analytic differentiable form.

**Ran** `propka_pkas()` timed on 1TNF (456 res), 3WD5 (579), 3ALQ (1818).

**Result** 0.77 s / 1.13 s / 10.43 s — superlinear. The real case (75-residue
binder + TNF dimer = 389 residues) is ~0.6 s per complex, so scoring 1,000
designs is ~12 minutes on one core. **Tier 2 ranking is effectively free.**
Labelling 20k structures for a surrogate is ~2.8 CPU-hours, i.e. ~90 s on one
128-core node — so data generation was never the bottleneck for distillation;
engineering and validation were.

---

## 4 — Gravitational epitope loss (ORBIT-style)

**When** 2026-10-12, committed 11:06 as `54aa18b`.

**Why** `BinderTargetContact` scores distogram contact against epitope columns
and treats them all alike, so it cannot express "bind these residues,
specifically not those". ORBIT puts the field in coordinate space with
per-residue importance and notes it extends to repulsion — which is the part
worth having, given six epitope residues actively work against criterion #2.

**Ran** Smoke test on synthetic geometry (now folded into `selftest.py`).

**Result** 28 weighted residues: 2 at +2.0 (E23, Q67 — contacted in all three
complexes and conserved), 9 at +1.0, 13 at +0.3, and **4 repulsive** (P20, S71,
H73, I97 — the mouse-divergent residues inside the TNFR2 groove; A111 and D140
are in the antibody footprints but not the groove, so they do not appear).
Gradients finite.

**Implementation notes** Virtual C-beta from backbone N/CA/C, because
design-time features carry no sidechains. Soft-min over binder positions rather
than a mean, so the term cannot collapse the binder onto one hotspot;
flat-bottomed wells so it stops pulling at contact distance instead of driving a
clash.

---

## 5 — Differentiable pH loss (JustHISpKa-style)

**When** 2026-10-12, committed 11:19 as `072838e`.

**Why** `ph_switch.py` optimizes a proxy (how many histidines land in the
interface). Hogues/Wei/Sulea (JCIM 2025, JustHISpKa, 0.4 pH-unit MUE) gave a
better blueprint than distilling PROPKA: it assumes electrostatics dominates and
drops the H-bond geometry term — necessary anyway without sidechains — and its
energy→pKa map is *linear*, so gradients are clean and the fit has few
parameters. No labels, no training, no distribution shift.

**Ran** Sign tests on synthetic geometry (now `selftest.py`).

**Three bugs caught before any design run.** None would have raised an error;
two would have produced plausible trajectories optimizing nonsense.
1. A fixed low dielectric put an Asp–His pair at **+6.4 pKa units** instead of
   ~1. The reaction field cancels Coulomb for solvated pairs and I had applied it
   only to the self-energy. Fixed by interpolating epsilon with burial and
   decaying with range.
2. **Selectivity was identically zero** as a consequence — at pKa 12 the
   histidine is protonated at both pH values, so nothing switches.
3. **NaN coordinate gradients** from masking self-distances with `jnp.inf`: inf
   poisons the backward pass, and `norm` of a zero vector has no derivative.

**Result after fixing** Asp/Glu shifts pKa_free by +1.26, Lys/Arg by −1.26,
approaching the target drives the shift to −0.74 with selectivity rising
monotonically. The JAX linkage relation agrees with `ph_score`'s numpy version to
9.4e-08, so the in-loop objective and the ranking score are consistent by
construction.

**Lesson** Sign tests that state an expected *magnitude* in advance, not just a
sign, are worth more than any amount of re-reading the code. "Asp raises pKa"
passes for a model wrong by 6x.

---

## 6 — Negative controls on the Tier 2 score

**When** 2026-10-14, committed 15:51 as `0f3d62c`.

**Why** A score nobody has tested against a known negative is not a score.
Adalimumab and infliximab bind the receptor groove and neither is pH-dependent,
so Tier 2 must place them near zero. Both have crystal structures in complex with
TNF, so this costs no prediction and no GPU, and the rules explicitly permit
known binders for filter calibration.

**Ran**
```bash
$PY -I controls.py --data-dir pdb_cache
```

**Result — PASS.** Adalimumab **−0.312** log10, infliximab **+0.385**, against
the ~2 the challenge asks for.

**The more useful finding** Infliximab's CDR-L3 His92 has a genuine **−2.35 pKa
shift** on binding and still contributes only +0.326 log10, because its
free-state pKa is 6.09. That is entry 2's design rule confirmed on real antibody
data rather than synthetic geometry. It also answers the obvious objection to a
one-sided control: the score is not returning ~0 indiscriminately — it detects a
2.35-unit shift and correctly discounts it. Adalimumab going mildly negative is a
reminder that selectivity has a sign.

---

## 7 — Calibrating the physics term, run 1

**When** 2026-10-14. Fetch at ~15:35; Slurm job **19396246** submitted
15:41:30, `compute`, 16 cores, ran 00:26:42, COMPLETED.

**Why** The physics term's parameters were physically scaled guesses. Needed to
know whether to trust it as an objective or only as a gradient direction.

**Ran**
```bash
# download on the login node (compute nodes may have no network)
$PY -I calibrate.py --fetch 600 --data-dir pdb_cache --fetch-only
./submit_calibrate.sh        # -> job 19396246
```

**Key realization that reordered the plan** Calibration needs neither designs nor
a GPU. The term consumes backbone only, so it can be evaluated on crystal
structures with sidechains discarded, while PROPKA labels come from the full
structure — features restricted to what design sees, labels from the best
available source. It also sidesteps the distribution shift that makes distilling
PROPKA awkward.

**Result** 314 complexes, 1266 histidines, 236/78 split. Test MUE(pKa) 2.875 →
0.701, MUE(shift) 0.581 → 0.468, r(sel) 0.386 → 0.421. Train and test nearly
identical, so no overfitting.

**Why this was not trusted** The fitted parameters looked like a model retreating
to a constant: `epsilon_solvent` pinned to its upper bound (119.998 of 120) and
`alpha_desolv` collapsed 3.0 → 0.82, suppressing *both* energy terms. A MUE(pKa)
of 0.70 had looked "PROPKA-comparable", but if the model had flattened toward
"predict 6.5 always" that number would mean nothing. A published benchmark of
fast pKa methods reported none beating such nulls with confidence. So nulls were
added and the run repeated.

---

## 8 — Calibration run 2, with explicit nulls

**When** 2026-10-14. Slurm job **19397373** submitted 16:29:23, `compute`, 16
cores, ran 00:21:32, COMPLETED.

**Why** To find out whether run 1's numbers beat a model that had learned
nothing.

**Ran**
```bash
./submit_calibrate.sh        # -> job 19397373, now reporting null baselines
```

**Result** 316 complexes, 1272 histidines, 237/79 split.

| metric (test) | null | fitted | gain |
|---|---|---|---|
| MUE(pKa) | 0.821 (constant 6.5) | 0.727 | 11% |
| MUE(shift) | 0.570 (zero shift) | 0.457 | 20% |
| MUE(selectivity) | 0.125 | 0.101 | 19% |
| r(selectivity) | — | +0.379 | weak |

It had **not** collapsed — it beats every null, consistently. But the nulls
changed the reading completely: predicting 6.5 for every histidine scores 0.821,
so nearly all the apparent absolute accuracy is the prior on histidine pKa, and
the model adds 11%. The 20% on the shift is the meaningful gain, but 0.457 log
units of error per site against a 0.5–1.2 per-histidine effect means **error is
comparable to signal for one site**; it only averages down over 3–4.

**Verdict** `ph_physics` stays a gradient direction. `ph_score` remains the
ranking score. No change to the submission plan.

**Correction to entry 7** The `epsilon_solvent` pinning did *not* recur: run 2
fitted 91.4 (bounds 40–120). The parameter actually straining is
`screening_length`, 3.38 against a lower bound of 3.0 — the fit wants
electrostatics to act almost only on immediate neighbours, which together with
`alpha_desolv` 0.82 means it has reduced the SIE continuum it was meant to
approximate to something close to a nearest-neighbour model. *An observation from
one fit is not a finding.*

**Second fix for a revisit** `r(sel)` *fell* with fitting (+0.416 → +0.379) while
MUE improved. The least-squares objective is misaligned with the rank correlation
the score is actually used for — fit Spearman.

**Lesson** Report an explicit null alongside every fitted metric. Without one, a
0.73 MUE reads as success.

**Housekeeping** Labels cached to `pdb_cache/labels.npz`, so a refit is seconds
instead of ~25 minutes. Use `python -u` under `sbatch` — run 1's progress was
invisible for 26 minutes because stdout was buffered to a file.

---

## 9 — Self-test suite, and closing the calibration loop

**When** 2026-10-14, after job 19397373.

**Why** The sign tests from entries 4 and 5 were ad-hoc inline Python, so "what
exactly we ran" was unreproducible. Writing them to a file also surfaced a gap.

**Ran**
```bash
$PY -I selftest.py
```

**Result** 16 checks, all passing. Each asserts an expected *magnitude*, not just
a sign.

**Gap it surfaced** The suite was exercising the *default* parameters — we had
calibrated and never wired the fitted values in. Added
`ph_physics.load_params()`, which `design.py`'s `physics` pool now uses. Fitted
values in force: `epsilon_solvent` 91.4, `epsilon_interior` 17.6,
`screening_length` 3.38, `alpha_desolv` 0.82, `offset` ≈ 0 — substantially
different from the defaults (80 / 10 / 8.0 / 3.0), so which set is active
genuinely changes the gradient. *Calibrating and then not loading the result is
an easy trap.*

---

## 10 — Design runs launched; one GPU node is bad

**When** 2026-10-14 17:17. Jobs **19399012** (affinity L65), **19399013** (ph
L75), **19399014** (ph_hot L75), **19399015** (physics L75), all `cnu`,
`--gpus=a100:1`, 8 cores, 64 GB, 6 h.

**Why** Critical path. Nothing can be submitted to the challenge without
sequences, and the deadline is 2026-10-18.

**Ran**
```bash
./submit.sh affinity 8 65 ; ./submit.sh ph 8 75
./submit.sh ph_hot   8 75 ; ./submit.sh physics 8 75
```

**Result** Three started normally and reported `epitope=24 positions`, which is
the conserved two-protomer groove as expected. **19399012 FAILED** after ~4
minutes, exit 1:0.

**Cause — the node, not the code.** `bp1-gpu031` advertises an `a100` to Slurm
but CUDA cannot initialize on it:

```
Failed to create stream executor for device CUDA:0: CUDA_ERROR_UNKNOWN
RuntimeError: Unable to initialize backend 'cuda': INTERNAL: no supported
devices found for platform CUDA
```

The three jobs that landed on `bp1-gpu038` were fine, so this is specific to
`bp1-gpu031`. Note the failure mode: JAX raises during backend init, *before*
`design.py`'s own `platform == "gpu"` guard can report anything useful, so the
guard does not help here.

**Fix** `submit.sh` now passes `--exclude="$EXCLUDE"`, defaulting to
`bp1-gpu031`; override with the `EXCLUDE` env var. Affinity pool resubmitted as
**19399131**, which landed on `bp1-gpu038` and started cleanly.

**Lesson** A requested-and-granted GPU allocation does not mean a working GPU.
Watch for a failure signature, not just for the success marker — a monitor
grepping only for progress lines would have sat silent through this.

---

## 11 — All four design jobs failed: a cold-cache race and a real bug

**When** 2026-10-14 17:17–17:30. Jobs 19399013/14/15 and 19399131 all FAILED.
Zero designs written. Relaunched as **19399317** (ph), **19399318** (ph_hot),
**19399319** (physics), **19399320** (affinity).

**Why it looked like GPU contention and was not** All four were on one node, each
requesting `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`, so memory contention was the
obvious suspect. The tracebacks said otherwise — two separate causes.

### Cause A — cold-cache download race (3 jobs)

Boltz-2 lazily downloads a 2.2 GB checkpoint and a 1.8 GB CCD tarball into one
shared cache (`~/.cache/mosaic/boltz`) on first use. Four jobs started against an
empty cache and raced on the same paths. Cache mtimes tell the story: `mols.tar`
17:22, `boltz2_conf.ckpt` 17:23, `boltz2_aff.ckpt` 17:24, `mols/` still
extracting at 17:25.

- 19399131 read `boltz2_conf.ckpt` while it was being written:
  `PytorchStreamReader failed reading zip archive: failed finding central
  directory`.
- 19399014 and 19399015 read `mols/` mid-extraction: `ValueError: CCD component
  ASP not found!` — ASP simply had not been extracted yet.

Nothing was permanently corrupt. The cache is now complete (`ASP.pkl` present,
45,227 files, count stable), populated by the one job that won the race.

**Fix** `submit.sh` now refuses to submit against a cold or incomplete cache and
tells you to run one `WARMUP=1` job first.

### Cause B — a real bug in design.py (the 4th job, and all of them in truth)

19399013 survived the race, loaded Boltz-2, and then died **7m28s in** on the
first optimizer step:

```
mosaic/losses/transformations.py line 17, in __call__
    assert seq.shape[-1] == 19
AssertionError
```

`NoCys` takes a **19**-wide simplex and reinserts a zero column at the cysteine
index, so the optimizer must work in 19 dimensions and the result must be mapped
back through `NoCys.sequence` before decoding. `design.py` initialized at 20 and
decoded as 20. Its docstring says so explicitly: *"be sure to call
`loss.sequence(...)` after optimization to get the final sequence!"* — I had read
the class, used it, and missed the contract.

Had the cache race not happened, all four jobs would have failed this way
anyway. The race merely hid it behind a different error.

**Fix** Initialize at `(binder_length, 19)`, decode via `NoCys.sequence`, and
assert the decoded sequence contains no cysteine. Plus five new `selftest.py`
checks covering the simplex width, the restored alphabet, the zeroed cysteine
column, mass preservation, and a cysteine-free decode — 21 checks total, seconds
to run.

**Lessons**
- A loss wrapper can change the *shape contract of the optimizer*. Check what a
  wrapper expects, not just what it does.
- This failed after the model had loaded, so each attempt cost ~7 GPU-minutes to
  learn one assertion. Anything checkable without a GPU belongs in `selftest.py`,
  and `selftest.py` should be run before every submit.
- Lazy shared caches and parallel job launches do not mix. Warm the cache with
  one job.

---

## 12 — Relaunched jobs hang in XLA compilation (not compiling, idle)

**When** 2026-10-14 17:30–18:06. Jobs 19399317/18/19/20, cancelled at 35:20 with
zero designs. Probe **19400552** submitted.

**What went right** All four got past the 7m28s `NoCys` assertion from entry 11,
so that fix works. They reached the MSA server, built complex and monomer
features, and entered XLA compilation.

**What went wrong** They then sat for 35 minutes producing nothing. Last log
output is XLA's Triton GEMM autotuner emitting invalid kernels, repeatedly:

```
loc("dot.5964"): error: 'tt.dot' op expected the output shape to be the
concatenation of the last dimension of the first operand and ...
```

**How "slow compile" was ruled out** A 30-minute Boltz-2 compile is plausible, so
this needed distinguishing from a hang. Three measurements:

- `sacct` reported `TotalCPU 00:00:00` at 33 minutes elapsed (though accounting
  may simply not have been sampling, since `MaxRSS` was blank too).
- The log files did not grow by a single byte over 45 s.
- Decisive: `scontrol show node bp1-gpu038` gave **`CPULoad=2.64`** with
  `CPUAlloc=32`. Four compiling jobs would drive load to 8–32; 2.64 is four
  mostly-sleeping processes. `FreeMem=437563` of 515046 MB confirmed almost
  nothing was resident. `AllocTRES` showed `gres/gpu=4` on a 4-GPU node, so each
  job did have its own device — GPU contention was not the cause either.

Node load is the cheap, reliable way to tell a silent compile from a hang when
you cannot get a shell on the node (`srun --overlap` failed here with "task 0
launch failed").

**Hypothesis under test** XLA's Triton GEMM autotuner compiles and benchmarks
candidate kernels on-device; the invalid `tt.dot` kernels it is generating are a
plausible way to hang that process while consuming no CPU. Probe 19400552 runs
one small design (binder 60, 20 steps) alone on a node with
`XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`, which forces cuBLAS instead.

**Next hypothesis if that fails** `use_msa=True` makes every job call
`api.colabfold.com`. Blocking on a network read also shows as idle-with-no-CPU.
The logs show the MSA completing before the Triton errors, so it is the weaker
explanation, but a `use_msa=False` run is the next probe — the target is
templated on 1TNF anyway, so the MSA is contributing less than it would
otherwise.

**Standing risk to the plan** Boltz-2 hallucination on a 389-residue complex
(binder + TNF dimer) may be too expensive for a 4-day deadline even once it
runs. `mosaic` ships a much cheaper ESMFold2 design path
(`examples/esmfold_minibinder.py`, `biohub_design`) that is the documented fast
route. If the probe does not produce a design quickly, switch to that rather
than keep paying for Boltz-2 compiles.

---

## Open / next

- **No designs exist yet.** Probe 19400552 is testing whether Boltz-2 can run at
  all here (entry 12). Fall back to the ESMFold2 path if it cannot.
- Run `$PY -I selftest.py` before every design submit.
- Repredict designs, populate Tier 2 scores on the JSONL records, then
  `$PY -I rank.py designs/*.jsonl --out submission.csv --limit 40`.
- Decide the submission split across the four pools.

**Not attempted, and why**
- A learned pKa surrogate — ~2–3 days of engineering and it inherits a
  distribution-shift problem the analytic form avoids (PROPKA labels come from
  full-sidechain structures; design predictions have none).
- JustHISpKa itself for rescoring the final 40 — registration plus AmberTools,
  but 0.4 vs PROPKA's ~0.7 MUE would be worth it for 40 designs, if not for
  screening hundreds.
- Fitting Spearman rather than least squares (entry 8), which is the single
  cheapest improvement to the physics term.
