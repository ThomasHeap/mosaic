# Modelling choices: the design objective

What every loss term in `design.py` does, why it is there, and what breaks without
it. Written for someone fluent in optimization and deep learning but not in
structural biology, so the protein-specific background is spelled out rather than
assumed.

Companion files: `design.py` (the objective and schedule), `gravity.py`,
`ph_switch.py`, `ph_physics.py` (the challenge-specific terms), `target.py` (what
we are binding to), `LOG.md` (what actually happened on the GPU).

---

## 1. The optimization problem, in ML terms

This is **input optimization against a frozen predictor** — the same shape of
problem as activation maximization, adversarial example generation, or CLIP-guided
image synthesis. Nothing is trained. The weights of every network involved are
fixed; the *input* is the decision variable.

| | |
|---|---|
| **Decision variable** | `x`, shape `[L, 19]`, each row on the probability simplex. `L = 75` by default. |
| **Frozen differentiable "renderer"** | Boltz-2, a structure prediction model. Maps (sequence, target) → 3D coordinates + per-residue and per-pair confidence tensors. |
| **Loss** | A weighted sum of ~15 scalar functions of the renderer's output (and of `x` directly). |
| **Optimizer** | Accelerated projected gradient on the simplex, 165 steps in 3 annealed stages (`mosaic.optimizers.simplex_APGM`). |
| **Output** | `argmax` over each row of the final `x`, read out as a 75-character string. |

### The decision variable is a relaxation

A protein is a chain of **residues** (amino acids), each drawn from an alphabet of
20. So a protein of length 75 is a sequence in `{1..20}^75` — a discrete object,
and the search space is `20^75 ≈ 10^97`.

To get gradients we relax it: each position holds a *distribution* over amino
acids rather than a choice. `x[i]` is a point on the 20-simplex (19 here, see
below), and the structure predictor's embedding layer is linear in the one-hot
input, so feeding it a soft distribution is well-defined and differentiable. This
object is called a **PSSM** (position-specific scoring matrix) in the protein
literature; think of it as the logit-free, pre-argmax state of a sequence.

The relaxation is the source of most of the subtle failure modes in section 6:
intermediate `x` is a *mixture* that no physical protein realizes, and terms that
are well-behaved on one-hot inputs can be gamed by spreading mass around. The
annealing schedule (section 7) exists to force `x` back to one-hot by the end.

**Why 19 and not 20.** The objective is wrapped in `NoCys` — a
reparameterization, not a penalty — so the optimizer works on a 19-wide simplex
and cysteine is *unreachable by construction*. Cysteine is the one amino acid that
forms covalent bonds to other cysteines; a lone unpaired cysteine in a secreted
protein will cross-link to whatever it finds, including another copy of itself,
which ruins expression and assay behaviour. Making it structurally impossible is
cheaper than penalizing it and then discovering the penalty was too weak.
`design.py:270` asserts the decode honours this.

### Why a frozen predictor is a reasonable proxy at all

The thing we actually care about — does this molecule bind tightly to that one? —
is a wet-lab measurement. The bet of this entire approach is that a structure
predictor's *confidence* in a co-folded complex correlates with real binding
affinity. That bet is empirically decent and is why the confidence terms in
section 5 carry the weight they do, but it is a proxy, and every term here is a
proxy for a proxy. Worth keeping in view when reading the numbers.

---

## 2. Minimal protein vocabulary for reading the code

**Residue / position.** One link in the chain, one row of `x`. Indexed 0..L-1 for
the binder.

**Target vs binder.** The **target** is the molecule we want to stick to — here,
human TNF-alpha, a fixed 157-residue sequence (`target.HUMAN_TNF`). The **binder**
is what we are designing. They are co-folded as a single system: `mosaic` lays out
the tokens as `[binder, target_chain_1, target_chain_2]`, so in a tensor of shape
`[N, N, ...]` the block `[:L, L:]` is binder→target and `[L:, :L]` is
target→binder. Nearly every loss term starts by slicing out the block it cares
about.

**Epitope / paratope.** The **epitope** is the patch on the *target* we want
contacted. The **paratope** is the corresponding patch on the *binder*. We specify
the epitope (section 4) and let the optimizer discover the paratope.

**TNF-alpha is a trimer.** Three identical copies assemble into one functional
unit, and the receptor-binding site — the groove we are targeting — sits in the
*seam between two adjacent copies*. This has a hard consequence recorded at
`target.py:47-50`: a binder designed against a single TNF chain is being designed
against a surface that does not exist in the real assay. So we always co-fold at
least a dimer (`tnf_template_chains(2)`), which is why the target region is 314
residues and why epitope indices carry a `k * 157` protomer offset.

**Residue numbering is a minefield.** `target.py` fixes the convention: all
residue numbers are 1-indexed on the 157-aa mature construct, `groove_epitope_idx`
converts to 0-indexed offsets into the concatenated target region, and the two
must not be mixed. Several terms take `epitope_idx` and index a target axis with
it directly; an off-by-one here is silent and produces a binder aimed at the wrong
face of the protein.

### What the predictor outputs

These are the tensors the losses consume (`StructureModelOutput`,
`losses/structure_prediction.py:41`):

- **`distogram_logits` `[N, N, 64]`** — for every pair of residues, a categorical
  distribution over 64 distance bins. This is the model's belief about pairwise
  geometry, and it is *richer than coordinates*: it carries uncertainty. Most
  contact terms work here rather than on coordinates, because a distribution over
  distances gives a smoother gradient than a single point estimate.
- **`pae` `[N, N]` and `pae_logits` `[N, N, 64]`** — Predicted Aligned Error, in
  ångströms (Å; 1 Å = 0.1 nm, roughly one bond length). `pae[i,j]` is the expected
  positional error at residue `j` if you superimpose the prediction on truth at
  residue `i`. **Lower is better.** It is *asymmetric* and that asymmetry is
  meaningful, which is why there are separate binder→target and target→binder
  terms.
- **`plddt` `[N]`** — per-residue local confidence, 0..1 in this codebase.
  **Higher is better.**
- **`backbone_coordinates` `[N, 4, 3]`** — actual 3D coordinates of 4 backbone
  atoms per residue. Used by the two terms that need real geometry rather than
  distance distributions.
- **`full_sequence` `[N, 20]`** — per-token identity distribution, including the
  target's known identity. Used by the electrostatics term to get target charges.

**iPTM** is a derived scalar: the predicted TM-score restricted to *cross-chain*
pairs. TM-score is a 0..1 structural agreement measure; iPTM is the field's
standard single-number answer to "does the model believe this complex is real?"
and it is the quantity that correlates best with experimental binding success.
**Higher is better.**

### Sign convention

Everything is **minimized**. Terms wrapping a quantity where higher is better
return its negation (`PLDDTLoss` returns `-plddt`, `IPTMLoss` returns `-iptm`).
Terms wrapping an error return it directly (the PAE terms). Terms targeting a
specific value return a squared deviation. When reading weights, remember that
the *scale* of each term differs by orders of magnitude — PAE is in ångströms
(~5-30), iPTM is in [0,1], the contact terms are log-probabilities (~-5 to 0) —
so the weights are partly unit conversion and only partly priority.

---

## 3. What we are optimizing *for*

Three challenge criteria, in the organizers' priority order:

1. **pH selectivity** — bind human TNF-alpha at pH 7.4, with no detectable
   binding at pH 6.0.
2. **Mouse cross-reactivity** — the same binder should also bind mouse TNF-alpha
   (79% identical to human, `target.py:21`).
3. **Human affinity** — bind tightly.

Criterion 3 is what the confidence terms serve, and it is the one the ML machinery
is actually designed for. Criteria 1 and 2 are the interesting ones, because
**neither is expressible in a structure predictor's output at all**:

- Boltz-2 has no notion of pH. There is no input to vary and no output that
  responds to it.
- Boltz-2 has no notion of "and also bind this other, similar protein."

So both get handled by **encoding the known physical mechanism as a constraint on
where and what**, rather than by optimizing the endpoint. That is the single most
important idea in this objective, and sections 4 and 6 are the two instances of
it.

---

## 4. Criterion 2 (cross-reactivity), solved as epitope geometry

Human and mouse TNF are 79% identical. A binder cross-reacts if its contact
footprint lands on positions that are *the same in both*. The existing clinical
antibodies are a worked example of the failure: adalimumab is human-only
precisely because it leans on positions that differ in mouse.

`compute_epitope.py` computes the epitope from three reference crystal structures
(adalimumab, infliximab, and the natural receptor TNFR2 bound to TNF), taking TNF
residues within 4.5 Å of the partner and intersecting across complexes. Residues
appearing in only one complex are dropped as antibody-specific or crystal
artefacts. The result splits three ways (`target.py:36-45`):

| Set | Residues | Status |
|---|---|---|
| `CORE_EPITOPE` | 20 of them | contacted in ≥2 of 3 complexes |
| `MOUSE_DIVERGENT` | 20, 71, 73, 97, 111, 140 | core, but **differ** in mouse |
| `CONSERVED_EPITOPE` | the other 14 | core and conserved |
| `TRIPLE_CONSERVED` | 23, 67 | all 3 complexes **and** conserved |

This split then enters the objective in two places, at different resolutions:

- **Coarse, as a mask.** `groove_epitope_idx(conserved_only=True)` simply deletes
  the divergent positions from the index list passed to the contact terms. They
  become invisible: no reward for touching them.
- **Fine, as a signed field.** Invisible is not the same as discouraged — a binder
  can still sprawl across a divergent residue en route to a conserved one. The
  gravitational term (section 5.4) makes them **actively repulsive**.

One more trap, `target.py:54-57`: residue 73 is the *only* histidine anywhere in
the epitope, and histidine is exactly the residue the pH switch is built from. It
would be the obvious thing to exploit — and it is lost to an insertion/deletion in
mouse. So criterion 1 must be built from histidines on the **binder**, never from
this one. Criteria 1 and 2 are in direct tension at this position, and the
resolution is recorded in the target definition rather than left to be
rediscovered.

---

## 5. The terms

### 5.1 Binding geometry — weight 1.0 each

**`BinderTargetContact(epitope_idx=..., contact_distance=20.0)`**
`losses/structure_prediction.py:303`

```
log_contact = contact_cross_entropy(distogram[:L, L:], 20.0, bins)  # [L, n_target]
log_contact = log_contact[:, epitope_idx]                           # [L, n_epitope]
per_residue = sort(log_contact, desc)[:, :3].mean(-1)               # top-3 per binder residue
return -per_residue.mean()
```

`contact_cross_entropy` re-normalizes the distogram over just the bins below the
cutoff and dots that with the log-probabilities — so it rewards both *mass* in
the contact bins and *concentration* within them. A diffuse "could be anywhere
under 20 Å" scores worse than a confident "15 Å".

The design choice that matters is **top-3 per binder residue, then mean**. Read it
as: *every* binder position should be confidently near *a few* epitope positions.
Two alternatives and why they are worse:

- **Mean over all epitope columns** would ask every binder residue to be near
  every epitope residue — geometrically impossible, and the gradient pushes toward
  an averaged, shapeless blob.
- **Max over columns** gives gradient to exactly one pair at a time, which is slow
  and brittle early on.

The 20 Å cutoff is generous (contact in the physical sense is ~5 Å). This is
deliberate: at step 0 the binder is nowhere near the target, and a tight cutoff
gives a vanishing gradient. This term's job is *gross positioning*; the real
geometry is enforced by the gravitational term at 8 Å and by the confidence terms.

**`WithinBinderContact()`** `losses/structure_prediction.py:240`

The same machinery on the binder→binder block `[:L, :L]`, with 14 Å cutoff, top-25
contacts per residue, and a mask requiring `|i - j| > 8`.

This is the **"be a compact folded domain"** term. Without it, the optimizer is
perfectly happy to return an extended string draped over the epitope: that
satisfies the contact term and is not a protein. Rewarding many medium-range
contacts per residue is a proxy for globularity.

The sequence-separation mask is the load-bearing detail. Residues adjacent in the
chain are *necessarily* close in space — that is free, carries no information
about folding, and there are `O(L)` such pairs versus `O(L²)` long-range ones.
Without the mask the term is maximized by a straight line.

### 5.2 Designability — weight 10.0, the largest in the objective

**`InverseFoldingSequenceRecovery(mpnn, temp=0.001)`** `losses/protein_mpnn.py:295`

This one needs the most context. **Inverse folding** is the reverse problem:
given a 3D backbone, what sequence would fold into it? ProteinMPNN is a small,
fast, frozen graph network that does this well. Here it is used as a *critic*:

```
samples = vmap(inverse_fold(mpnn, backbone, temp=0.001))(keys)  # 16 samples, one-hot
consensus = stop_gradient(samples.mean(0))                      # [L, 20]
return -<x, consensus>                                          # inner product
```

**Why this is necessary.** The structure predictor can be confidently wrong in a
specific, exploitable way: input optimization will happily find sequences that
*the predictor believes* fold into a beautiful binder but that physically do not.
These are adversarial examples against the predictor. ProteinMPNN is an
independent model with a different architecture and training objective, so
agreement between them is much harder to fake than either alone. Textbook
ensemble disagreement as a regularizer.

Note `stop_gradient` on the consensus: gradient flows only through `x`, not back
through MPNN's view of the backbone. The term is a moving target that is treated
as fixed within each step — closer to a self-distillation step than a joint
objective.

**Why weight 10.0 and temperature 0.001.** Both are deliberate departures from the
reference recipe (`design.py:18-21`). Early runs sat at sequence recovery
0.04-0.17 — i.e. the designs were ones ProteinMPNN considered implausible *for
their own backbone*, exactly the adversarial regime above. The near-zero
temperature makes MPNN near-deterministic, so the consensus is a sharp target
rather than a blur.

The documented cost: this weight encourages **homopolymers** (runs of one amino
acid), since the inner product is maximized by piling onto MPNN's single favourite
residue per position. 10.0 is the upper end of usable and the comment says so.

### 5.3 Confidence — the terms that track the actual metric

| Term | Weight | Returns |
|---|---|---|
| `BinderTargetPAE` | 0.05 | `pae[:L, L:].mean()` |
| `TargetBinderPAE` | 0.05 | `pae[L:, :L].mean()` |
| `WithinBinderPAE` | 0.4 | `pae[:L, :L].mean()`, diagonal zeroed |
| `IPTMLoss` | 0.025 | `-iptm` |
| `pTMEnergy` | 0.025 | pre-expectation pTM surrogate |
| `PLDDTLoss` | 0.1 | `-plddt[:L].mean()` |

These were **entirely absent** from the complex branch before
(`design.py:13-17`), which meant the pipeline optimized hand-built proxies for
binding while ignoring the quantities that actually correlate with the contest
metric. Adding them was the single largest improvement.

Four notes on the choices:

**Both PAE directions.** PAE is asymmetric: confidence in placing the target given
the binder's reference frame is a different question from the reverse. Including
only one leaves a degree of freedom unconstrained.

**`WithinBinderPAE` is weighted 8× the interface PAE terms** (0.4 vs 0.05). The
priority is a rigid, confidently-predicted binder *first*. A floppy binder cannot
bind anything, and interface confidence conditioned on a floppy binder is not
meaningful. This also partly substitutes for the monomer branch that was removed
(section 8).

**`pTMEnergy` alongside `IPTMLoss`** is a gradient-quality choice, not a
redundancy. iPTM takes an expectation over the PAE distribution and then a `max`,
and it **saturates** — once it is near 1.0 the gradient nearly vanishes while
there is still real structure to improve. `pTMEnergy` applies the TM kernel
`g(d) = 1/(1 + d²/d₀²)` *inside* a `logsumexp` over the raw logits, so it stays
responsive in that regime. Same underlying quantity, different saturation
behaviour; the standard trick of keeping a log-domain surrogate next to a bounded
metric.

**The weights are small** (0.025-0.4) relative to the contact and MPNN terms. Part
of this is unit scale — PAE is in ångströms and runs 5-30, so 0.05 × 20 ≈ 1 is
comparable to a log-probability term. Part is that confidence is a *ranking*
signal: it is excellent at telling good designs from bad ones but a weak
*search* signal early, when everything is bad.

### 5.4 The gravitational term — weight 1.0

`gravity.py:85`, after ORBIT's "gravitational binding loss."

This is the one term operating in **coordinate space** rather than on the
distogram, and the reason is a genuine expressiveness limit: `BinderTargetContact`
takes a *set* of epitope columns and treats every member identically. It has no
way to say "and specifically not this one." Our epitope is not uniform — six of
its residues are ones we need to *avoid* (section 4) — so we need a **signed**
field.

```
cb       = pseudo_cb(backbone_coordinates)              # [N, 3]
d        = ||target_cb[idx][:, None] - binder_cb[None]||  # [K, L]
soft_min = -T * logsumexp(-d / T, axis=-1)              # [K], T = 2.0 Å
pull     = relu(w)  * softplus(soft_min - 8.0)
push     = relu(-w) * softplus(12.0 - soft_min)
loss     = (pull + push).sum() / |w|.sum()
```

Weights from the conservation split (`tnf_epitope_weights`):

| Residues | Weight |
|---|---|
| E23, Q67 — all 3 complexes, conserved | **+2.0** |
| other conserved core | **+1.0** |
| in the groove but not core | **+0.3** |
| P20, S71, H73, I97, A111, D140 — mouse-divergent | **−1.0** |

Four implementation details, each fixing a specific failure:

**Virtual C-beta** (`gravity.py:44`). Distances between residues should be
measured between side chains, not backbones — but design-time features in
`mosaic` carry no side chains at all (the reference-atom channel is UNK/G), so the
real atom is unavailable inside the loop. The standard substitute is a *virtual*
C-beta placed by a fixed linear combination of the three backbone atoms. It is
smooth in the backbone coordinates and defined even for glycine, which has no
real C-beta.

**Soft-min over binder positions, not mean** (`gravity.py:129`). Each target
residue cares about its *nearest* binder residue — that is what contact means. A
hard `min` gives gradient to one position at a time; a `mean` asks every binder
residue to approach every weighted target residue, which collapses the binder onto
one hotspot. The log-sum-exp soft-min at `T = 2.0 Å` interpolates: gradient
reaches several nearby positions early and concentrates as geometry resolves. `T`
is a reach-versus-sharpness knob.

**Flat-bottomed wells via softplus.** `softplus(soft_min - 8.0)` is ≈0 once the
distance is under 8 Å and grows ≈linearly beyond it. So attraction **places the
binder and then stops pulling**. A plain distance penalty would keep pulling past
contact and drive a steric clash — the predictor would then have to resolve an
impossible geometry, and the confidence terms would collapse. Repulsion is the
mirror image, pushing until 12 Å and then flat.

**Normalizing by `Σ|w|`** keeps the term's magnitude independent of how many
residues are weighted, so retuning the epitope sets does not silently retune this
term's effective weight against the rest of the objective.

The aux dict reports `d_attract` and `d_repel` separately, because a single scalar
cannot distinguish "failing to engage" from "failing to avoid" — opposite
problems needing opposite fixes.

### 5.5 Criterion 1 (pH selectivity), the composition route — weight 2.0

`ph_switch.py:204`. Pool-dependent: `his_target` is `None` (off), 3.0, or 5.0.

**The mechanism.** pH measures proton concentration. **Histidine** is the one
amino acid whose charge flips in the physiological range: its side chain has a pKa
around 6.0-6.5, so it is neutral at pH 7.4 and acquires a positive charge as pH
drops to 6.0. Bury a histidine in a protein-protein interface and that charge
appears somewhere with nowhere to go — energetically expensive, and the interface
comes apart. This is how pH-dependent "recycling" antibodies are engineered; it is
established, not speculative.

Since the predictor cannot see pH, these terms shape **paratope composition** —
put a controlled number of histidines exactly where the binder touches the target
— and leave actual pH discrimination to post-hoc scoring. The module docstring is
explicit that these are hypothesis generators, not a pH oracle.

Four sub-terms:

**`InterfaceHistidineCount`** (1.0). Expected interface histidine count, penalized
toward a target:

```
p_contact = 1 - Π_j (1 - P(contact(i,j)))   # P(residue i touches any epitope residue)
expected  = Σ_i p_contact(i) * x[i, HIS]
return (expected - target_count)**2
```

Note `p_contact` is computed as the complement of "touches nothing", using
`contact_log_probability` (a true `logsumexp` over contact bins, unlike the
cross-entropy variant above) — with a clip, because `log1p` of the complement is
numerically touchy once a contact is near-certain.

**Squared, not `-log p_His`**, and the docstring says why: a plain reward would
drive *every* interface position to histidine. That does not make a better switch,
it destroys the interface. We want a target count, so the loss is a squared
deviation. The useful range is 2-4: one histidine rarely gives a clean switch,
more than four costs too much affinity at pH 7.4.

**`BuriedHistidinePacking`** (0.5) decides *where* those histidines go. A
histidine on the surface titrates at its free-solution pKa and changes nothing on
binding; the switch needs the side chain **desolvated** — wedged against the
partner, water excluded. Burial is proxied by the mean log-contact-probability
over the 4 closest epitope residues (top-4 rather than a sum, so one very tight
contact cannot masquerade as genuine burial), averaged with histidine probability
as the weight.

**`NoHistidineOutsideInterface`** (0.3) penalizes histidine mass where contact
probability is low. Stray histidines add pH-dependent *folding* behaviour and
aggregation risk while buying no binding selectivity, so the histidine budget is
pushed into the paratope.

**`TotalHistidineBudget`** (2.0) is a hard cap, weighted **above** the count term
it protects. Its docstring records the exploit it closes, and it is the cleanest
example in this codebase of the relaxation being gamed:

> `InterfaceHistidineCount` penalizes `(Σ_i p_contact(i)·p_His(i) − target)²`, and
> while the PSSM is still diffuse every position has only a small `p_His`, so the
> least-effort way to raise that sum is to push histidine up *everywhere* rather
> than place a few histidines where contact is high.

Probe 19400994 did exactly that: **24 histidines in a 75-mer, 32% of the
sequence, against a target of 3.** A 32%-histidine protein will not express or
fold. `NoHistidineOutsideInterface` discourages the same thing but is scaled by
per-position contact probability and was too weak to hold it, so this term is an
unscaled `relu(total − (target + 1.5))²` on the raw column sum. The 1.5 tolerance
keeps it from fighting the count term over the last fraction of a residue.

The signature comment at `ph_switch.py:191` is worth heeding: this term needs no
structural input, but it is composed into a `LinearCombination` alongside terms
that do, and every member is called with `(sequence, output, key)`. A
sequence-only signature cost five GPU jobs.

### 5.6 Criterion 1, the physics route — weight 3.0, `physics` pool only

`ph_physics.py:246`. This optimizes **the quantity that actually gets scored** —
log₁₀ selectivity between pH 7.4 and 6.0 — through a differentiable pKa model,
rather than the histidine-count proxy above.

```
burial    = Σ_j sigmoid((cutoff - d_ij) / width)        # smooth neighbour count
eps       = eps_solvent - (eps_solvent - eps_int) * burial   # burial-dependent dielectric
energy    = COULOMB_K * Σ_j q_j * exp(-d/λ) / (eps * d)
pka       = HIS_PKA + offset - energy / RT_ln10 - alpha_desolv * burial
```

Evaluated twice from **one** structure: summing over binder neighbours only gives
the *free* state (binder alone), adding the target gives the *bound* state. The
difference is exactly the target's electrostatic contribution, which is the
quantity selectivity depends on. `linkage_selectivity` then converts the pKa shift
to log₁₀ selectivity via the exact proton-linkage relation — and is deliberately
identical to `ph_score.selectivity_log10`, so the in-loop objective and the
offline ranking score agree **by construction rather than by coincidence**.

The loss is a one-sided hinge, `relu(2.0 - expected_selectivity)`: reward up to
~2 log units ("no detectable binding") and nothing beyond, because chasing more
only costs affinity at pH 7.4.

Three things to know before trusting this term:

- **It is not calibrated.** `calibrate()` at `ph_physics.py:317` raises
  `NotImplementedError` and documents the intended fit (PROPKA labels,
  Nelder-Mead over three scalars). `load_params` falls back to defaults if
  `pka_params.json` is absent — and its docstring warns the fitted values differ
  *substantially* from the physically scaled guesses (far more screening, far less
  desolvation), so which set is in force changes the gradient direction. The
  honest status: a gradient direction, not a measurement. Rank on `ph_score.py`.
- **It only beats the nulls by 11-20%** (`LOG.md` entry 8), which is why it runs
  as its own pool to be compared against the proxy pools rather than replacing
  them.
- **The count proxy stays on at 0.5** underneath it. Without any pressure to put
  histidines near the interface at all, a selectivity term that is multiplied by
  `p_His` has essentially nothing to act on early in a trajectory. It is a prior
  that gets the term into a regime where it has gradient.

Two numerical details worth stealing: `_pairwise_dist` adds an epsilon *before*
the `sqrt` because `norm` of a zero vector has a NaN derivative and masking
afterwards does not help — `inf` poisons the backward pass too; and `_coulomb`
floors distances at 2 Å to bound the `1/d` gradient.

---

## 6. How the terms interact

The recurring pattern: **each hand-built term has a cheap exploit, and the fix is
a second term that is unscaled where the first is scaled.**

| Term | Exploit | Counterweight |
|---|---|---|
| `BinderTargetContact` | extended chain draped over the epitope | `WithinBinderContact` |
| contact terms generally | sprawl across mouse-divergent residues | gravity's negative weights |
| structure predictor confidence | adversarial sequences | ProteinMPNN critic |
| `InterfaceHistidineCount` | raise histidine everywhere while diffuse | `TotalHistidineBudget` |
| `InterfaceHistidineCount` | histidines *near* but not *buried* | `BuriedHistidinePacking` |
| gravity attraction | over-tighten into a clash | flat-bottomed softplus well |

Three tensions are real and unresolved by design:

- **MPNN weight 10.0 vs sequence diversity.** It pushes toward homopolymers. 10.0
  is knowingly at the edge.
- **Histidine count vs affinity.** Every interface histidine is a residue not
  chosen for binding. More switch, less affinity — hence `target_count` is a
  *pool* parameter, hedged across submission slots rather than guessed once.
- **Criterion 1 vs criterion 2 at residue 73** (section 4), resolved in the target
  definition.

Because every term returns an aux dict, the per-step logs carry `target_contact`,
`intra_contact`, `grav` / `d_attract` / `d_repel`, `interface_his` / `total_his` /
`his_burial`, `iptm`, `plddt`, `bt_pae` separately. This is the only practical way
to tell *which* term a stalled trajectory is losing to — and given six terms that
exist solely to counterweight other terms, you will need it.

---

## 7. The optimizer, and why it is part of the objective

`simplex_APGM` — accelerated projected gradient on the simplex. The schedule
(`design.py:86-90`) is three stages, `x` carried forward:

| Stage | Steps | Stepsize | Momentum | `scale` | Space |
|---|---|---|---|---|---|
| 1 | 100 | `0.2·√L` | 0.3 | 1.00 | simplex |
| 2 | 50 | `0.5·√L` | 0.0 | 1.25 | log |
| 3 | 15 | `0.5·√L` | 0.0 | 1.40 | log |

Gradients are clipped to norm **1.0** before the step (`MAX_GRAD_NORM`); the
library default is `√L`, which with momentum is what diverged.

The mechanism that matters for understanding the losses is **`scale > 1` in log
space**. In log space the update is `x_new = scale * (v - stepsize * g)`, with
`softmax` applied at evaluation. Multiplying logits by `scale > 1` is a
temperature *decrease* — it sharpens the distribution. Formally it is **negative
entropic regularization**: instead of the usual entropy bonus encouraging
exploration, we penalize entropy to force `x` toward one-hot.

This is load-bearing because of section 1's problem: intermediate `x` is a mixture
no physical protein realizes, and several terms are gameable precisely while it is
diffuse (the histidine exploit is the clearest case). Stage 1 explores on the
simplex; stages 2-3 anneal to a near-discrete sequence, so the final `argmax`
loses little. `pssm_max_mean` is logged per design (`design.py:284`) as the
sharpness diagnostic — near 1.0 means converged; well below means the reported
sequence is an argmax over a still-fuzzy distribution and the loss values do not
describe the sequence you are reading.

Consequently: **the final PSSM is the answer, not the best iterate**
(`design.py:34-35`). Taking an early low-loss iterate would discard the annealing
and hand back a diffuse distribution.

Two more choices that affect gradient quality:

**`num_samples=4`** in `build_multisample_loss`. The predictor is stochastic, so
each evaluation is a noisy estimate. Four samples in one forward pass are far
cheaper than four runs and cut gradient variance substantially.

**Gumbel initialization at random temperature** (`design.py:255-260`):
`softmax(T · Gumbel)` with `T ~ U(0.25, 0.75)`. Starting every design from the
same near-uniform simplex point makes a "pool" of designs correlated; this gives
each one a genuinely different starting basin, and the random temperature varies
*how* committed the start is.

---

## 8. What is deliberately *not* in the objective

**No monomer branch** (`design.py:30-33`). A natural term would co-fold the binder
*alone* and require it to predict the same structure as in the complex — catching
designs that only fold when the target holds them. That check was removed from the
loss and moved to `rescore.py`. Reasons: it roughly doubles the forward cost, and
the check is better *targeted* post hoc (discard designs that change conformation
in isolation) than averaged into a gradient. `WithinBinderPAE` at 0.4 carries some
of the load in-loop.

**No explicit stability, aggregation, or expression term.** These ride on the MPNN
critic (sequences MPNN likes tend to be expressible), `NoCys`, and the histidine
budget.

**pH discrimination itself.** By construction impossible in-loop (section 3).
Ranked afterwards by `ph_score.py` / `rescore.py`.

**Mouse cross-reactivity itself.** Approached entirely through epitope geometry
(section 4) rather than by co-folding against mouse TNF.

The general shape: **in-loop terms are for search, post-hoc scoring is for
selection.** A term earns its place in the objective only if it provides useful
gradient; if it is only a good filter, it belongs in `rescore.py`. Worth
remembering when reading `LOG.md` entry 14 — "measure that filtering was never the
problem" — the filters were fine; the objective was the bottleneck.

---

## 9. Summary table

All terms are minimized. Weights as of `CONFIG = "nipah-recipe+hiscap-v1"`.

| Term | Weight | Reads | Purpose |
|---|---|---|---|
| `BinderTargetContact` | 1.0 | distogram `[:L, L:]` | contact the conserved epitope |
| `WithinBinderContact` | 1.0 | distogram `[:L, :L]` | be a compact folded domain |
| `InverseFoldingSequenceRecovery` | 10.0 | backbone → MPNN | sequence must encode the fold |
| `BinderTargetPAE` | 0.05 | `pae[:L, L:]` | interface confidence |
| `TargetBinderPAE` | 0.05 | `pae[L:, :L]` | interface confidence, other direction |
| `WithinBinderPAE` | 0.4 | `pae[:L, :L]` | rigid binder |
| `IPTMLoss` | 0.025 | `pae_logits` | complex plausibility (metric proxy) |
| `pTMEnergy` | 0.025 | `pae_logits` | same, non-saturating |
| `PLDDTLoss` | 0.1 | `plddt[:L]` | local confidence |
| `GravitationalEpitope` | 1.0 | `backbone_coordinates` | bind here, **not** there |
| `InterfaceHistidineCount` | 2.0 × 1.0 | distogram + `x` | N histidines in the interface |
| `BuriedHistidinePacking` | 2.0 × 0.5 | distogram + `x` | buried, not merely nearby |
| `NoHistidineOutsideInterface` | 2.0 × 0.3 | distogram + `x` | keep the budget in the paratope |
| `TotalHistidineBudget` | 2.0 × 2.0 | `x` only | hard cap on global histidine |
| `ElectrostaticPKaSelectivity` | 3.0 | coords + charges | optimize scored selectivity (`physics` pool) |
| `NoCys` | — | reparameterization | cysteine unreachable |

Pools (`design.py:67-76`) vary only `his_target` and `physics`: `affinity`
(switch off), `ph` (3.0), `ph_hot` (5.0), `physics` (2.0 + the pKa model).
