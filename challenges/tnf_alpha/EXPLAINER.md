# The proton-linkage equation, and why the design looks the way it does

Background notes for the TNF-α pH-selectivity work. Covers where the linkage
relation comes from, what each term means, the two limits it imposes, and how
those limits determined concrete design choices.

## 1. The problem

The challenge asks for a binder that holds TNF-α at pH 7.4 and releases it at
pH 6.0. Expressed as a measurable quantity, that is a ratio of association
constants:

    S = log10 [ Ka(pH 7.4) / Ka(pH 6.0) ]

S = 1 means binding is 10× weaker at pH 6.0, S = 2 means 100× weaker. "No
detectable binding" on a standard BLI or SPR readout, starting from low-nanomolar
affinity, needs roughly S ≥ 2.

Note that S is a *ratio*. It says nothing about how tightly the binder holds TNF
at either pH. A binder that is weak at both pH values can have a large S and be
useless, which is why the ranking requires a binding threshold before S is
considered at all.

## 2. Why histidine

Binding strength changes with pH only if something on the protein changes
protonation state over the relevant range. Of the twenty amino acids, only
histidine has a side-chain pKa near the window of interest: the imidazole group
titrates at roughly 6.0–6.5 in water. Aspartate and glutamate are ~4 and
lysine/arginine are ~10.5 and ~12.5, so all of them are fully ionised at both
7.4 and 6.0 and contribute nothing to the pH *dependence*. They matter only
indirectly, by shifting histidine pKa values.

So the question becomes: how much pH-dependent binding can a given number of
histidines produce, and where do they have to be?

## 3. Where the linkage equation comes from

Consider one histidine on the binder. It can be protonated or not, and the
binder can be free or in complex. That gives four states and a closed
thermodynamic cycle:

```
        B  +  T  ──── Ka⁰ ────▶  BT
        │                        │
     pKa_free                 pKa_bound
        │                        │
       BH⁺ +  T  ───────────▶  BTH⁺
```

Write the protonation equilibria using the standard definition. For the acid
dissociation BH⁺ → B + H⁺ with dissociation constant Ka_a and pKa = −log10 Ka_a:

    [BH⁺] / [B] = [H⁺] / Ka_a = 10^(pKa − pH)

So the total free binder, summing both protonation states, is

    [B]_total = [B] + [BH⁺] = [B] · ( 1 + 10^(pKa_free − pH) )

and likewise for the complex,

    [BT]_total = [BT] · ( 1 + 10^(pKa_bound − pH) )

What an experiment measures is the association constant over *all* protonation
states:

    Ka_obs(pH) = [BT]_total / ( [B]_total · [T] )

Substituting the two expressions above, the deprotonated-species concentrations
cancel into Ka⁰ = [BT] / ([B][T]), which is pH-independent:

    Ka_obs(pH) = Ka⁰ · ( 1 + 10^(pKa_bound − pH) ) / ( 1 + 10^(pKa_free − pH) )

Taking logs:

    log10 Ka_obs(pH) = log10 Ka⁰
                     + log10( 1 + 10^(pKa_bound − pH) )
                     − log10( 1 + 10^(pKa_free − pH) )

Define the pH-dependent part as

    δ(pH) = log10(1 + 10^(pKa_bound − pH)) − log10(1 + 10^(pKa_free − pH))

and the selectivity is the difference between two pH values:

    S = δ(7.4) − δ(6.0)

Ka⁰ cancels. This is the key structural feature: **the selectivity depends only
on the pKa values, not on how tightly the binder binds.** Affinity and pH
selectivity are separable, which is why they are separate criteria and why
optimising one does not automatically help the other.

For several independent ionisable groups the same derivation gives a product of
such factors, so the log-selectivities add:

    S_total = Σ_i S_i

That additivity is why the question "how many histidines?" has a definite answer.

## 4. What the terms mean

The factor `1 + 10^(pKa − pH)` is a *binding polynomial* for the proton. It sums
over the available protonation states: the `1` is the deprotonated state and the
`10^(pKa − pH)` is the protonated state, weighted by its relative population at
that pH. Its logarithm is the free energy, in log₁₀ units, made available by
having that protonation equilibrium accessible.

Reading the two terms:

- The **numerator** is how much the complex gains from being able to take up a
  proton.
- The **denominator** is the same quantity for the free binder.

Three cases follow directly.

**pKa_bound = pKa_free.** The terms cancel and S = 0. If binding does not change
the pKa, then protonation does not care about binding and binding does not care
about pH. This is the symmetric statement of linkage, and it is why a histidine
that sticks out into solvent is useless no matter how many you add.

**pKa_bound < pKa_free** (binding suppresses protonation). At low pH the free
binder wants a proton but the complex cannot accommodate it, so forming the
complex requires stripping that proton off. That costs energy, so binding is
weaker at low pH. S > 0, which is the switch we want.

**pKa_bound > pKa_free** (binding favours protonation). The inverse: binding is
*stronger* at low pH. S < 0. This is a real failure mode, not a hypothetical —
23 of our first 124 designs came out negative, which is why the sign is checked
rather than assumed.

## 5. Two limits, and what they force

### Limit 1: across all pH, S cannot exceed the pKa shift

Taking pH → −∞ gives δ → pKa_bound − pKa_free, and pH → +∞ gives δ → 0. So
measured between extremes,

    S_max(full range) = pKa_free − pKa_bound

Verified numerically: pKa 6.5 → 4.0 gives exactly 2.500 over pH 0–14, and
7.0 → 2.0 gives 4.996.

### Limit 2: across a narrow window, S cannot exceed the window width

This is the limit that actually binds, and it is not obvious. The
Henderson–Hasselbalch term changes by at most 1 log unit per pH unit, so over a
window of width ΔpH no single group can contribute more than ΔpH. For
7.4 → 6.0, ΔpH = 1.4:

| pKa_free (pKa_bound = 0) | S per histidine |
|---|---|
| 6.5 | 0.568 |
| 7.0 | 0.896 |
| 8.0 | 1.307 |
| 9.0 | 1.390 |
| 12.0 | 1.400 |

The ceiling is exactly 1.4, the window width, and is approached only with an
absurdly elevated free-state pKa. **One histidine cannot reach S = 2 even in
principle.** Two is the theoretical minimum and requires near-perfect placement.

With realistic numbers:

| pKa_free → pKa_bound | S | fold | His needed for S = 2 |
|---|---|---|---|
| 6.5 → 5.5 | 0.454 | 2.8× | 5 |
| 6.5 → 4.0 | 0.564 | 3.7× | 4 |
| 7.0 → 4.0 | 0.892 | 7.8× | 3 |
| 7.5 → 3.0 | 1.159 | 14.4× | 2 |

This table is the reason the submitted designs carry 4–11 histidines rather than
one or two. A single well-placed histidine was never going to be enough, and
that was knowable before any design was run.

## 6. The two levers

S increases when pKa_free rises and when pKa_bound falls. These are physically
distinct, and both are controllable.

**Raising pKa_free.** The protonated imidazole carries +1, so nearby negative
charge stabilises it and raises its pKa. Placing Asp or Glu near the histidine
*in the free binder* therefore increases S. Note that the table above is much
more sensitive to pKa_free than to pKa_bound: dropping pKa_bound from 5.5 to 4.0
buys 0.11 log units, while raising pKa_free from 6.5 to 7.5 buys 0.60. This is
the lever that a "count the histidines" objective cannot see, and it is why
several of the submitted designs are deliberately acidic (net charge −12 to +3).

**Suppressing pKa_bound.** Burying the imidazole against the target desolvates
it, and a buried charge is costly, so the protonated form is destabilised and the
pKa drops. Nearby positive charge on the target does the same. This requires the
histidine to be at the *interface*, not merely on the binder, which is why the
scanning procedure restricts substitutions to positions in contact with the
epitope.

Both levers were confirmed on measured data before being relied on. Infliximab's
CDR-L3 His92 shows a −2.35 pKa drop on binding — a large, genuine shift — yet
contributes only +0.33 log units, because its free-state pKa is 6.09. TNF's own
His15 and His78 sit at pKa_free 5.06 and 4.01 and contribute 0.044 and 0.003.
Large shifts on low-pKa histidines buy almost nothing.

## 7. What is exact and what is not

The linkage relation is exact. It is a consequence of the definitions of Ka and
pKa plus the assumption that the system is at equilibrium; there is no fitting
and no free parameter. If the pKa values are right, S is right.

All the uncertainty is in the pKa values. We used PROPKA 3.5.1, whose error
against experiment is roughly 0.7–1.0 pKa units, and measured its error on the
quantity that matters — the *shift* between end states — at 0.46 log₁₀ per site
on held-out complexes. Since 0.46 is comparable to a single histidine's entire
contribution, individual per-site numbers are not trustworthy. Totals over 4–11
sites average down, which is the regime we are in, and selectivity differences
below 0.5 log₁₀ are treated as unresolvable in the ranking.

A differentiable version of the pKa model was built so the linkage relation could
be optimised directly during design rather than only evaluated afterwards. It
improves on a constant-predictor null by 11% on absolute pKa and 20% on the
shift, which is too weak to rank on, so it was used only to supply a gradient
direction while PROPKA did all the ranking.

## 8. How it is used end to end

1. **Before designing:** evaluate the relation to establish how many histidines
   are needed and which lever matters more (section 5–6). This set the histidine
   budgets of 3 and 5 for the in-loop objectives.
2. **During design:** the differentiable pKa model estimates pKa_free and
   pKa_bound from one complex prediction — binder neighbours alone give the free
   state, adding the target gives the bound state — and the linkage relation
   turns them into an objective. This step does not work. Across 133 designed
   parents the median number of histidines at the interface is zero, against a
   random-placement expectation of 1.29, while total histidine content sits at
   about 6 per design. The histidines are placed one residue from a contacting
   position (2.41 per design against 1.04 expected), which on a helix is the
   off-contact face. The cause is not ProteinMPNN suppressing histidine in the
   paratope: measured over 40 complexes, MPNN assigns P(His) 0.0295 at interface
   positions against 0.0243 elsewhere, slightly favouring it. The remaining
   candidates are the affinity terms, which see histidine at a buried contact as
   a predicted-iPTM cost, and a pose shift between design at one recycling step
   and rescoring at three.
3. **After designing:** PROPKA is run on the predicted complex and the predicted
   free binder, and the same relation converts the two pKa sets into S. This is
   the number used for ranking. Because step 2 does not place histidines,
   interface histidine scanning after the fact is what actually produces the
   switch: raw design parents clear all three challenge criteria at 0.6%,
   against 5.8% for scanned variants, and 13 of the 15 all-rounders in the first
   submission were variants rather than parents.
4. **Validation:** two known non-pH-dependent binders of this epitope,
   adalimumab and infliximab, score −0.312 and +0.385 against a target of ~2.0.

Steps 2 and 3 use the same relation, implemented twice — once in JAX and once in
NumPy — and they agree to 9.4 × 10⁻⁸, so the objective and the ranking score
cannot drift apart.

## References

- Wyman, J. (1964) Linked functions and reciprocal effects in hemoglobin.
  *Adv. Protein Chem.* 19:223 — the original linkage formalism.
- Igawa, T. et al. (2010) Antibody recycling by engineered pH-dependent antigen
  binding. *Nat. Biotechnol.* 28:1203 — the therapeutic rationale, and the
  origin of interface histidine scanning.
- Hogues, H., Wei, W. & Sulea, T. (2025) Improved structure-based histidine pKa
  prediction for pH-responsive protein design. *J. Chem. Inf. Model.* — the
  electrostatics-only, linear-readout pKa model our differentiable term follows.
- Olsson, M. H. M. et al. (2011) PROPKA3. *J. Chem. Theory Comput.* 7:525.
