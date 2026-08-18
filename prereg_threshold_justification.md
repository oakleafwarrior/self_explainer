# Threshold justification for the NB00 §8 preregistration

Companion to `reports/preregistration.json` (v2). Every threshold in that file is stated as a
number; this document is the argument for the number. Where the argument does not survive the
arithmetic, it says so — a preregistration whose thresholds are below the resolution of the
eval set is not a preregistration, it is a wish.

Written against `EVAL_SIZE = 1024`, `PAPER_WITH_ACTIVATION = 0.640`,
`PAPER_NO_ACTIVATION = 0.599`.

---

## 1. The resolution that governs everything

Every normalized threshold is implicitly a raw-score threshold, because `destroyed` and
`retained` divide by a span of about 4.1 exact-match points. So each one has to be checked
against what n=1024 can actually see.

| quantity | value |
|---|---|
| independent 95% half-width at p=0.5 | ±3.06 pts |
| independent 95% half-width at p=0.64 | ±2.94 pts |
| **difference of two independent arms** | **±4.33 pts** |
| the activation's entire contribution (span) | **4.10 pts** |

The two-arm independent comparison is wider than the whole effect being measured. NB00 §6
prints `resolvable = worst < activation_worth`, which compares a *single-arm* half-width to the
span and passes; the comparison the design actually makes is `worst * sqrt(2)` against the span,
and that fails (0.0433 > 0.0410). The conclusion in §6 is still right, but for the reason stated
two paragraphs down rather than the one the boolean tests: **the readings are decidable only
because they are paired.**

Every arm is evaluated on the same held-out items, so the operative statistic is McNemar's
paired delta (`se_common.paired_delta`), with `se = sqrt(b + c) / n` over the discordant pairs:

| discordance rate `d` | paired 95% half-width | as a fraction of the 4.1-pt span |
|---|---|---|
| 0.05 | ±1.37 pts | 0.33 |
| 0.10 | ±1.94 pts | 0.47 |
| 0.15 | ±2.37 pts | 0.58 |
| 0.20 | ±2.74 pts | 0.67 |
| 0.30 | ±3.35 pts | 0.82 |

`d` is the single unmeasured quantity that determines whether half the thresholds below are
reachable. §5 says how to measure it without compromising the preregistration.

### 1.1 The ratio is less stable than either of its parts

`retained = (score − floor) / (reference − floor)` is a ratio whose denominator is measured, is
small (~4 pts), and carries its own CI. By the delta method, with paired SEs of 0.0121 on both
numerator and denominator (d = 0.15):

| true `retained` | SE(`retained`) | 95% CI |
|---|---|---|
| 0.0 | 0.295 | ±0.579 |
| 0.5 | 0.330 | ±0.647 |
| 1.0 | 0.417 | ±0.818 |

**A CI of ±0.58 on a scale whose decision thresholds are 0.15 and 0.35 is the central problem
with the preregistration as written.** The thresholds are separated by less than one standard
error. This is a Fieller-type problem: with a noisy small denominator, the normalized statistic
is not approximately normal and its interval can be unbounded. Two implications:

- Report `destroyed`/`retained` with a **Fieller interval**, not a delta-method one, or report
  the raw paired delta as primary and the normalized figure as the interpretive gloss.
- The normalization is still the right *reporting* choice — a 1-point raw gap really is 24% of
  the mechanism under study, and the raw scale really does understate the effect ~15×. But
  normalization improves interpretability, not precision, and the preregistration currently
  reads as if it did both.

---

## 2. Thresholds, one at a time

| threshold | value | raw equiv. | verdict |
|---|---|---|---|
| `overlap_destroyed` | 0.15 | 0.62 pts | **unresolvable** — needs n ≈ 15,200 |
| `separated_destroyed` | 0.35 | 1.44 pts | **marginal** — needs n ≈ 2,800 |
| `floor_match_retained` | 0.10 | 0.41 pts | **unresolvable** as a point estimate |
| `below_floor_retained` | −0.10 | −0.41 pts | **unresolvable** as a point estimate |
| `recovery_fraction` | 0.90 | 3.69 pts | sound — the coarse ladder absorbs the noise |
| `rank_artifact_closure` | 0.50 | — | sound, needs a precondition |
| `baseline_match_gap` | 0.02 | 2 pts | sound via paired TOST only |
| `overlap_gap_raw` | 0.02 | 2 pts | sound via paired TOST only |
| `separated_gap_raw` | 0.05 | 5 pts | **dead key, and wider than the span** |
| `ridge_identity_tol` | 1e-2 | — | too loose if the solve is float64 |
| `exactness_tol` | 0.02 | 2 pts | measured in the wrong place |

Sample sizes are paired, at d = 0.15.

### `overlap_destroyed = 0.15` — "the curves overlap"

**The argument.** 15% of a mechanism's total contribution is a defensible operational "nothing
here." It is small enough that a reader will accept "no recoverable effect" and large enough not
to be defeated by rounding.

**Whether it holds.** As an effect size, yes. As a decision rule at n=1024, no: detecting a
0.62-point difference needs n ≈ 15,200. Worse, the rule is stated as `abs(destroyed) < 0.15`
on a point estimate carrying ±0.58 of noise, so a *true zero effect* will fall outside ±0.15
roughly half the time. `no_effect` is therefore both the easiest reading to trigger by luck and
the one you would least want to publish by luck.

**Fix.** Make it an equivalence claim: `no_effect` fires when the 95% paired CI on `destroyed`
lies **within** ±0.15, not when the point estimate does. At n=1024 that will essentially never
fire, which is the honest answer — it routes the outcome to `all_inside_band`, which is already
in the file and is the correct home for "smaller than we can see."

### `separated_destroyed = 0.35` — "the curves are separated"

**The argument.** A third of the mechanism's contribution is a substantial effect; if rotation
destroys that much, the paper's §3.4 correlation is doing real work.

**Whether it holds.** Best-behaved of the normalized four. 1.44 points needs n ≈ 2,800 paired —
roughly 2.7× the current eval set, so it is *marginal* rather than impossible, and a true effect
at 0.5+ would be visible. Keep the value.

**Fix.** State it one-sided: fires when the 95% paired CI lies **entirely above** 0.35.

### `floor_match_retained = 0.10` / `below_floor_retained = −0.10`

**The argument.** Symmetric ±0.10 around the measured floor. `collapse_to_floor` is the dramatic
outcome (the activation's entire contribution was coordinate frame); `below_floor` is a stop
condition.

**Whether it holds.** 0.41 raw points, a tenth of the manipulation's range — far under
resolution. This matters most for `below_floor`, whose reading is *"Bug, or the rotated vector
is actively misleading the explainer — investigate before reporting anything."* A point-estimate
rule at −0.10 will halt the project on noise several times over.

**Fix.** `below_floor` fires only when the paired CI on `retained` lies entirely below 0.
`collapse_to_floor` fires only when the CI lies entirely below `floor_match_retained`.

### `recovery_fraction = 0.90`, `low_rank_recovery` at rank ≤ 8

**The argument.** The minimum rank whose map recovers 90% of the unrotated reference locates the
dimensionality of the frame mismatch.

**Whether it holds.** Yes — and it survives where the others fail for a structural reason worth
stating: the ladder is {0, 8, 128, 512, full}, with 16× and 4× gaps between rungs. The answer is
a choice among five discrete values, and the CI (±0.58 normalized) is small compared to the
*spacing between rungs* even though it is large compared to 0.90. Coarse grids buy robustness.

**Caveat.** Report "≤ 8", never "at 8". With nothing between 8 and 128, a true recovery rank of
32 reports as 128, and the reading ("suggests a very cheap alignment recipe") would be
understated by 4×.

### `rank_artifact_closure = 0.50`

**The argument.** If lifting the rank cap closes more than half the P-ridge − P-rand gap, the
paper's random-vs-pretrained projector difference is substantially a rank artifact. Half is the
natural dividing line for "substantially."

**Whether it holds.** The value is right, but this is another ratio with a measured denominator.
If `score_ridge − score_rand` comes out small, the closure fraction is unstable in exactly the
way `retained` is, and NB07 will print a confident number computed from noise over noise.

**Fix.** Add a precondition: evaluate `rank_artifact` only when `score_ridge − score_rand`
exceeds twice its own paired CI. Below that, the reading is "the paper's gap is too small at
this n to attribute to anything," which is itself publishable given the 70B numbers rest on it.

### `baseline_match_gap = 0.02` and `overlap_gap_raw = 0.02`

**The argument.** Two raw exact-match points as the "these are the same" margin for comparisons
that are levels rather than ranges (`ridge_frozen_works`, `ridge_works`, `rank_not_binding`,
`collapse_to_baseline`).

**Whether it holds.** These are the best-calibrated numbers in the file — but by a margin that
is invisible in the file itself. A paired TOST at a ±2-point equivalence margin needs n ≈ 1,015
at d = 0.15. `EVAL_SIZE` is 1024. The design works with 9 items to spare, and only under the
paired analysis; the independent-CI requirement the notebook itself prints is n ≥ 2,401.

Two consequences:

1. **They must be TOST statements**, not `abs(gap) < 0.02`. As written they read absence of
   evidence as evidence of equivalence, and `ridge_frozen_works` — the headline practical
   result, "the alternative recipe works with NO per-target training" — rests on exactly that
   inference. Correct form: the 90% CI of the paired difference lies entirely within ±0.02.
2. **The margin is reachable only if d ≤ ~0.15.** At d = 0.30 the requirement is n ≈ 2,030 and
   the equivalence claims become unsupportable. See §5.

### `separated_gap_raw = 0.05`

**Delete or rescope.** Two independent problems:

- It is referenced by no reading in `preregistration.json` and appears nowhere in NB05–NB07.
  It is a dead key.
- 5 points exceeds the entire 4.1-point dynamic range of the manipulation, so no rotation arm
  can ever cross it. If it is meant for the cross-model comparisons, where scores span a wider
  range, say so in the key name and scope it there.

`use_seed_band: True` is likewise unconsumed — `attainable_optimum` mentions the band only in
prose.

### `ridge_identity_tol = 1e-2`

**The argument.** Relative Frobenius error on `‖Π^Q − Π Qᵀ‖`, the integrity check that the ridge
implementation is doing what the algebra says.

**Whether it holds.** Depends entirely on the dtype of the solve, which the threshold does not
name. In bf16 (eps ≈ 3.9e-3) a 1e-2 relative error is ~2.5 eps — tight but reachable. In float64
it is roughly ten orders of magnitude looser than it should be, and a genuinely broken
implementation could pass.

**Fix.** Solve the ridge system in float64, assert at 1e-6 there, and keep 1e-2 as a separate
post-cast check on the bf16 matrix that actually gets used.

### `exactness_tol = 0.02`

**The argument.** Oracle/R-Q composes Qᵀ with Q and must reproduce C0/R-id up to bf16 noise. Any
larger difference is a plumbing bug and everything downstream is contaminated.

**Whether it holds.** The threshold is fine; the *measurement point* is wrong. Oracle/R-Q and
C0/R-id are two separately trained runs, so bf16 divergence compounds over the entire
trajectory. Two points of exact-match difference after 8,192 training examples confounds
"plumbing bug" with "arithmetic divergence amplified by training," and the reading — *"Stop"* —
is too expensive to trigger on the latter.

**Fix.** Assert exactness on **one forward pass at initialization**, where the two configurations
are mathematically identical and KL should be ≲1e-3. Keep the 2-point post-training metric check
as a downstream sanity test with a softer reading than "stop."

### The `has_changed_f1` gap

`metrics_disagree` compares the signs of three metrics, but `paired_delta` supports
`exact_match` and `content_match` only — F1 is not a per-example binary and has no paired-CI
implementation. The rule therefore compares two quantities with error bars against one without.
Add an item-level bootstrap (resample eval items, recompute F1, take the 2.5/97.5 percentiles)
before this reading can fire.

---

## 3. Structural gaps in the preregistration

**No precedence rule.** `not_mutually_exclusive: True` is honest but incomplete.
`collapse_to_floor`, `collapse_to_baseline`, and `attainable_optimum` can all fire on the same
results, and nothing says which is reported. Add an explicit order:

```
integrity → below_floor → collapse_to_floor → collapse_to_baseline
          → attainable_optimum → sample_efficiency → no_effect
```

**A dead zone with no reading.** `no_effect` requires `destroyed < 0.15` at every N;
`sample_efficiency` requires `> 0.35` at min N. A result sitting at `destroyed = 0.25`
everywhere maps to no row at all. Add a named `indeterminate` reading covering [0.15, 0.35] so
the outcome is reported rather than silently dropped.

**Multiplicity.** The rules quantify over 7 N values × 3 metrics. The "for every N" forms are
conjunctions and therefore conservative, but the single-point forms (`destroyed(min N) > 0.35`)
are not. Naming `exact_match` as the primary metric is the right control and is already in the
file; add a **primary N** for the single-point rules so they are one test rather than seven.

**The floor has no band.** `retained` divides by `(reference − floor)` and §1.1 shows the
denominator's noise dominates the ratio — but NB03's floor loop runs single-seed. The
denominator of every number this project reports is currently the least-replicated quantity in
it. Fixed by §4.

---

## 4. The seed grid: 3 seeds at every N

**Superseding v2 §3** ("3 training seeds at N ∈ {512, 8192}; 1 seed elsewhere"). All three seeds
run at all seven N values. Implemented in `se_config.py` — `MULTI_SEED_N = list(N_TRAIN_VALUES)`,
consumed through `C.seeds_for(n)` by NB03 and NB05.

**Why.** Two reasons, one of which fixes a defect that would have blocked the modal reading.

1. **The readings quantify over N, but the bands did not.** `sample_efficiency` — flagged in the
   preregistration as the modal outcome — turns on `destroyed(min N)` at N=128.
   `attainable_optimum` requires a gap wider than the seed band *at all N*. `all_inside_band`
   compares against the band by construction. Under the old grid, five of seven N values had no
   band, so three readings leaned hardest on exactly the points where the uncertainty was
   unmeasured.

2. **Pooling makes the band tighter than a per-N estimate could ever be.** With 3 seeds at 7 N
   values, seed variance can be pooled across N (checking first that it is roughly constant in
   N), giving 14 df instead of 2:

   | estimate | t | band half-width |
   |---|---|---|
   | per-N, 3 seeds | t(2) = 4.303 | 2.484 σ |
   | pooled over 7 N | t(14) = 2.145 | **1.238 σ** |

   A 2.01× tightening, for free, from data being collected anyway. This also retires a
   methodological weak point: a min/max "band" from 3 draws has an expected range of 1.69 σ,
   which *understates* the honest 95% interval by ~1.5×. Define the band once — mean ±
   t·σ̂/√3 with σ̂ pooled — and use that definition in NB03, NB04, NB06 and NB07. NB06 currently
   forms its band from `|seed_a − seed_b|`, two points, which is weaker still.

**What it enables: seed-matched pairing.** With three seeds at every N, R-id and R-Q can be
compared *seed by seed* — seed 67's R-id against seed 67's R-Q, and so on. The training seed
controls data order and LoRA init, so a matched pair differs only in the rotation. This removes
between-run training noise from the contrast, leaving item noise plus the seed×rotation
interaction, and yields three independent estimates of the gap at each N instead of one
difference of two noisy means. **Report the seed-matched paired delta as the primary statistic
at each N.** This was not possible under the old grid at five of seven N values.

**Cost.**

All of NB03, including §3's 4-run init control (unchanged by this):

| | runs | training examples |
|---|---|---|
| old grid | 56 | 186,880 |
| **new grid** | **100** | **264,704** |
| new grid, floor left single-seed | 86 | 232,192 |

NB03 goes 1.79× in run count and 1.42× in training examples. Eval cost scales with run count,
not with N, so it goes the full 1.85× — and at `EVAL_SIZE = 1024` eval is a `generate()` pass
per run, so this is the larger share of the increase. NB04, NB05 and NB06 are unaffected: their
arms run at `N_TRAIN_LADDER`, which already had three seeds.

This has to be paid for. The v2 §8 cut list, in its stated order, is the budget: the R-Qs
distributional arm, then the C64/C512 interior ladder rungs, then the Cfull-rand init control.
Cutting in that order preserves every arm the §9 readings depend on. NB00 §9's budget block
(19.5 h including a 2.0 h buffer, with the buffer already spent by the injection bug) is stale
and should be recomputed against 96 runs before NB03 starts.

**The floor arm is included.** NB03's no-activation floor loop iterated `FLOOR_N` directly and
did not call `C.seeds_for(n)`, so the config change would not have reached it. It now does, for
the reason in §3: the floor is the denominator of every normalized reading, and a single-seed
denominator is the weakest link in the ratio. That is the difference between the 100-run and
86-run lines above.

NB03 §1 now keeps both `no_activation_floor_runs.csv` (one row per N × seed) and
`no_activation_floor.csv` (the per-N mean, which is what everything downstream divides by), plus
`no_activation_floor_spread.csv`. The spread is printed under the floor table because it *is*
the denominator's own uncertainty, and §1.1 is the argument for why that number deserves to be
visible rather than averaged away silently.

---

## 5. Measuring the discordance rate `d` first

`d` is the fraction of eval items on which two arms disagree. It sets the paired CI, and
therefore whether `overlap_gap_raw = 0.02` is reachable at n=1024 (yes at d ≤ 0.15, no at
d ≥ 0.30) and how far off `separated_destroyed = 0.35` is. It is currently a guess, and the
whole NB03 grid is about to be run against thresholds whose reachability depends on it.

**It can be measured without compromising the preregistration.** Run `paired_delta` on **two
training seeds of the same arm** — `(R-id, Cfull, seed 67)` against `(R-id, Cfull, seed 68)` —
and read `discordant_a + discordant_b`. Both runs are on the same side of the manipulation, so
the comparison contains no information about the rotation contrast, about R-Q, or about any
preregistered reading. Nothing is unblinded. What it yields is the run-to-run item-level
disagreement rate, which is an upper bound on the discordance you should expect between arms
that differ only in rotation, and hence a conservative input to the power calculation.

Under the new seed grid these two runs are the first two cells of NB03 anyway, so the
measurement costs nothing beyond reading a number off runs already scheduled.

**Decision rule, fixed in advance:**

| measured `d` | paired half-width | consequence |
|---|---|---|
| ≤ 0.15 | ≤ ±2.37 pts | proceed; the ±0.02 equivalence margins are supported |
| 0.15 – 0.25 | ±2.4 – 3.1 pts | proceed, but widen the equivalence margin to ±0.03 and say so **before** any R-Q result is read |
| > 0.25 | > ±3.1 pts | raise `EVAL_SIZE` to 2048 before running the grid; the equivalence readings are otherwise unsupportable |

Recording this table before the measurement is what keeps the adjustment from being a
post-hoc one. The rule is a function of `d` alone, and `d` is estimable from arms that carry no
information about the hypothesis.

---

## 6. What is right and needs no defense

For completeness, since the preceding sections are all criticism:

- **`EVAL_SIZE = 1024`** is correct and correctly derived. At n=128 the half-width (±8.7 pts)
  is twice the entire dynamic range and the experiment would have been undecidable regardless of
  outcome. 1024 also lands within 1% of the optimum for the binding constraint (a paired TOST at
  ±2 points needs n ≈ 1,015). The 8× eval cost buys decidability, not vanity precision.
- **Normalizing to the measured floor rather than the paper's 0.599.** Our floor is at our N,
  our eval items, our recipe. The preregistration is explicit about this and it is right.
- **Reporting on `destroyed`/`retained` rather than raw points.** Rotation acts only on `v`, so
  the achievable range is the activation's contribution, not the score. A 1-point raw gap is 24%
  of the mechanism under study; on the raw scale it reads as noise. (Precision comes from the
  paired analysis, not from the normalization — see §1.1.)
- **`primary_metric: exact_match` with two named secondaries** — the control that keeps 3
  metrics × 7 N from becoming 21 free shots.
- **C0 as a faithful arm rather than a strawman**, carried by Appendix F.1 with file:line
  citations. This converts a contestable claim into a checkable one and is the strongest part of
  the document.
- **`PREREG_VERSION` refusing to silently overwrite.** The failure mode a preregistration exists
  to prevent is a post-hoc rewrite; the v1/v2 branch handles it correctly (keep v1 binding, mark
  v2 exploratory).
- **`USE_4BIT = False`, the untied-embedding assert, `vec_dim == hidden_size`** — correctness
  preconditions, not thresholds. Each fails loudly.
- **`assert injection_is_noop` in NB00 §1** — the notebook's own premise is asserted, so if the
  bug ever stops reproducing, the reuse verdict fails loudly instead of standing on a stale
  claim.

---

## 7. Unrelated defect found while checking §7 of NB00

**The KL noise floor measures zero, not a floor.** `R.invariance_gate(m, m, tok, ...)` passes
the same model object twice, over the same batches, at the same batch size. Both forward passes
hit identical kernels with identical inputs, so the logits are bitwise identical and `mean_kl`
is 0.0 — for the `batch_size=1` and the `batch_size=4` call alike. The section's own note is
half-right: "invariance_gate runs both models batch-identically, so this measures within-batching
noise." Within-batching noise for one model against itself is not small, it is nil.

The floor the gate needs is the arithmetic cost of the **folding**, not of the forward pass.
Construct it by folding with Q and then unfolding with Qᵀ through the same code path: the result
is mathematically M but built by the same dense matmuls as M_Q, so gating M against it isolates
exactly the error the rotation introduces. That number is what the 1e-4 threshold in
`rotate.invariance_gate` should be calibrated against — v2 §5.5 says to accept relative to the
measured floor rather than an absolute bar, and a floor of 0 makes "within one order of
magnitude" vacuous.

Minor, same section: the markdown says 256 sequences; the code breaks at 64.
