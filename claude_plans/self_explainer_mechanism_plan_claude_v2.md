# Where Does the Self-Explainer Advantage Come From?

Testing whether an explainer succeeds because of privileged access to a model it *is*, or because that model's activations arrive in a coordinate frame the explainer already reads.

**MATS 12.0 application project.** ~18 hours of work (hard cap 20), plus 2 for the executive summary. Builds on `oakleafwarrior/introspection_replication`.

---

## 1. The claim, and why the answer changes what you'd do

Li et al. (arXiv:2511.08579) report that a target model $M$ is best explained by fine-tuned versions of itself, and that this self-explanation is ~100× more sample-efficient than a nearest-neighbours baseline. The proposed mechanism is privileged access.

Two alternatives explain the same numbers without any privilege:

- **Behavioral self-simulation.** $E$ is initialized from $M$'s weights and is close enough in function space to predict $M$'s outputs directly.
- **Basis compatibility.** $E$ decodes $v$ well because $v$ arrives in coordinates $E$ already reads fluently. No self-knowledge required, just a shared frame.

The paper's §3.4 and Appendix C.2 establish a correlation between activation alignment and explainer quality. That is basis compatibility, presented as supporting evidence for privileged access rather than as the competing explanation it is.

**The consequence is a practical one, and it is the reason to run this.** If the mechanism is basis compatibility, then the recipe for cheap interpretability is not "fine-tune each model to explain itself" — it is "take one good explainer and fit it a cheap linear map into each new target's frame." Those two recommendations differ enormously in cost. The first requires a separately trained explainer per target model, retrained on every checkpoint. The second requires one explainer plus a ridge regression per target. If basis compatibility is doing the work, the paper's most useful finding is being filed under the wrong heading, and the deployment story changes completely.

There is also a narrower stake for oversight. If self-explanation is geometric rather than introspective, "the model told us what it was doing" carries no more epistemic weight than any other trained probe, and the CoT-faithfulness framing in §5.2 loses its distinctive force.

### 1.1 Scope of the claim this project can actually support

State this up front in the write-up, not in a footnote. **This project tests the mechanism on the activation-patching task with Qwen3-8B as target, using LoRA-plus-adapter explainers.** The paper's headline Table 1 result is the *feature description* task on Llama-3.1-8B with full-parameter SFT. Three gaps follow, and all three belong in the abstract:

1. Different task. Patching, not feature description (see Appendix B for why porting is the wrong use of the budget).
2. Different training regime. LoRA + explicit input map, not full SFT (Appendix A).
3. Different model family.

The defensible claim is therefore: *on the task and family where we can measure it, the self-explainer advantage is substantially or wholly a coordinate-frame effect, and a closed-form linear fit recovers most of it.* Anything broader is unearned.

---

## 2. The wedge

An orthogonal transformation $Q$ of $M$'s residual stream separates the hypotheses. RMSNorm transformers are computationally invariant to such a transform, so $M$ and $M_Q$ compute the **identical function** — behavioral similarity between $E$ and $M_Q$ is exactly what it was for $M$. But every extracted activation becomes $Qv$, and the coordinate alignment measured in the paper's Table 4 goes to chance.

### 2.1 The explainer is initialized from *unrotated* $M$. This is not optional.

The self-explainer of $M_Q$ is, by the paper's own definition, a fine-tune of $M_Q$. Build it that way and $E$ is rotated too, alignment is perfectly preserved, and the experiment is a mathematical no-op that will produce identical numbers in both arms.

**The design is: $E$ is fine-tuned from unrotated $M$; the activations it receives come from $M_Q$.** Because $M_Q \equiv M$ as a function, this holds behavioral similarity *exactly* fixed — not approximately, identically — while destroying coordinate correspondence.

The consequence for framing: strictly, this does not measure "the self-explainer advantage" as the paper defines it. It measures **the contribution of coordinate correspondence to that advantage**, by constructing a target with identical behavior and no shared frame. That is the sharper experiment, but the write-up must say which quantity is on the axis. A reader who notices the ambiguity before you name it will discount the whole result.

| Hypothesis | Predicts for $E(\,\text{from }M\,)$ reading $M_Q$ |
|---|---|
| Behavioral self-simulation | Advantage survives — behavior is bit-identical |
| Basis compatibility | Advantage collapses toward the cross-model baseline |

**Scope.** This applies to the activation-patching task (§2.3 of the paper). Input ablation (§2.4) passes no activation to the explainer at all, so there is nothing to rotate — which makes it a free control whose scores must be *exactly* invariant.

### 2.2 Only the activations handed to $E$ need to be rotated

The scientific content of the rotation arm is: *feed $E$ the vector $Qv$ instead of $v$.* That requires no weight surgery — it is a transform on cached tensors (Phase 2).

What the weight-level construction (Phase 1) buys is the **validity argument**: that $Qv$ is a genuine activation of a real model computing exactly the same function, rather than an arbitrary perturbation. That argument is worth several hours, but it is an argument, not a pipeline dependency. If Phase 1 overruns, it degrades gracefully — see the cut order in §8.

---

## 3. The arm matrix

Everything below refers to this table. Write it into the repo README before running anything; the combinatorics are where budgets die.

**Target** $M$ = Qwen3-8B, throughout. The reason is data, not architecture: the cached `act_patch_qwen3_8b_counterfact` dataset exists for this target and Phase 2 reuses it wholesale. (Tied embeddings are *not* a constraint — see Appendix C.3, which corrects a claim in the previous draft.)

**Explainers**

| Tag | Model | Activation path | Init |
|---|---|---|---|
| `E_self` | Qwen3-8B | direct injection + trainable input map | from unrotated $M$ |
| `E_cross` | Qwen3-4B (primary), 1.7B (secondary) | per-layer full-rank $\Pi_\ell$ | independent pretrained weights |

**Rotation condition** (applied to activations handed to $E$)

| Tag | Transform |
|---|---|
| `R-id` | $v \mapsto v$ |
| `R-Q` | $v \mapsto Qv$, single global $Q \in O(4096)$ |
| `R-Qs` | $v \mapsto \mathrm{diag}(s)\,Qv$, $s$ log-uniform — **input-side only**, see §7.3 |

**Capacity ladder** (input map on the injected $v$, `E_self` only)

| Arm | Input map | Can represent $Q^{-1}$? | Trainable? |
|---|---|---|---|
| `C0` | none (LoRA only) | No | — |
| `C8` / `C64` / `C512` | rank 8 / 64 / 512 | No / No / Partially | Yes |
| `Cfull` | full-rank $d\times d$, init $I$ | Exactly | Yes |
| `Cfull-rand` | full-rank $d\times d$, random orthogonal init | Exactly | Yes |
| `Oracle` | frozen at $Q^\top$ | Exactly | **No** |

**Projector init** (`E_cross` only)

| Tag | Meaning |
|---|---|
| `P-rand` | random init, then train — the paper's Qwen3-8B condition |
| `P-ridge` | closed-form ridge init, then train |
| `P-ridge-frozen` | closed-form ridge fit, **explainer never trained on this target** |

**Sweep.** `N_TRAIN` $\in [128, 256, 512, 1024, 2048, 4096, 8192]$, matched data ordering across arms.

**Seeds.** 3 training seeds at $N \in \{512, 8192\}$ for every core arm; 1 seed elsewhere. Every figure carries the seed band. This is non-negotiable — §9's readings all turn on "curves overlap" vs "separated," and neither is decidable without it.

### 3.1 Why `Cfull-rand`, `Oracle`, and the cross-model baseline all exist

`Cfull` under `R-id` is initialized at identity — that is, **at the solution**. `Cfull` under `R-Q` is initialized at identity when the truth is $Q^\top$. So "R-id separates from R-Q at low $N$ and converges at high $N$" is close to a mathematical consequence of the initialization, not a finding about self-explanation. Three counters, in increasing order of importance:

- **`Cfull-rand`** puts both rotation arms the same distance from their solutions at init, so a residual gap is not an artifact of the head start.
- **`Oracle`** separates *representability* from *learnability*. It also gives an exact bug check: `Oracle` under `R-Q` is algebraically identical to `C0` under `R-id`, since $Q^\top(Qv) = v$ and the rest of the network is untouched. If those two runs don't agree to bf16 noise, there is a plumbing bug — find it before spending the sweep.
- **The cross-model baseline is the yardstick.** "R-id beats R-Q" is uninterpretable on its own. "R-Q falls to where an unrelated 4B model sits" is the claim. `E_cross` under `P-rand` is therefore a first-class budgeted arm, not a reference number quoted from the paper.

Correspondingly, the honest reading of the capacity ladder is narrower than "the minimum rank at which performance recovers measures how much of the advantage is coordinate-frame." At $N=128$ you are fitting a $4096^2$ map from ~128 injected vectors; low ranks will win at low $N$ and high ranks at high $N$ for pure bias–variance reasons, with the basis effect sitting inside that. The ladder plus `Oracle` is what disentangles them. Say so in the write-up rather than letting a reviewer say it for you.

---

## 4. Phase 0 — Audit before spending GPU time

**Goal:** know exactly which existing runs are reusable, before any of the budget is committed. *Est. 1.0 h.*

- [ ] Enumerate `finished_notebooks/` runs and record, per run: explainer model, capacity arm, `N_TRAIN`, seed, data ordering, LoRA config, eval set and its size.
- [ ] Decide per run: reusable as-is / reusable with re-eval / must rerun.
- [ ] **Explicitly check whether any `E_self` run already has a $d\times d$ input map.** If not — and it probably does not — then `Cfull` under `R-id` is a *fresh run*, and the "identity condition is free" assumption from the previous draft is false for exactly the arm that matters. Only the `E_cross` projector runs are genuinely free.
- [ ] Record the eval-set size $n$ and compute the binomial 95% CI half-width for the exact-match metric at that $n$. **Any effect smaller than this is not observable and must not be pre-registered as a reading.**
- [ ] Establish the numerical noise floor: run $M$ twice on the same 256 sequences at different batch sizes, report mean per-token KL. Phase 1's gate is calibrated against this number, not against an absolute.

**Exit criteria:** a table of reusable runs, a stated eval $n$ and CI, a measured KL noise floor. If the CI half-width is larger than the effect sizes in the paper's Table 2 margins, stop and enlarge the eval set before proceeding.

---

## 5. Phase 1 — Build $M_Q$ and prove it is the same function

**Goal:** a rotated model that is provably behaviorally identical, establishing that $Qv$ is a real activation. *Est. 3.5 h.* Construction details in Appendix C; do not re-derive from scratch, the SliceGPT reference implementation (Ashkboos et al., 2024) exists.

**Debug on Qwen3-0.6B, not 8B.** It is the same architecture family, it fits comfortably, and iteration is minutes rather than tens of minutes. Tied embeddings are not an obstacle (Appendix C.3). Promote to 8B only once 0.6B passes the gate.

### 5.1 Fold the RMSNorm gains

- [ ] Fold `input_layernorm` gains into `q_proj`/`k_proj`/`v_proj` of the same block.
- [ ] Fold `post_attention_layernorm` gains into `gate_proj`/`up_proj`.
- [ ] Fold final `model.norm` into `lm_head`. **This is the step that breaks weight tying** — materialize `lm_head` as its own tensor first.
- [ ] Confirm all folded gains are set to $\mathbf{1}$.
- [ ] **Verify logits unchanged with folding alone, before $Q$ exists.** Debugging two transforms at once is the most reliable way to lose an afternoon.

### 5.2 Apply $Q$

- [ ] Sample $Q \in O(4096)$ via QR of a Gaussian in float64. Record the seed; checkpoint $Q$.
- [ ] Write paths (left-multiply): `o_proj`, `down_proj` $\leftarrow QW$; embedding `[vocab, d]` $\leftarrow EQ^\top$.
- [ ] Read paths (right-multiply): `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj` $\leftarrow WQ^\top$; `lm_head` `[vocab, d]` $\leftarrow W_UQ^\top$.
- [ ] Confirm no remaining module reads or writes the residual stream unrotated — including any linear biases, if the config has them.
- [ ] Build and apply in float64, cast to bf16 last. **No quantization on rotated runs** — invariance is exact in fp64, close in bf16, meaningless in 4-bit.

### 5.3 Gate: invariance check

**Do not proceed without a pass.** Highest-value 30 minutes in the project.

- [ ] 256 FineWeb sequences at 512 tokens; logits under $M$ and $M_Q$.
- [ ] Report mean per-token KL, max absolute logit delta, top-1 agreement.
- [ ] **Accept relative to the Phase 0 noise floor**, not an absolute bar: KL within ~1 order of magnitude of the same-model reruns, top-1 agreement $\geq 99.9\%$. Dense $Q$ destroys whatever near-zero structure the weights had, so bf16 error will exceed the unrotated model's own — a fixed $10^{-4}$ threshold may be unreachable for reasons that are not bugs.
- [ ] Second check: recompute has-changed labels for ~200 patching examples under $M_Q$ and confirm they match cache. This validates the transform through the hook path you actually use.

**Failure triage, in likelihood order:** a missed gain fold, a transposed `[vocab, d]` convention, an unnoticed tied tensor, a bias term reading the residual.

**Exit criteria:** gate passes on 8B; $Q$ and its seed checkpointed; gate numbers written down for the report (§10, deliverable 6).

---

## 6. Phase 2 — Rotated activation dataset and the exactness checks

**Goal:** the rotated inputs the experiment actually consumes, plus the two checks that catch plumbing bugs for free. *Est. 1.0 h.*

- [ ] Load cached tensors from `act_patch_qwen3_8b_counterfact`, apply $v \mapsto Qv$, write out as a sibling dataset. Because $M_Q \equiv M$, **every `has_changed` label and every content string is unchanged and the counterfact patching pipeline does not need rerunning.**
- [ ] Spot-check that the rotated cache matches activations extracted live from $M_Q$ on ~32 examples, within bf16 tolerance. This is what licenses the offline shortcut.
- [ ] **Exactness check 1 (the `Oracle` identity):** run `Oracle`/`R-Q` and `C0`/`R-id` with matched seeds and data order. They are algebraically the same computation. If the metrics differ beyond bf16 noise, stop and debug — everything downstream is contaminated.
- [ ] Note the injection-point property: a single global $Q$ means one $Q^\top$ inverts the frame at *every* layer, so multi-layer injection needs no per-layer bookkeeping.

**Exit criteria:** rotated dataset written; live-vs-offline spot check passes; `Oracle`/`R-Q` $\equiv$ `C0`/`R-id`.

---

## 7. Phase 3 — The measurement phases

### 7.1 Phase 3a — Core: does the advantage survive rotation? *Est. 4.0 h*

The irreducible experiment. Everything else is elaboration.

- [ ] `E_self` / `Cfull` / `R-id` across the full `N_TRAIN` sweep.
- [ ] `E_self` / `Cfull` / `R-Q` across the full sweep.
- [ ] `E_cross` / `P-rand` across the full sweep — **the baseline that makes the other two interpretable.**
- [ ] 3 seeds at $N \in \{512, 8192\}$ for all three; plot with seed bands.
- [ ] `E_self` / `Cfull-rand` / {`R-id`, `R-Q`} at $N \in \{512, 8192\}$ — controls for the init head start.

**Exit criteria:** one figure, three curves plus bands, with the cross-model baseline drawn as a horizontal reference at each $N$. This figure alone answers the question.

### 7.2 Phase 3b — Capacity ladder *Est. 2.0 h*

- [ ] `E_self` / {`C0`, `C64`, `C512`} / `R-Q` at $N \in \{512, 8192\}$.
- [ ] `E_self` / `Oracle` / `R-Q` at the same $N$ — the representability ceiling.
- [ ] Plot score vs input-map rank at fixed $N$, with `Oracle` and `Cfull`/`R-id` as horizontal references.
- [ ] Report the gap `Oracle` − `Cfull`/`R-Q` as the *learnability* cost, distinct from the representability cost. State the bias–variance caveat from §3.1 in the figure caption.
- [ ] Add `C8` if time permits; it is the cheapest arm and pins the low end.

### 7.3 Phase 3c — Distributional control (optional) *Est. 0.5 h*

$Q$ preserves norms, angles, anisotropy, effective rank, and the entire covariance spectrum — it destroys *coordinate* correspondence specifically. A version of basis compatibility in which $E$ benefits because $M$'s activations are merely distributionally *shaped* like its own would survive $Q$ untouched. `R-Qs` breaks that too, and the gap between `R-Q` and `R-Qs` isolates coordinate alignment from distributional compatibility.

- [ ] Apply `R-Qs` **as an input-side transform on the cached activations only.**

> **Correction to the previous draft.** $\mathrm{diag}(s)$ does **not** commute with $x/\mathrm{rms}(x)$ — per-token RMS changes non-uniformly — so there is no weight-level $M_{Qs}$ computing the same function. Listing `R-Qs` alongside `R-Q` as a model transform would have silently destroyed the one property that makes the wedge clean. As an input-side perturbation it is perfectly coherent: labels still come from unmodified $M$, and the claim is about what $E$ can decode, not about a second model. Say this explicitly, or cut the arm.

### 7.4 Phase 3d — The constructive arm: price the alternative recipe *Est. 2.0 h*

This is where the "so what" gets teeth, and it is where the practical result lives.

The paper half-shows this: Llama-3.1-70B with a pretrained projection recovers 14% on SAE explanations, 2.7× on real activations, 1.7× on activation differences relative to random init — reported as evidence *for* privileged access rather than as the alternative recipe it implies. Meanwhile Qwen3-8B, the main cross-model comparison in Table 1, was only ever run with a **randomly initialized** projector, so the headline cross-model gap and the projector-quality effect are confounded.

- [ ] Cache paired activations $h^E_{\ell,t}(x), h^M_{\ell,t}(x)$ on a few thousand FineWeb sequences. Requires a shared tokenizer for token alignment — true within Qwen3.
- [ ] Decide and record the layer correspondence: `E_cross` at 4B has fewer blocks than $M$ at 8B, so the $\ell \leftrightarrow \ell$ mapping is a design choice, not a given. State the rule (proportional depth is the obvious default).
- [ ] Fit $\Pi_\ell$ in closed form — ridge regression, not a training run, minutes on cached tensors, no SGD and no hyperparameter search beyond $\lambda$:
  $$\Pi_\ell = H^E (H^M)^\top \left(H^M (H^M)^\top + \lambda I\right)^{-1}$$
- [ ] **`P-ridge-frozen`: evaluate the ridge-mapped activations through an explainer that is never fine-tuned on this target.** This is the arm that actually establishes §1's cost claim. Training a cross-model explainer from ridge init is still a per-target training run and does *not* show "you only need a regression." Closed-form fit plus eval — essentially free, and it is the headline practical deliverable.
- [ ] `P-ridge`: train from ridge init and sweep $N$, for comparison against `P-rand` and against `E_self`.

**The free analytic result — check it numerically, then state it.** Fitting the ridge map against rotated activations gives *exactly* $\Pi Q^\top$:

$$\Pi^{Q} = H^E (QH^M)^\top\!\left(QH^M H^{M\top}\!Q^\top + \lambda I\right)^{-1} = H^E H^{M\top}\!\left(H^M H^{M\top} + \lambda I\right)^{-1}\!Q^\top = \Pi Q^\top$$

- [ ] Verify $\lVert \Pi^{Q} - \Pi Q^\top \rVert_F \approx 0$ numerically as a unit test.
- [ ] Report the implication: **the closed-form recipe is provably as good on rotated activations as on unrotated ones.** So if ridge closes the gap, the coordinate frame is not merely the mechanism — it is fully and cheaply recoverable, shown analytically and numerically rather than by a training curve.

### 7.5 Phase 3e — Diagnostics and controls *Est. 1.0 h*

- [ ] **Learned map vs $Q^\top$.** After training `Cfull`/`R-Q`, report $\lVert \Pi Q - I \rVert_F / \sqrt{d}$ and the principal angles between the row spaces of $\Pi$ and $Q^\top$. If the model recovered by learning approximately $Q^\top$, say so with numbers — it converts a performance claim into a mechanistic one.
- [ ] **Input-ablation invariance control.** Input ablation passes no activation, so rotation cannot touch it. Run it under both `R-id` and `R-Q`. Movement beyond the seed band is a bug; movement *within* the band is expected if the explainer is retrained, so state the tolerance rather than "must not move at all."

---

## 8. Budget

| Phase | Task | Hours |
|---|---|---|
| 0 | Audit reusable runs, eval CI, KL noise floor | 1.0 |
| 1 | Fold gains, apply $Q$, invariance gate (debug on 0.6B) | 3.5 |
| 2 | Rotated dataset + `Oracle`$\equiv$`C0` exactness checks | 1.0 |
| 3a | **Core:** `Cfull` × {`R-id`, `R-Q`} + cross-model baseline, full sweep, seeds | 4.0 |
| 3b | Capacity ladder + `Oracle` | 2.0 |
| 3c | Distributional control (optional) | 0.5 |
| 3d | Ridge fit, `P-ridge-frozen`, `P-ridge` | 2.0 |
| 3e | Learned-map diagnostic, input-ablation control | 1.0 |
| — | **Buffer** (see below) | 2.0 |
| 4 | Figures, write-up | 2.5 |
| | **Total** | **19.5** |

**The buffer is a line item, not slack to be optimistically reclaimed.** The near-universal failure mode in this kind of work is a transposed convention at the gate costing three hours. Phase 0's audit and the 0.6B debug path exist to shrink that risk; the buffer exists because it does not go to zero.

**Cut order, in order:**

1. `R-Qs` distributional arm (Phase 3c).
2. Intermediate ladder ranks — keep `C0`, `Cfull`, `Oracle`; drop `C64`/`C512`.
3. `Cfull-rand` init control.
4. **Phase 1 on 8B.** Demonstrate weight-level invariance on Qwen3-0.6B only, run the science off the rotated cache, and present the 8B invariance as an argument supported by a smaller-scale check. This is the most compressible 3.5 hours in the plan and the previous draft did not treat it as cuttable — §2.2 is why it is.
5. Extra $Q$ seeds. **Never cut training seeds before $Q$ seeds** — $Q$ seeds test robustness to one arbitrary choice; training seeds are what make every figure readable.
6. Low end of the $N$ sweep.

**Irreducible core (~8 h):** Phase 0, Phase 1 on 0.6B, Phase 2, and `Cfull` × {`R-id`, `R-Q`} plus the cross-model baseline at three values of $N$ with 3 seeds. That alone answers the question.

---

## 9. Pre-registered readings

*(Left for the author to revise — flagged issues: rows below are not mutually exclusive, there is no row for "all metrics inside the seed band," there is no row for "metrics disagree with each other," and if any `R-id` results from `finished_notebooks/` have already been inspected, pre-registration is partially compromised for those cells and the write-up should say so.)*

| Observation | Reading |
|---|---|
| `Cfull` curves for `R-id` and `R-Q` overlap at all $N$ | Coordinate alignment contributes nothing; §3.4's correlation is confounded |
| `R-Q` separated at low $N$, converging by 8192 | Basis compatibility is a **sample-efficiency** effect — reframes the paper's §4 data-efficiency claim as the real finding |
| `R-Q` strictly below `R-id` at all $N$, gap $>$ seed band | Basis compatibility affects the attainable optimum; strongest form of the critique |
| `R-Q` falls to the `E_cross`/`P-rand` baseline | Table 1's ordering is a statement about geometry, not selfhood |
| Recovery threshold at low rank (`C8`/`C64`) | The frame mismatch is low-dimensional; suggests a very cheap alignment recipe |
| `Oracle` $\gg$ `Cfull`/`R-Q` at high $N$ | The map is representable but not learnable at this $N$; the ladder is measuring optimization, not basis |
| `P-ridge-frozen` matches or approaches `E_self` | The alternative recipe works with no per-target training; primary practical result |
| Input-ablation control moves beyond the seed band | Bug. Stop. |

Row 2 is the modal outcome. Plan the write-up around curve *shape*, not a headline number.

---

## 10. Deliverables

1. `rotate.py` — gain folding, $Q$ construction, weight transform, invariance gate as a test.
2. Rotated activation dataset; $Q$ and seed checkpointed.
3. Sweep curves per metric, `R-id` vs `R-Q` vs cross-model baseline, with seed bands, faceted by capacity arm.
4. Capacity-recovery plot: score vs input-map rank at fixed $N$, with `Oracle` reference line.
5. Ridge-projector comparison — `P-ridge-frozen` against `E_self`, plus the $\Pi^Q = \Pi Q^\top$ numerical check.
6. Invariance-gate numbers and the Phase 0 noise floor reported in the write-up, not buried in a log.
7. Honest failure log — whatever ate three hours, write it down.
8. A stated position on which row of §9 the results landed on, and what it implies for anyone planning to use self-explanation in practice.

**Negative-result path — decide now, not at hour 17.** If the gate fails on 8B, or if every arm sits inside the seed band, the deliverable is still a paper-shaped object: the invariance construction as a reusable tool, the measured effect-size floor at this eval size, and the statement *"the coordinate-frame contribution is smaller than run-to-run variance at this scale, and here is the $N$ at which it would become detectable."* That is a real result about measurement sensitivity, and writing it confidently is worth more than an inconclusive hedge.

**Executive summary leads with §1's consequence and Phase 3d's frozen-explainer result.** The rotation is the method; neither of those is the method.

---
---

# Appendix A — On full SFT: don't, and say why

The paper full-parameter fine-tunes its same-dimension explainers. The natural question is whether to match that.

**Single A100 80GB, Qwen3-8B, AdamW:** 16GB bf16 weights + 16GB gradients + 64GB optimizer states + 32GB fp32 master ≈ 128GB. Does not fit. With 8-bit Adam, gradient checkpointing, batch size 1 and short sequences you land around 50–60GB — technically feasible, slow, and fragile. **Multi-GPU pod (4–8× A100):** fine under FSDP or ZeRO-3, but the config debugging is a real time sink against the budget and it buys nothing scientific.

The reason to skip it is not just cost. The only thing full SFT gives you *that matters here* is a full-rank trainable path from $v$ into the explainer — and the `Cfull` arm gets that for 16.8M parameters, under 2% of the LoRA-adjacent budget, in precisely the parameter class where $Q^\top$ lives. Spending the pod on full SFT converts an explicit, measurable variable back into an implicit one. If the multi-GPU pod is available anyway, spend it running more arms of §3's matrix in parallel, not on making one arm heavier.

State this reasoning in the write-up. "I could have matched the paper's training setup and chose not to, for this reason" is a prioritization signal; silently using LoRA is not.

---

# Appendix B — On porting to Llama: no

Tempting, since Llama-3.1-8B is the paper's actual Table 1 target and refuting it there would land harder. Don't, for this application.

The Table 1 result is the **feature description** task, which the repo doesn't implement. Reproducing it means building the autointerp pipeline, the trained simulator, LlamaScope SAE ingestion, and Neuronpedia label handling — days, not hours. Restricting to patching instead means regenerating the entire counterfact dataset for a new target, which is exactly the cost Phase 2 was designed to avoid, plus HF gating on Llama-3.1-8B.

Architecturally Llama is marginally *easier* — RMSNorm, no QK-norm — but that saves 30 minutes against many hours of data regeneration. And a second architecture is a robustness check, not a new claim. The capacity ladder and the ridge-projector arm both produce claims. Spend the hours there.

Say this explicitly in the write-up. Choosing not to chase the more impressive-sounding version because the marginal evidence per hour is worse is the kind of call the application is testing for.

---

# Appendix C — Rotation construction reference

Standard computational invariance, as in SliceGPT (Ashkboos et al., 2024). Reference implementation exists; don't derive it from scratch.

## C.1 Why folding is required

$\mathrm{RMSNorm}(x) = \frac{x}{\mathrm{rms}(x)} \odot g$. The elementwise gain doesn't commute with $Q$, so absorb it into every linear map that consumes the normalized output: $W \leftarrow W\,\mathrm{diag}(g)$, $g \leftarrow \mathbf{1}$. After folding, normalization is pure $x/\mathrm{rms}(x)$, which commutes with any orthogonal $Q$ since $\lVert Qx \rVert = \lVert x \rVert$. Then $WQ^\top(Qh) = Wh$ everywhere.

## C.2 Qwen3-specific notes

- **QK-norm.** Qwen3 RMSNorms head-dimension slices of $W_Qh$ and $W_Kh$, *inside* the head and downstream of the read matrix. Unaffected. Do not fold or touch these gains.
- **RoPE.** Operates on head dimensions post-projection. Unaffected.
- **GQA.** `k_proj`/`v_proj` have fewer output heads but full input dimension $d$, so $WQ^\top$ is well-formed. No special handling.
- **Biases.** Qwen3 is expected to have no linear biases in the attention projections (QK-norm replaces that role), but verify against `config.json` — a bias on a residual-write path would need $b \leftarrow Qb$.

## C.3 Tied embeddings are not a blocker — correcting the previous draft

The earlier plan claimed that `tie_word_embeddings: true` on Qwen3-0.6B/1.7B/4B prevents rotation, "which is why the target must be 8B." That reasoning is wrong.

Both tensors receive the *same* transform. Embedding rows write into the residual: row $e^\top \mapsto (Qe)^\top = e^\top Q^\top$, so $E \leftarrow EQ^\top$. The unembedding reads the residual: $W_U(Qh) = W_h$ requires $W_U \leftarrow W_UQ^\top$. Identical operation — a shared tensor is fine, applied once.

What actually breaks the tie is the **final-norm gain fold**: $W_U \leftarrow W_U\,\mathrm{diag}(g)$ hits `lm_head` but not the embedding. And that is not a blocker either — untie by materializing `lm_head` as its own tensor before folding.

The practical consequence matters: **Qwen3-0.6B is available as a fast substrate for debugging the fold and the gate**, which is where Phase 1's hours will actually go. The target remains 8B, but for the data reason in §3 (the cached counterfact dataset), not an architectural one.

---

# Appendix D — The base/instruct arm, and why it is not the dual

Rotation kills basis while holding behavior fixed. The dual — kill behavior while holding basis fixed — would complete the argument.

The tempting cheap version: run Qwen3-8B-Base and Qwen3-8B-Instruct as explainers of each other on input ablation. They share a frame almost exactly (the paper measures 85% dot-product similarity for the Llama instruct pair) but differ behaviorally, and relabelling is cheap — regenerating MMLU responses is what `iteration.ipynb` already does.

**But this does not bracket the question, and the previous draft's claim that it does was wrong.** Input ablation passes no activation, so alignment is *inert* on that task — it is not a variable that can take the value 0.85 there. The rotation point (behavioral divergence 0, alignment 0, patching task) and the base/instruct point (behavioral divergence > 0, patching-task alignment ≈ 0.85, but measured on input ablation) do not sit on the same surface and cannot be compared.

The genuine dual requires base/instruct on **patching**, which means regenerating every ground-truth patching label for the base model — the exact cost Phase 2 exists to avoid.

The base/instruct-on-input-ablation experiment is still worth ~3 hours if they exist, but as its own claim: *isolating the contribution of behavioral similarity on a task where basis compatibility cannot operate by construction.* There is a live thread there — the paper's Table 2 shows the Qwen3 self-margin is **larger** on input ablation (83.4 vs 58.1 exact match) than on patching (64.0 vs 54.1). The task with no internals shows the bigger effect. That is already awkward for the privileged-access reading and nobody has followed it up.

---

# Appendix E — Future work

The rigorous version of the behavioral dual: LoRA-tune $M'$ from $M$ with a dot-product alignment term in the objective, producing a model that is behaviorally divergent but frame-matched by construction. It does not fit this budget, and naming why is worth more than doing it badly:

- The regularizer fights the behavioral objective with no principled $\lambda$.
- Mean dot-product is a weak proxy for "same basis" — it is exactly the quantity rotation shows can be destroyed without touching function.
- Every ground-truth patching label must be regenerated per $M'$.
