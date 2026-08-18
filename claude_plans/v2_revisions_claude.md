# Revised sections — drop-in replacements for v2

Rewritten in light of a direct audit of `TransluceAI/introspective-interp`. Evidence is in the new **Appendix F**; §§3, 4, 7 and the affected rows of §9 all now cite it rather than inferring from the paper's prose.

Three findings drive the changes:

1. **The paper's activation-patching configs create no projector at all** — on either side. `C0` is therefore not a strawman arm, it is the paper's actual configuration.
2. **Under LoRA, projectors are LoRA target modules** at rank 128. A "full-rank $\Pi_\ell$" is not full-rank unless you explicitly exclude it. This is an implementation trap that would silently invalidate the central arm.
3. **A pretrained Qwen projector exists in the repo** and is loaded by the default config. The v2 claim that Qwen was "only ever run with a randomly initialized projector" is wrong — but what replaces it is stronger, and it reframes Phase 3d around a much sharper hypothesis.

---

## 3. The arm matrix

Everything below refers to this table. Write it into the repo README before running anything; the combinatorics are where budgets die.

**Target** $M$ = Qwen3-8B, throughout. The reason is data, not architecture: the cached `act_patch_qwen3_8b_counterfact` dataset exists for this target and Phase 2 reuses it wholesale. (Tied embeddings are *not* a constraint — see Appendix C.3.)

### 3.0 What the paper's configuration actually is

This determines what counts as a faithful arm and what counts as an addition, so it comes first.

For activation patching, all four of the paper's configs (`config/act_patch/*.yaml`) omit `use_embed_proj`. `train.py:76` reads `config.get("use_embed_proj", False)` and passes the result explicitly to the constructor, so the `False` default overrides `ContinuousQwen`'s class-level default of `True`. And because Qwen3-8B and Llama-3.1-8B are both 4096-dimensional, the dimension-mismatch fallback at `continuous_base.py:120` does not fire either. Result: `embed_projs = None`.

**So on the patching task the target activation is injected raw into the explainer's embedding layer, with no learned map, on both the self and cross-model sides.** Two consequences:

- The comparison in the paper's Tables 2 and 5 is *symmetric*. There is no projector asymmetry to indict, and any suggestion of one should be struck.
- The only trainable adaptation between $v$ and the explainer is rank-128 LoRA on downstream weights (`peft_lora: true`, `lora_r: 128`). That configuration is exactly `C0` below. **It cannot represent $Q^{-1}$, and that is a fact about the paper's setup, not an artifact of ours.**

This reframes the capacity ladder. `C0` is the faithful replication; `Cfull` is a deliberate augmentation introduced *so that the question becomes answerable at all*. Both belong in the results, labelled that way.

### 3.1 Explainers

| Tag | Model | Activation path | Init |
|---|---|---|---|
| `E_self` | Qwen3-8B | direct injection; capacity arm per §3.3 | from unrotated $M$ |
| `E_cross` | Qwen3-4B (primary), 1.7B (secondary) | per-layer $\Pi_\ell$, **explicitly excluded from LoRA** | independent pretrained weights |

> **Implementation trap — read before writing any training code.** `model/utils.py:252–253` appends `embed_projs.{i}` to the LoRA `target_modules` list whenever the projector is trainable. Under `peft_lora: true` with `lora_r: 128`, a projector is therefore **frozen at its initialization plus a rank-128 update** — not full-rank trainable. The paper's footnote 7 says exactly this and is accurate.
>
> For $d = 4096$, rank 128 cannot approximate $Q^\top$. If `Cfull` or `E_cross`'s $\Pi_\ell$ ends up inside `target_modules`, the central arm silently becomes `C128` and the whole experiment measures LoRA rank. **Exclude it and set `requires_grad_(True)` directly, then assert the trainable parameter count matches $32 \times 4096^2 \approx 537$M (or $d_E \times d_M$ per layer for `E_cross`) before training.** Add that assertion as a test, not a comment.

This also means v2's description of `E_cross` as having "per-layer full-rank $\Pi_\ell$" was true only by accident of how the replication was configured. Verify it in Phase 0 rather than assuming.

### 3.2 Rotation condition (applied to activations handed to $E$)

| Tag | Transform |
|---|---|
| `R-id` | $v \mapsto v$ |
| `R-Q` | $v \mapsto Qv$, single global $Q \in O(4096)$ |
| `R-Qs` | $v \mapsto \mathrm{diag}(s)\,Qv$, $s$ log-uniform — **input-side only**, see §7.3 |

### 3.3 Capacity ladder (input map on the injected $v$, `E_self` only)

| Arm | Input map | Can represent $Q^{-1}$? | Trainable? | Status |
|---|---|---|---|---|
| `C0` | none | No | — | **the paper's configuration** |
| `C128` | rank 128 | No | Yes | what `Cfull` degrades to if LoRA-wrapped |
| `C8` / `C512` | rank 8 / 512 | No / Partially | Yes | ladder interior |
| `Cfull` | full-rank $d\times d$, init $I$ | Exactly | Yes, full-rank | primary augmented arm |
| `Cfull-rand` | full-rank $d\times d$, random orthogonal init | Exactly | Yes, full-rank | init control |
| `Oracle` | frozen at $Q^\top$ | Exactly | **No** | representability ceiling |

`C128` earns a row because it is not hypothetical — it is the failure mode of a plausible implementation, and running it deliberately turns a possible silent bug into a measured data point. It is also the rank the paper itself uses everywhere, which makes it the right comparison for any claim about the paper's own numbers.

### 3.4 Projector init (`E_cross` only)

| Tag | Meaning |
|---|---|
| `P-rand` | random init + rank-128 update — **the paper's cross-model condition, faithfully** |
| `P-rand-full` | random init, full-rank trainable |
| `P-ridge` | closed-form ridge init, then train |
| `P-ridge-frozen` | closed-form ridge fit, **explainer never trained on this target** |

`P-rand` vs `P-rand-full` is a new arm and it is cheap. Its purpose is in §7.4: it separates "random initialization" from "random initialization *under a rank cap*," and those are very different handicaps.

**Sweep.** `N_TRAIN` $\in [128, 256, 512, 1024, 2048, 4096, 8192]$, matched data ordering across arms.

**Seeds.** 3 training seeds at $N \in \{512, 8192\}$ for every core arm; 1 seed elsewhere. Every figure carries the seed band. Non-negotiable — §9's readings all turn on "curves overlap" vs "separated," and neither is decidable without it.

### 3.5 Why `Cfull-rand`, `Oracle`, and the cross-model baseline all exist

`Cfull` under `R-id` is initialized at identity — that is, **at the solution**. `Cfull` under `R-Q` is initialized at identity when the truth is $Q^\top$. So "R-id separates from R-Q at low $N$ and converges at high $N$" is close to a mathematical consequence of the initialization, not a finding about self-explanation. Three counters, in increasing order of importance:

- **`Cfull-rand`** puts both rotation arms the same distance from their solutions at init, so a residual gap is not an artifact of the head start.
- **`Oracle`** separates *representability* from *learnability*, and gives an exact bug check: `Oracle` under `R-Q` is algebraically identical to `C0` under `R-id`, since $Q^\top(Qv) = v$ and the rest of the network is untouched. If those two runs don't agree to bf16 noise, there is a plumbing bug — find it before spending the sweep.
- **The cross-model baseline is the yardstick.** "R-id beats R-Q" is uninterpretable alone. "R-Q falls to where an unrelated 4B model sits" is the claim. `E_cross` under `P-rand` is a first-class budgeted arm, not a number quoted from the paper.

Correspondingly, the honest reading of the ladder is narrower than "the minimum rank at which performance recovers measures how much of the advantage is coordinate-frame." At $N=128$ you are fitting a $4096^2$ map from ~128 injected vectors; low ranks will win at low $N$ and high ranks at high $N$ for pure bias–variance reasons, with the basis effect sitting inside that. The ladder plus `Oracle` is what disentangles them. Say so in the write-up rather than letting a reviewer say it for you.

---

## 4. Phase 0 — Audit before spending GPU time

**Goal:** know exactly which existing runs are reusable, and confirm the capacity plumbing, before any of the budget is committed. *Est. 1.5 h.*

### 4.1 Reusable-run inventory

- [ ] Enumerate `finished_notebooks/` runs and record, per run: explainer model, capacity arm, `N_TRAIN`, seed, data ordering, LoRA config, eval set and its size.
- [ ] Decide per run: reusable as-is / reusable with re-eval / must rerun.
- [ ] **`Cfull` under `R-id` is a fresh run. Assume nothing else.** Appendix F.1 establishes that the paper's patching configs build no projector, so no replication of them can contain a $d\times d$ input map. v2's "the identity condition is free" is false for exactly the arm that matters. Only `E_cross` projector runs are candidates for reuse.
- [ ] **Audit your own replication's `target_modules`.** If your per-layer $\Pi_\ell$ for the 1.7B/4B explainers was passed through PEFT rather than trained directly, those existing numbers are rank-capped and every description of them as "full-rank" needs correcting — in the write-up and in the repo README. This is a five-minute check that determines whether hours of prior work is reusable.

### 4.2 Measurement floor

- [ ] Record the eval-set size $n$ and compute the binomial 95% CI half-width for exact match at that $n$. **Any effect smaller than this is not observable and must not be pre-registered as a reading.**
- [ ] Establish the numerical noise floor: run $M$ twice on the same 256 sequences at different batch sizes, report mean per-token KL. Phase 1's gate is calibrated against this number, not an absolute.
- [ ] Compute the **task's own dynamic range** before deciding the sweep is worth running. Rotation can only destroy the value of $v$; it cannot degrade what the explainer gets from the prompt. The paper's `– activation` ablation (Table 5) bounds that value at $64.0 \to 59.9$ exact match for Qwen3-8B self-explanation — about 4.1 points against a self-vs-cross margin of ~10. So the entire measurable range of the rotation arm is ~4 points, and **all results should be reported normalized to the no-activation floor**: "rotation destroyed $X\%$ of what the activation was contributing." If your eval CI half-width is not comfortably under 4 points, enlarge the eval set now.

**Exit criteria:** a table of reusable runs with the `target_modules` question settled, a stated eval $n$ and CI, a measured KL noise floor, and the no-activation floor established (rerun it yourself at your $N$ rather than quoting 59.9 — the paper's number is at their $N$ and eval set).

---

## 7. Phase 3 — The measurement phases

### 7.1 Phase 3a — Core: does the advantage survive rotation? *Est. 4.0 h*

The irreducible experiment. Everything else is elaboration.

- [ ] `E_self` / `Cfull` / `R-id` across the full `N_TRAIN` sweep.
- [ ] `E_self` / `Cfull` / `R-Q` across the full sweep.
- [ ] `E_self` / `C0` / {`R-id`, `R-Q`} at $N \in \{512, 8192\}$ — **the paper's own configuration under rotation.** Expect collapse; report it as a capacity result, explicitly labelled, per §3.0. This arm exists so that a reader who asks "what happens in the setup the paper actually used?" gets an answer rather than an argument.
- [ ] `E_cross` / `P-rand` across the full sweep — **the baseline that makes everything else interpretable.**
- [ ] 3 seeds at $N \in \{512, 8192\}$ for all of the above; plot with seed bands.
- [ ] `E_self` / `Cfull-rand` / {`R-id`, `R-Q`} at $N \in \{512, 8192\}$ — controls for the init head start.
- [ ] Normalize every curve against the no-activation floor from Phase 0 §4.2, and show the raw scale alongside.

**Exit criteria:** one figure — `Cfull` under both rotation conditions, `C0` under both, cross-model baseline as a horizontal reference at each $N$, seed bands throughout, second axis showing fraction-of-activation-contribution-retained.

### 7.2 Phase 3b — Capacity ladder *Est. 2.0 h*

- [ ] Assert full-rank trainability before running anything: parameter count check per §3.1's trap box.
- [ ] `E_self` / {`C8`, `C128`, `C512`} / `R-Q` at $N \in \{512, 8192\}$.
- [ ] `E_self` / `Oracle` / `R-Q` at the same $N$ — the representability ceiling.
- [ ] Plot score vs input-map rank at fixed $N$, with `Oracle`, `Cfull`/`R-id`, and the no-activation floor as horizontal references.
- [ ] Report the gap `Oracle` − `Cfull`/`R-Q` as the *learnability* cost, distinct from the representability cost. State the bias–variance caveat from §3.5 in the figure caption.
- [ ] Mark `C0` and `C128` on the rank axis as "the paper's configuration" — this is the single most legible way to show why the ladder is necessary rather than ornamental.

### 7.3 Phase 3c — Distributional control (optional) *Est. 0.5 h*

$Q$ preserves norms, angles, anisotropy, effective rank, and the entire covariance spectrum — it destroys *coordinate* correspondence specifically. A version of basis compatibility in which $E$ benefits because $M$'s activations are merely distributionally *shaped* like its own would survive $Q$ untouched. `R-Qs` breaks that too, and the gap between `R-Q` and `R-Qs` isolates coordinate alignment from distributional compatibility.

- [ ] Apply `R-Qs` **as an input-side transform on the cached activations only.**

> $\mathrm{diag}(s)$ does **not** commute with $x/\mathrm{rms}(x)$ — per-token RMS changes non-uniformly — so there is no weight-level $M_{Qs}$ computing the same function. Listing `R-Qs` alongside `R-Q` as a model transform would silently destroy the one property that makes the wedge clean. As an input-side perturbation it is coherent: labels still come from unmodified $M$, and the claim is about what $E$ can decode, not about a second model. Say this explicitly, or cut the arm.

### 7.4 Phase 3d — The constructive arm, and the sharpest available claim *Est. 2.5 h*

**This section is materially rewritten. The v2 motivation was factually wrong and the correct version is stronger.**

v2 asserted that Qwen3-8B was only ever run with a randomly initialized projector. The repo says otherwise. `config/feature_descriptions/qwen_131k.yaml` sets `use_embed_proj: true` *and* loads `alignment_outputs/qwen_llama_3.1_8b_base/final_alignment_model.pt`, writing to a checkpoint directory named `aligned_qwen_pretrained`. The random-init condition is a separate config, `qwen_131k_nonpretrainedembed.yaml`, writing to `nonaligned_qwen`. Full evidence in Appendix F.2.

Three things follow, and they are better than the claim they replace.

**(a) The paper's description of its own method is inaccurate.** §3.1 states that pretrained projections were included only for Llama-3.1-70B. The code shows a pretrained Qwen projector was built and used. Worth an issue on the repo; worth a sentence in the write-up either way.

**(b) The most direct available test of basis compatibility was run and never reported.** `aligned_qwen_pretrained` vs `nonaligned_qwen` is same explainer, same target, same data, alignment as the only variable — precisely the contrast this whole project exists to construct by other means. It sits unreported in the config directory. And Table 1's Qwen row may not be internally consistent about which condition it reflects: only the LM-judge config carries the pretrained path, while `_simcor`, `_ood_fw`, and `_ood_diff` do not and point at differently-named checkpoint directories, with `peft_lora` differing across them too.

**(c) The new hypothesis, which is cheap to test and sharper than anything in v2.** Because projectors are LoRA target modules at rank 128 (§3.1), a *randomly initialized* $4096 \times 4096$ projector under LoRA is close to unfixable — rank 128 cannot correct a random dense map — whereas a *pretrained* one only needs refinement. So the paper's §3.4 finding, stated as "activation alignment improves explainer performance," may substantially be:

> **Under low-rank training, projector initialization dominates, and the random-vs-pretrained gap is largely a rank-128 artifact rather than evidence about activation alignment.**

That would reinterpret the paper's own central mechanistic argument, and the 70B random-vs-pretrained numbers it rests on (14% on SAE, 2.7× on activations, 1.7× on differences) become a measurement of rank capacity. It is testable with one extra cheap arm.

- [ ] **`P-rand` vs `P-rand-full`.** Same random init, one under rank-128 LoRA and one full-rank trainable. If `P-rand-full` closes most of the gap to `P-ridge`, hypothesis (c) is supported and the paper's alignment-vs-performance correlation needs restating. This is the highest-value-per-hour arm in the plan.
- [ ] Cache paired activations $h^E_{\ell,t}(x), h^M_{\ell,t}(x)$ on a few thousand FineWeb sequences. Requires a shared tokenizer for token alignment — true within Qwen3.
- [ ] Record the layer correspondence rule: `E_cross` at 4B has fewer blocks than $M$ at 8B, so $\ell \leftrightarrow \ell$ is a design choice. Proportional depth is the obvious default; state it.
- [ ] Fit $\Pi_\ell$ in closed form — ridge regression, minutes on cached tensors, no SGD, no hyperparameter search beyond $\lambda$:
  $$\Pi_\ell = H^E (H^M)^\top \left(H^M (H^M)^\top + \lambda I\right)^{-1}$$
- [ ] **`P-ridge-frozen`: evaluate ridge-mapped activations through an explainer never fine-tuned on this target.** This is the arm that establishes §1's cost claim. Training from ridge init is still a per-target training run and does *not* show "you only need a regression." Closed-form fit plus eval — essentially free, and the headline practical deliverable.
- [ ] `P-ridge`: train from ridge init and sweep $N$, for comparison against `P-rand`, `P-rand-full`, and `E_self`.
- [ ] No alignment-training script shipped with the repo — only the loading path for `LinearAlignmentModule` state dicts keyed `alignments.{layer}.weight` (`model/utils.py:134–156`). Emit your ridge fit in that format so it drops into their loader unmodified. Cheap, and it makes the artifact directly reusable by anyone replicating.

**The free analytic result — check it numerically, then state it.** Fitting the ridge map against rotated activations gives *exactly* $\Pi Q^\top$:

$$\Pi^{Q} = H^E (QH^M)^\top\!\left(QH^M H^{M\top}\!Q^\top + \lambda I\right)^{-1} = H^E H^{M\top}\!\left(H^M H^{M\top} + \lambda I\right)^{-1}\!Q^\top = \Pi Q^\top$$

- [ ] Verify $\lVert \Pi^{Q} - \Pi Q^\top \rVert_F \approx 0$ numerically as a unit test.
- [ ] Report the implication: **the closed-form recipe is provably as good on rotated activations as unrotated.** If ridge closes the gap, the coordinate frame is not merely the mechanism — it is fully and cheaply recoverable, shown analytically and numerically rather than by a training curve. Note the contrast with `C0`/`Cfull` under `R-Q`, where recovery is an optimization problem: the same rotation is free for a closed-form fit and expensive for gradient descent.

### 7.5 Phase 3e — Diagnostics and controls *Est. 1.0 h*

- [ ] **Full-rank assertion.** Log trainable parameter counts per arm and confirm `Cfull` is not inside `target_modules`. A one-line check that protects the entire result.
- [ ] **Learned map vs $Q^\top$.** After training `Cfull`/`R-Q`, report $\lVert \Pi Q - I \rVert_F / \sqrt{d}$ and the principal angles between the row spaces of $\Pi$ and $Q^\top$. If the model recovered by learning approximately $Q^\top$, say so with numbers — it converts a performance claim into a mechanistic one.
- [ ] **Input-ablation invariance control.** Input ablation passes no activation, so rotation cannot touch it. The paper's configs set `use_embed_proj: false` there explicitly, confirming this. Run under both `R-id` and `R-Q`. Movement beyond the seed band is a bug; movement *within* the band is expected if the explainer is retrained, so state the tolerance rather than "must not move at all."

---

## 9. Pre-registered readings — revised rows

Replaces the corresponding rows in v2. Rows are grouped because they are not mutually exclusive; record which fire.

**Primary — does coordinate correspondence matter?**

| Observation | Reading |
|---|---|
| `Cfull` curves for `R-id` and `R-Q` overlap at all $N$, both above the no-activation floor | Coordinate alignment contributes nothing recoverable; §3.4's correlation is confounded |
| `R-Q` separated at low $N$, converging by 8192 | Coordinate frame is a **sample-efficiency** effect — reframes the paper's §4 claim as the real finding |
| `R-Q` strictly below `R-id` at all $N$, gap $>$ seed band | Frame affects the attainable optimum; strongest form of the critique |
| `R-Q` falls to the no-activation floor | The activation's *entire* contribution was coordinate-frame |
| `R-Q` falls below the no-activation floor | Bug, or the rotated vector is actively misleading the explainer — investigate before reporting |
| All arms inside the seed band | Effect is smaller than run-to-run variance at this scale; report the sensitivity result per §10's negative-result path |
| Metrics disagree (e.g. has-changed F1 moves, content match doesn't) | Report all three separately and do not aggregate; state which component the frame affects |

**Capacity — is the collapse about basis or about rank?**

| Observation | Reading |
|---|---|
| `C0` and `C128` collapse under `R-Q`, `Cfull` does not | Expected. The paper's configuration cannot represent $Q^{-1}$; label as capacity, not evidence |
| `Oracle` $\gg$ `Cfull`/`R-Q` at high $N$ | Map is representable but not learnable at this $N$; the ladder is measuring optimization |
| Recovery threshold at low rank (`C8`) | Frame mismatch is low-dimensional; suggests a very cheap alignment recipe |
| `Cfull-rand` gap $\approx$ `Cfull` gap | The `R-id` head start was not driving the result |

**Constructive — does the alternative recipe work?**

| Observation | Reading |
|---|---|
| `P-ridge-frozen` matches or approaches `E_self` | Alternative recipe works with no per-target training; primary practical result |
| `P-rand-full` $\gg$ `P-rand`, approaching `P-ridge` | **The paper's random-vs-pretrained projector gap is largely a rank-128 artifact.** Reinterprets §3.4's mechanism |
| `P-rand-full` $\approx$ `P-rand` | Rank is not the binding constraint; initialization quality genuinely reflects alignment, and the paper's reading stands |
| $\lVert \Pi^Q - \Pi Q^\top\rVert_F \not\approx 0$ | Bug in the ridge implementation. Stop. |

**Integrity checks**

| Observation | Reading |
|---|---|
| `Oracle`/`R-Q` $\neq$ `C0`/`R-id` beyond bf16 noise | Plumbing bug. Stop. |
| Input-ablation control moves beyond the seed band | Bug. Stop. |
| Any `R-id` cell was inspected before pre-registration | Pre-registration is compromised for that cell; say so explicitly in the write-up |

---

# Appendix F — Code audit of `TransluceAI/introspective-interp`

Findings from reading the released code at `main`. These are a deliverable in their own right (§10) — several are not inferable from the paper, and two contradict it.

## F.1 The activation-patching task uses no projector on either side

- `config/act_patch/{base_base, base_qwen, qwen_base, qwen_qwen}_act_patch_cf.yaml` all omit `use_embed_proj`.
- `train.py:76` — `use_embed_proj=config.get("use_embed_proj", False)`.
- `model/utils.py:116` passes that value explicitly to the constructor, so it overrides `ContinuousQwen`'s class default of `True` at `model/continuous_qwen.py:25`. (`ContinuousLlama` defaults to `False` at `model/continuous_llama.py:22`.)
- `model/continuous_base.py:120` builds projectors if `use_embed_proj` **or** `subject_embed_dim != hidden_size`. Qwen3-8B and Llama-3.1-8B are both 4096, so the fallback does not fire.
- Therefore `embed_projs = None` for all four patching configs.

**Consequence.** Tables 2 and 5 are symmetric — no projector asymmetry between self and cross-model explainers. Any critique alleging one is wrong. But the paper's self-explainer on patching has *no* learned map on the activation, only rank-128 LoRA downstream, which is why `C0` is both the faithful arm and an arm that cannot answer the rotation question.

## F.2 A pretrained Qwen projector exists; the paper says it does not

- `config/feature_descriptions/qwen_131k.yaml` — `use_embed_proj: true`, `embed_proj_path: .../alignment_outputs/qwen_llama_3.1_8b_base/final_alignment_model.pt`, `output_dir: .../checkpoints/aligned_qwen_pretrained`, `peft_lora: false`.
- `config/feature_descriptions/qwen_131k_nonpretrainedembed.yaml` — same but no `embed_proj_path`, `output_dir: .../checkpoints/nonaligned_qwen`.
- `qwen_131k_simcor.yaml` (`peft_lora: false`), `qwen_131k_ood_fw.yaml` and `qwen_131k_ood_diff.yaml` (`peft_lora: true`) — all `use_embed_proj: true`, none with `embed_proj_path`, pointing at `..._qwen3_131k_dec` / `..._qwen_131k_dec`.

**Consequence.** The paper's §3.1 claim that pretrained projections were included only for Llama-3.1-70B is contradicted. Both Qwen conditions were trained. Which one Table 1 reports is ambiguous, and the row may mix conditions across columns — differing both in projector init and in `peft_lora`.

## F.3 Projectors are LoRA target modules, not full-rank

- `model/utils.py:252–253` — for each trainable projector, `target_modules.append(f"embed_projs.{i}")`.
- `LoraConfig` constructed at `model/utils.py:255–261`; configs set `lora_r: 128`.
- `model/continuous_base.py:138` sets `requires_grad_(True)`; `model/continuous_peft.py:94` saves projector weights alongside adapter weights.

**Consequence.** Under LoRA a projector is frozen-at-init plus a rank-128 update. The paper's footnote 7 states this accurately. For $d=4096$ this cannot represent an arbitrary orthogonal map, which is the basis of hypothesis (c) in §7.4 and the reason for the assertion in §3.1's trap box.

## F.4 No alignment-training code shipped

Only the loading path exists (`model/utils.py:121–156`), consuming `LinearAlignmentModule` state dicts keyed `alignments.{layer}.weight`. The `.pt` artifacts are not public. So the pretrained-projector condition is not directly reproducible from the release, and any ridge fit is yours to write — but the expected format is documented by the loader, so emit it in that shape.
