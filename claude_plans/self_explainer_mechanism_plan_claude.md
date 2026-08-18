# Where Does the Self-Explainer Advantage Come From?

Testing whether models are better at explaining themselves because of privileged access, or because their activations arrive in a coordinate frame they already read.

**MATS 12.0 application project.** ~16 hours (max 20), plus 2 for the executive summary. Builds on `oakleafwarrior/introspection_replication`.

---
## 1. The claim, and why the answer changes what you'd do

Li et al. (arXiv:2511.08579) report that a target model $M$ is best explained by fine-tuned versions of itself, and that this self-explanation is ~100× more sample-efficient than a nearest-neighbours baseline. The proposed mechanism is privileged access.

Two alternatives explain the same numbers without any privilege:

- **Behavioral self-simulation.** $E$ is initialized from $M$'s weights and is close enough in function space to predict $M$'s outputs directly.
- **Basis compatibility.** $E$ decodes $v$ well because $v$ arrives in coordinates $E$ already reads fluently. No self-knowledge required, just a shared frame.

The paper's §3.4 and Appendix C.2 establish a correlation between activation alignment and explainer quality. That is basis compatibility, presented as supporting evidence for privileged access rather than as the competing explanation it is.

**The consequence is a practical one, and it is the reason to run this.** If the mechanism is basis compatibility, then the recipe for cheap interpretability is not "fine-tune each model to explain itself" — it is "take one good explainer and fit it a cheap linear map into each new target's frame." Those two recommendations differ enormously in cost. The first requires a separately trained explainer per target model, retrained on every checkpoint. The second requires one explainer plus a ridge regression per target. If basis compatibility is doing the work, the paper's most useful finding is being filed under the wrong heading, and the deployment story changes completely.

There is also a narrower stake for oversight. If self-explanation is geometric rather than introspective, "the model told us what it was doing" carries no more epistemic weight than any other trained probe, and the CoT-faithfulness framing in §5.2 loses its distinctive force.

## 2. The wedge

An orthogonal transformation $Q$ of $M$'s residual stream separates the hypotheses. RMSNorm transformers are computationally invariant to such a transform, so $M$ and $M_Q$ compute the **identical function** — behavioral similarity between $E$ and $M_Q$ is exactly what it was. But every extracted activation becomes $Qv$, and the coordinate alignment measured in the paper's Table 4 goes to chance.

| Hypothesis | Predicts under rotation |
|---|---|
| Behavioral self-simulation | Self-advantage survives |
| Basis compatibility | Self-advantage collapses toward the cross-model baseline |

**Scope.** This applies to the activation-patching task (§2.3). Input ablation (§2.4) passes no activation to the explainer at all, so there is nothing to rotate — which makes it a free control whose scores must be *exactly* invariant.

---

## 3. Capacity is the experiment, not a nuisance

This section is load-bearing. Get it wrong and the whole project reads as a mistake.

If the explainer's activation-ingestion path cannot represent $Q^{-1}$, then a performance collapse under rotation tells you about parameter capacity and nothing about basis. A rank-$r$ LoRA update cannot invert a full-rank $d \times d$ rotation. So the naive setup — 8B explains 8B, dimensions match, activation injected directly, LoRA only — will collapse under rotation *for a completely uninteresting reason*.

The fix is not full-parameter SFT (see §6). It is one extra trainable $d \times d$ matrix on the injected activation, initialized at identity, trained alongside the LoRA. For $d = 4096$ that is 16.8M parameters — under 2% of the LoRA-adjacent budget — and it is precisely the parameter class in which $Q^\top$ lives.

Once you have that knob, capacity stops being a confound and becomes the measurement:

| Arm | Input map on $v$ | Can represent $Q^{-1}$? |
|---|---|---|
| **C0** | none (LoRA only) | No |
| **C8** | rank 8 | No |
| **C64** | rank 64 | No |
| **C512** | rank 512 | Partially |
| **Cfull** | full-rank $d\times d$, init identity | Exactly |

**The minimum rank at which performance recovers is a direct measure of how much of the self-explainer advantage is coordinate-frame.** If Cfull recovers and C0 doesn't, the advantage was a frame. If nothing recovers, something survives rotation that isn't linear-frame alignment. If C8 already recovers, the effective dimensionality of the mismatch is tiny and the whole "privileged access" story is about a handful of directions.

That ladder is a better result than the binary the rotation alone gives you, and it costs one adapter config per arm on otherwise identical runs.

**Diagnostic worth 20 minutes:** after training Cfull, compare the learned map $\Pi$ against $Q^\top$. Report $\lVert \Pi Q - I \rVert_F / \sqrt{d}$ and the principal angles between their row spaces. If the model recovered by learning approximately $Q^\top$, say so with numbers — it converts a performance claim into a mechanistic one.

Your existing projector explainers (Qwen3-1.7B/4B with full-rank per-layer $\Pi_\ell$) already sit at Cfull by construction. They are the natural starting arm and the cheapest to run.

---

## 4. The constructive arm: does a fitted projector buy you the advantage?

This is where the "so what" gets teeth, and it is nearly free.

If basis compatibility is the mechanism, a non-self explainer given a *pre-fitted* map into $M$'s frame should close most of the gap to the self-explainer. The paper half-shows this already: Llama-3.1-70B with a pretrained projection recovers 14% on SAE explanations, 2.7× on real activations, 1.7× on activation differences relative to random init. But this is reported as evidence *for* privileged access rather than as the alternative recipe it implies. Meanwhile Qwen3-8B — the main cross-model comparison in Table 1 — was only ever run with a **randomly initialized** projector, so the headline cross-model gap and the projector-quality effect are confounded.

**Fix the confound and price the alternative recipe:**

1. Cache paired activations $h^E_{\ell,t}(x), h^M_{\ell,t}(x)$ on a few thousand FineWeb sequences.
2. Fit $\Pi_\ell$ in closed form — this is ridge regression, not a training run:
   $$\Pi_\ell = H^E (H^M)^\top \left(H^M (H^M)^\top + \lambda I\right)^{-1}$$
   Minutes on cached tensors, per layer. No SGD, no hyperparameter search beyond $\lambda$.
3. Train the cross-model explainer from that initialization instead of random.
4. Report the sample-efficiency curve against the self-explainer.

If a closed-form ridge fit closes most of the gap, the practical conclusion writes itself: *you do not need a self-explainer per target; you need one explainer and a linear regression.* That is a positive, actionable finding, and it reframes the project from "this paper's claim is confounded" to "here is the cheaper thing that actually works."

It also composes with the rotation. Fit the projector on *rotated* activations; if it recovers, the entire phenomenon is a linear-frame problem and you have shown it two independent ways.

---

## 5. The construction

Standard computational invariance, as in SliceGPT (Ashkboos et al., 2024). Reference implementation exists; don't derive it from scratch.

### 5.1 Fold the RMSNorm gains

$\mathrm{RMSNorm}(x) = \frac{x}{\mathrm{rms}(x)} \odot g$. The elementwise gain doesn't commute with $Q$, so absorb it into every linear map that consumes the normalized output:

$$W \leftarrow W \,\mathrm{diag}(g), \qquad g \leftarrow \mathbf{1}$$

- `input_layernorm` → `q_proj`, `k_proj`, `v_proj` of the same block
- `post_attention_layernorm` → `gate_proj`, `up_proj`
- final `model.norm` → `lm_head`

After folding, normalization is pure $x/\mathrm{rms}(x)$, which commutes with any orthogonal $Q$ since $\lVert Qx \rVert = \lVert x \rVert$.

**Verify folding alone before introducing $Q$.** Debugging two transforms at once wastes hours you don't have.

### 5.2 Apply $Q$

Sample $Q \in O(d)$, $d = 4096$, via QR of a Gaussian in float64. Fix and record the seed.

Write paths (contribute to the residual) — left-multiply:
- `o_proj`, `down_proj`: $W \leftarrow QW$
- embedding, stored `[vocab, d]`: $E \leftarrow EQ^\top$

Read paths (consume the residual) — right-multiply:
- `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`: $W \leftarrow WQ^\top$
- `lm_head`, stored `[vocab, d]`: $W_U \leftarrow W_UQ^\top$

Then $WQ^\top(Qh) = Wh$ everywhere.

### 5.3 Qwen3-specific notes

- **QK-norm.** Qwen3 RMSNorms head-dimension slices of $W_Qh$ and $W_Kh$, *inside* the head and downstream of the read matrix. Unaffected. Do not fold or touch these gains.
- **RoPE.** Operates on head dimensions post-projection. Unaffected.
- **Tied embeddings.** Qwen3-0.6B/1.7B/4B set `tie_word_embeddings: true`; 8B does not. Check `config.json` for anything you rotate — if the tensor is shared you cannot apply $EQ^\top$ and $W_UQ^\top$ independently. This is why the target must be 8B.
- **No quantization.** Invariance is exact in float64, close in bfloat16, meaningless in 4-bit. Build and apply $Q$ in float64, cast last. Rotated runs unquantized.
- **GQA.** `k_proj`/`v_proj` have fewer output heads but full input dimension $d$, so $WQ^\top$ is well-formed. No special handling.

### 5.4 Rotate the cached activations

The step that makes the budget work. $M_Q$ is functionally identical to $M$, so every `has_changed` label and every content string is unchanged and **the counterfact patching pipeline does not need rerunning**. Load the cached tensors from `act_patch_qwen3_8b_counterfact`, apply $v \mapsto Qv$, write out as a sibling dataset. Minutes.

### 5.5 Gate: invariance check

**Do not proceed without a pass.** Highest-value 30 minutes in the project.

1. ~256 FineWeb sequences, 512 tokens.
2. Logits under $M$ and $M_Q$.
3. Report mean per-token KL, max absolute logit delta, top-1 agreement.

Accept at mean KL $\lesssim 10^{-4}$, top-1 agreement $\geq 99.9\%$ in bfloat16. Failures are nearly always a missed gain fold, a transposed `[vocab, d]` convention, or an unnoticed tied embedding.

Second check: recompute has-changed labels for ~200 patching examples under $M_Q$ and confirm they match cache. This validates the transform through the hook path you actually use.

---

## 6. On full SFT: don't, and say why

The paper full-parameter fine-tunes its same-dimension explainers. The natural question is whether to match that.

**Single A100 80GB, Qwen3-8B, AdamW:** 16GB bf16 weights + 16GB gradients + 64GB optimizer states + 32GB fp32 master ≈ 128GB. Does not fit. With 8-bit Adam, gradient checkpointing, batch size 1 and short sequences you land around 50–60GB — technically feasible, slow, and fragile. **Multi-GPU pod (4–8× A100):** fine under FSDP or ZeRO-3, but the config debugging is a real time sink against a 16-hour budget and it buys you nothing scientific.

The reason to skip it is not just cost. The only thing full SFT gives you *that matters here* is a full-rank trainable path from $v$ into the explainer — and §3 gets that for 16.8M parameters. Spending the pod on full SFT converts an explicit, measurable variable back into an implicit one. If you have the multi-GPU pod anyway, spend it on running more arms of the capacity ladder in parallel, not on making one arm heavier.

State this reasoning in the write-up. "I could have matched the paper's training setup and chose not to, for this reason" is a prioritization signal; silently using LoRA is not.

---

## 7. Conditions and decision rules

### 7.1 Arms

**Rotation:** identity (control, reuse existing runs) · random orthogonal $Q$ · *optional* $Q \circ \mathrm{diag}(s)$ with $s$ log-uniform.

The optional third arm exists because $Q$ preserves norms, angles, anisotropy, effective rank, and the whole covariance spectrum — it destroys *coordinate* correspondence specifically. A version of basis compatibility in which $E$ benefits because $M$'s activations are distributionally shaped like its own survives rotation untouched. Adding a scaling breaks that too, and the gap between the arms isolates coordinate alignment from distributional compatibility.

**Capacity:** C0 / C8 / C64 / C512 / Cfull per §3.

**Projector init:** random vs closed-form ridge per §4.

**Sweep:** existing `N_TRAIN` grid `[128, 256, 512, 1024, 2048, 4096, 8192]`, matched seeds across arms.

### 7.2 Pre-registered readings

Commit to these before looking at results, and say in the write-up that you did.

| Observation | Reading |
|---|---|
| Cfull curves for identity and $Q$ overlap at all $N$ | Coordinate alignment contributes nothing; §3.4's correlation is confounded |
| $Q$ separated at low $N$, converging by 8192 | Basis compatibility is a **sample-efficiency** effect — this reframes the paper's §4 data-efficiency claim as the real finding |
| $Q$ strictly below identity at all $N$ | Basis compatibility affects the attainable optimum; strongest form of the critique |
| $Q$ collapses to cross-family baseline | Table 1's ordering is a statement about geometry, not selfhood |
| Recovery threshold sits at low rank (C8/C64) | The frame mismatch is low-dimensional; suggests a very cheap alignment recipe |
| Ridge-init cross-model explainer matches self-explainer | The alternative recipe works; primary practical result |
| Input-ablation control moves at all | Bug. Stop. |

Row 2 is the modal outcome. Plan the write-up around curve *shape*, not a headline number.

---

## 8. Budget

| Task | Hours |
|---|---|
| Fold RMSNorm gains; verify logits unchanged | 2.0 |
| Implement and apply $Q$ | 1.5 |
| Invariance gate (both checks) | 1.0 |
| Rotate cached activation dataset | 0.5 |
| **Core:** Cfull × {identity, $Q$}, full `N_TRAIN` sweep | 3.5 |
| Capacity ladder: C0 / C64 / C512 at two values of $N$ | 2.0 |
| Ridge-fit projector (closed form) + one training run | 1.5 |
| Learned-map vs $Q^\top$ diagnostic | 0.5 |
| Input-ablation invariance control | 0.5 |
| Figures, write-up | 2.5 |
| **Total** | **15.5** |

Identity-condition runs are free *if* seeds, data ordering, and LoRA config match `finished_notebooks/`. **Verify this before spending any GPU time** — a mismatch turns a free control into a 3.5-hour rerun.

**Cut order if it runs long:** distributional-scaling arm → extra $Q$ seeds → intermediate ladder ranks → low end of the sweep. Irreducible core is Cfull at identity vs $Q$ across three values of $N$, plus the gate. That alone answers the question and is roughly 8 hours.

---

## 9. If there's spare time: the other half of the dissociation

Rotation kills basis while holding behavior fixed. The dual — kill behavior while holding basis — completes the argument.

The cheapest version exploits a structural fact: **input ablation gives the explainer no activation at all.** Basis compatibility cannot explain any self-advantage there, by construction. Yet the paper's Table 2 shows the self-margin for Qwen3 is *larger* on input ablation (83.4 vs 58.1 exact match) than on patching (64.0 vs 54.1) — the task with no internals shows the bigger effect. That is already awkward for the privileged-access reading and nobody has followed it up.

So: run Qwen3-8B-Base and Qwen3-8B-Instruct as explainers of each other on input ablation. They share a frame almost exactly (the paper measures 85% dot-product similarity for the Llama instruct pair) but differ behaviorally. Relabelling is cheap — regenerating MMLU responses is the thing `iteration.ipynb` already does, which is why you chose input ablation for the iteration experiment in the first place. Roughly 3 hours.

Together the two experiments bracket the question: rotation at (behavioral divergence 0, alignment 0), base/instruct at (behavioral divergence > 0, alignment ~0.85). If explainer score tracks alignment in one and behavior in the other, you have the decomposition.

The critique's original Experiment 1 — LoRA-tuning $M'$ with a dot-product term in the objective — is the rigorous version of this and does not fit the budget. Put it in future work, with the reasoning: the regularizer fights the behavioral objective with no principled $\lambda$, mean dot-product is a weak proxy for "same basis," and every ground-truth patching label has to be regenerated per $M'$. Naming why you didn't do it is worth more than doing it badly.

## 10. On porting to Llama: no

Tempting, since Llama-3.1-8B is the paper's actual Table 1 target and refuting it there would land harder. Don't, for this application.

The Table 1 result is the **feature description** task, which your repo doesn't implement. Reproducing it means building the autointerp pipeline, the trained simulator, LlamaScope SAE ingestion, and Neuronpedia label handling — days, not hours. Restricting to patching instead means regenerating the entire counterfact dataset for a new target, which is exactly the cost §5.4 was designed to avoid, plus HF gating on Llama-3.1-8B.

Architecturally Llama is marginally *easier* — RMSNorm, no QK-norm — but that saves 30 minutes against many hours of data regeneration. And a second architecture is a robustness check, not a new claim. The capacity ladder and the ridge-projector arm both produce claims. Spend the hours there.

Say this explicitly in the write-up. Choosing not to chase the more impressive-sounding version because the marginal evidence per hour is worse is the kind of call the application is testing for.

---

## 11. Deliverables

1. `rotate.py` — gain folding, $Q$ construction, weight transform, invariance gate as a test.
2. Rotated activation dataset; $Q$ and seed checkpointed.
3. Sweep curves per metric, identity vs $Q$, faceted by capacity arm.
4. Capacity-recovery plot: score vs input-map rank at fixed $N$.
5. Ridge-projector comparison against the self-explainer.
6. Invariance-gate numbers reported in the write-up, not buried in a log.
7. Honest failure log — whatever ate three hours, write it down.
8. A stated position on which row of §7.2 the results landed on, and what it implies for anyone planning to use self-explanation in practice.

**Executive summary should lead with §1's consequence and §3's capacity finding.** The rotation is the method; neither of those is the method.
