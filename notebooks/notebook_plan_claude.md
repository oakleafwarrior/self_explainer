# Where Does the Self-Explainer Advantage Come From?

Notebook implementation of `self_explainer_mechanism_plan_v2.md`: testing whether an explainer
succeeds because of privileged access to a model it *is*, or because that model's activations
arrive in a coordinate frame the explainer already reads.

Built against **v2** of the plan. Where v1 and v2 disagree, v2 wins — the `Oracle` and
`Cfull-rand` arms, seed bands, the cross-model baseline as a first-class arm, and
`P-ridge-frozen` all come from v2 and are absent from v1.

Builds on [`oakleafwarrior/introspection_replication`](https://github.com/oakleafwarrior/introspection_replication).
Target model throughout is Qwen3-8B; the wedge is an orthogonal transform `Q` of its residual
stream, which leaves the computed function identical while sending every extracted activation
to `Qv`.

## The sequence

| Notebook | Phase | Budget | Pod | What it decides |
|---|---|---|---|---|
| `NB00_preflight` | 0 (§4) | 1.0 h | CPU, + GPU for §7 | The injection bug (nothing prior is reusable), what effect size is observable, the KL noise floor. Preregister the readings. |
| `NB01_rotation_gate` | 1 (§5) | 3.5 h | 1 × 80 GB (48 GB works) | Debug on 0.6B, build `Q`, fold gains, rotate, **pass the invariance gate**. |
| `NB02_rotate_activations` | 2 (§6) | 1.0 h | CPU, + GPU for the spot check | `v ↦ Qv` on the cached dataset; live-vs-offline spot check; matched training sets per arm. |
| `NB03_core_sweep` | 3a (§7.1) | 4.0 h | 1 × 80 GB | **The irreducible core:** `Cfull` × {R-id, R-Q} **plus the cross-model baseline**, full sweep, seed bands. |
| `NB04_capacity_ladder` | 3b (§7.2) | 2.0 h | 1 × 80 GB | `Oracle`≡`C0` exactness check, then C0→Cfull plus `Oracle`. |
| `NB05_alignment_and_ridge` | 3d (§7.4) | 2.0 h | 1 × 80 GB | Alignment; closed-form `Π_ℓ`; the `Π^Q = Π Q^T` identity; **`P-ridge-frozen`**. |
| `NB06_diagnostics_and_controls` | 3e (§7.5) | 1.0 h | CPU for §1, 1 × 80 GB for §2 | Did the map learn `Q^T`? The input-ablation control, read against the seed band. |
| `NB07_figures_and_writeup` | 4 (§10) | 2.5 h | CPU | Figures with bands, preregistered readings applied mechanically, failure log. |
| `NB08_base_instruct_dissociation` | Appx D | ~3 h | 1 × 80 GB | Optional: behavioral similarity on a task where basis cannot operate. |

Run them in order. NB01 gates NB02; NB02 gates everything downstream. If NB04's exactness check
or NB06's control fails, stop and debug rather than reading any results.

The pod column is what each notebook's setup cell asserts, so a mismatch is a loud error at
cell 2 rather than an OOM three hours in. §2 of NB06 trains, which the plan's budget table
does not say out loud.

**The explainer is initialized from unrotated `M`** and fed activations from `M_Q`. Building it
from `M_Q` would rotate the explainer too, preserve alignment exactly, and make the whole
experiment a no-op producing identical numbers in both arms. What is on the axis is therefore
*the contribution of coordinate correspondence to the self-explainer advantage*, not the
advantage itself — NB03 says so, and the write-up must too.

## The arm matrix

v2 §3: *"Write it into the repo README before running anything; the combinatorics are where
budgets die."* These tags are the directory names under `runs/`, via `se_config.run_dir`.

**Target** `M` = Qwen3-8B throughout. The reason is data, not architecture: the cached
`act_patch_qwen3_8b_counterfact` dataset exists for this target and NB02 reuses it wholesale.

| Explainer | Model | Activation path | Init |
|---|---|---|---|
| `E_self` | Qwen3-8B | direct injection + trainable input map | from **unrotated** `M` |
| `E_cross` | Qwen3-4B (primary), 1.7B (secondary) | per-chunk full-rank `Π_ℓ` | independent pretrained weights |

| Rotation | Transform applied to the activations handed to `E` |
|---|---|
| `R-id` | `v ↦ v` |
| `R-Q` | `v ↦ Qv`, single global `Q ∈ O(4096)` |
| `R-Qs` | `v ↦ diag(s) Qv` — **input-side only**, see correction 4 |

| Capacity (`E_self`) | Input map | Represents `Q^{-1}`? | Trainable |
|---|---|---|---|
| `C0` | none (LoRA only) | no | — |
| `C8` / `C64` / `C512` | rank 8 / 64 / 512 | no / no / partially | yes |
| `Cfull` | full-rank `d×d`, init `I` | exactly | yes |
| `Cfull-rand` | full-rank `d×d`, random orthogonal init | exactly | yes |
| `Oracle` | frozen at `Q^T` | exactly | **no** |

| Projector init (`E_cross`) | Meaning |
|---|---|
| `P-rand` | random init, then train — the paper's Qwen3-8B condition |
| `P-ridge` | closed-form ridge init, then train |
| `P-ridge-frozen` | closed-form ridge fit, explainer **never trained on this target** |

**Sweep** `N_TRAIN ∈ [128, 256, 512, 1024, 2048, 4096, 8192]`, matched data ordering across
arms. **Seeds:** 3 at `N ∈ {512, 8192}` for every core arm, 1 elsewhere; every figure carries
the band. The readings turn on "curves overlap" vs "separated", and neither is decidable
without it.

Two arms exist to keep the core honest. `Cfull` under `R-id` is initialized *at its solution*,
so low-`N` separation is partly an initialization artifact — `Cfull-rand` starts both rotation
arms equidistant from theirs. And `Oracle` separates representability from learnability, while
giving an exact bug check: `Oracle`/`R-Q` is algebraically identical to `C0`/`R-id`, since
`Q^T(Qv) = v`. NB04 runs that check before the ladder.

## Running it on RunPod

One pod, one notebook at a time; the volume carries state between them.

1. **Deploy a single-GPU pod** from a `runpod/pytorch` template. NB03–NB06 and NB08 train an
   8B explainer and were written for 80 GB (A100 80 GB, H100 80 GB); NB01 holds two 8B copies
   at once and fits in 48 GB (A40, L40S, A6000); NB00, NB02 and NB07 need no GPU. Prefer one
   GPU: `device_map="auto"` will cheerfully shard across four, which is correct but makes
   throughput and memory numbers incomparable between arms. `SE_PIN_GPU=0` pins it.
2. **Attach a network volume of 150 GB or more**, mounted at `/workspace`. It holds the HF
   cache (Qwen3-8B is 16 GB, 4B 8 GB, 0.6B 1.5 GB), the rotated activation datasets, and
   ~0.2 GB per finished run. A *network* volume survives terminate; the container volume does
   not, and neither does a four-hour sweep sitting on it.
3. **Set `HF_TOKEN`** in the pod template's environment variables, or write it to
   `/workspace/.hf_token`. It is optional — every checkpoint and dataset here is public.
4. **Clone onto the volume and provision**, from the pod's web terminal:

   ```bash
   cd /workspace && git clone <this repo> self_explainer
   python /workspace/self_explainer/se/se_env.py --install   # deps, then a pod report
   python /workspace/self_explainer/se/test_rotate.py        # CPU, seconds
   python /workspace/self_explainer/se/test_input_map.py
   ```

5. **Open JupyterLab** on the pod and start with `notebooks/NB00_preflight.ipynb`. Every
   notebook's first two cells install what is missing and print the pod they actually got —
   GPU, VRAM, volume, cache, token, library versions, and a copy in
   `reports/environment.json`. If that report disagrees with the notebook's header, stop there.
6. **Stop the pod between notebooks.** Nothing is held in a live kernel: `Q` (134 MB), the
   rotated datasets, and `runs/` are all on the volume.

Interruption is expected rather than exceptional. Every run writes its own directory and is
skipped when already complete, so re-running a sweep cell after a preemption resumes it — which
is what makes the interruptible pods usable for the 4-hour NB03.

**Two things Colab did for free that a pod does not.** `bitsandbytes` was preinstalled there,
and `se_common` trains with `optim="paged_adamw_8bit"`, so without it every training cell dies
at trainer construction — `se_env.ensure_deps()` installs it, pinning the image's torch as a
pip constraint so that resolving it cannot swap a CUDA-matched build for a generic wheel. And
`HF_HOME` was already pointing somewhere sane: it is read into module constants when
`huggingface_hub` is first imported, so the cache has to be configured before any
`from transformers import ...` runs. That is why setup is two cells, and why `se_env` says so
if the kernel imported the hub first.

## Shared modules

Notebooks are thin drivers over `se/`, so the arms differ in configuration rather than in
copied code:

- **`se/se_env.py`** — the pod: dependency floors (installed around the image's torch, never
  over it), the volume, the HF caches, the token, and the GPU check each notebook declares.
- **`se/rotate.py`** — deliverable 1: gain folding, `Q` construction, the weight transform,
  the invariance gate, and the learned-map diagnostics.
- **`se/se_common.py`** — dataset construction, the capacity-ladder input map, activation
  injection, training, evaluation, the closed-form ridge fit, and the input-ablation path.
  Prompt templates, chunking, LoRA config and seeds are held identical to the base repo.
- **`se/se_config.py`** — every constant the notebooks would otherwise disagree about.
- **`se/mini_qwen3.py`** — a pure-torch replica of the Qwen3 block structure, for testing.

## Tests — run these before any GPU time

```bash
python se/test_rotate.py       # folding, rotation, the gate, the diagnostics
python se/test_input_map.py    # the capacity ladder, and the injection path
```

Both run on CPU in seconds and validate the parts that are cheap to get wrong and expensive to
debug at 8B: that folding and rotation preserve logits exactly, that hidden states move by
exactly `Q`, that the gate can fail, that each rank-`r` arm attains its truncation bound, and
that the injected activation actually reaches the forward pass.

## Corrections found while implementing

v2 independently reaches the same conclusion as this repo on two of these (tied embeddings,
Appendix C.3; the scaled arm, §7.3). The injection bug is the one that is new, and it is the
expensive one.

**1. The base repo's activation injection was a no-op.**
`build_inputs_embeds_projected` calls the out-of-place `Tensor.masked_scatter` and returns the
unmodified embeddings, so the projected activation never entered the forward pass and the
projection never received a gradient. NB00 asserts this at runtime rather than taking it on
faith.

v2 §4 already expects `Cfull`/R-id to be a fresh run (no finished `E_self` run has a `d × d`
input map), so this does not add an arm. What it costs is the *anchor*: the finished patching
numbers measure prediction of patch outcomes from the prompt text alone, so they cannot be
compared against anything here. NB03 turns that into an asset by running the text-only floor
deliberately — it is the right floor for this experiment, and if the identity arm barely clears
it, the activation was never doing much work in the first place.

**2. Tied embeddings block the gain fold, not the rotation.** Both matrices need the *same*
transform (`Q^{-1} = Q^T`), so tying is fine for the rotation as long as the tensor is
transformed once; what breaks is folding `model.norm`'s gain into `lm_head`, which writes to
the shared tensor and corrupts the embedding. `untie_embeddings` handles it, which is what
makes Qwen3-0.6B usable as NB01's debug substrate. (v2 Appendix C.3 reaches this independently;
`se/test_rotate.py` demonstrates both halves.)

**3. The full-rank input map is not "under 2% of the LoRA-adjacent budget".** 16.8M is one
`d × d` matrix at `d = 4096`; the pipeline learns one per layer chunk (`N_LAYER_CHUNKS = 4`),
so Cfull is ~67M against a ~44M LoRA budget — larger, not negligible. The argument for Cfull
stands (it is exactly the parameter class in which `Q^T` lives), but "capacity is not a
confound because the map is tiny" is not available. "Capacity is not a confound because we
measured it across a ladder" is, and that is what §3 actually builds.

**4. The optional `Q ∘ diag(s)` arm cannot be a model-side transform.** Computational
invariance rests on `‖Mx‖ = ‖x‖`. A diagonal scaling breaks it, so no model computes `M`'s
function with activations `Sv`; `apply_rotation` refuses non-orthogonal maps. The arm survives
as a data-side distortion, which still separates coordinate alignment from distributional
compatibility but is not an invariance argument. (v2 §7.3 states the same correction.)

**5. "Principal angles between row spaces" is vacuous for square full-rank maps.** `Π` and
`Q^T` both span all of `R^d`, so every principal angle is 0 regardless of what was learned.
`compare_to_inverse` reports relative Frobenius error, row cosines, and the orthogonality
defect of `ΠQ` instead — the last sees through a residual rotation the explainer could absorb
downstream, which is a *more* interesting form of recovery. `subspace_angles` keeps the angle
diagnostic for the low-rank arms, where the update really is rank-`r` and the question has
content.

One smaller note: the `N_TRAIN` grid of seven points is not what the finished 8B run used; it
used four. That no longer needs recording, because nothing from that run is reused — the grid
is chosen for what the readings need.

## Configuration

Paths and constants live in `se/se_config.py`; the pod is set up by `se/se_env.py`, which the
notebooks call in their first two cells.

**Everything writes to the volume.** `se_env.persistent_root()` resolves `/workspace` (GPU
pod), `/runpod-volume` (serverless), or `$SE_PERSIST_ROOT`, and falls back to `~` off-RunPod so
the CPU-only notebooks run on a laptop. Outputs land in
`<volume>/self_explainer` — the same directory the repo is cloned into on a pod, so the four
output trees are .gitignore'd — and the HF cache in `<volume>/hf_cache`.

All of these are optional:

| Variable | Default | What it does |
|---|---|---|
| `HF_TOKEN` | — | pod-template environment variable; `<volume>/.hf_token` works too |
| `SE_REPO_DIR` | `/workspace/self_explainer` | where this repo is checked out |
| `SE_PERSIST_ROOT` | `/workspace` | the volume everything is written under |
| `SE_ROOT` | `<volume>/self_explainer` | overrides the output tree outright |
| `SE_PIN_GPU` | — | `0` restricts the process to one device on a multi-GPU pod |

**One clone, and it is this repo.** Nothing reads `introspection_replication` or
`TransluceAI/introspective-interp` at runtime. NB00 §1 proves the base repo's activation
injection is a no-op by reproducing the nine lines inline, so every finished run from the
previous project is void and there is nothing to reuse; the paper's code audit was run once
and is frozen with its file:line citations in `se/paper_audit.json`. Provisioning a pod is
`git clone` this repo onto the volume and nothing else.

`USE_4BIT` is off and should stay off: invariance is exact in float64, close in bfloat16, and
meaningless in 4-bit.

**Negative-result path**, decided now rather than at hour 17: if the gate fails on 8B, or every
arm sits inside the seed band, the deliverable is still paper-shaped — the invariance
construction as a reusable tool, the measured effect-size floor at this eval size, and the
statement *"the coordinate-frame contribution is smaller than run-to-run variance at this
scale, and here is the N at which it would become detectable."* NB00 computes that floor and
NB07 has a preregistered row for it (`all_inside_band`).

## Deliverables (plan §11)

1. `se/rotate.py`, with the invariance gate as a test — `se/test_rotate.py`
2. Rotated activation datasets; `Q` and its seed checkpointed by NB01
3. Sweep curves per metric, R-id vs R-Q vs the cross-model baseline, with seed bands,
   faceted by capacity — NB03 and NB07 figure 1
4. Capacity-recovery plot with the `Oracle` reference line — NB04 and NB07 figure 2
5. Ridge comparison — `P-ridge-frozen` against `E_self`, plus the `Π^Q = Π Q^T` numerical
   check — NB05, NB07 figure 3
6. Invariance-gate numbers and the Phase 0 noise floor reported, not buried —
   `reports/invariance_gate.json`, `reports/noise_floor.json`, reprinted in NB07
7. Honest failure log — `reports/failure_log.json`, seeded with the four corrections above
8. A stated position on which §7.2 row the results landed on — NB07 applies the rules from
   `reports/preregistration.json`, written in NB00 before any results existed
