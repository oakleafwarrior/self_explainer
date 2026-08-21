# The scale question, resolved — and what it leaves open

Written 2026-08-21. **Supersedes the N=100k jump this file previously proposed.** That plan was
built on the premise that the paper trained on ~100,000 activation-patching examples and we
stopped at 16,384. The premise was wrong. Do not run it.

## 0. What the paper actually trained on

`2511.08579v3.pdf`, **Table 7** (Appendix G.2, "Dataset statistics for the activation patching
data on the Llama-3.1-8B target model") tabulates the split by token type, layer chunk and
has-changed label. Summed:

| split | total | unchanged | changed |
|---|---|---|---|
| **Train** | **13,454** | 6,513 | 6,941 |
| Test | 5,600 | 2,757 | 2,843 |

**Our N=16,384 already exceeds the paper's entire activation-patching training set by ~22%.**
The paper's regime sits *between* our top two points — N=8,192 below it, N=16,384 above — and
both are measured. There is no scale gap to close.

Where the ~123k came from: that is the raw pool in
`Transluce/act_patch_qwen3_8b_counterfact` before the filtering and balancing G.2 describes
("we filter the dataset to ensure roughly equal representation across each has-changed
category while avoiding over-representation of any specific (token, layer) combination").
Table 7's near-even 6,513/6,941 split is that balancing having been applied. Independently,
`se_config.ACT_DATASET_PREFIX`'s comment records that the base repo maps only the first 10,000
shuffled rows — smaller still, and the operative constraint for the Qwen pipeline.

**Caveat, stated because it is real:** Table 7 is the *Llama-3.1-8B* target. The paper gives no
equivalent table for Qwen3-8B activation patching (Table 8 is input ablations, a different
task). The two corroborating figures — 13,454 tabulated and 10,000 mapped by the release — sit
either side of the same order of magnitude, and our sweep is at or above both, so the
conclusion holds either way. Do not write "the paper trained Qwen on 13,454"; write that the
tabulated Llama split is 13,454 and the release maps 10,000.

**Do not cite the "0.8% / 1,024 samples" figure here.** That claim (abstract item 2, §4 line
834, Figure 5) is about *feature descriptions* — "0.8% of all training features (1024 samples
per layer)" against a nearest-neighbours baseline on SAE features. It is not the
activation-patching sweep and does not transfer to it.

## 1. What this changes

**The writeup's scope statement gets stronger, not weaker.** The bounded phrasing this file
previously recommended — "at N ≤ 16,384 we cannot speak to the paper's regime" — is no longer
the honest one and understates the result. The honest claim is now:

> Our sweep spans and exceeds the paper's activation-patching training set. At and above the
> scale the paper itself used, on this task and family, the injected activation contributes
> nothing measurable over a v=0 floor, and rotating it away changes nothing.

**The four N=16,384 seed runs become the most important runs in the project**, because N=16,384
is the closest point above the paper's train size and N=8,192 the closest below. The paper's
regime is bracketed by exactly those two, and the lift over floor at both is ~0 (−0.002 at
8,192, +0.017 at 16,384). Whatever the seeds say about the single-seed N=16,384 effect, they
are saying it at the most load-bearing point on the curve.

**Retired, with the GPU time it frees:**

| retired | cost avoided |
|---|---|
| Stage 1 at N=100k (floor + Cfull/R-id) | ~36 h |
| Stage 2 at N=100k (Cfull/R-Q + C0/R-id) | ~36 h |
| `ACT_DATASET_PREFIX` migration + tree re-eval | ~5 h + all the breakage in §A |
| N=32,768 as a trend point | ~12 h |

Nothing above needs running. `ACT_DATASET_PREFIX = 20_000` is already ~1.5× the paper's
tabulated train split and 2× what the release maps; it does not need raising.

## 2. What is still open

- **The four N=16,384 seeds** (running). The only live claim from NB03: at one seed,
  exact_match +0.026 and content_match +0.033 outside the NB04 §1b null band, `has_changed_f1`
  inside it, predicted-changed rate flat (0.542 → 0.533), both per-class deltas positive — i.e.
  *not* the response-bias artifact that explains the low-N points (at N=512 the arms straddle
  the 45.9% base rate at 36.3% and 55.1%, and the per-class deltas split +0.150 / −0.096).
  Re-run `se/bootstrap_paired.py` when they land; it picks up new seeds through `C.seeds_for`.
  If it evaporates, NB03 is null end to end and the writeup says so.
- **The NB04 ladder** (running). Its premise — that `C0` collapses under rotation and the
  ladder traces the recovery — was falsified by NB03, where `C0 · R-Q` beat `C0 · R-id` at all
  three N and `C0 · R-id` never cleared the floor by more than 1.1 points. Read the rungs as a
  capacity result about the paper's configuration, not as the basis finding; NB03 §5's exit
  criteria already say to label it that way.
- **NB05 §1–§4.** No training at all. §3's held-out ridge residual is a continuous quantity
  with no floor to normalize against — the one measurement in the project immune to the
  resolution problem that flattened NB03 — and §2 reports the paper's own Table 4 alignment
  quantity. Highest information per GPU-hour available, and it supplies the cross-model anchor
  NB03's exit criteria asked for and never got. **This is where the freed time should go.**
- **The stale eval group.** `se/bootstrap_paired.py --pair C0` asserted at N=512: two C0 arms
  scored on disjoint held-out items. Run `--audit`, delete the minority group's
  `eval_scores.json` / `eval_records.json`, re-run those cells. Adapters and training sets are
  unaffected (see §A). Do this before reading anything from the C0 arm.

## 3. Operational notes that still apply

These were written for the 18-hour runs and outlive them.

**Checkpointing.** `TrainingArguments` sets `save_strategy="no"` (se_common.py:754) and the
adapter is written only by `trainer.save_model` after `trainer.train()` returns.
`run_training`'s resume is all-or-nothing — it skips when `metrics.json` exists and otherwise
starts from scratch. A run that dies late loses everything. Tolerable at 3 h; fix it before any
run materially longer.

**Concurrency.** Two training runs on one 80 GB A100 is plausible but unverified. Measure peak
VRAM from a single run (`torch.cuda.max_memory_allocated()`, or watch `nvidia-smi` during the
current N=16,384 runs) before pairing. Expect well under 2× throughput — an 8B model at batch
4 × seq 512 with gradient checkpointing is likely compute-bound, so two processes may share SMs
at close to 1× aggregate. Note that `se_env.bootstrap(gpu="required", min_vram_gb=80)` checks
*total* VRAM, not free, so it will happily let a second process start into a collision. Never
pair two runs that write the same `run_dir`.

**Cost model, for any future N.** No `num_train_epochs` and no `max_steps` are set, so training
takes HF's 3-epoch default and wall clock scales linearly with `n_train`: 3,072 optimizer steps
at N=16,384 (eff. batch 16, 3 epochs) ≈ 3 h measured.

## Appendix A — prefix-migration mechanics, if it is ever needed

Not needed now. Recorded because it was worked out and because `--audit` exists to catch its
failure mode.

Raising `ACT_DATASET_PREFIX` moves the eval window and breaks every eval in the tree:

- **Adapters and training sets survive.** `build_act_dataset` does
  `load_dataset(...).shuffle(seed=C.SEED).select(range(prefix))`. The shuffle is a
  deterministic permutation and `select` takes a prefix of it, so rows 0–19,999 are
  byte-identical at any prefix; the template `rng` is consumed in `map` order, so those rows
  keep their templates. Training is `select(range(n_train))` from the front (se_common.py:720)
  — unchanged. Nothing needs retraining.
- **Every eval dies.** Eval is `select(range(len(dataset) − EVAL_SIZE, len(dataset)))`
  (se_common.py:831) — the *last* 1,024 rows, which move wholesale. Fix per run by deleting
  `eval_scores.json` and `eval_records.json` and re-running the cell (~5 min each).

If a raise ever happens, make the window prefix-independent in the same change: anchor it at a
fixed absolute range above the maximum N (`EVAL_START = 102_400`, eval = rows
`[EVAL_START, EVAL_START + EVAL_SIZE)`, prefix ≥ 103,424) so it stops moving. Train stays
anchored at the front, so no adapter is invalidated. **Do not** instead reserve rows 0–1023 for
eval — that shifts the training rows and would invalidate every adapter in the tree.
