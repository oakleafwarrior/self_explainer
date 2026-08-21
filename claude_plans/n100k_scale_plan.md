# Plan: closing the scale gap with a jump to N_TRAIN = 100k

Written 2026-08-21, to be picked up when the four N=16384 seed runs finish. Companion to
`prereg_threshold_justification.md`. Nothing here is preregistered — the readings still come
from `reports/preregistration.json` via NB07.

## 0. Why this exists

NB03 resolved to null. After the paired bootstrap (`se/bootstrap_paired.py`), the wrong-signed
R-Q advantage at N ≤ 1024 is a response-bias artifact — the two arms sit on opposite sides of
the 45.9% "changed" base rate (R-id predicts 36.3%, R-Q 55.1% at N=512), and the per-class
deltas split in opposite directions (+0.150 on `changed`, −0.096 on `unchanged`). Macro F1
rewards that without any information having changed hands. Everything at N=2048–8192 falls
inside the NB04 §1b null band. One unreplicated point at N=16384 survives.

Separately, the activation's lift over the v=0 floor is ~0 across the whole sweep: +0.002 at
N=256, +0.046 at N=1024, +0.009 at N=4096, −0.002 at N=8192, +0.017 at N=16384, against a
floor that climbs 0.317 → 0.691. There is no advantage to decompose, so the rotation cannot
discriminate basis compatibility from behavioural self-simulation.

**The strongest attack on that conclusion is the scale gap.** The paper trained on ~100,000
examples. Our top point is 16,384 — 6× short. A reader can say the null is a low-data null and
that the self-explainer advantage only exists at the scale the paper actually used. One clean
measurement at N=100k closes that hole, and no number of additional points at N ≤ 16384 does.

Walking the ladder up piecewise is not worth it: the middle of the sweep already resolved to
null, and each intermediate point costs the same order as the endpoint. Jump straight to 100k.

## 1. What it costs

`TrainingArguments` in `se_common.run_training` (se_common.py:754) sets neither
`num_train_epochs` nor `max_steps`, so it takes the HuggingFace default of **3 epochs** and
wall-clock scales linearly with `n_train`:

| N | optimizer steps (eff. batch 16, 3 epochs) | wall clock |
|---|---|---|
| 16,384 | 3,072 | ~3 h (measured) |
| 100,000 | 18,750 | **~18 h** (6.1×) |

Budget in units of 18 h, not 3 h. This is the number most likely to be misremembered later.

## 2. The runs, staged

The question is "does the high-N regime destroy any signal whatsoever." That is a
lift-over-floor question, not an R-id vs R-Q question, so **the floor run is not optional** —
without it there is no scale to read the result on. But the rotation arm is only worth buying
once there is a signal for it to act on.

### Stage 1 — is there any signal at the paper's scale? (2 runs, ~36 h)

| run | rotation | capacity | init | why |
|---|---|---|---|---|
| no-activation floor | `zerovec` | `Cfull` | `identity` | the scale; non-negotiable |
| `E_self · Cfull · R-id` | `identity` | `Cfull` | `identity` | the with-activation arm |

One seed (67) each. Read `lift = score − floor` on all three metrics, with
`se/bootstrap_paired.py`'s null band as the yardstick (half-width ~0.016 on exact_match).

Both outcomes are worth having and neither is reachable from more points at 16384:

- **Lift ≈ 0.** The null holds at the paper's own training scale. This is the result that
  makes the whole project's conclusion attack-proof on scale, and it is the likelier outcome
  given the floor was already 0.691 at N=16384 and still climbing.
- **Lift > 0.** The effect lives at high N and the ladder below was measuring an unlearned
  task. The experiment relocates to N=100k and Stage 2 becomes the real measurement.

### Stage 2 — only if Stage 1 shows lift (2 runs, ~36 h)

| run | rotation | capacity | init | why |
|---|---|---|---|---|
| `E_self · Cfull · R-Q` | `Q` | `Cfull` | `identity` | the rotation comparison, at the N where the effect lives |
| `E_self · C0 · R-id` | `identity` | `C0` | `identity` | the paper's exact configuration at the paper's exact scale |

`C0 · R-id` at N=100k is the most faithful replication point in the project — their config,
their budget, our family and task. Worth having in the writeup even if Stage 1 is null, if
budget ever allows.

## 3. The prefix migration

N=100k needs a ready dataset of ≥ 101,024 rows (100,000 train + 1,024 eval).
`ACT_DATASET_PREFIX` is currently 20,000.

### What survives

`build_act_dataset` does `load_dataset(...).shuffle(seed=C.SEED).select(range(prefix))`. The
shuffle is a deterministic permutation of the full dataset and `select` takes a prefix of it,
so **rows 0–19,999 are byte-identical at any prefix**. The template `rng` is consumed in `map`
order, so those rows keep their templates too. Training is `select(range(n_train))` from the
front (se_common.py:720), so **every existing adapter stays valid. Nothing needs retraining.**

### What breaks

Eval is `select(range(len(dataset) − EVAL_SIZE, len(dataset)))` (se_common.py:831) — the *last*
1,024 rows. Raising the prefix moves it from rows 18,976–19,999 to the new tail, which is
disjoint. **Every eval in the tree becomes non-comparable**: all NB03 sweep runs, the floor
runs, NB04's §1b gate and ladder. Each needs re-scoring — delete `eval_scores.json` and
`eval_records.json`, leave the adapter, re-run the cell. ~5 min each, ~5 h for the tree.

This is the same defect `se/bootstrap_paired.py --audit` was written to catch, and it is why
`--pair C0` asserted at N=512.

### Fix the split while paying for it

Since the re-eval is being paid once anyway, make the eval window prefix-independent so this
cannot recur: anchor it at a fixed absolute range above the maximum N rather than at the end of
the dataset.

```
EVAL_START = 102_400                       # fixed; above any N_TRAIN we will use
eval  = rows [EVAL_START, EVAL_START + EVAL_SIZE)
train = rows [0, n_train)                  # unchanged, so adapters are unaffected
ACT_DATASET_PREFIX = 103_424
```

Train stays anchored at the front, so no adapter is invalidated. Eval stops moving no matter
what the prefix does later. Do **not** instead reserve rows 0–1023 for eval — that shifts the
training rows and would invalidate every adapter in the tree.

### Order of operations

1. Let the four N=16384 seed runs and the NB04 ladder finish **before** touching the prefix,
   or they land on the old split and need re-evaluating with everything else.
2. Confirm the ceiling — needs ≥ 103,424, not just ≥ 101,024:
   ```python
   from datasets import load_dataset
   print(load_dataset("Transluce/act_patch_qwen3_8b_counterfact", split="train").num_rows)
   ```
3. Change `ACT_DATASET_PREFIX` and the eval-window anchor in `se_config.py`.
4. Re-run NB02 to rebuild both ready datasets. Cheap in time (the Q transform was 3 s at 10k
   rows) but the vectors are ~1.6 GB per arm at 100k — check volume headroom.
5. Re-run NB01's §1a-equivalent check and `se/bootstrap_paired.py --audit`; the audit must
   report **one** eval set before anything is read.
6. Re-eval the existing tree (delete the two score files per run, re-run the notebook cells).
7. Start Stage 1.

## 4. Before an 18-hour run: make it survivable

`TrainingArguments` sets `save_strategy="no"` and the adapter is written only by
`trainer.save_model(save_dir)` after `trainer.train()` returns. `run_training`'s resume is
all-or-nothing at run granularity — it skips when `metrics.json` exists and otherwise starts
from scratch. **An 18 h run that dies at hour 17 loses all 17 hours.**

At N=16384 that risk was worth ignoring. At N=100k it is not. Before Stage 1, set
`save_strategy="steps"` with a `save_steps` giving ~1 h granularity (~1,000 steps) and wire
`resume_from_checkpoint` into `run_training`. Adapter checkpoints are small (LoRA + input map,
~0.5 GB). This does not change the training math.

## 5. Concurrency: measure before relying on it

Two concurrent training runs on one 80 GB A100 is plausible but unverified, and the failure
mode is an OOM that costs a full run under §4's no-checkpoint behaviour.

- Measure peak VRAM from a single run first (`torch.cuda.max_memory_allocated()`, or watch
  `nvidia-smi` during the current N=16384 runs) and only pair runs if two peaks plus headroom
  fit well inside 80 GB.
- Expect less than 2× throughput. An 8B model at batch 4 × seq 512 with gradient checkpointing
  is likely compute-bound, so two runs may share SMs at close to 1× aggregate. Verify on a
  short run before assuming 36 h of Stage 1 becomes 18 h.
- `se_env.bootstrap(gpu="required", min_vram_gb=80)` will still pass for both processes; it
  checks total VRAM, not free VRAM. It will not protect against this.
- If concurrency works, the natural pairing is Stage 1's two runs, which are independent.
  Do not pair two runs that write the same `run_dir`.

Fix §4 first regardless. Checkpointing is what makes a failed experiment cost an hour instead
of a day, and it matters more under concurrency, not less.

## 6. Open items to fold in on return

- **The four N=16384 seeds.** The only live claim from NB03: at one seed, exact_match +0.026
  and content_match +0.033 outside the null band, `has_changed_f1` inside it, predicted-changed
  rate flat (0.542 → 0.533) and both per-class deltas positive — i.e. *not* the response-bias
  artifact that explains the low-N points. Re-run `se/bootstrap_paired.py` once they land; it
  picks up new seeds automatically through `C.seeds_for`. If it evaporates, NB03 is null
  end-to-end and the writeup says so.
- **The NB04 ladder** (running now). Its premise — that `C0` collapses under rotation and the
  ladder traces the recovery — was falsified by NB03, where `C0 · R-Q` beat `C0 · R-id` at all
  three N and `C0 · R-id` never cleared the floor by more than 1.1 points. Read the rungs as a
  capacity result about the paper's configuration, not as the basis finding; the exit criteria
  in NB03 §5 already say to label it that way. Note it lands on the *old* eval split.
- **NB05 §1–§4.** No training at all, and §3's held-out ridge residual is a continuous quantity
  with no floor to normalize against — the one measurement in the project immune to the
  resolution problem that flattened NB03. Highest information per GPU-hour available. Also
  supplies the cross-model anchor NB03's exit criteria asked for and never got.
- **Scope statement for the writeup.** Until Stage 1 lands, the honest claim is bounded: *at
  N ≤ 16,384, on this task and family, there is no measurable activation advantage to
  decompose.* Do not write it as a claim about the paper's regime.
