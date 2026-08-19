"""Shared configuration for the self-explainer mechanism experiments.

One place for every constant the notebooks disagree about if left to themselves:
seeds, paths, the N_TRAIN grid, and the names of the experimental arms.

Import this at the top of every notebook. Do not redefine these inline.
"""

import os

# --- identity of the run -----------------------------------------------------

SEED = 67                       # inherited from introspection_replication/replication.ipynb;
                                # nothing is reused from it, so this is provenance, not a constraint
Q_SEED = 20260816               # seed for the orthogonal matrix Q; recorded in every artifact name

TARGET_MODEL_ID = "Qwen/Qwen3-8B"      # model act_patch_qwen3_8b_counterfact was captured from
EXPLAINER_MODEL_ID = "Qwen/Qwen3-8B"   # self-explainer arm; NB05 also uses a cross-model explainer

# --- paths -------------------------------------------------------------------
# RunPod convention: the pod's volume is the only filesystem that outlives the pod, so
# every artifact lands under it. se_env.persistent_root() resolves it (/workspace on a
# pod, /runpod-volume under serverless, SE_PERSIST_ROOT to override) and falls back to
# the home directory off-RunPod, so the CPU-only notebooks still run on a laptop.
# Set SE_ROOT to override the whole tree at once.

try:
    from se_env import persistent_root
except ImportError:                      # imported as a package rather than flat
    from .se_env import persistent_root

PERSIST_ROOT = persistent_root()

# `<volume>/self_explainer`, which on a pod is the same directory the repo is cloned into. That
# is deliberate — one folder on the volume holds the code and everything it produces — but it
# means the four output directories below land inside the git working tree, so .gitignore
# excludes them by name. Do not rename them here without renaming them there.
#
# (Was `<volume>/explanations_interp/self_explainer`. The extra level was a namespace inherited
# from the earlier project and grouped nothing.)
SE_ROOT = os.environ.get("SE_ROOT", f"{PERSIST_ROOT}/self_explainer")

ROTATION_DIR = f"{SE_ROOT}/rotation"          # Q, folded/rotated weights, gate reports
RUNS_DIR = f"{SE_ROOT}/runs"                  # one subdir per (task, rotation, capacity, init, n_train)
FIGURES_DIR = f"{SE_ROOT}/figures"
REPORTS_DIR = f"{SE_ROOT}/reports"            # gate numbers, preregistration, failure log

# Nothing outside this repo is read at runtime. Neither oakleafwarrior/introspection_replication
# nor TransluceAI/introspective-interp needs to exist on the pod:
#
#   - The base repo supplied no reusable artifact. Its activation injection discards the
#     result of an out-of-place `masked_scatter` (finished_notebooks/no_quantization/
#     notebook_Qwen3_8B_no_quant.ipynb, `build_inputs_embeds_projected`), so every finished
#     patching run was trained on the prompt text alone. NB00 §1 reproduces that code and
#     asserts the no-op, which is a claim about a snippet, not about a checkout.
#   - The paper's release was audited once and the findings are frozen in paper_audit.json
#     next to this file. They set the arm matrix (C0, C128, the P-* arms), so they travel
#     with the code rather than with a clone.
PAPER_AUDIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_audit.json")

# --- datasets ----------------------------------------------------------------

ACT_DATASET = "Transluce/act_patch_qwen3_8b_counterfact"
ABLATION_DATASET = "Transluce/input_ablation_qwen3_8b_mmlu_hint"

ACT_DATASET_PREFIX = 20_000     # base repo maps only the first 10k shuffled rows; keep identical

# Held out from every run so N_TRAIN comparisons share an eval set.
#
# 1024, not the 128 this started at. Appendix F/§4.2: rotation can only destroy the value of
# the injected activation, and the paper's own `- activation` ablation bounds that value at
# 64.0 -> 59.9 exact match for Qwen3-8B self-explanation. The whole measurable range is ~4
# points, so an eval set whose 95% binomial half-width is +/-8.7 (n=128) cannot resolve the
# experiment at all. At n=1024 the half-width is ~3.0 points at p=0.5 and ~2.9 at the
# operating point, which is what §4.2's "comfortably under 4 points" asks for. NB00 recomputes
# this and will refuse to preregister thresholds the eval set cannot resolve.
#
# The cost is real: eval is a generate() pass per run, so this is 8x the eval time of the
# original. It buys the only thing that makes the readings decidable.
EVAL_SIZE = 1024

# The paper's Table 5 `- activation` ablation for Qwen3-8B self-explanation on patching, and
# the Table 2 self-vs-cross margin. Reference points only: NB03 measures our own floor at our
# own N and eval set, because these are at theirs (§4.2's exit criterion says so explicitly).
PAPER_WITH_ACTIVATION = 0.640
PAPER_NO_ACTIVATION = 0.599       # the floor; the activation is worth ~4.1 points to them
PAPER_SELF_CROSS_MARGIN = 0.640 - 0.541

# --- sweep grid --------------------------------------------------------------
# Every point is a fresh run (see PAPER_AUDIT_PATH's note: nothing from the base repo is
# reusable), so the grid is chosen for what the readings need rather than for overlap with
# a finished sweep. v2 §3's seven points, all of them ours.

N_TRAIN_VALUES = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]   # v2 §3: the full sweep
N_TRAIN_LADDER = [512, 2048, 16384]     # capacity ladder and secondary-arm N values (v2 §3, §7.2)

# v2 §3 asked for "3 training seeds at N in {512, 8192} for every core arm; 1 seed elsewhere."
# Superseded: 3 seeds at EVERY N of the sweep. Two reasons, both in
# prereg_threshold_justification.md §4.
#
#   1. The modal preregistered reading (`sample_efficiency`) turns on destroyed(min N) at
#      N=128, and `attainable_optimum` requires a gap wider than the seed band "at all N".
#      Under the old grid five of the seven N values had no band, so the readings that quantify
#      over N were undecidable at exactly the points they lean on hardest.
#   2. With 3 seeds at 7 N values the seed variance can be pooled across N (14 df instead of
#      2), which halves the band half-width: t(2)/sqrt(3) = 2.48 sigma against
#      t(14)/sqrt(3) = 1.24 sigma. The band gets tighter than a per-N estimate could ever be,
#      which is what makes "inside the band" a claim rather than a shrug.
#
# Cost: NB03 goes 52 -> 96 runs (1.8x) and 169k -> 247k training examples (1.46x). Paid for
# out of the v2 §8 cut list, in its stated order.
SEEDS = [67, 68, 69]
MULTI_SEED_N = list(N_TRAIN_VALUES[:-1])      # every N is a multi-seed N now

# --- training hyperparameters (must match the base repo for the control to be valid) ---

MAX_SEQ_LEN = 512
LORA_R = 16
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# The paper's own LoRA rank (`lora_r: 128` in its configs). It matters here for one reason:
# `model/utils.py:252-253` appends every trainable projector to `target_modules`, so the
# paper's projectors are frozen-at-init plus a rank-128 update, never full-rank (Appendix
# F.3, and the paper's footnote 7 says so). The C128 arm exists to measure that cap, and
# LORA_TARGET_MODULES above deliberately does not contain the input map.
PAPER_LORA_R = 128
LEARNING_RATE = 1e-4
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
N_LAYER_CHUNKS = 4              # paper's 4-chunk patching protocol (footnote 4, §2.3)

USE_4BIT = False                # §5.3: invariance is meaningless in 4-bit. Never flip this on.

# --- experimental arms -------------------------------------------------------

# v2 tags R-id / R-Q / R-Qs; the directory names stay as below and ROTATION_LABEL maps them.
ROTATIONS = ["identity", "Q", "Qscaled"]
ROTATION_LABEL = {"identity": "R-id", "Q": "R-Q", "Qscaled": "R-Qs"}

# Capacity ladder. Oracle is frozen at Q^T (representability ceiling); Cfull-rand starts at a
# random orthogonal map so both rotation arms sit the same distance from their solution at
# init, which Cfull-at-identity does not.
#
# C0 is not a strawman: Appendix F.1 establishes that all four of the paper's act_patch configs
# omit `use_embed_proj`, `train.py:76` defaults it to False, and both models are 4096-dim so
# the dimension-mismatch fallback never fires. The paper injects the target activation raw,
# with no learned map, on the self *and* cross sides. C0 is therefore the faithful replication
# and Cfull is a deliberate augmentation, introduced so the rotation question is answerable at
# all. C128 is what Cfull degrades to if the map is ever passed through PEFT (F.3), which makes
# it both the paper's real capacity and the failure mode of a plausible implementation.
CAPACITIES = ["C0", "C8", "C128", "C512", "Cfull", "Cfull-rand", "Oracle"]
CAPACITY_RANK = {"C0": 0, "C8": 8, "C128": PAPER_LORA_R, "C512": 512,
                 "Cfull": "full", "Cfull-rand": "full", "Oracle": 0}

# What each rung is for, printed in tables and figure legends so a reader never has to guess
# which arms are the paper's and which are ours.
CAPACITY_STATUS = {
    "C0": "the paper's configuration (no map at all)",
    "C8": "ladder interior",
    "C128": "the paper's rank cap on a projector (F.3)",
    "C512": "ladder interior",
    "Cfull": "primary augmented arm",
    "Cfull-rand": "init control",
    "Oracle": "representability ceiling",
}

# how the input map starts. "oracle" and "ridge" supply explicit matrices; "ridge-frozen"
# is "ridge" with no training at all, evaluated straight through (§7.4).
PROJECTOR_INITS = ["identity", "random", "orthogonal", "ridge", "oracle"]
CAPACITY_DEFAULT_INIT = {"C0": "identity", "C8": "identity", "C128": "identity",
                         "C512": "identity", "Cfull": "identity",
                         "Cfull-rand": "orthogonal", "Oracle": "oracle"}

# Cross-model projector arms as (capacity, init) pairs, so NB03, NB05 and NB07 cannot disagree
# about what "P-rand" means. The distinction is the point of §7.4's new arm: P-rand is the
# paper's condition *faithfully* — a random dense projector correctable only by a rank-128
# update — while P-rand-full lifts the cap and nothing else. If P-rand-full closes most of the
# gap to P-ridge, the paper's random-vs-pretrained projector gap is largely a rank artifact
# rather than evidence about activation alignment.
PROJECTOR_ARMS = {
    "P-rand": ("C128", "random"),
    "P-rand-full": ("Cfull", "random"),
    "P-ridge": ("Cfull", "ridge"),
    "P-ridge-frozen": ("C0", "ridge"),
}

# §7.4: E_cross at 4B has a different depth from M at 8B, so l <-> l is a design choice.
# chunk_hidden_states splits each model's own layer list into N_LAYER_CHUNKS contiguous groups,
# which *is* proportional depth; recorded here so the write-up states the rule.
LAYER_CORRESPONDENCE = "proportional_depth"

# --- special tokens (inherited verbatim from the base repo) ------------------

PLACEHOLDER_TOKEN = "<|patch_v|>"
FEATURE_START_TOKEN = "<|feat_s|>"
FEATURE_END_TOKEN = "<|feat_e|>"


def run_dir(task, rotation, capacity, init, n_train, explainer=None, seed=None, root=None):
    """Canonical output directory for one training run.

    Every notebook writes here and NB07 reads the whole tree back, so the naming has to
    be mechanical rather than descriptive. The explainer is part of the path because
    NB05 trains two different ones through the same grid — without it, the cross-model
    run and the self-explainer run would collide on identical (rotation, capacity, init,
    n_train) coordinates.
    """
    root = root or RUNS_DIR
    tag = (explainer or EXPLAINER_MODEL_ID).split("/")[-1]
    # seed is part of the path only when it is not the default, so single-seed runs keep
    # the same directory they had before seed bands were added
    suffix = "" if seed in (None, SEED) else f"/seed_{seed}"
    return (f"{root}/{task}/expl_{tag}/rot_{rotation}/cap_{capacity}"
            f"/init_{init}/n_train_{n_train}{suffix}")


def seeds_for(n_train):
    """Every N of the sweep runs all three training seeds.

    Kept as a function, and kept keyed on MULTI_SEED_N, so the grid can be cut back to a
    subset of N under budget pressure without touching NB03 or NB05 — they call this rather
    than iterating SEEDS directly. Arms whose N values are N_TRAIN_LADDER already ran three
    seeds and are unaffected.
    """
    return list(SEEDS) if n_train in MULTI_SEED_N else [SEED]


def describe_arm(rotation, capacity, init):
    """Human-readable arm label used in figure legends."""
    return f"{rotation} · {capacity} · init={init}"
