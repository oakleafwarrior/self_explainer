"""Shared machinery for the self-explainer mechanism experiments.

Lifted from oakleafwarrior/introspection_replication's `replication.ipynb` (activation
patching half) and generalized along the three axes the plan varies:

    rotation  identity / Q / Qscaled   — applied to the cached activation v
    capacity  C0 / C8 / C64 / C512 / Cfull — rank of the trainable input map on v
    init      identity / random / ridge    — what that map starts as

Everything else — prompt templates, chunking, LoRA config, eval metrics, seeds — is
held byte-identical to the base repo so the identity arm is comparable to its finished
runs. Deviations from the base repo are marked `DEVIATION:` and explained.

Notebooks import from here rather than redefining; NB00 verifies the config actually
matches what the finished runs used.
"""

import gc
import json
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn

import se_config as C


# ---------------------------------------------------------------------------
# model / tokenizer
# ---------------------------------------------------------------------------

def compute_dtype():
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# The input map runs in float32 while the explainer runs in `compute_dtype()`. Two
# independent reasons, both measured rather than assumed (NB04 §1):
#
#   1. Exactness. `Oracle`/R-Q and `C0`/R-id are the same computation only if Pi is applied
#      to the vector as stored. Casting v to bfloat16 *before* the map makes Oracle compute
#      Q^T bf16(Qv) against C0's bf16(v): 82% of the 4096 coordinates differ, at a relative
#      L2 error of 2.5e-3 -- larger than bf16 quantization itself. No tolerance on an
#      end-to-end score can absorb that, because it perturbs training, not just eval.
#   2. Trainability of `Cfull`. bfloat16 spacing at 1.0 is 2^-8 = 3.9e-3, and AdamW at
#      lr=1e-4 produces updates of ~1e-4. An identity-initialized map's diagonal entries
#      are therefore rounded straight back to 1.0 on every step and never move, while the
#      off-diagonals (starting at 0.0, where bf16 has range to spare) move freely. The arm
#      whose whole purpose is to represent Q^{-1} exactly would be optimizing under an
#      arbitrary dtype-induced constraint, in a direction the rotation arm needs most.
#
# Cost is ~0.5 GB of VRAM for Cfull (4 x 4096^2 fp32, doubled by PEFT's modules_to_save
# copy). The projected vector is cast to the embedding dtype in build_inputs_embeds, so
# the explainer's own arithmetic is unchanged.
MAP_DTYPE = torch.float32


def load_tokenizer(model_id=C.EXPLAINER_MODEL_ID):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.add_special_tokens({
        "additional_special_tokens": [C.PLACEHOLDER_TOKEN, C.FEATURE_START_TOKEN, C.FEATURE_END_TOKEN]
    })
    return tok


def special_ids(tokenizer):
    return {
        "placeholder": tokenizer.convert_tokens_to_ids(C.PLACEHOLDER_TOKEN),
        "feature_start": tokenizer.convert_tokens_to_ids(C.FEATURE_START_TOKEN),
        "feature_end": tokenizer.convert_tokens_to_ids(C.FEATURE_END_TOKEN),
    }


FEATURE_SPAN = f"{C.FEATURE_START_TOKEN}{C.PLACEHOLDER_TOKEN}{C.FEATURE_END_TOKEN}"


def load_base_model(tokenizer, model_id=C.EXPLAINER_MODEL_ID, use_4bit=C.USE_4BIT):
    """Fresh weights every call, so LoRA adapters never stack across runs."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    dtype = compute_dtype()
    quant = None
    if use_4bit:
        raise ValueError(
            "USE_4BIT is set. Computational invariance is exact in float64, close in "
            "bfloat16, and meaningless in 4-bit (plan §5.3). Rotated runs must be "
            "unquantized, and the identity control has to match them."
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant, dtype=dtype, device_map="auto",
    )
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    return model


# ---------------------------------------------------------------------------
# activation-patching dataset (verbatim prompt construction from the base repo)
# ---------------------------------------------------------------------------

ACTIVATION_TEMPLATES = [
    "If feature {feat} at layer {layer} is added to tokens {xt} when processing the text <<<{x}>>>, how would the output change?",
    "When feature {feat} at layer {layer} is added at tokens {xt} in the input <<<{x}>>>, what happens to the model's output?",
    "Consider the input text: <<<{x}>>>. If we steer layer {layer} towards feature {feat} at tokens {xt}, how does this affect the generated continuation?",
    "Given the text <<<{x}>>>, what would be the effect on the output if feature {feat} at layer {layer} is added to tokens {xt}?",
    "If we steer towards feature {feat} at layer {layer} and tokens {xt} when processing <<<{x}>>>, how would the model's response differ?",
]


def format_layer_range(layers):
    return str(layers[0] if len(layers) == 1 else f"{layers[0]}-{layers[-1]}")


def build_activation_target(original_text, ablated_text):
    if original_text != ablated_text:
        return f"The most likely output would change to <<<{ablated_text}>>>."
    return f"The output would remain unchanged from <<<{original_text}>>>."


def chunk_id_for_layers(layers):
    """Dataset layer range -> input-map index in [0, N_LAYER_CHUNKS)."""
    return int(layers[0] // 9)


def build_act_dataset(tokenizer, seed=C.SEED, prefix=C.ACT_DATASET_PREFIX):
    """Shuffle, truncate, and render the counterfact patching set into chat messages.

    The template `rng` is seeded and consumed in dataset order exactly as the base repo
    does it, so a given row gets the same template here as it did there. That is what
    makes the identity arm comparable to the finished runs.
    """
    from datasets import load_dataset

    rng = random.Random(seed)

    def build_user_turn(layers, xt, x):
        return rng.choice(ACTIVATION_TEMPLATES).format(
            feat=FEATURE_SPAN, layer=format_layer_range(layers), xt=xt, x=x
        )

    def to_messages(example):
        input_text = tokenizer.convert_tokens_to_string(example["input_tokens"])
        xt = tokenizer.convert_tokens_to_string([example["patch_position"]["orig_text_token"]])
        original_text = "".join(example["original_continuation"])
        ablated_text = "".join(example["ablated_continuation"])
        return {
            "messages": [
                {"role": "user", "content": build_user_turn(example["layer"], xt, input_text)},
                {"role": "assistant", "content": build_activation_target(original_text, ablated_text)},
            ],
            "chunk_id": chunk_id_for_layers(example["layer"]),
        }

    ds = load_dataset(C.ACT_DATASET, split="train").shuffle(seed=seed)
    ds = ds.select(range(prefix))     # base repo maps only the first 10k; keep identical
    return ds.map(to_messages)


def transform_vectors(dataset, M, batch_size=512):
    """Apply the basis change v -> M v to every cached activation.

    This is the whole cost of the rotation arm (§5.4): M_Q computes the identical
    function as M, so every has_changed label and every content string is unchanged and
    the counterfact patching pipeline does not need rerunning.
    """
    M = M.double()

    def _batch(rows):
        v = torch.tensor(
            [p["intervention_vector"] for p in rows["patch_position"]], dtype=torch.float64
        )
        Mv = (v @ M.T).tolist()
        patched = []
        for p, new_v in zip(rows["patch_position"], Mv):
            p = dict(p)
            p["intervention_vector"] = new_v
            patched.append(p)
        return {"patch_position": patched}

    return dataset.map(_batch, batched=True, batch_size=batch_size)


def ready_dataset_path(rotation, root=None):
    return f"{root or C.ROTATION_DIR}/act_ready_{rotation}"


def save_ready_dataset(dataset, rotation, root=None):
    path = ready_dataset_path(rotation, root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset.save_to_disk(path)
    return path


def load_ready_dataset(rotation, root=None):
    from datasets import load_from_disk

    path = ready_dataset_path(rotation, root)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no prepared dataset for rotation={rotation} at {path}. Run NB02 first."
        )
    return load_from_disk(path)


def check_rotation_exactness(dataset_rot, dataset_id, M, n=256, tol=1e-5):
    """Is the rotated dataset actually `M v` of the identity dataset, row for row?

    The end-to-end `Oracle`/R-Q vs `C0`/R-id comparison in NB04 is two *training runs*, so
    it cannot separate a wiring fault from ordinary run-to-run variance. This can: it is
    the same algebra, in float64, on the stored vectors, with no model involved. It is the
    check that actually falsifies the three suspects worth suspecting —

        row misalignment        `.map` reordered rows, or the two arms were built from
                                different shuffles of the base set
        wrong or stale Q        the dataset on disk was rotated by a different Q than the
                                one loaded now (a re-run of NB01 with a changed Q_SEED)
        wrong orientation       M applied as M^T, which is self-consistent everywhere else
                                and only shows up here

    A failure here means everything downstream is contaminated. A pass means the data is
    right and any score difference is training noise, which is a claim about tolerance
    rather than about wiring.
    """
    k = min(n, len(dataset_rot), len(dataset_id))
    v_id = torch.tensor(
        [p["intervention_vector"] for p in dataset_id[:k]["patch_position"]], dtype=torch.float64
    )
    v_rot = torch.tensor(
        [p["intervention_vector"] for p in dataset_rot[:k]["patch_position"]], dtype=torch.float64
    )
    Md = M.double()
    resid = (v_rot - v_id @ Md.T).norm(dim=1) / v_id.norm(dim=1).clamp_min(1e-30)
    # the round trip the Oracle arm actually performs, in the precision it will run in
    back = (v_rot @ Md).to(MAP_DTYPE).double()
    round_trip = (back - v_id).norm(dim=1) / v_id.norm(dim=1).clamp_min(1e-30)
    orth = (Md @ Md.T - torch.eye(Md.shape[0], dtype=torch.float64)).abs().max().item()
    chunks_agree = dataset_rot[:k]["chunk_id"] == dataset_id[:k]["chunk_id"]
    out = {
        "n_checked": k,
        "max_rel_residual": resid.max().item(),
        "max_rel_round_trip": round_trip.max().item(),
        "max_orthogonality_error": orth,
        "chunk_ids_aligned": bool(chunks_agree),
        "tol": tol,
    }
    out["passed"] = (out["max_rel_residual"] <= tol and out["max_rel_round_trip"] <= tol
                     and out["chunk_ids_aligned"])
    return out


# ---------------------------------------------------------------------------
# the input map on v — this is the capacity ladder (§3)
# ---------------------------------------------------------------------------

class ChunkInputMap(nn.Module):
    """Per-layer-chunk map from target activation space into explainer embedding space.

    Parameterized as  Pi = base + update,  where

        base    fixed, untrained: identity (dims match), a ridge-fitted Pi (§4), or a
                random projection (dims differ, the base repo's default)
        update  trainable, of the given rank:
                  rank == 0        no update at all                       -> C0
                  rank == r        B @ A with B zero-init, so Pi starts at base -> C8/C64/C512
                  rank == "full"   the whole d_E x d_M matrix, init at base     -> Cfull

    The minimum rank at which performance recovers under rotation is the measurement:
    a rank-r update cannot invert a full-rank d x d rotation, so recovery at low r means
    the frame mismatch is low-dimensional.
    """

    def __init__(self, d_target, d_explainer, n_chunks, rank, base="identity",
                 base_matrices=None, init_scale=0.02, seed=C.SEED):
        super().__init__()
        self.d_target, self.d_explainer = d_target, d_explainer
        self.n_chunks, self.rank, self.base_mode = n_chunks, rank, base

        if base in ("identity", "orthogonal") and d_target != d_explainer:
            raise ValueError(
                f"base={base!r} needs matching dims, got d_target={d_target} "
                f"d_explainer={d_explainer}. Use base='ridge' or base='random'."
            )
        if base in ("ridge", "oracle") and base_matrices is None:
            raise ValueError(
                f"base={base!r} requires base_matrices (one [d_E, d_M] per chunk). "
                "For the Oracle arm pass [Q.T] * n_chunks."
            )

        gen = torch.Generator().manual_seed(seed)
        self.identity_base = base == "identity"

        def base_matrix(c):
            if base == "identity":
                return torch.eye(d_explainer, d_target)
            if base in ("ridge", "oracle"):
                return torch.as_tensor(base_matrices[c]).float()
            if base == "random":
                return torch.randn(d_explainer, d_target, generator=gen) / (d_target ** 0.5)
            if base == "orthogonal":
                # Cfull-rand: a random orthogonal start, so the rotated and unrotated arms
                # sit the same distance from their solutions at init. Cfull-at-identity does
                # not — under R-id it is initialized *at* the answer (v2 §3.1).
                a = torch.randn(d_target, d_target, generator=gen, dtype=torch.float64)
                q, r = torch.linalg.qr(a)
                return (q * torch.sign(torch.diagonal(r)).unsqueeze(0)).float()
            raise ValueError(f"unknown base mode {base!r}")

        if rank == "full":
            # one fully trainable linear per chunk, initialized at the base
            self.full = nn.ModuleList(
                [nn.Linear(d_target, d_explainer, bias=False) for _ in range(n_chunks)]
            )
            for c, lin in enumerate(self.full):
                lin.weight.data.copy_(base_matrix(c))
            self.A = self.B = None
        else:
            self.full = None
            # an identity base needs no storage: 4 x 4096 x 4096 of explicit identity
            # would cost ~134 MB of VRAM to express `v -> v`
            if not self.identity_base:
                self.register_buffer(
                    "base", torch.stack([base_matrix(c) for c in range(n_chunks)])
                )
            if rank == 0:
                self.A = self.B = None
            else:
                self.A = nn.Parameter(
                    torch.randn(n_chunks, rank, d_target, generator=gen) * init_scale
                )
                # B starts at zero so the map is exactly `base` at step 0: the arms differ
                # only in how far they are allowed to move from a common starting point
                self.B = nn.Parameter(torch.zeros(n_chunks, d_explainer, rank))

    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def matrix_for_chunk(self, c):
        """The effective Pi for chunk c as a dense [d_E, d_M] tensor (NB06's diagnostic)."""
        if self.full is not None:
            return self.full[c].weight.detach().float()
        if self.identity_base:
            Pi = torch.eye(self.d_explainer, self.d_target)
        else:
            Pi = self.base[c].detach().float().cpu().clone()
        if self.A is not None:
            Pi = Pi + (self.B[c].detach().float().cpu() @ self.A[c].detach().float().cpu())
        return Pi

    def _dtype(self):
        for t in self.parameters():
            return t.dtype
        for t in self.buffers():
            return t.dtype
        return None

    def forward(self, v, chunk_ids):
        # Trainer's compute_loss context is a nullcontext on GPU (accelerate wraps only
        # model.forward), so today this map already runs outside autocast -- but the map is
        # called from three places and float32 here is a correctness requirement, not a
        # preference. Pin it rather than depend on where it happens to be called from.
        with torch.autocast(device_type=v.device.type, enabled=False):
            return self._forward(v, chunk_ids)

    def _forward(self, v, chunk_ids):
        if self.full is not None:
            dtype = self.full[0].weight.dtype
            out = v.new_empty(v.shape[0], self.d_explainer, dtype=dtype)
            for i in chunk_ids.unique():
                mask = chunk_ids == i
                out[mask] = self.full[i](v[mask].to(dtype))
            return out

        dtype = self._dtype() or v.dtype
        if self.identity_base and self.A is None:      # C0 with matching dims: pass v through
            return v.to(dtype)

        out = v.new_empty(v.shape[0], self.d_explainer, dtype=dtype)
        for i in chunk_ids.unique():
            mask = chunk_ids == i
            vi = v[mask].to(dtype)
            yi = vi if self.identity_base else vi @ self.base[i].T
            if self.A is not None:
                yi = yi + (vi @ self.A[i].to(dtype).T) @ self.B[i].to(dtype).T
            out[mask] = yi
        return out


def build_input_map(capacity, init, d_target, d_explainer, n_chunks=C.N_LAYER_CHUNKS,
                    ridge_matrices=None, seed=C.SEED):
    """Construct the arm's input map from the (capacity, init) labels used in run_dir.

    `Oracle` (rank 0, base frozen at Q^T) is the representability ceiling: it can express
    the inverse exactly and never has to learn it, so it separates "the map is not
    representable" from "the map is representable but not learnable at this N".
    """
    rank = C.CAPACITY_RANK[capacity]
    return ChunkInputMap(
        d_target, d_explainer, n_chunks, rank,
        base=init, base_matrices=ridge_matrices, seed=seed,
    )


def oracle_matrices(Q, n_chunks=C.N_LAYER_CHUNKS):
    """The Oracle arm's frozen map: Q^T, repeated per chunk.

    A single global Q means one Q^T inverts the frame at every layer, so multi-layer
    injection needs no per-layer bookkeeping (v2 §6).
    """
    return [Q.T.float() for _ in range(n_chunks)]


# ---------------------------------------------------------------------------
# the trap box (§3.1): the input map must be trained directly, never through LoRA
# ---------------------------------------------------------------------------

def expected_input_map_params(capacity, d_target, d_explainer, n_chunks=C.N_LAYER_CHUNKS):
    """Trainable parameters the arm is *supposed* to have.

        Cfull / Cfull-rand   n_chunks * d_E * d_M     the full-rank map
        C<r>                 n_chunks * r * (d_M + d_E)   a rank-r update, base frozen
        C0 / Oracle          0                        nothing trainable in the map

    The paper's projectors are per *layer*, so §3.1's trap box quotes 32 * 4096^2 ~ 537M.
    Ours are per layer *chunk* (N_LAYER_CHUNKS = 4, the paper's own patching protocol), so
    the same argument lands at 4 * 4096^2 ~ 67M. The shape of the check is what matters:
    a number that comes out at a few million when it should be tens of millions means the
    map was rank-capped somewhere.
    """
    rank = C.CAPACITY_RANK[capacity]
    if rank == "full":
        return n_chunks * d_explainer * d_target
    if rank == 0:
        return 0
    return n_chunks * rank * (d_target + d_explainer)


def check_input_map_trainability(named_parameters, capacity, d_target, d_explainer,
                                 n_chunks=C.N_LAYER_CHUNKS, strict=True):
    """§3.1's assertion: the map is full-rank trainable and PEFT never wrapped it.

    `model/utils.py:252-253` of the paper's release appends every trainable projector to the
    LoRA `target_modules` list, so a "full-rank" projector is really frozen-at-init plus a
    rank-128 update — the paper's footnote 7 is accurate about this. At d = 4096 a rank-128
    update cannot approximate Q^T, so if our `Cfull` ever ends up inside `target_modules` the
    central arm silently becomes `C128` and the experiment measures LoRA rank instead of
    basis. That is a silent failure, which is why this is an assertion on every run rather
    than a comment in the config.

    Takes an iterable of (name, parameter) pairs rather than a model so it can be tested
    without PEFT or a GPU. Returns the audit dict that goes into metrics.json.
    """
    trainable, frozen, lora_wrapped = 0, 0, []
    for name, param in named_parameters:
        if "input_map" not in name:
            continue
        if "lora_" in name:
            lora_wrapped.append(name)
        if param.requires_grad:
            trainable += param.numel()
        else:
            frozen += param.numel()

    expected = expected_input_map_params(capacity, d_target, d_explainer, n_chunks)
    audit = {
        "capacity": capacity,
        "input_map_trainable_params": trainable,
        "input_map_frozen_params": frozen,
        "input_map_expected_params": expected,
        "input_map_lora_wrapped": lora_wrapped,
        "full_rank_ok": trainable == expected and not lora_wrapped,
    }
    if strict:
        assert not lora_wrapped, (
            f"the input map was wrapped by LoRA: {lora_wrapped[:4]}. Under PEFT it is then "
            f"frozen-at-init plus a rank-{C.LORA_R} update, so this arm is not {capacity} "
            f"— it is C{C.LORA_R}. Remove it from target_modules and train it directly "
            f"(§3.1's trap box; Appendix F.3)."
        )
        assert trainable == expected, (
            f"{capacity} should train {expected:,} input-map parameters "
            f"({n_chunks} chunks x {d_explainer} x {d_target} for a full-rank map) but "
            f"{trainable:,} require grad. A count in the low millions where tens of millions "
            f"were expected means the map was rank-capped; zero means it was frozen."
        )
    return audit


# ---------------------------------------------------------------------------
# the no-activation floor (§4.2): the scale every result is reported against
# ---------------------------------------------------------------------------

def fraction_retained(score, floor, *, span=C.CONTRIBUTION_SPAN):
    """Fraction of the activation's contribution an arm keeps, on a FIXED scale.

        0.0   the arm is at the no-activation floor: rotation destroyed everything the
              injected vector was contributing
        1.0   the arm recovers the whole activation contribution the paper measured
              (Table 5's `- activation` ablation, 4.1 exact-match points)
        < 0   worse than passing no activation at all — a bug, or the rotated vector is
              actively misleading the explainer (§9)

    Rotation can only destroy the *value of v*; it cannot touch what the explainer gets
    from the prompt text. The paper's ablation bounds that value at ~4 exact-match points,
    so raw score differences understate the effect by roughly 15x and are not the scale the
    readings should be stated on (§4.2).

    `span` is `C.CONTRIBUTION_SPAN`, a constant. It is deliberately NOT the per-N
    `reference - floor` this function divided by before 2026-08-24: that quantity's own 95%
    CI contains zero at every N in the sweep and is negative at N=256, which makes the ratio
    a Fieller statistic with no finite mean (see the CONTRIBUTION_SPAN comment in
    se_config.py, and prereg_threshold_justification.md §1.1). With a constant denominator
    this is an exact linear rescale of `score - floor`, so a CI on the raw difference maps
    to a CI here by dividing by the same number.

    1.0 is therefore no longer "our unrotated arm at this N". Where that arm lands on this
    scale is a measurement, and NB03/NB07 report it as one.

    `span` is keyword-only on purpose. The old signature took `reference` third and
    positionally, so a stale `fraction_retained(v, floor, ref)` raises TypeError here
    instead of silently dividing a 4-point effect by a ~0.7 accuracy.
    """
    if not span or span != span:
        return float("nan")
    if not 0 < span < 0.25:
        raise ValueError(
            f"span={span!r} is not a plausible activation contribution. This is almost "
            f"certainly a per-N `reference - floor` (or a raw accuracy) reaching a "
            f"parameter that now expects a constant. Pass C.CONTRIBUTION_SPAN, or see "
            f"se_config.CONTRIBUTION_SPAN for why the per-N span was retired."
        )
    return (score - floor) / span


# 95% two-sided t quantiles. Seed counts here are 3 per arm, so the normal 1.96 understates
# every interval; df is small enough that the difference matters (t(4) = 2.78).
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 12: 2.179, 14: 2.145, 16: 2.120, 20: 2.086, 24: 2.064,
        30: 2.042, 40: 2.021, 60: 2.000}


def t_crit_95(df):
    """Two-sided 95% t quantile, rounded up to the nearest tabulated df (conservative)."""
    if df < 1:
        return float("inf")
    for k in sorted(_T95):
        if df <= k:
            return _T95[k]
    return 1.96


def closure_with_ci(rand, full, ridge):
    """The rank-artifact closure, with the check that its denominator is real.

        closure = (score(P-rand-full) - score(P-rand)) / (score(P-ridge) - score(P-rand))

    NB05 §5 and NB07 both read this, so it lives here rather than in either of them.

    **Why this is not a division.** The denominator is a difference of two measured arms, so
    it carries its own seed noise. `fraction_retained` used to have exactly this shape and it
    produced values of +2.59 and -13.67 once the denominator's own CI covered zero (see
    se_config.CONTRIBUTION_SPAN). There the fix was to divide by a constant; here there is no
    constant available, because the `P-rand -> P-ridge` gap is the quantity of interest. So
    the ratio has to be reported as a ratio — which means Fieller, not a point estimate.

    Numerator and denominator SHARE `rand`, so they are positively correlated:
    cov = var(mean(rand)). Ignoring that would misstate the interval in both directions.

    Fieller inverts the pivot (a - rho*b) / sqrt(v11 - 2 rho v12 + rho^2 v22) ~ t, giving the
    quadratic (b^2 - t^2 v22) rho^2 - 2(ab - t^2 v12) rho + (a^2 - t^2 v11) = 0. Its leading
    coefficient A is positive exactly when |b| / se(b) > t, i.e. exactly when the gap's own CI
    excludes zero. So "is the denominator resolvable" and "does Fieller give a bounded
    interval" are the same question, and `gap_resolvable` below reports it once.

    Args:
        rand, full, ridge: per-seed score sequences for P-rand, P-rand-full, P-ridge. Pass
            every seed, not a mean -- the spread is the point.

    Returns a dict. `closure` is None when the gap is not resolvable: a point estimate whose
    denominator could be zero is not a number to put in a table, and the honest report is the
    raw `full - rand` difference alongside the unresolvable gap.
    """
    import numpy as _np

    def arm(xs):
        a = _np.asarray([x for x in xs if x is not None], dtype=float)
        k = len(a)
        return {"mean": float(a.mean()) if k else None,
                "sd": float(a.std(ddof=1)) if k > 1 else None,
                "band": float(a.max() - a.min()) if k else None,
                "var_mean": float(a.var(ddof=1) / k) if k > 1 else None,
                "n_seeds": k}

    R, F, G = arm(rand), arm(full), arm(ridge)
    out = {"P-rand": R, "P-rand-full": F, "P-ridge": G}

    if None in (R["mean"], F["mean"], G["mean"]):
        return {**out, "gap": None, "gap_resolvable": False, "closure": None,
                "note": "an arm is missing"}

    a = F["mean"] - R["mean"]                 # numerator
    b = G["mean"] - R["mean"]                 # denominator: the P-rand -> P-ridge gap
    out["numerator"], out["gap"] = a, b

    if None in (R["var_mean"], F["var_mean"], G["var_mean"]):
        return {**out, "gap_resolvable": False, "closure": None,
                "note": "single seed on at least one arm -- the gap has no measured spread, "
                        "so the ratio cannot be checked. Run C.seeds_for(n) on all three."}

    v11 = F["var_mean"] + R["var_mean"]       # var(numerator)
    v22 = G["var_mean"] + R["var_mean"]       # var(denominator)
    v12 = R["var_mean"]                       # cov: both differences contain `rand`
    df = sum(x["n_seeds"] - 1 for x in (R, F, G))
    t = t_crit_95(df)

    se_gap = v22 ** 0.5
    out.update({"gap_se": se_gap, "df": df, "t_crit": t,
                "gap_ci95": [b - t * se_gap, b + t * se_gap],
                "seed_band_max": max(x["band"] for x in (R, F, G))})

    A = b * b - t * t * v22
    out["gap_resolvable"] = bool(A > 0)
    if A <= 0:
        return {**out, "closure": None,
                "note": (f"the P-rand -> P-ridge gap is {b:+.4f} with se {se_gap:.4f} "
                         f"(t{df} CI [{out['gap_ci95'][0]:+.4f}, {out['gap_ci95'][1]:+.4f}], "
                         f"which includes 0). Closure is undefined: report the raw "
                         f"P-rand-full - P-rand difference of {a:+.4f} instead, and say the "
                         f"gap it would normalize could not be measured apart from zero.")}

    B = a * b - t * t * v12
    Cq = a * a - t * t * v11
    disc = B * B - A * Cq
    if disc < 0:
        return {**out, "closure": a / b, "closure_ci95": None,
                "note": "Fieller discriminant < 0: no bounded solution despite a resolvable "
                        "gap. Report the point estimate as indicative only."}
    root = disc ** 0.5
    return {**out, "closure": a / b,
            "closure_ci95": sorted([(B - root) / A, (B + root) / A]),
            "note": ""}


def paired_delta(records_a, records_b, key="exact_match"):
    """Paired difference between two runs scored on the same eval items, with a CI.

    Not in the plan; added because §4.2's arithmetic leaves the independent binomial CI
    (~3 points at n=1024) uncomfortably close to the task's whole ~4-point dynamic range.
    Every arm here is evaluated on the *same* held-out examples, so the decidable quantity
    is the paired difference: McNemar's discordant counts give a standard error of
    sqrt(b + c) / n, which is several times tighter than sqrt(2) * the per-arm half-width.

    Takes the `eval_records.json` lists written by `eval_run`. Returns the delta, its 95%
    CI, and the discordant counts, so "separated" can be read against something.
    """
    def correct(rec):
        if key == "exact_match":
            return (rec["generated_text"].replace(" ", "")
                    == rec["target_text"].replace(" ", ""))
        if key == "content_match":
            return (rec.get("pred_content") or "").replace(" ", "") == \
                (rec.get("true_content") or "").replace(" ", "")
        raise ValueError(f"paired_delta supports exact_match / content_match, not {key!r}")

    n = min(len(records_a), len(records_b))
    a = [correct(r) for r in records_a[:n]]
    b = [correct(r) for r in records_b[:n]]
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    delta = (only_a - only_b) / n if n else float("nan")
    se = ((only_a + only_b) ** 0.5) / n if n else float("nan")
    return {"n": n, "delta": delta, "se": se,
            "ci95": (delta - 1.96 * se, delta + 1.96 * se),
            "discordant_a": only_a, "discordant_b": only_b,
            "score_a": sum(a) / n if n else float("nan"),
            "score_b": sum(b) / n if n else float("nan")}


def load_eval_records(n_train, rotation, capacity, init, task="patching", seed=C.SEED,
                      explainer_model_id=C.EXPLAINER_MODEL_ID, root=None):
    """The per-example eval records for one run, for `paired_delta`."""
    save_dir = C.run_dir(task, rotation, capacity, init, n_train,
                         explainer=explainer_model_id, seed=seed, root=root)
    path = f"{save_dir}/eval_records.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_alignment_state_dict(mats, path, n_layers, n_chunks=C.N_LAYER_CHUNKS):
    """Write a ridge fit in the format the paper's own loader consumes (Appendix F.4).

    `model/utils.py:121-156` loads `LinearAlignmentModule` state dicts keyed
    `alignments.{layer}.weight`, one entry per layer. No alignment-*training* code shipped
    with the release and the .pt artifacts are not public, so the pretrained-projector
    condition is not reproducible from it — but the expected format is documented by the
    loader, so emitting that shape makes our fit drop into their pipeline unmodified.

    Our maps are per layer chunk, so each layer gets its chunk's matrix under the
    proportional-depth rule (§7.4).
    """
    base, rem = divmod(n_layers, n_chunks)
    state, layer = {}, 0
    for c in range(n_chunks):
        size = base + (1 if c < rem else 0)
        for _ in range(size):
            state[f"alignments.{layer}.weight"] = mats[c].float().cpu().clone()
            layer += 1
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    return {"path": path, "n_layers": layer, "n_chunks": n_chunks,
            "keys": [f"alignments.{i}.weight" for i in range(min(3, layer))] + ["..."]}


# ---------------------------------------------------------------------------
# injection, tokenization, collation
# ---------------------------------------------------------------------------

def build_inputs_embeds(model, input_map, input_ids, intervention_vectors, chunk_ids,
                        placeholder_id):
    """Replace the <|patch_v|> embedding with Pi v.

    DEVIATION (bug fix): the base repo calls `input_embeds.masked_scatter(...)` and
    then returns `input_embeds`, discarding the result. `Tensor.masked_scatter` is
    out-of-place, so in the base repo the projected activation never entered the forward
    pass: the explainer saw only the prompt text, and the projection received no
    gradient. NB00 asserts this at runtime rather than taking it on faith. Here the
    return value is captured.
    """
    input_embeds = model.get_input_embeddings()(input_ids)
    mask = (input_ids == placeholder_id).unsqueeze(-1)
    v = input_map(intervention_vectors, chunk_ids)
    input_embeds = input_embeds.masked_scatter(mask, v.to(input_embeds.dtype))
    return input_embeds


def make_tokenize_fn(tokenizer):
    def tokenize(example):
        prompt_ids = tokenizer.apply_chat_template(
            example["messages"][:1], tokenize=True, add_generation_prompt=True, enable_thinking=False
        )["input_ids"]
        full_ids = tokenizer.apply_chat_template(
            example["messages"], tokenize=True, add_generation_prompt=False, enable_thinking=False
        )["input_ids"]
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        return {
            "input_ids": full_ids,
            "labels": labels,
            "intervention_vector": example["patch_position"]["intervention_vector"],
            "chunk_id": example["chunk_id"],
        }

    return tokenize


def make_collator(tokenizer, dtype=None):
    # MAP_DTYPE, not compute_dtype: the vector is quantized once, on the far side of the
    # input map, in build_inputs_embeds. Quantizing it here would quantize Pi's *input*.
    dtype = dtype or MAP_DTYPE

    def collate(features):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(f["input_ids"]) for f in features],
            batch_first=True, padding_value=tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(f["labels"]) for f in features],
            batch_first=True, padding_value=-100,
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": (input_ids != tokenizer.pad_token_id).long(),
            "intervention_vector": torch.tensor(
                [f["intervention_vector"] for f in features], dtype=dtype
            ),
            "chunk_id": torch.tensor([f["chunk_id"] for f in features], dtype=torch.long),
        }

    return collate


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def _warmup_kwarg(args_cls, ratio=0.03):
    """`warmup_ratio` was folded into `warmup_steps` (a float < 1 now means a ratio of total
    steps, same semantics) in newer transformers/trl; `TrainingArguments.__init__` and
    `SFTConfig.__init__` raise TypeError on the old name there. se_env.py's floors are floors,
    not pins, so an older install that still satisfies them may keep `warmup_ratio` as its own
    field — detect which this install has rather than assuming one API generation.
    """
    import inspect

    params = inspect.signature(args_cls.__init__).parameters
    return {"warmup_ratio": ratio} if "warmup_ratio" in params else {"warmup_steps": ratio}


def make_trainer_class(placeholder_id):
    from transformers import Trainer

    class InjectingTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            chunk_id = inputs.pop("chunk_id")
            intervention_vector = inputs.pop("intervention_vector")
            input_embeds = build_inputs_embeds(
                model, model.input_map, inputs["input_ids"],
                intervention_vector, chunk_id, placeholder_id,
            )
            outputs = model(
                inputs_embeds=input_embeds,
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
            )
            return (outputs.loss, outputs) if return_outputs else outputs.loss

    return InjectingTrainer


def run_training(n_train, rotation, capacity, init, tokenizer, dataset=None,
                 ridge_matrices=None, task="patching", seed=C.SEED, resume=True,
                 explainer_model_id=C.EXPLAINER_MODEL_ID, d_target=None):
    """One training run = one cell of the (rotation x capacity x init x n_train) grid.

    Returns the metrics dict and writes it, plus the adapter, under `run_dir(...)`.
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import TrainingArguments

    save_dir = C.run_dir(task, rotation, capacity, init, n_train,
                         explainer=explainer_model_id, seed=seed)
    metrics_path = f"{save_dir}/metrics.json"
    if resume and os.path.exists(metrics_path):
        print(f"[skip] {save_dir} already done")
        with open(metrics_path) as f:
            return json.load(f)

    os.makedirs(save_dir, exist_ok=True)
    dataset = dataset if dataset is not None else load_ready_dataset(rotation)
    tokenize = make_tokenize_fn(tokenizer)
    dataset = dataset.map(tokenize)

    eval_dataset = dataset.select(range(len(dataset) - C.EVAL_SIZE, len(dataset)))
    train_subset = dataset.select(range(n_train))

    tokenizer.padding_side = "right"
    base_model = load_base_model(tokenizer, explainer_model_id)
    d_explainer = base_model.config.hidden_size
    d_target = d_target or _target_hidden_size()

    input_map = build_input_map(capacity, init, d_target, d_explainer,
                                ridge_matrices=ridge_matrices, seed=seed)
    input_map.to(base_model.device, dtype=MAP_DTYPE)
    base_model.input_map = input_map

    peft_config = LoraConfig(
        r=C.LORA_R, lora_alpha=C.LORA_R * 2, lora_dropout=0.05,
        target_modules=C.LORA_TARGET_MODULES,
        # modules_to_save, NOT target_modules: the map is trained whole. Appendix F.3 —
        # the paper's release appends trainable projectors to target_modules, which caps
        # them at a rank-128 update; §3.1 says assert against that rather than trust it.
        modules_to_save=["input_map"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, peft_config)

    map_audit = check_input_map_trainability(
        model.named_parameters(), capacity, d_target, d_explainer
    )
    print(f"  input map: {map_audit['input_map_trainable_params']:,} trainable "
          f"(expected {map_audit['input_map_expected_params']:,}), "
          f"{map_audit['input_map_frozen_params']:,} frozen (PEFT's modules_to_save "
          f"keeps the original alongside its trainable copy), LoRA-wrapped: "
          f"{map_audit['input_map_lora_wrapped'] or 'no'}")

    trainer = make_trainer_class(special_ids(tokenizer)["placeholder"])(
        model=model,
        args=TrainingArguments(
            output_dir=save_dir, remove_unused_columns=False,
            per_device_train_batch_size=C.PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=C.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=C.LEARNING_RATE, lr_scheduler_type="cosine",
            optim="paged_adamw_8bit",
            bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="epoch", logging_steps=5, save_strategy="no",
            report_to="none", seed=seed, data_seed=seed,
            **_warmup_kwarg(TrainingArguments),
        ),
        train_dataset=train_subset,
        eval_dataset=eval_dataset,
        data_collator=make_collator(tokenizer),
    )

    train_result = trainer.train()
    eval_result = trainer.evaluate()
    trainer.save_model(save_dir)

    metrics = {
        "n_train": n_train, "rotation": rotation, "capacity": capacity, "init": init,
        "task": task, "seed": seed, "explainer": explainer_model_id,
        "input_map_trainable_params": input_map.trainable_parameter_count(),
        **{k: v for k, v in map_audit.items() if k != "capacity"},
        **train_result.metrics, **eval_result,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    del trainer, model, base_model
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def _target_hidden_size():
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(C.TARGET_MODEL_ID).hidden_size


# ---------------------------------------------------------------------------
# evaluation (paper's three metrics, §5)
# ---------------------------------------------------------------------------

CONTENT_RE = re.compile(r"<<<(.*?)>>>", re.DOTALL)


def parse_activation_generation(text):
    if "would change to" in text:
        verdict = "changed"
    elif "would remain unchanged from" in text:
        verdict = "unchanged"
    else:
        verdict = None
    match = CONTENT_RE.search(text)
    return verdict, (match.group(1).strip() if match else None)


@torch.no_grad()
def eval_run(n_train, rotation, capacity, init, tokenizer, dataset=None, ridge_matrices=None,
             task="patching", batch_size=16, max_new_tokens=32, seed=C.SEED, resume=True,
             explainer_model_id=C.EXPLAINER_MODEL_ID, d_target=None):
    """Score one saved run. Set n_train=None for the untrained baseline."""
    from peft import PeftModel
    from sklearn.metrics import f1_score

    save_dir = C.run_dir(task, rotation, capacity, init, n_train,
                         explainer=explainer_model_id, seed=seed)
    scores_path = f"{save_dir}/eval_scores.json"
    if resume and os.path.exists(scores_path):
        with open(scores_path) as f:
            return json.load(f)

    dataset = dataset if dataset is not None else load_ready_dataset(rotation)
    eval_dataset = dataset.select(range(len(dataset) - C.EVAL_SIZE, len(dataset)))

    base_model = load_base_model(tokenizer, explainer_model_id)
    d_explainer = base_model.config.hidden_size
    d_target = d_target or _target_hidden_size()
    base_model.input_map = build_input_map(
        capacity, init, d_target, d_explainer, ridge_matrices=ridge_matrices, seed=seed
    ).to(base_model.device, dtype=MAP_DTYPE)

    model = PeftModel.from_pretrained(base_model, save_dir) if n_train is not None else base_model
    model.eval()
    model.config.use_cache = True
    tokenizer.padding_side = "left"
    pid = special_ids(tokenizer)["placeholder"]

    records = []
    for start in range(0, len(eval_dataset), batch_size):
        batch = eval_dataset[start:start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(m[:1], tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)
            for m in batch["messages"]
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        vectors = torch.tensor(
            [p["intervention_vector"] for p in batch["patch_position"]], dtype=MAP_DTYPE
        ).to(model.device)
        chunk_ids = torch.tensor(batch["chunk_id"], dtype=torch.long).to(model.device)

        inputs_embeds = build_inputs_embeds(
            model, model.input_map, inputs["input_ids"], vectors, chunk_ids, pid
        )
        output_ids = model.generate(
            inputs_embeds=inputs_embeds, attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        # generate() with inputs_embeds has no discrete prompt ids to prepend, so
        # output_ids is already just the continuation
        generations = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for gen, is_diff, orig, abl in zip(generations, batch["is_different"],
                                           batch["original_continuation"],
                                           batch["ablated_continuation"]):
            orig_text, abl_text = "".join(orig), "".join(abl)
            verdict, content = parse_activation_generation(gen)
            records.append({
                "generated_text": gen.strip(),
                "target_text": build_activation_target(orig_text, abl_text),
                "pred_verdict": verdict,
                "true_verdict": "changed" if is_diff else "unchanged",
                "pred_content": content,
                "true_content": abl_text if is_diff else orig_text,
            })

    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()

    parseable = [r for r in records if r["pred_content"] is not None]
    result = {
        "n_train": n_train, "rotation": rotation, "capacity": capacity, "init": init,
        "task": task, "explainer": explainer_model_id, "seed": seed,
        "exact_match": float(np.mean([
            r["generated_text"].replace(" ", "") == r["target_text"].replace(" ", "")
            for r in records
        ])),
        "has_changed_f1": float(f1_score(
            [r["true_verdict"] for r in records],
            [r["pred_verdict"] or "unparseable" for r in records],
            labels=["changed", "unchanged"], average="macro",
        )),
        "content_match": float(np.mean([
            r["pred_content"].replace(" ", "") == r["true_content"].replace(" ", "")
            for r in parseable
        ])) if parseable else 0.0,
        "unparseable_rate": 1 - len(parseable) / len(records),
    }

    os.makedirs(save_dir, exist_ok=True)
    with open(scores_path, "w") as f:
        json.dump(result, f, indent=2)
    with open(f"{save_dir}/eval_records.json", "w") as f:
        json.dump(records, f, indent=2)
    return result


# ---------------------------------------------------------------------------
# input ablation — the free control (§2) and the base/instruct arm (§9)
# ---------------------------------------------------------------------------

ABLATION_SYSTEM_PROMPT = (
    "The following are multiple choice questions (with a correct answer). "
    "Output only the answer letter (A, B, C, or D) and nothing else, "
    "in the format Answer: x, where x is one of A, B, C, or D."
)

ANSWER_RE = re.compile(r"<<<\s*Answer:\s*([A-D])\s*>>>", re.IGNORECASE)


def build_ablation_user_turn(q, hint):
    return f"Question: {q}\nHint: {hint}\n\nIf the hint were removed how would the assistant answer change?"


def build_ablation_target(answer_no_hint, answer_with_hint):
    if answer_no_hint != answer_with_hint:
        return f"The most likely output would change to <<<Answer: {answer_no_hint}>>>."
    return f"The output would remain unchanged from <<<Answer: {answer_no_hint}>>>."


def build_ablation_dataset(seed=C.SEED, labels=None):
    """The MMLU-hint ablation set, rendered into chat messages.

    `labels` optionally overrides the ground truth with a dict
    {index: (answer_no_hint, answer_with_hint)} — that is how NB08 relabels the task
    for a different target model without touching anything else.
    """
    from datasets import load_dataset

    def derive(example, idx):
        no_hint = example["zeroshot_prediction"]
        with_hint = example["random_hint_prediction"]
        if labels is not None and idx in labels:
            no_hint, with_hint = labels[idx]
        q = example["original_user_prompt"].strip()
        hint = example["hint"].split(": ", 1)[-1]
        return {
            "q": q, "hint": hint,
            "answer_no_hint": no_hint, "answer_with_hint": with_hint,
            "messages": [
                {"role": "system", "content": ABLATION_SYSTEM_PROMPT},
                {"role": "user", "content": build_ablation_user_turn(q, hint)},
                {"role": "assistant", "content": build_ablation_target(no_hint, with_hint)},
            ],
        }

    ds = load_dataset(C.ABLATION_DATASET, split="train").shuffle(seed=seed)
    return ds.map(derive, with_indices=True)


def parse_ablation_generation(text):
    if "would change to" in text:
        verdict = "changed"
    elif "would remain unchanged from" in text:
        verdict = "unchanged"
    else:
        verdict = None
    match = ANSWER_RE.search(text)
    return verdict, (match.group(1).upper() if match else None)


def run_ablation_training(n_train, tag, tokenizer, dataset, seed=C.SEED, resume=True,
                          explainer_model_id=C.EXPLAINER_MODEL_ID, task="ablation"):
    """LoRA SFT on the input-ablation task. No activation is injected — that is the point.

    `tag` names the arm (e.g. "rot_identity", "base_explains_instruct"); the run lands
    under runs/<task>/<tag>/n_train_<n>.
    """
    from peft import LoraConfig, TaskType
    from trl import SFTConfig, SFTTrainer

    save_dir = f"{C.RUNS_DIR}/{task}/{tag}/n_train_{n_train}"
    metrics_path = f"{save_dir}/metrics.json"
    if resume and os.path.exists(metrics_path):
        print(f"[skip] {save_dir} already done")
        with open(metrics_path) as f:
            return json.load(f)
    os.makedirs(save_dir, exist_ok=True)

    tokenizer.padding_side = "right"
    eval_dataset = dataset.select(range(len(dataset) - C.EVAL_SIZE, len(dataset)))
    train_subset = dataset.select(range(n_train))
    base_model = load_base_model(tokenizer, explainer_model_id)

    trainer = SFTTrainer(
        model=base_model,
        args=SFTConfig(
            output_dir=save_dir, max_length=C.MAX_SEQ_LEN,
            assistant_only_loss=True, packing=False,
            per_device_train_batch_size=C.PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=C.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=C.LEARNING_RATE, lr_scheduler_type="cosine",
            optim="paged_adamw_8bit",
            bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="epoch", logging_steps=5, save_strategy="no",
            report_to="none", seed=seed, data_seed=seed,
            **_warmup_kwarg(SFTConfig),
        ),
        peft_config=LoraConfig(
            r=C.LORA_R, lora_alpha=C.LORA_R * 2, lora_dropout=0.05,
            target_modules=C.LORA_TARGET_MODULES, task_type=TaskType.CAUSAL_LM,
        ),
        train_dataset=train_subset,
        eval_dataset=eval_dataset,
    )

    train_result = trainer.train()
    eval_result = trainer.evaluate()
    trainer.save_model(save_dir)

    metrics = {"n_train": n_train, "tag": tag, "task": task, "seed": seed,
               **train_result.metrics, **eval_result}
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    del trainer, base_model
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


@torch.no_grad()
def eval_ablation(n_train, tag, tokenizer, dataset, batch_size=16, max_new_tokens=32,
                  explainer_model_id=C.EXPLAINER_MODEL_ID, task="ablation", resume=True,
                  keep_generations=True):
    """Score an ablation run. Returns the three metrics plus the raw generations.

    NB06 compares generations byte-for-byte across nominally-identical arms: input
    ablation passes no activation, so any difference under a rotation label is a leak.
    """
    from peft import PeftModel
    from sklearn.metrics import f1_score

    save_dir = f"{C.RUNS_DIR}/{task}/{tag}/n_train_{n_train}"
    scores_path = f"{save_dir}/eval_scores.json"
    if resume and os.path.exists(scores_path):
        with open(scores_path) as f:
            return json.load(f)

    eval_dataset = dataset.select(range(len(dataset) - C.EVAL_SIZE, len(dataset)))
    base_model = load_base_model(tokenizer, explainer_model_id)
    model = PeftModel.from_pretrained(base_model, save_dir) if n_train is not None else base_model
    model.eval()
    model.config.use_cache = True
    tokenizer.padding_side = "left"

    records = []
    for start in range(0, len(eval_dataset), batch_size):
        batch = eval_dataset[start:start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(m[:2], tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)
            for m in batch["messages"]
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
        for gen, no_hint, with_hint in zip(
            tokenizer.batch_decode(new_tokens, skip_special_tokens=True),
            batch["answer_no_hint"], batch["answer_with_hint"],
        ):
            verdict, letter = parse_ablation_generation(gen)
            records.append({
                "generated_text": gen.strip(),
                "target_text": build_ablation_target(no_hint, with_hint),
                "pred_verdict": verdict,
                "true_verdict": "changed" if no_hint != with_hint else "unchanged",
                "pred_letter": letter, "true_letter": no_hint,
            })

    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()

    parseable = [r for r in records if r["pred_letter"] is not None]
    result = {
        "n_train": n_train, "tag": tag, "task": task,
        "exact_match": float(np.mean([
            r["generated_text"].replace(" ", "") == r["target_text"].replace(" ", "")
            for r in records
        ])),
        "has_changed_f1": float(f1_score(
            [r["true_verdict"] for r in records],
            [r["pred_verdict"] or "unparseable" for r in records],
            labels=["changed", "unchanged"], average="macro",
        )),
        "content_match": float(np.mean([
            r["pred_letter"] == r["true_letter"] for r in parseable
        ])) if parseable else 0.0,
        "unparseable_rate": 1 - len(parseable) / len(records),
    }
    if keep_generations:
        result["generations"] = [r["generated_text"] for r in records]

    os.makedirs(save_dir, exist_ok=True)
    with open(scores_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def aggregate_seeds(df, group_cols=("rotation", "capacity", "init", "n_train"),
                    metrics=("exact_match", "has_changed_f1", "content_match")):
    """Collapse per-seed rows into mean / min / max, which is what the seed band plots.

    v2 §9's readings turn on "curves overlap" vs "separated", and neither is decidable
    without this. Rows with a single seed get a zero-width band and are labelled as such.
    """
    cols = [c for c in group_cols if c in df.columns]
    if "explainer_tag" in df.columns:
        cols = cols + ["explainer_tag"]
    metrics = [m for m in metrics if m in df.columns]
    agg = df.groupby(cols, dropna=False)[metrics].agg(["mean", "min", "max", "count"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg.reset_index().rename(
        columns={f"{metrics[0]}_count": "n_seeds"}
    ).drop(columns=[f"{m}_count" for m in metrics[1:]], errors="ignore")


def collect_results(task="patching", root=None):
    """Walk the runs tree and return every eval_scores.json as a DataFrame (NB07)."""
    import pandas as pd

    root = root or C.RUNS_DIR
    rows = []
    for dirpath, _, filenames in os.walk(f"{root}/{task}"):
        if "eval_scores.json" in filenames:
            with open(os.path.join(dirpath, "eval_scores.json")) as f:
                row = json.load(f)
            # the explainer is in the path, so older runs without the field still resolve
            row.setdefault("seed", C.SEED)
            for part in dirpath.split(os.sep):
                if part.startswith("expl_"):
                    row["explainer_tag"] = part[len("expl_"):]
                elif part.startswith("seed_"):
                    row["seed"] = int(part[len("seed_"):])
            metrics_path = os.path.join(dirpath, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    row["input_map_trainable_params"] = json.load(f).get(
                        "input_map_trainable_params"
                    )
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# closed-form ridge projector (§4)
# ---------------------------------------------------------------------------

def chunk_hidden_states(hidden_states, n_chunks):
    """Split the per-layer hidden states into n_chunks contiguous groups and average.

    Mirrors how the cached intervention vectors were built: a chunk's activations are
    averaged into a single vector before injection.

    This is also the layer-correspondence rule (§7.4, `C.LAYER_CORRESPONDENCE`). Applied to
    each model's own layer list, chunk k of a 36-layer 4B explainer covers the same
    *fractional* depth as chunk k of the 8B target, so cross-model fits are proportional-depth
    rather than l <-> l. Depth differs across Qwen3 sizes (1.7B has 28 blocks), so this is a
    design choice and the write-up has to state it.
    """
    layers = hidden_states[1:]        # drop the embedding output
    L = len(layers)
    base, rem = divmod(L, n_chunks)
    chunks, start = [], 0
    for c in range(n_chunks):
        size = base + (1 if c < rem else 0)
        chunks.append(torch.stack(layers[start:start + size]).mean(0))
        start += size
    return chunks


@torch.no_grad()
def fit_ridge_projector(explainer_model, target_model, tokenizer, texts, n_chunks=C.N_LAYER_CHUNKS,
                        lam=1e-2, batch_size=4, max_length=256, vector_transform=None):
    """Closed-form Pi_l = H^E (H^M)^T (H^M (H^M)^T + lam I)^{-1}, per chunk.

    This is ridge regression on cached tensors, not a training run: accumulate the two
    Gram matrices in one streaming pass, then solve. Minutes, no SGD, no hyperparameter
    search beyond lam.

    `vector_transform` applies the arm's basis change to the target activations before
    fitting, so the same function produces both the identity-arm and rotated-arm
    projectors. If a ridge fit on rotated activations recovers, the phenomenon is a
    linear-frame problem and you have shown it two independent ways (§4).
    """
    d_M = target_model.config.hidden_size
    d_E = explainer_model.config.hidden_size
    G = [torch.zeros(d_M, d_M, dtype=torch.float64) for _ in range(n_chunks)]   # H^M H^M^T
    Cx = [torch.zeros(d_E, d_M, dtype=torch.float64) for _ in range(n_chunks)]  # H^E H^M^T
    n_tokens = 0

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length)
        mask = enc["attention_mask"].bool()

        t_out = target_model(**{k: v.to(target_model.device) for k, v in enc.items()},
                             output_hidden_states=True)
        e_out = explainer_model(**{k: v.to(explainer_model.device) for k, v in enc.items()},
                                output_hidden_states=True)

        t_chunks = chunk_hidden_states(t_out.hidden_states, n_chunks)
        e_chunks = chunk_hidden_states(e_out.hidden_states, n_chunks)

        for c in range(n_chunks):
            hm = t_chunks[c].cpu()[mask].double()      # [tokens, d_M]
            he = e_chunks[c].cpu()[mask].double()      # [tokens, d_E]
            if vector_transform is not None:
                hm = hm @ vector_transform.double().T
            G[c] += hm.T @ hm
            Cx[c] += he.T @ hm
        n_tokens += int(mask.sum())

    mats = [ridge_solve(G[c], Cx[c], lam, n_tokens) for c in range(n_chunks)]
    return mats, {"n_tokens": n_tokens, "lam": lam, "n_chunks": n_chunks}


def ridge_solve(G, Cx, lam, n_tokens):
    """Pi = Cx (G + lam n I)^{-1}, solved rather than inverted.

    Split out from `fit_ridge_projector` because it carries the analytic result in v2 §7.4:
    fitting against rotated activations gives *exactly* Pi Q^T. With H^M -> Q H^M,

        Cx' = H^E (Q H^M)^T           = Cx Q^T
        G'  = Q H^M H^M^T Q^T + lam I = Q (G + lam I) Q^T          (Q orthogonal)
        Pi' = Cx Q^T (Q (G + lam I) Q^T)^{-1} = Cx (G + lam I)^{-1} Q^T = Pi Q^T

    So the closed-form recipe is provably as good on rotated activations as on unrotated
    ones. `test_input_map.py` checks this numerically.
    """
    eye = torch.eye(G.shape[0], dtype=torch.float64)
    return torch.linalg.solve((G.double() + lam * n_tokens * eye).T, Cx.double().T).T


def ridge_residual(mats, explainer_model, target_model, tokenizer, texts, **kw):
    """Fraction of explainer-activation variance the fitted map fails to explain.

    Sanity check on the fit before spending a training run on it: a projector that
    cannot predict held-out activations will not close the gap to the self-explainer.
    """
    n_chunks = len(mats)
    num = [0.0] * n_chunks
    den = [0.0] * n_chunks
    batch_size = kw.get("batch_size", 4)
    max_length = kw.get("max_length", 256)
    transform = kw.get("vector_transform")

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            enc = tokenizer(texts[start:start + batch_size], return_tensors="pt",
                            padding=True, truncation=True, max_length=max_length)
            mask = enc["attention_mask"].bool()
            t_out = target_model(**{k: v.to(target_model.device) for k, v in enc.items()},
                                 output_hidden_states=True)
            e_out = explainer_model(**{k: v.to(explainer_model.device) for k, v in enc.items()},
                                    output_hidden_states=True)
            t_chunks = chunk_hidden_states(t_out.hidden_states, n_chunks)
            e_chunks = chunk_hidden_states(e_out.hidden_states, n_chunks)
            for c in range(n_chunks):
                hm = t_chunks[c].cpu()[mask].double()
                he = e_chunks[c].cpu()[mask].double()
                if transform is not None:
                    hm = hm @ transform.double().T
                pred = hm @ mats[c].double().T
                num[c] += ((he - pred) ** 2).sum().item()
                den[c] += (he ** 2).sum().item()

    return [n / d for n, d in zip(num, den)]
