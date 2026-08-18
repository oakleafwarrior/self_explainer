"""rotate.py — computational-invariance transform for RMSNorm transformers (Qwen3).

Deliverable 1 of the plan (§11). Three pieces:

  1. `fold_rmsnorm_gains`   absorb every RMSNorm elementwise gain into the linear
                            maps that consume its output, leaving pure x/rms(x)
  2. `apply_rotation`       left/right multiply the write/read paths by Q
  3. `invariance_gate`      the §5.5 check that M and M_Q compute the same function

After folding, normalization is x/rms(x), which commutes with any orthogonal Q
because ||Qx|| = ||x||. So M_Q computes the identical function as M, while every
activation read off the residual stream becomes Qv.

Construction follows SliceGPT (Ashkboos et al., 2024) §3.

Everything is built and applied in float64 and cast back at the end: invariance is
exact in float64, close in bfloat16, and meaningless in 4-bit (see §5.3).
"""

import json
import os

import torch


# ---------------------------------------------------------------------------
# module plumbing
# ---------------------------------------------------------------------------

def _decoder_layers(model):
    """The list of transformer blocks, for either a bare model or a *ForCausalLM."""
    inner = getattr(model, "model", model)
    return inner.layers


def _inner(model):
    return getattr(model, "model", model)


def qwen3_module_map(model):
    """Group every rotation-relevant weight by how it touches the residual stream.

    read  : consumes the residual, so W <- W Q^T
    write : contributes to the residual, so W <- Q W
    vocab : stored [vocab, d]; both embed_tokens and lm_head consume/produce a
            residual-space vector in the same orientation, so both are X <- X Q^T
    """
    inner = _inner(model)
    read, write, vocab = [], [], []

    for layer in _decoder_layers(model):
        read += [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj,
                 layer.mlp.gate_proj, layer.mlp.up_proj]
        write += [layer.self_attn.o_proj, layer.mlp.down_proj]

    vocab.append(inner.embed_tokens)
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None:
        vocab.append(lm_head)

    return {"read": read, "write": write, "vocab": vocab}


def tied_embeddings(model):
    """True when embed_tokens and lm_head share storage.

    Tying blocks the *fold*, not the rotation. Both matrices need the same rotation
    (X <- X Q^T, since Q^T = Q^{-1}), so a shared tensor is fine there as long as it is
    transformed once. But folding the final `model.norm` gain into lm_head writes to
    that same tensor, silently corrupting the embedding — which is what makes a tied
    model unusable here. Call `untie_embeddings` first. Qwen3-8B is untied already;
    0.6B/1.7B/4B are not (§5.3).
    """
    inner = _inner(model)
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        return False
    return inner.embed_tokens.weight.data_ptr() == lm_head.weight.data_ptr()


def untie_embeddings(model):
    """Give lm_head its own copy of the weight. Behavior-preserving.

    Costs one extra [vocab, d] tensor (~1.2 GB for Qwen3 in bf16), which is why the
    plan routes around it by targeting 8B rather than paying for it.
    """
    if not tied_embeddings(model):
        return model
    lm_head = model.lm_head
    lm_head.weight = torch.nn.Parameter(
        lm_head.weight.data.clone(), requires_grad=lm_head.weight.requires_grad
    )
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False
    return model


# ---------------------------------------------------------------------------
# 1. fold the RMSNorm gains
# ---------------------------------------------------------------------------

def _fold_gain_into(linear_modules, norm_module):
    """W <- W diag(g) for every consumer, then g <- 1.

    torch Linear stores weight as [out, in], and the normalized activation enters
    on the `in` axis, so scaling column j by g[j] is the correct absorption.
    """
    g = norm_module.weight.data.double()
    for lin in linear_modules:
        w = lin.weight.data
        lin.weight.data = (w.double() * g.unsqueeze(0)).to(w.dtype)
    norm_module.weight.data = torch.ones_like(norm_module.weight.data)


def fold_rmsnorm_gains(model):
    """Absorb all residual-stream RMSNorm gains into their downstream linears.

    Deliberately does NOT touch Qwen3's q_norm/k_norm: those normalize head-dimension
    slices *after* the read matrix, inside the head, so they are untouched by a
    residual-stream rotation (§5.3).

    Mutates `model` in place and returns it.
    """
    if tied_embeddings(model):
        raise RuntimeError(
            "embed_tokens and lm_head share storage: folding model.norm's gain into "
            "lm_head would corrupt the embedding matrix. Call untie_embeddings(model) "
            "first, or use an untied target (Qwen3-8B)."
        )

    for layer in _decoder_layers(model):
        _fold_gain_into(
            [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
            layer.input_layernorm,
        )
        _fold_gain_into(
            [layer.mlp.gate_proj, layer.mlp.up_proj],
            layer.post_attention_layernorm,
        )

    inner = _inner(model)
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None:
        _fold_gain_into([lm_head], inner.norm)
    else:
        raise RuntimeError("no lm_head found; final model.norm gain has no consumer to fold into")

    return model


def assert_no_residual_gains(model, atol=1e-6):
    """Post-condition for folding: every residual-stream norm is now the identity gain."""
    bad = []
    for i, layer in enumerate(_decoder_layers(model)):
        for name in ("input_layernorm", "post_attention_layernorm"):
            g = getattr(layer, name).weight.data
            if not torch.allclose(g.double(), torch.ones_like(g.double()), atol=atol):
                bad.append(f"layers.{i}.{name}")
    g = _inner(model).norm.weight.data
    if not torch.allclose(g.double(), torch.ones_like(g.double()), atol=atol):
        bad.append("model.norm")
    if bad:
        raise AssertionError(f"gains not folded: {bad}")


# ---------------------------------------------------------------------------
# 2. build and apply Q
# ---------------------------------------------------------------------------

def random_orthogonal(d, seed):
    """Haar-uniform Q in O(d), via QR of a float64 Gaussian.

    The diagonal sign correction matters: raw torch.linalg.qr is not Haar-uniform
    without it.
    """
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    a = torch.randn(d, d, generator=gen, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    return q


def random_orthogonal_scaled(d, seed, log_scale=1.0):
    """The optional third arm (§7.1): Q · diag(s) with s log-uniform.

    IMPORTANT — this is a DATA-SIDE transform only. It cannot be pushed into the
    model. Computational invariance rests on ||Mx|| = ||x||, which makes x/rms(x)
    commute with M; a diagonal scaling breaks that, so there is no model M_S that
    computes the same function as M and whose activations are S v. `apply_rotation`
    refuses non-orthogonal M for exactly this reason.

    The arm is still worth running, with a narrower claim: it distorts the activation
    distribution the explainer is fed while leaving the labels attached to the original
    M, so the Q vs Qscaled gap separates coordinate alignment from distributional
    compatibility. It just is not an invariance argument. See NB02.
    """
    gen = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
    q = random_orthogonal(d, seed)
    s = torch.exp((torch.rand(d, generator=gen, dtype=torch.float64) - 0.5) * 2 * log_scale)
    return q * s.unsqueeze(0)      # columns scaled: M = Q diag(s)


def apply_rotation(model, M, orthogonality_atol=1e-8):
    """Apply the residual-stream change of basis h -> M h, for orthogonal M.

    Mutates `model` in place and returns it. Requires folded gains — call
    `fold_rmsnorm_gains` first or the transform is wrong (the elementwise gain does
    not commute with M).
    """
    assert_no_residual_gains(model)

    M = M.double()
    d = M.shape[0]
    orthogonality_err = (
        M.T @ M - torch.eye(d, dtype=torch.float64, device=M.device)
    ).abs().max().item()
    if orthogonality_err > orthogonality_atol:
        raise ValueError(
            f"M is not orthogonal (max |M^T M - I| = {orthogonality_err:.2e}). "
            "RMSNorm only commutes with norm-preserving maps, so a non-orthogonal basis "
            "change does not leave the function invariant. Use random_orthogonal(), or "
            "apply the scaled map to cached activations only (see random_orthogonal_scaled)."
        )
    Minv = M.T

    # `random_orthogonal` builds M on the CPU while the model may be anywhere, and under
    # device_map="auto" an 8B model is sharded across devices, so the operand device is a
    # per-weight question rather than a global one. One copy of M is cached per device touched;
    # for a single-GPU pod that is exactly one extra d x d float64 tensor (128 MB at d=4096).
    _per_device = {}

    def _operands(device):
        if device not in _per_device:
            _per_device[device] = (M.to(device), Minv.to(device))
        return _per_device[device]

    groups = qwen3_module_map(model)
    seen = set()

    def _once(module):
        ptr = module.weight.data_ptr()
        if ptr in seen:
            return False
        seen.add(ptr)
        return True

    # read paths: input becomes M h, so W (M h) must equal W_old h  ->  W <- W M^{-1}
    for lin in groups["read"]:
        if not _once(lin):
            continue
        w = lin.weight.data
        _, Minv_d = _operands(w.device)
        lin.weight.data = (w.double() @ Minv_d).to(w.dtype)

    # write paths: output must land in the rotated frame -> W <- M W (and bias <- M b)
    for lin in groups["write"]:
        if not _once(lin):
            continue
        w = lin.weight.data
        M_d, _ = _operands(w.device)
        lin.weight.data = (M_d @ w.double()).to(w.dtype)
        if getattr(lin, "bias", None) is not None:
            b = lin.bias.data
            M_b, _ = _operands(b.device)
            lin.bias.data = (M_b @ b.double()).to(b.dtype)

    # embed_tokens rows are residual contributions: row <- M row, i.e. E <- E M^T
    # lm_head rows consume the residual:                              W_U <- W_U M^{-1}
    # For orthogonal M these coincide (M^T = M^{-1}), which is why a tied tensor is safe.
    inner = _inner(model)
    lm_head = getattr(model, "lm_head", None)

    emb = inner.embed_tokens
    if _once(emb):
        w = emb.weight.data
        M_d, _ = _operands(w.device)
        emb.weight.data = (w.double() @ M_d.T).to(w.dtype)
    if lm_head is not None and _once(lm_head):
        w = lm_head.weight.data
        _, Minv_d = _operands(w.device)
        lm_head.weight.data = (w.double() @ Minv_d).to(w.dtype)

    return model


def save_rotation(M, seed, path):
    """Checkpoint Q and its seed (deliverable 2). Stored in float64."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"M": M.double().cpu(), "seed": int(seed), "d": int(M.shape[0])}, path)
    with open(path + ".json", "w") as f:
        json.dump({"seed": int(seed), "d": int(M.shape[0]), "path": path}, f, indent=2)
    return path


def load_rotation(path):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    return blob["M"], blob["seed"]


# ---------------------------------------------------------------------------
# 3. the invariance gate (§5.5)
# ---------------------------------------------------------------------------

@torch.no_grad()
def invariance_gate(model_ref, model_rot, tokenizer, texts, max_length=512, batch_size=4,
                    kl_threshold=1e-4, top1_threshold=0.999):
    """Compare M and M_Q token-by-token. Returns a dict with a pass/fail flag.

    Accept at mean KL <~ 1e-4 and top-1 agreement >= 99.9% in bfloat16 (§5.5).
    Failures are nearly always a missed gain fold, a transposed [vocab, d] convention,
    or an unnoticed tied embedding.

    Both models must be resident, and statistics are reduced to scalars per batch:
    holding 256 x 512 tokens of Qwen3 logits would be ~79 TB.
    """
    kl_sum = kl_max = 0.0
    max_delta = 0.0
    agree_num = agree_den = 0

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding="max_length", truncation=True,
                        max_length=max_length)
        m = enc["attention_mask"].bool()

        lr = model_ref(**{k: v.to(model_ref.device) for k, v in enc.items()}).logits.float()
        lq = model_rot(**{k: v.to(model_rot.device) for k, v in enc.items()}).logits.float()
        lq = lq.to(lr.device)
        mm = m.to(lr.device)

        p = torch.log_softmax(lr, dim=-1)
        q = torch.log_softmax(lq, dim=-1)
        kl = (p.exp() * (p - q)).sum(-1)[mm]               # KL(ref || rot), per token

        kl_sum += kl.sum().item()
        kl_max = max(kl_max, kl.max().item())
        max_delta = max(max_delta, (lr - lq).abs()[mm].max().item())
        agree_num += (lr.argmax(-1) == lq.argmax(-1))[mm].sum().item()
        agree_den += int(mm.sum().item())

        del lr, lq, p, q, kl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "n_sequences": len(texts),
        "n_tokens": agree_den,
        "mean_kl": kl_sum / max(agree_den, 1),
        "max_kl": kl_max,
        "max_abs_logit_delta": max_delta,
        "top1_agreement": agree_num / max(agree_den, 1),
        "kl_threshold": kl_threshold,
        "top1_threshold": top1_threshold,
    }
    report["passed"] = bool(
        report["mean_kl"] <= kl_threshold and report["top1_agreement"] >= top1_threshold
    )
    return report


@torch.no_grad()
def disagreement_profile(model_ref, model_rot, tokenizer, texts, max_length=512,
                         batch_size=2, confident_gap=1.0,
                         gap_edges=(0.0, 0.05, 0.25, 1.0, 4.0, float("inf"))):
    """Where do the top-1 flips sit — on ties, or on decided predictions?

    `invariance_gate` reports one agreement number, and that number cannot tell a
    rounding artifact from a broken fold. This can.

    For every token it takes the *reference's* top-1 minus top-2 logit gap — how decided
    that prediction was before anything was touched — and buckets the flips by it.
    Re-rounding perturbs logits by a bounded amount, so it can only flip a prediction
    whose gap was already inside that amount. The signature is: flips pile up in the
    gap ~ 0 bin, the reference's top-1 is still the rotated model's *second* choice when
    it does flip, and agreement on decided tokens (gap > `confident_gap`) is 1.

    A real construction bug — a missed gain fold, a transposed [vocab, d] convention, an
    unnoticed tied embedding — perturbs logits by O(1) or worse and therefore flips
    decided tokens too. That is what this separates out, and it is the thing the raw
    agreement number cannot do, because a bad-enough tie distribution and a mild bug
    produce the same scalar.
    """
    edges = list(gap_edges)
    n_bins = len(edges) - 1
    bin_tokens, bin_flips = [0] * n_bins, [0] * n_bins
    gaps_at_flip = []
    n_tokens = n_flips = flips_still_rank2 = 0
    conf_tokens = conf_flips = 0

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding="max_length", truncation=True,
                        max_length=max_length)
        m = enc["attention_mask"].bool()

        lr = model_ref(**{k: v.to(model_ref.device) for k, v in enc.items()}).logits.float()
        lq = model_rot(**{k: v.to(model_rot.device) for k, v in enc.items()}).logits.float()
        lq = lq.to(lr.device)
        mm = m.to(lr.device)

        top2_r = lr.topk(2, dim=-1)
        top2_q = lq.topk(2, dim=-1)

        gap = (top2_r.values[..., 0] - top2_r.values[..., 1])[mm]      # [n]
        ref_top1 = top2_r.indices[..., 0][mm]
        q_top1 = top2_q.indices[..., 0][mm]
        q_top2 = top2_q.indices[..., 1][mm]

        flip = ref_top1 != q_top1
        n_tokens += int(gap.numel())
        n_flips += int(flip.sum().item())
        flips_still_rank2 += int((flip & (ref_top1 == q_top2)).sum().item())
        gaps_at_flip.append(gap[flip].cpu())

        conf = gap > confident_gap
        conf_tokens += int(conf.sum().item())
        conf_flips += int((conf & flip).sum().item())

        for b in range(n_bins):
            in_bin = (gap >= edges[b]) & (gap < edges[b + 1])
            bin_tokens[b] += int(in_bin.sum().item())
            bin_flips[b] += int((in_bin & flip).sum().item())

        del lr, lq, top2_r, top2_q, gap, ref_top1, q_top1, q_top2, flip
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    gaps_at_flip = (torch.cat(gaps_at_flip) if gaps_at_flip else torch.zeros(0))
    return {
        "n_tokens": n_tokens,
        "n_flips": n_flips,
        "top1_agreement": 1.0 - n_flips / max(n_tokens, 1),
        "confident_gap": confident_gap,
        "n_confident_tokens": conf_tokens,
        "n_confident_flips": conf_flips,
        # NaN, not 1.0, when nothing was decided: an empty set has no disagreements in it, and
        # a caller asserting `> 0.999` must not read that as a pass. NaN fails every comparison.
        "confident_agreement": (1.0 - conf_flips / conf_tokens) if conf_tokens else float("nan"),
        "flip_gap_median": float(gaps_at_flip.median()) if gaps_at_flip.numel() else 0.0,
        "flip_gap_p90": (float(gaps_at_flip.quantile(0.9)) if gaps_at_flip.numel() else 0.0),
        "flip_gap_max": float(gaps_at_flip.max()) if gaps_at_flip.numel() else 0.0,
        "frac_flips_still_rank2": flips_still_rank2 / max(n_flips, 1),
        "gap_edges": edges,
        "bin_tokens": bin_tokens,
        "bin_flips": bin_flips,
    }


def format_disagreement_profile(prof, title="disagreement profile"):
    edges = prof["gap_edges"]
    rows = []
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        n, f = prof["bin_tokens"][b], prof["bin_flips"][b]
        label = f"[{lo:g}, {hi:g})" if hi != float("inf") else f"[{lo:g}, inf)"
        rows.append(f"    {label:>14} : {f:7d} / {n:7d} flipped"
                    f"   = {(f / n if n else 0.0):.5f}")
    return (
        f"{title}\n"
        f"  tokens / top-1 flips   : {prof['n_tokens']} / {prof['n_flips']} "
        f"(agreement {prof['top1_agreement']:.5f})\n"
        f"  flip ref top-2 gap     : median {prof['flip_gap_median']:.4f}, "
        f"p90 {prof['flip_gap_p90']:.4f}, max {prof['flip_gap_max']:.4f}\n"
        f"  flips where ref top-1 is still the rotated model's 2nd choice : "
        f"{prof['frac_flips_still_rank2']:.5f}\n"
        f"  agreement on decided tokens (gap > {prof['confident_gap']:g}) : "
        f"{prof['confident_agreement']:.5f}  "
        f"({prof['n_confident_flips']}/{prof['n_confident_tokens']} flipped)\n"
        f"  flip rate by reference top-2 logit gap:\n" + "\n".join(rows)
    )


def gate_against_floor(report, floor_report, kl_ratio=3.0, top1_ratio=3.0,
                       kl_threshold=1e-4, top1_threshold=0.999):
    """Re-judge an invariance report against a measured quantization floor.

    `M ≡ M_Q` is exact in exact arithmetic, so every deviation the gate sees is arithmetic.
    In bfloat16 that deviation is large and irreducible: `apply_rotation` computes `W Qᵀ` in
    float64 and stores it back with `.to(w.dtype)`, so every weight is re-rounded to 8
    mantissa bits *after* a dense mix of all `d` coordinates. On Qwen3-0.6B that alone puts
    mean KL at ~1.2e-3, an order above the absolute 1e-4 bar, for reasons that are not bugs.
    v2 §5.5 called this in advance: "a fixed 1e-4 threshold may be unreachable for reasons
    that are not bugs."

    `floor_report` supplies the scale. The right null is the *folded* model: folding is also
    an exact identity in exact arithmetic, and it is re-rounded through the same path, but it
    involves no change of basis. What it costs is what rewriting every weight costs. A
    rotation that stays within a few multiples of that has not broken anything the arithmetic
    had not broken already. Measured on the mini_qwen3 replica in bf16, fold-only is 6.9e-06
    and fold+rotate is 1.3e-05 — a ratio of 1.9.

    The bar never loosens below the absolute one — it is `max(absolute, ratio * floor)` — so
    in float32, where the floor is ~3e-09, this reduces exactly to the original 1e-4 test.
    One formula covers both regimes, and the failure it is meant to catch (a missed gain fold,
    a transposed [vocab, d] convention, a tied embedding) lands orders above the floor rather
    than a few multiples of it, so the relative form keeps the power the absolute one had.
    """
    floor_kl = float(floor_report["mean_kl"])
    floor_disagree = 1.0 - float(floor_report["top1_agreement"])

    kl_bar = max(kl_threshold, kl_ratio * floor_kl)
    disagree_bar = max(1.0 - top1_threshold, top1_ratio * floor_disagree)

    out = dict(report)
    out["floor_mean_kl"] = floor_kl
    out["floor_top1_agreement"] = float(floor_report["top1_agreement"])
    out["kl_ratio_to_floor"] = (report["mean_kl"] / floor_kl) if floor_kl > 0 else float("inf")
    out["kl_threshold"] = kl_bar
    out["top1_threshold"] = 1.0 - disagree_bar
    out["relative_to"] = "folded model (same re-rounding, no change of basis)"
    out["passed"] = bool(
        report["mean_kl"] <= kl_bar
        and (1.0 - report["top1_agreement"]) <= disagree_bar
    )
    return out


def format_gate_report(report, title="invariance gate", show_verdict=True):
    """Render a gate report.

    `show_verdict=False` prints the numbers without the accept bars or the PASS/FAIL line, for
    a report being used as a *measurement* rather than a test — the folded model in bf16, whose
    `passed` field is computed against an absolute threshold it is not being asked to meet.
    Printing "FAIL — do not proceed" under a number that is doing its job is worse than
    printing nothing.
    """
    floor = ""
    if "floor_mean_kl" in report:
        floor = (
            f"  quantization floor     : {report['floor_mean_kl']:.3e} mean KL "
            f"({report['relative_to']})\n"
            f"  this run / floor       : {report['kl_ratio_to_floor']:.2f}x\n"
        )
    kl_accept = f"   (accept <= {report['kl_threshold']:.2e})" if show_verdict else ""
    top1_accept = f"   (accept >= {report['top1_threshold']:.5f})" if show_verdict else ""
    verdict = (
        f"\n  VERDICT                : "
        f"{'PASS' if report['passed'] else 'FAIL — do not proceed'}"
    ) if show_verdict else ""
    return (
        f"{title}\n"
        f"  sequences / tokens     : {report['n_sequences']} / {report['n_tokens']}\n"
        + floor +
        f"  mean per-token KL      : {report['mean_kl']:.3e}{kl_accept}\n"
        f"  max per-token KL       : {report['max_kl']:.3e}\n"
        f"  max |logit delta|      : {report['max_abs_logit_delta']:.3e}\n"
        f"  top-1 agreement        : {report['top1_agreement']:.5f}{top1_accept}"
        + verdict
    )


# ---------------------------------------------------------------------------
# diagnostic: did the explainer learn Q^T? (§3)
# ---------------------------------------------------------------------------

def compare_to_inverse(Pi, M):
    """How close is a learned input map Pi to the true inverse of the basis change?

    The plan asks for "principal angles between their row spaces". That measure is
    vacuous here: Pi and M^{-1} are both square and (generically) full rank, so both
    row spaces are all of R^d and every principal angle is 0 regardless of whether the
    model learned anything. What actually discriminates:

      rel_frobenius       ||Pi M - I||_F / sqrt(d).   0 = exact recovery.
                          ~1.4 for an unrelated orthogonal map (E||Pi M - I||_F^2 = 2d).
      row_cosine_*        cosine between corresponding rows of Pi and M^{-1}.
                          1 = the map is literally Q^T row by row.
      singular_*          spectrum of Pi M. All ones <=> Pi M is orthogonal, i.e. Pi
                          inverted the basis change up to a residual rotation the
                          explainer is free to absorb into its own weights.

    The last one is the honest test. An explainer does not need Pi = Q^T exactly; it
    needs Pi Q to be *something it can read*, and any orthogonal residual can be
    absorbed downstream. Report all three.
    """
    Pi = Pi.double().cpu()
    M = M.double().cpu()
    d = M.shape[0]

    prod = Pi @ M
    rel_frob = ((prod - torch.eye(d, dtype=torch.float64)).norm() / (d ** 0.5)).item()

    target = torch.linalg.inv(M)
    cos = torch.nn.functional.cosine_similarity(Pi, target, dim=1)
    sv = torch.linalg.svdvals(prod)

    return {
        "rel_frobenius": rel_frob,
        "row_cosine_mean": cos.mean().item(),
        "row_cosine_median": cos.median().item(),
        "frac_rows_cos_above_0.5": (cos > 0.5).double().mean().item(),
        "singular_mean": sv.mean().item(),
        "singular_std": sv.std().item(),
        "singular_min": sv.min().item(),
        "singular_max": sv.max().item(),
        "orthogonality_defect": (sv - 1.0).abs().mean().item(),   # 0 <=> Pi M orthogonal
    }


def subspace_angles(A, B, k=None):
    """Principal angles (degrees) between the leading-k row spaces of A and B.

    This is the version of the angle diagnostic that carries information: for the
    low-rank ladder arms the learned update is genuinely rank-r, so asking whether its
    row space lines up with the top-r right-singular subspace of Q^T - I is a real
    question. Pass k = the arm's rank.
    """
    def leading_rowspace(X, k):
        X = X.double().cpu()
        _, _, vh = torch.linalg.svd(X, full_matrices=False)
        return vh[:k].T                      # [d, k], orthonormal columns

    k = k or min(A.shape[0], B.shape[0])
    qa, qb = leading_rowspace(A, k), leading_rowspace(B, k)
    sv = torch.linalg.svdvals(qa.T @ qb).clamp(-1.0, 1.0)
    angles = torch.rad2deg(torch.arccos(sv))
    return {
        "k": int(k),
        "angle_mean_deg": angles.mean().item(),
        "angle_median_deg": angles.median().item(),
        "angle_min_deg": angles.min().item(),
        "frac_under_45deg": (angles < 45).double().mean().item(),
    }
