"""CPU smoke test for rotate.py — run this before spending any GPU time.

Folds the gains of a tiny RMSNorm transformer, rotates it, and checks the function is
unchanged, the residual stream really moved, and the diagnostics read the way NB06
assumes. If this fails, the 8B gate in NB01 will fail too, and debugging it here costs
seconds instead of minutes.

Uses a real Qwen3 built from config when `transformers` is importable, and falls back
to the pure-torch replica in mini_qwen3.py otherwise (same module layout, no download).

    python se/test_rotate.py
"""

import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mini_qwen3  # noqa: E402
from rotate import (  # noqa: E402
    apply_rotation,
    assert_no_residual_gains,
    compare_to_inverse,
    fold_rmsnorm_gains,
    invariance_gate,
    random_orthogonal,
    random_orthogonal_scaled,
    subspace_angles,
    tied_embeddings,
    untie_embeddings,
)


def build_model(d, tied, seed=0, backend="auto"):
    """Prefer the real architecture; fall back to the replica so this always runs."""
    if backend in ("auto", "transformers"):
        try:
            from transformers import Qwen3Config, Qwen3ForCausalLM

            torch.manual_seed(seed)
            cfg = Qwen3Config(
                vocab_size=256, hidden_size=d, intermediate_size=2 * d,
                num_hidden_layers=3, num_attention_heads=4,
                num_key_value_heads=2,            # exercise GQA (§5.3)
                head_dim=d // 4, max_position_embeddings=64,
                tie_word_embeddings=tied,
            )
            m = Qwen3ForCausalLM(cfg).to(torch.float64).eval()
            for name, p in m.named_parameters():
                if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
                    p.data = 1.0 + 0.3 * torch.randn_like(p.data)
            return m, "transformers"
        except Exception as exc:  # broken install, no network, version skew
            if backend == "transformers":
                raise
            print(f"  (transformers unavailable: {type(exc).__name__}; using mini_qwen3 replica)")
    return mini_qwen3.build(d=d, tied=tied, seed=seed), "mini_qwen3"


class _Encoding(dict):
    """Stand-in for a tokenizer BatchEncoding: supports .to() and dict unpacking."""

    def to(self, *a, **k):
        return self


class _StubTokenizer:
    """Deterministic fake tokenizer so invariance_gate can be exercised without a real one."""

    def __call__(self, batch, max_length=8, **kw):
        gen = torch.Generator().manual_seed(len(batch))
        return _Encoding(
            input_ids=torch.randint(0, 256, (len(batch), max_length), generator=gen),
            attention_mask=torch.ones(len(batch), max_length, dtype=torch.long),
        )


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    return bool(ok)


def main():
    all_ok = True
    d = 64
    ids = torch.randint(0, 256, (4, 32))
    Q = random_orthogonal(d, seed=1234)

    orth = (Q.T @ Q - torch.eye(d, dtype=torch.float64)).abs().max().item()
    print("\nQ construction")
    all_ok &= check("Q is orthogonal", orth < 1e-10, f"max |Q^T Q - I| = {orth:.2e}")
    all_ok &= check("Q is reproducible from its seed",
                    torch.equal(Q, random_orthogonal(d, seed=1234)))

    for tied in (False, True):
        model, backend = build_model(d, tied)
        print(f"\n{backend}, tie_word_embeddings={tied}")

        with torch.no_grad():
            ref_logits = model(ids).logits

        all_ok &= check("tie detection", tied_embeddings(model) == tied)

        if tied:
            # folding writes into lm_head, which IS the embedding when tied
            try:
                fold_rmsnorm_gains(copy.deepcopy(model))
                all_ok &= check("folding a tied model is refused", False, "no error raised")
            except RuntimeError:
                all_ok &= check("folding a tied model is refused", True)
            model = untie_embeddings(copy.deepcopy(model))
            with torch.no_grad():
                untied_logits = model(ids).logits
            err = (ref_logits - untied_logits).abs().max().item()
            all_ok &= check("untying preserves logits", err < 1e-12, f"max |delta| = {err:.2e}")

        # --- §5.1: verify folding alone before introducing Q ---
        folded = fold_rmsnorm_gains(copy.deepcopy(model))
        assert_no_residual_gains(folded)
        with torch.no_grad():
            fold_logits = folded(ids).logits
        err = (ref_logits - fold_logits).abs().max().item()
        all_ok &= check("folding preserves logits", err < 1e-8, f"max |delta| = {err:.2e}")

        # --- §5.2: folding + Q leaves the function identical ---
        rotated = apply_rotation(copy.deepcopy(folded), Q)
        with torch.no_grad():
            rot_out = rotated(ids, output_hidden_states=True)
            fold_out = folded(ids, output_hidden_states=True)
        err = (ref_logits - rot_out.logits).abs().max().item()
        all_ok &= check("rotation preserves logits", err < 1e-7, f"max |delta| = {err:.2e}")

        # --- but the residual stream really did move, and moved by exactly Q ---
        h_ref, h_rot = fold_out.hidden_states[2], rot_out.hidden_states[2]
        moved = (h_ref - h_rot).abs().max().item()
        matched = (h_rot - h_ref @ Q.T).abs().max().item()
        all_ok &= check("hidden states moved", moved > 1e-3, f"max |delta| = {moved:.2e}")
        all_ok &= check("hidden states equal Q h", matched < 1e-7,
                        f"max |h_rot - Q h| = {matched:.2e}")

        # --- the gain fold is load-bearing: rotating without it must be refused ---
        try:
            apply_rotation(copy.deepcopy(model), Q)
            all_ok &= check("rotating an unfolded model is refused", False, "no error raised")
        except AssertionError:
            all_ok &= check("rotating an unfolded model is refused", True)

        # --- the gate agrees with the direct logit comparison ---
        rep = invariance_gate(folded, rotated, _StubTokenizer(), ["x"] * 4,
                              max_length=16, batch_size=2)
        all_ok &= check("gate passes on M vs M_Q", rep["passed"], f"mean KL = {rep['mean_kl']:.2e}")

    # --- the gate must be able to fail, or it is not a gate ---
    folded = fold_rmsnorm_gains(build_model(d, tied=False, seed=1)[0])
    broken = copy.deepcopy(folded)
    broken.model.layers[1].self_attn.q_proj.weight.data += 0.05
    rep = invariance_gate(folded, broken, _StubTokenizer(), ["x"] * 4, max_length=16, batch_size=2)
    print("\ngate sensitivity")
    all_ok &= check("gate fails on a perturbed model", not rep["passed"],
                    f"mean KL = {rep['mean_kl']:.2e}, top-1 = {rep['top1_agreement']:.3f}")

    # --- non-orthogonal maps are refused: RMSNorm does not commute with them (§7.1) ---
    S = random_orthogonal_scaled(d, seed=7, log_scale=0.5)
    try:
        apply_rotation(fold_rmsnorm_gains(build_model(d, tied=False, seed=2)[0]), S)
        all_ok &= check("scaled map refused as a model transform", False, "no error raised")
    except ValueError:
        all_ok &= check("scaled map refused as a model transform", True)

    # --- the learned-map diagnostic reads the way NB06 assumes (§3) ---
    print("\nlearned-map diagnostic")
    exact = compare_to_inverse(Q.T, Q)
    junk = compare_to_inverse(random_orthogonal(d, seed=99), Q)
    all_ok &= check("Q^T scores as exact recovery", exact["rel_frobenius"] < 1e-10,
                    f"rel_frob = {exact['rel_frobenius']:.2e}")
    all_ok &= check("an unrelated map does not", junk["rel_frobenius"] > 0.1,
                    f"rel_frob = {junk['rel_frobenius']:.3f}")
    all_ok &= check("row cosines separate the two",
                    exact["row_cosine_mean"] > 0.99 and abs(junk["row_cosine_mean"]) < 0.2,
                    f"exact = {exact['row_cosine_mean']:.3f}, junk = {junk['row_cosine_mean']:.3f}")

    # an orthogonal residual still reads as "inverted the basis, up to a rotation"
    residual = random_orthogonal(d, seed=5) @ Q.T
    res = compare_to_inverse(residual, Q)
    all_ok &= check("orthogonality_defect sees through a residual rotation",
                    res["orthogonality_defect"] < 1e-10 and res["rel_frobenius"] > 0.1,
                    f"defect = {res['orthogonality_defect']:.2e}, "
                    f"rel_frob = {res['rel_frobenius']:.3f}")

    # low-rank arms: does the rank-r update point where Q^T - I points?
    U, S, Vh = torch.linalg.svd(Q.T - torch.eye(d, dtype=torch.float64))
    aligned = (U[:, :8] * S[:8]) @ Vh[:8]           # best rank-8 approximation
    unrelated = random_orthogonal(d, seed=11)[:, :8] @ random_orthogonal(d, seed=12)[:8]
    a_ang, u_ang = subspace_angles(aligned, Q.T - torch.eye(d, dtype=torch.float64), k=8), \
        subspace_angles(unrelated, Q.T - torch.eye(d, dtype=torch.float64), k=8)
    all_ok &= check("subspace angles separate aligned from unrelated rank-8 updates",
                    a_ang["angle_mean_deg"] < 1.0 and u_ang["angle_mean_deg"] > 45.0,
                    f"aligned = {a_ang['angle_mean_deg']:.1f}deg, "
                    f"unrelated = {u_ang['angle_mean_deg']:.1f}deg")

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
