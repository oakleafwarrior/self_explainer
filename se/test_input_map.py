"""CPU smoke test for the capacity ladder and the activation-injection path.

Checks the three things the experiment's validity rests on:

  1. every arm starts at its base map (so arms differ only in what they may learn)
  2. a rank-r arm really is confined to a rank-r update, and only Cfull can express Q^T
  3. the injected activation actually reaches the forward pass

(3) is the one that matters most: the base repo's injection was a silent no-op, and a
rotation experiment run on top of that would produce a beautifully flat null result for
entirely the wrong reason.

    python se/test_input_map.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mini_qwen3  # noqa: E402
import se_config as C  # noqa: E402
from rotate import random_orthogonal  # noqa: E402
from se_common import ChunkInputMap  # noqa: E402


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    return bool(ok)


def fit_map(capacity_rank, target, d, n_chunks=1, steps=400, lr=0.05, base="identity"):
    """Train an arm to invert a known basis change, and report how close it gets.

    This is the capacity ladder in miniature: the question is not whether the optimizer
    works but whether the arm's parameterization can express the answer at all.
    """
    torch.manual_seed(0)
    m = ChunkInputMap(d, d, n_chunks, capacity_rank, base=base).double()
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr) \
        if any(p.requires_grad for p in m.parameters()) else None

    # fit and score on disjoint samples: a train residual measured on 512 finite rows can
    # beat the population bound by exploiting the sample covariance
    v, v_test = torch.randn(4096, d, dtype=torch.float64), torch.randn(4096, d, dtype=torch.float64)
    want, want_test = v @ target.T, v_test @ target.T
    chunk_ids = torch.zeros(len(v), dtype=torch.long)

    if opt is not None:
        for _ in range(steps):
            opt.zero_grad()
            loss = ((m(v, chunk_ids) - want) ** 2).mean()
            loss.backward()
            opt.step()

    with torch.no_grad():
        resid = ((m(v_test, chunk_ids) - want_test) ** 2).mean()
    return (resid / (want_test ** 2).mean()).item()


def main():
    all_ok = True
    d = 32

    # --- 1. every arm starts at its base ---------------------------------------
    print("\narms start at their base map")
    v = torch.randn(8, d, dtype=torch.float64)
    ids = torch.zeros(8, dtype=torch.long)
    for cap in (0, 4, 8, "full"):
        m = ChunkInputMap(d, d, 1, cap, base="identity").double()
        err = (m(v, ids) - v).abs().max().item()
        all_ok &= check(f"rank={cap!r} starts at identity", err < 1e-12, f"max |delta| = {err:.1e}")

    ridge = [torch.randn(d, d, dtype=torch.float64) * 0.1]
    for cap in (0, 4, "full"):
        m = ChunkInputMap(d, d, 1, cap, base="ridge", base_matrices=ridge).double()
        err = (m(v, ids) - v @ ridge[0].T.double()).abs().max().item()
        all_ok &= check(f"rank={cap!r} starts at the ridge map", err < 1e-6,
                        f"max |delta| = {err:.1e}")

    # --- 2. rank is a real constraint -----------------------------------------
    print("\nrank is a real capacity constraint (fitting Pi -> Q^T, d=32)")
    Q = random_orthogonal(d, seed=3)
    results = {}
    for cap in (0, 2, 8, 16, "full"):
        results[cap] = fit_map(cap, Q.T, d)
        print(f"      rank={str(cap):>4}  relative residual = {results[cap]:.4f}")

    all_ok &= check("C0 cannot invert the rotation at all", results[0] > 0.5,
                    f"residual = {results[0]:.3f}")
    all_ok &= check("Cfull inverts it essentially exactly", results["full"] < 1e-3,
                    f"residual = {results['full']:.2e}")
    all_ok &= check("residual decreases monotonically with rank",
                    results[0] > results[2] > results[8] > results[16] > results["full"])
    # The arm must not merely be worse than Cfull, it must be as good as its rank allows:
    # the best a rank-r update can do is I + trunc_r(Q^T - I), so the achievable residual
    # is the tail of the spectrum of Q^T - I. If the fitted arm sits well above this, the
    # ladder measures optimizer trouble rather than capacity.
    sv = torch.linalg.svdvals(Q.T - torch.eye(d, dtype=torch.float64))
    for r in (2, 8, 16):
        optimal = (sv[r:] ** 2).sum().item() / d
        all_ok &= check(f"rank-{r} arm attains its truncation bound",
                        abs(results[r] - optimal) < 0.05,
                        f"got {results[r]:.3f}, best possible {optimal:.3f}")

    # --- 3. the injected activation reaches the forward pass -------------------
    print("\nactivation injection")
    model = mini_qwen3.build(d=d, seed=1)
    placeholder_id = 7
    ids_in = torch.randint(0, 256, (3, 12))
    ids_in[:, 5] = placeholder_id
    vectors = torch.randn(3, d, dtype=torch.float64) * 5.0
    chunk_ids = torch.zeros(3, dtype=torch.long)
    imap = ChunkInputMap(d, d, 1, "full", base="identity").double()

    plain = model.get_input_embeddings()(ids_in)

    # the base repo's version: result discarded
    def buggy(model, input_map, input_ids, vectors, chunk_ids, placeholder_id):
        e = model.get_input_embeddings()(input_ids)
        mask = (input_ids == placeholder_id).unsqueeze(-1)
        e.masked_scatter(mask, input_map(vectors, chunk_ids).to(e.dtype))
        return e

    from se_common import build_inputs_embeds

    bugged = buggy(model, imap, ids_in, vectors, chunk_ids, placeholder_id)
    fixed = build_inputs_embeds(model, imap, ids_in, vectors, chunk_ids, placeholder_id)

    all_ok &= check("the base repo's injection is a no-op",
                    torch.equal(bugged, plain),
                    "masked_scatter is out-of-place; its result was discarded")
    all_ok &= check("the fixed injection changes the embedding",
                    not torch.equal(fixed, plain))
    all_ok &= check("only the placeholder position changed",
                    torch.equal(fixed[:, [0, 1, 2, 3, 4, 6, 7]], plain[:, [0, 1, 2, 3, 4, 6, 7]]))
    all_ok &= check("the placeholder holds exactly Pi v",
                    (fixed[:, 5] - vectors).abs().max().item() < 1e-12)

    # gradient actually flows back into the map
    imap2 = ChunkInputMap(d, d, 1, 8, base="identity").double()
    out = build_inputs_embeds(model, imap2, ids_in, vectors, chunk_ids, placeholder_id)
    out.sum().backward()
    all_ok &= check("gradient reaches the input map",
                    imap2.B.grad is not None and imap2.B.grad.abs().sum().item() > 0)
    # LoRA-style init: B = 0 means A has zero gradient on the very first step and starts
    # moving once B leaves zero. Expected, and the rank sweep above confirms it converges.
    all_ok &= check("A is zero-gradient at step 0 (B starts at zero, as intended)",
                    imap2.A.grad is not None and imap2.A.grad.abs().sum().item() == 0)

    # --- the Oracle arm: frozen at Q^T, and algebraically identical to C0 under R-id ---
    print("\nOracle arm")
    from se_common import build_input_map, oracle_matrices

    Qs = random_orthogonal(d, seed=13)
    oracle = build_input_map("Oracle", "oracle", d, d, n_chunks=2,
                             ridge_matrices=oracle_matrices(Qs, 2)).double()
    v_test = torch.randn(16, d, dtype=torch.float64)
    ids2 = torch.zeros(16, dtype=torch.long)
    undone = oracle(v_test @ Qs.T, ids2)          # feed it Qv, expect v back
    all_ok &= check("Oracle inverts Q exactly", (undone - v_test).abs().max().item() < 1e-6,
                    f"max |delta| = {(undone - v_test).abs().max().item():.2e}")
    all_ok &= check("Oracle has no trainable parameters",
                    oracle.trainable_parameter_count() == 0)

    c0 = build_input_map("C0", "identity", d, d, n_chunks=2).double()
    all_ok &= check("Oracle(Qv) == C0(v), the exactness check v2 §6 demands",
                    (oracle(v_test @ Qs.T, ids2) - c0(v_test, ids2)).abs().max().item() < 1e-6)

    # --- Cfull-rand starts orthogonal, so both rotation arms start equidistant ---
    rand_arm = build_input_map("Cfull-rand", "orthogonal", d, d, n_chunks=2).double()
    W = rand_arm.matrix_for_chunk(0).double()
    all_ok &= check("Cfull-rand initializes to an orthogonal map",
                    (W.T @ W - torch.eye(d, dtype=torch.float64)).abs().max().item() < 1e-5)

    # --- the trap box (§3.1): full-rank means full-rank ------------------------
    # The paper's release appends every trainable projector to LoRA's target_modules
    # (Appendix F.3), which caps a "full-rank" map at a rank-128 update. §3.1 says to assert
    # the parameter count before training and to make that assertion a test, not a comment.
    print("\nfull-rank trainability (§3.1's trap box)")
    from se_common import check_input_map_trainability, expected_input_map_params

    # deliberately mismatched dims: the cross-model arms are where the count is easiest to
    # get wrong, since d_E x d_M is not symmetric
    d_M, d_E, n_chunks = 64, 48, 3
    for cap in C.CAPACITIES:
        rank = C.CAPACITY_RANK[cap]
        if cap == "Oracle":
            mats = [torch.zeros(d_E, d_M, dtype=torch.float64)] * n_chunks
            m = ChunkInputMap(d_M, d_E, n_chunks, rank, base="oracle", base_matrices=mats)
            dims = (d_M, d_E)
        elif cap == "Cfull-rand":
            m = ChunkInputMap(d_M, d_M, n_chunks, rank, base="orthogonal")
            dims = (d_M, d_M)
        else:
            m = ChunkInputMap(d_M, d_E, n_chunks, rank, base="random")
            dims = (d_M, d_E)
        want = expected_input_map_params(cap, dims[0], dims[1], n_chunks)
        got = m.trainable_parameter_count()
        all_ok &= check(f"{cap:>10} trains exactly the parameters it should",
                        got == want, f"{got:,} == {want:,}")

    # the assertion has to be able to fail, or it is decoration
    named = [("base_model.model.input_map.full.0.weight",
              torch.nn.Parameter(torch.zeros(d_E, d_M)))]
    try:
        check_input_map_trainability(named, "Cfull", d_M, d_E, n_chunks=n_chunks)
        caught_short = False
    except AssertionError:
        caught_short = True
    all_ok &= check("a rank-capped map is caught", caught_short,
                    "one chunk's worth of parameters where three were expected")

    lora_named = [(f"base_model.model.input_map.full.{c}.lora_A.default.weight",
                   torch.nn.Parameter(torch.zeros(128, d_M))) for c in range(n_chunks)]
    lora_named += [(f"base_model.model.input_map.full.{c}.lora_B.default.weight",
                    torch.nn.Parameter(torch.zeros(d_E, 128))) for c in range(n_chunks)]
    try:
        check_input_map_trainability(lora_named, "Cfull", d_M, d_E, n_chunks=n_chunks)
        caught_lora = False
    except AssertionError:
        caught_lora = True
    all_ok &= check("a LoRA-wrapped map is caught", caught_lora,
                    "the F.3 failure mode: Cfull silently becomes C128")

    audit = check_input_map_trainability(
        [(f"base_model.model.input_map.full.{c}.weight",
          torch.nn.Parameter(torch.zeros(d_E, d_M))) for c in range(n_chunks)],
        "Cfull", d_M, d_E, n_chunks=n_chunks)
    all_ok &= check("a correctly wired Cfull passes", audit["full_rank_ok"])

    # C128 is the paper's projector: a frozen dense base plus a rank-128 update. It has to
    # be genuinely rank-capped, or the arm cannot measure the cap.
    Qd = random_orthogonal(d, seed=11)
    r_c128 = fit_map(8, Qd.T, d, base="random")          # stands in for rank << d at d=32
    r_cfull = fit_map("full", Qd.T, d, base="random")
    all_ok &= check("a rank-capped random projector cannot recover the frame",
                    r_c128 > 10 * max(r_cfull, 1e-12),
                    f"rank-8 residual {r_c128:.3f} vs full-rank {r_cfull:.2e}")

    # --- the ridge identity Pi^Q = Pi Q^T (v2 §7.4), which makes the recipe rotation-proof ---
    print("\nclosed-form ridge under rotation")
    from se_common import ridge_solve

    torch.manual_seed(0)
    n = 500
    HM = torch.randn(d, n, dtype=torch.float64)
    HE = torch.randn(d, n, dtype=torch.float64)
    Pi = ridge_solve(HM @ HM.T, HE @ HM.T, 1e-2, n)
    HMq = Qs @ HM
    PiQ = ridge_solve(HMq @ HMq.T, HE @ HMq.T, 1e-2, n)
    rel = ((PiQ - Pi @ Qs.T).norm() / (Pi @ Qs.T).norm()).item()
    all_ok &= check("fitting on rotated activations gives exactly Pi Q^T", rel < 1e-10,
                    f"relative error = {rel:.2e}")

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
