"""How much of `has_changed` is predictable from the prompt's surface features alone?

The paper filters its activation-patching data (Appendix G.2): "we filter the dataset to
ensure roughly equal representation across each has-changed category while avoiding
over-representation of any specific (token, layer) combination. Without this balancing ...
it would be incredibly easy for the model to perform well by simply picking up on these
surface-level correlations and predicting the majority label."

We do not filter -- `build_act_dataset` takes `shuffle(seed).select(range(prefix))` off the
raw pool. This script measures whether that matters, by computing the quantity the filter
is supposed to control and comparing it against the paper's own published numbers.

The statistic is the accuracy of a classifier that sees ONLY `(token_type, chunk_id)` and
predicts that cell's majority has-changed label. Both features are visible in the prompt, so
it is an operational floor on how well an explainer can do with no activation at all.

It is fit on the TRAIN rows and scored on the EVAL rows. An in-sample majority-per-cell is
not a leakage measurement: our `token_type` is the literal token string, not the paper's five
coarse categories, so the grid is ~2,000 cells over 20,000 rows and an in-sample majority
mostly memorises. The giveaway is that it falls monotonically as n grows (0.852 at n=256 with
1.9 rows/cell, 0.733 at n=20,000 with 10.0) -- real leakage would be flat in n.

Reference values, summed from the paper's Table 7 (activation patching, Llama-3.1-8B target,
train split, n=13,454):

    majority-per-cell : 0.814     <- the paper's data, AFTER its filtering
    global majority   : 0.516
    leakage           : +0.298

Their filtering left ~81% of the label recoverable from surface features. Compare only the
held-out number below against it, and even then loosely: their 20 cells (5 coarse token types
x 4 layer chunks) and our ~2,000 are not the same partition, so ours is the leakage available
to a model that can read the exact token, which is strictly more information.

The second thing this prints is composition, which turned out to matter more. The paper's five
categories are Subject Final, Relation, and Orig/Other/Changed Answer Option -- all content
positions. Our pool also patches at prompt-template tokens ("or", "else", "and", "Respond",
"nothing", "one", "of", "with", "assistant"), which are ~43% of rows and map to none of the
paper's five. If the paper restricted to those five, we are measuring the activation's
contribution on a mixture that is largely positions it excluded, and that dilution is a
candidate explanation for our near-zero lift over the floor independent of any leakage.

Reported per split, because only the eval split bears on the measured floor:
the eval rows are the last EVAL_SIZE (se_common.py:831) and the train rows the first n_train.

Usage (CPU-only, no GPU, no model):

    python se/audit_dataset_balance.py
    python se/audit_dataset_balance.py --rotation Q --out reports/dataset_balance.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import se_config as C  # noqa: E402

PAPER_MAJORITY_PER_CELL = 0.814     # Table 7, train split, summed
PAPER_GLOBAL_MAJORITY = 0.516
PAPER_TRAIN_N = 13_454

# Prompt-template tokens: positions that carry no counterfactual content and map to none of the
# paper's five categories (Subject Final, Relation, Orig/Other/Changed Answer Option). Listed
# explicitly rather than pattern-matched so the classification is auditable and arguable.
TEMPLATE_TOKENS = {'"or"', '"and"', '"else"', '"with"', '"Respond"',
                   '"nothing"', '"one"', '"of"', '"assistant"'}


def bucket(token_type):
    if token_type in TEMPLATE_TOKENS:
        return "prompt template"
    if token_type == "relation_suffix":
        return "relation"
    return "content token"


def contingency(rows):
    """(token_type, chunk_id) -> [n_false, n_true] on `is_different`."""
    cells = defaultdict(lambda: [0, 0])
    for tt, ch, is_diff in rows:
        cells[(tt, ch)][1 if is_diff else 0] += 1
    return cells


def leakage(cells):
    """In-sample majority-per-cell. Retained only to show how much it overfits; do not read it
    as leakage -- see held_out_leakage."""
    n = sum(a + b for a, b in cells.values())
    if not n:
        return None
    maj = sum(max(a, b) for a, b in cells.values())
    f = sum(a for a, _ in cells.values())
    t = sum(b for _, b in cells.values())
    glob = max(f, t) / n
    return {"n": n, "n_cells": len(cells), "majority_per_cell": maj / n,
            "global_majority": glob, "leakage": maj / n - glob,
            "frac_changed": t / n}


def held_out_leakage(fit_rows, score_rows):
    """Fit the per-cell majority rule on `fit_rows`, score it on `score_rows`.

    Cells unseen during fitting fall back to the global majority of the fit rows, which is what
    a model with no information about that cell would do. This is the honest analogue of the
    paper's Table 7 number and the only one worth comparing against it.
    """
    cells = contingency(fit_rows)
    rule = {k: (b > a) for k, (a, b) in cells.items()}
    f = sum(a for a, _ in cells.values())
    t = sum(b for _, b in cells.values())
    fallback = t > f
    if not score_rows:
        return None
    hits = unseen = 0
    for tt, ch, is_diff in score_rows:
        key = (tt, ch)
        if key in rule:
            pred = rule[key]
        else:
            pred = fallback
            unseen += 1
        hits += (pred == bool(is_diff))
    n = len(score_rows)
    t_s = sum(1 for _, _, d in score_rows if d)
    glob = max(t_s, n - t_s) / n
    return {"n": n, "acc": hits / n, "global_majority": glob, "leakage": hits / n - glob,
            "frac_unseen_cells": unseen / n, "n_fit_cells": len(cells)}


def report(name, cells, verbose=False):
    s = leakage(cells)
    if s is None:
        print(f"  {name:24} (empty)")
        return None
    flag = ""
    if s["majority_per_cell"] > PAPER_MAJORITY_PER_CELL + 0.05:
        flag = "  <- materially leakier than the paper's filtered data"
    print(f"  {name:24} n={s['n']:>6,}  cells={s['n_cells']:>3}  "
          f"changed={s['frac_changed']:.3f}  "
          f"majority-per-cell={s['majority_per_cell']:.3f}  "
          f"global={s['global_majority']:.3f}{flag}")
    if verbose:
        for (tt, ch), (a, b) in sorted(cells.items(), key=lambda kv: str(kv[0])):
            tot = a + b
            print(f"       {str(tt)[:26]:28} chunk {ch}  "
                  f"n={tot:>5,}  changed={b/tot:.3f}  majority={max(a,b)/tot:.3f}")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rotation", default="identity",
                    help="which ready dataset to read (vectors differ, labels do not)")
    ap.add_argument("--verbose", action="store_true", help="print every cell")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from datasets import load_from_disk

    # se_common.ready_dataset_path, inlined: that module imports torch at load time and this
    # audit needs no model, so it stays runnable anywhere the dataset is readable.
    path = f"{C.ROTATION_DIR}/act_ready_{args.rotation}"
    ds = load_from_disk(path)
    n = len(ds)
    cols = [c for c in ("token_type", "chunk_id", "is_different") if c in ds.column_names]
    missing = {"token_type", "chunk_id", "is_different"} - set(cols)
    if missing:
        sys.exit(f"dataset at {path} lacks {sorted(missing)}; columns are {ds.column_names}")

    rows = list(zip(ds["token_type"], ds["chunk_id"], ds["is_different"]))

    print(f"SURFACE-FEATURE LEAKAGE — {path}")
    print(f"{n:,} rows   (ACT_DATASET_PREFIX = {C.ACT_DATASET_PREFIX:,})")
    print("=" * 100)
    print(f"  {'PAPER Table 7 (train)':24} n={PAPER_TRAIN_N:>6,}  "
          f"{'':10}{'':14}majority-per-cell={PAPER_MAJORITY_PER_CELL:.3f}  "
          f"global={PAPER_GLOBAL_MAJORITY:.3f}")
    print("-" * 100)

    out = {"path": path, "n_rows": n, "prefix": C.ACT_DATASET_PREFIX,
           "paper": {"majority_per_cell": PAPER_MAJORITY_PER_CELL,
                     "global_majority": PAPER_GLOBAL_MAJORITY, "n": PAPER_TRAIN_N},
           "splits": {}}

    out["splits"]["whole pool"] = report("whole pool", contingency(rows), args.verbose)

    # the eval split is the one that bears on the measured floor
    eval_rows = rows[n - C.EVAL_SIZE:]
    out["splits"]["eval (last %d)" % C.EVAL_SIZE] = report(
        f"eval (last {C.EVAL_SIZE:,})", contingency(eval_rows), args.verbose)

    for n_train in C.N_TRAIN_VALUES:
        if n_train > n - C.EVAL_SIZE:
            continue
        out["splits"][f"train N={n_train}"] = report(
            f"train N={n_train:,}", contingency(rows[:n_train]))

    # marginals -- an uneven chunk_id distribution is the other thing the filter controls
    print("\n  chunk_id marginal :", dict(sorted(
        (k, sum(1 for r in rows if r[1] == k)) for k in set(r[1] for r in rows))))
    tt_counts = defaultdict(int)
    for tt, _, _ in rows:
        tt_counts[tt] += 1
    print("  token_type marginal:", dict(sorted(tt_counts.items(), key=lambda kv: -kv[1])))

    # --- the honest leakage number: fit on train, score on eval ---------------------
    print("\n" + "=" * 100)
    print("HELD-OUT LEAKAGE — majority rule fit on the train rows, scored on the eval rows")
    print("  (the in-sample column above overfits: ~2,000 cells over 20,000 rows, and it falls")
    print("   monotonically with n, which real leakage would not)")
    print("-" * 100)
    out["held_out"] = {}
    for n_train in C.N_TRAIN_VALUES:
        if n_train > n - C.EVAL_SIZE:
            continue
        h = held_out_leakage(rows[:n_train], eval_rows)
        out["held_out"][f"fit N={n_train}"] = h
        print(f"  fit on N={n_train:>6,}  acc={h['acc']:.3f}  "
              f"global={h['global_majority']:.3f}  leakage={h['leakage']:+.3f}  "
              f"(cells unseen in eval: {h['frac_unseen_cells']:.1%})")

    # --- composition: which patch positions are we even measuring on? ----------------
    comp = defaultdict(int)
    comp_eval = defaultdict(int)
    for tt, _, _ in rows:
        comp[bucket(tt)] += 1
    for tt, _, _ in eval_rows:
        comp_eval[bucket(tt)] += 1
    print("\n" + "=" * 100)
    print("POSITION COMPOSITION — the paper's five categories are all content positions")
    print("-" * 100)
    for k in sorted(comp, key=lambda k: -comp[k]):
        print(f"  {k:18} pool {comp[k]:>6,} ({comp[k]/n:5.1%})   "
              f"eval {comp_eval[k]:>5,} ({comp_eval[k]/len(eval_rows):5.1%})")
    out["composition"] = {"pool": dict(comp), "eval": dict(comp_eval)}

    best = max(out["held_out"].values(), key=lambda h: h["acc"]) if out["held_out"] else None
    print("\n" + "=" * 100)
    print("READING THIS")
    if best:
        print(f"  Held-out leakage tops out at acc={best['acc']:.3f} against a global majority "
              f"of {best['global_majority']:.3f}.")
        if best["acc"] > PAPER_MAJORITY_PER_CELL + 0.05:
            print("  That is materially above the paper's 0.814 on a strictly finer partition, "
                  "so the missing\n  filter IS inflating the floor. Rebuild with G.2's "
                  "balancing before quoting a normalized number.")
        else:
            print("  That is not materially above the paper's 0.814, so surface leakage is not "
                  "the reason the\n  v=0 floor is high. The leakage is in the task as the paper "
                  "built it.")
    tmpl = comp_eval.get("prompt template", 0) / len(eval_rows)
    print(f"\n  {tmpl:.1%} of eval rows patch at a prompt-template token, which maps to none of "
          "the paper's five\n  categories. If the paper restricted to those five, the "
          "activation's contribution is being\n  measured on a mixture largely made of "
          "positions it excluded -- a dilution that would depress\n  the lift over floor "
          "independently of any leakage. Test it with:\n"
          "      python se/stratify_by_position.py")

    dest = args.out or f"{C.REPORTS_DIR}/dataset_balance_{args.rotation}.json"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwritten to {dest}")


if __name__ == "__main__":
    main()
