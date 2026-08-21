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

Reference values, summed from the paper's Table 7 (activation patching, Llama-3.1-8B target,
train split, n=13,454):

    majority-per-cell : 0.814     <- the paper's data, AFTER its filtering
    global majority   : 0.516
    leakage           : +0.298

So the filtering left ~81% of the label recoverable from surface features. If our numbers
land near those, the unfiltered pool is not the reason our v=0 floor is high and the floor is
a property of the task. If ours are materially higher, the missing filter is a confound on
the floor and every reading normalized to it.

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


def contingency(rows):
    """(token_type, chunk_id) -> [n_false, n_true] on `is_different`."""
    cells = defaultdict(lambda: [0, 0])
    for tt, ch, is_diff in rows:
        cells[(tt, ch)][1 if is_diff else 0] += 1
    return cells


def leakage(cells):
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

    ev = out["splits"][f"eval (last {C.EVAL_SIZE})"]
    print("\n" + "=" * 100)
    print("READING THIS")
    print(f"  The eval split's majority-per-cell is {ev['majority_per_cell']:.3f}. That is an "
          "operational ceiling on\n  what an explainer can score on has-changed with no "
          "activation at all, from surface features only.")
    if ev["majority_per_cell"] > PAPER_MAJORITY_PER_CELL + 0.05:
        print("  It is materially above the paper's 0.814, so the missing filter IS inflating "
              "the floor.\n  Rebuild the ready dataset with G.2's balancing before any floor-"
              "normalized number is quoted.")
    else:
        print("  It is not materially above the paper's 0.814, so the missing filter is not "
              "why the floor\n  is high -- the leakage is in the task as the paper built it, "
              "and that belongs in the writeup.")

    dest = args.out or f"{C.REPORTS_DIR}/dataset_balance_{args.rotation}.json"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwritten to {dest}")


if __name__ == "__main__":
    main()
