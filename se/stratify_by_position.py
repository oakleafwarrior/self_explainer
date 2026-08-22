"""Does the activation contribute at the patch positions the paper actually used?

`se/audit_dataset_balance.py` turned up a composition problem. The paper's five patch-position
categories (Appendix G.2 / Table 7: Subject Final, Relation, Orig/Other/Changed Answer Option)
are all *content* positions. Our unfiltered pool also patches at prompt-template tokens --
`"or"`, `"else"`, `"and"`, `"with"`, `"Respond"`, `"nothing"`, `"one"`, `"of"`, `"assistant"` --
which are ~43% of rows and map to none of them.

If the activation carries little at a template token, then NB03's near-zero lift over the v=0
floor is partly a mixture effect: a real contribution on content positions, averaged down by a
large slab of positions where nothing could be expected. That is a different failure from
"there is no advantage to decompose", and it changes what the project concludes.

This splits every comparison NB03 already ran by patch-position bucket. No training, no GPU:
the eval split is the last `EVAL_SIZE` rows of the ready dataset in order (se_common.py:831)
and `eval_run` appends records in that same order, so record `i` joins to dataset row
`len(ds) - EVAL_SIZE + i` positionally. That join is asserted rather than assumed: every arm
and every seed has its recorded `target_text` compared against the dataset's, item by item
(`assert_join`). A length check alone is not enough -- a run scored against a different eval
split still has `EVAL_SIZE` records, and its bucket labels would then all be wrong.

Two quantities per bucket:

  lift  = score(Cfull, R-id) - score(zerovec floor)   the activation's contribution
  delta = score(R-Q) - score(R-id)                    the rotation effect

`lift` is the one that matters here. If it is ~0 on template tokens and clearly positive on
content tokens, NB03 should be re-scored on the paper's position subset before its null is
reported. If it is ~0 on both, the null stands and the composition issue is a non-finding.

Usage:

    python se/stratify_by_position.py
    python se/stratify_by_position.py --n-values 8192 16384 --n-boot 20000
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import se_config as C  # noqa: E402
from bootstrap_paired import (METRICS, collect, encode, metrics_on,  # noqa: E402
                              paired_bootstrap)
from audit_dataset_balance import bucket  # noqa: E402

# (label, arm_a, arm_b) -- delta reported is b - a
COMPARISONS = [
    ("lift  (Cfull R-id − floor)", ("zerovec", "Cfull", "identity"),
     ("identity", "Cfull", "identity")),
    ("lift  (Cfull R-Q  − floor)", ("zerovec", "Cfull", "identity"),
     ("Q", "Cfull", "identity")),
    ("delta (R-Q − R-id)", ("identity", "Cfull", "identity"),
     ("Q", "Cfull", "identity")),
]


def eval_position_labels(rotation="identity"):
    """Bucket label for each eval item, in eval-record order."""
    from datasets import load_from_disk

    ds = load_from_disk(f"{C.ROTATION_DIR}/act_ready_{rotation}")
    tt = ds["token_type"][len(ds) - C.EVAL_SIZE:]
    return [bucket(t) for t in tt], list(tt)


def eval_target_texts(rotation="identity"):
    """`target_text` for each eval item, in eval-record order.

    Reconstructed exactly as `eval_run` builds it (se_common.py:879) from the same two
    dataset columns, so it is comparable to a run's recorded targets character for character.
    """
    from datasets import load_from_disk
    from se_common import build_activation_target

    ds = load_from_disk(f"{C.ROTATION_DIR}/act_ready_{rotation}")
    tail = ds.select(range(len(ds) - C.EVAL_SIZE, len(ds)))
    return [build_activation_target("".join(o), "".join(a))
            for o, a in zip(tail["original_continuation"], tail["ablated_continuation"])]


def assert_join(enc, expected, n, where):
    """The positional join is only valid if the run scored exactly these items, in order.

    Item `i` of a run's records is joined to eval row `len(ds) - EVAL_SIZE + i` to get its
    bucket. A run evaluated against a different eval split -- the split is positional and
    moves wholesale when the dataset length changes (ACT_DATASET_PREFIX went 10,000 ->
    20,000 partway through this project, and the filtered-probe tree is shorter again) --
    still has EVAL_SIZE records, so a length check passes and every bucket label is then
    silently wrong. `bootstrap_paired.py --audit` finds this across the tree; this catches
    it for the one join actually being made here.
    """
    got = enc["target"]
    if len(got) != len(expected):
        sys.exit(f"eval records at N={n} ({where}) have {len(got)} items but the ready "
                 f"dataset's eval split has {len(expected)}; the positional join is "
                 "invalid. Re-run se/bootstrap_paired.py --audit.")
    if got != expected:
        bad = next(i for i, (g, e) in enumerate(zip(got, expected)) if g != e)
        sys.exit(
            f"eval records at N={n} ({where}) do not match the ready dataset's eval split: "
            f"first difference at item {bad}.\n"
            f"  recorded: {got[bad][:90]!r}\n"
            f"  dataset : {expected[bad][:90]!r}\n"
            "This run was scored against a different eval set, so its position labels would "
            "be wrong. Re-run se/bootstrap_paired.py --audit, delete the stale run "
            "directories, and re-evaluate them.")


def subset(enc, keep):
    """A view of `enc` restricted to the item indices in `keep`."""
    keep = np.asarray(keep, dtype=np.int64)
    return {k: (v[keep] if isinstance(v, np.ndarray) else [v[i] for i in keep])
            for k, v in enc.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--n-values", type=int, nargs="*", default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--rng-seed", type=int, default=20260816)
    ap.add_argument("--metric", default="exact_match", choices=list(METRICS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = args.root or C.RUNS_DIR
    n_values = args.n_values or C.N_TRAIN_VALUES
    labels, raw_tt = eval_position_labels()
    targets = eval_target_texts()
    buckets = sorted(set(labels), key=lambda b: -labels.count(b))

    print(f"PATCH-POSITION STRATIFICATION — metric: {args.metric}")
    print(f"runs tree: {root}   replicates: {args.n_boot}")
    print("=" * 100)
    print("  eval composition: " + "   ".join(
        f"{b} {labels.count(b):,} ({labels.count(b)/len(labels):.1%})" for b in buckets))
    report = {"metric": args.metric, "buckets": {b: labels.count(b) for b in buckets},
              "by_n": {}}

    for n in n_values:
        printed = False
        for label, arm_a, arm_b in COMPARISONS:
            pairs, seeds = collect(root, n, arm_a, arm_b, C.seeds_for(n))
            if not pairs:
                continue
            # Every arm, every seed -- not just the first. `collect` already asserts the
            # two arms of a pair agree with EACH OTHER; that is satisfied when both are
            # stale together, which is exactly the case this has to catch.
            for (ea, eb), s_ in zip(pairs, seeds):
                assert_join(ea, targets, n, f"{label} / {arm_a[0]}-{arm_a[1]} / seed {s_}")
                assert_join(eb, targets, n, f"{label} / {arm_b[0]}-{arm_b[1]} / seed {s_}")
            if not printed:
                print(f"\nN={n}  ({len(seeds)} seed(s): {seeds})")
                print("-" * 100)
                print(f"  {'comparison':28} {'bucket':16} {'n':>5}  {'delta':>8}  "
                      f"{'95% CI (items+seeds)':>24}")
                printed = True
            row = {}
            for b in buckets:
                keep = [i for i, x in enumerate(labels) if x == b]
                sub = [(subset(a, keep), subset(bb, keep)) for a, bb in pairs]
                rng = np.random.default_rng(args.rng_seed)
                r = paired_bootstrap(sub, args.n_boot, rng,
                                     resample_seeds=len(sub) > 1)[args.metric]
                row[b] = {"n_items": len(keep), **r}
                flag = "" if r["ci95"][0] <= 0 <= r["ci95"][1] else "  *"
                print(f"  {label:28} {b:16} {len(keep):>5}  {r['delta']:+8.4f}  "
                      f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]{flag}")
            report["by_n"].setdefault(str(n), {})[label] = row

    print("\n" + "=" * 100)
    print("  * = 95% CI excludes zero.")
    print("  Read the two `lift` rows first. A lift that is ~0 on 'prompt template' and clearly")
    print("  positive on 'content token' means NB03's null is a mixture effect and the sweep")
    print("  should be re-scored on the paper's position subset. A lift that is ~0 on both means")
    print("  the composition issue is a non-finding and the null stands as reported.")

    dest = args.out or f"{C.REPORTS_DIR}/stratify_by_position_{args.metric}.json"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwritten to {dest}")


if __name__ == "__main__":
    main()
