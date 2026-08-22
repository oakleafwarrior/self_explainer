"""Item-level paired bootstrap on the eval records NB03 already wrote.

Why this exists. NB03 left one live question: `E_self · Cfull · R-Q` scored at or above
`R-id` at 6 of 7 N, on all three metrics, and the effect is largest on `has_changed_f1`
(+0.133 vs +0.059 over the floor at N=512). `se_common.paired_delta` cannot speak to that
metric -- it supports exact_match and content_match only, because F1 is not a per-example
binary (prereg_threshold_justification.md, "The has_changed_f1 gap", which asks for exactly
this bootstrap before the `metrics_disagree` reading is allowed to fire).

Nothing here trains or evaluates anything. It reads `eval_records.json` off the runs tree,
so it is CPU-only and costs nothing to re-run.

Three things it does that a McNemar delta does not:

1. **Bootstraps `has_changed_f1`**, by resampling eval items and recomputing macro F1 from
   the same resampled indices for both arms. Paired: one index draw, both arms.

2. **Calibrates the null against NB04 §1b.** `Oracle`/R-Q and `C0`/R-id are the same
   computation in real arithmetic -- §1a passed at 2.6e-08, so any delta between them is
   run-to-run noise: bf16 divergence plus training nondeterminism. Those two runs disagreed
   on 68 of 1024 items. That is the floor a real effect has to clear, and it is wider than
   the McNemar SE because McNemar models sampling noise and not the retraining.
   Every reported delta is printed against this band.

3. **Splits the delta by `true_verdict`.** A macro-F1 gain that sits entirely on one class
   is a shift in how often the explainer says "changed", not evidence it read the activation
   better. That failure mode inflates macro F1 while carrying no information, and it is the
   most likely benign explanation for a wrong-signed result. If the R-Q advantage survives
   on both classes with the predicted-"changed" rate held roughly level, it is not that.

Usage, on the pod:

    python se/bootstrap_paired.py                    # Cfull R-id vs R-Q, every N
    python se/bootstrap_paired.py --pair C0          # the paper's own configuration
    python se/bootstrap_paired.py --n-boot 20000 --out reports/bootstrap_paired.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import se_config as C  # noqa: E402

VERDICTS = ("changed", "unchanged")
METRICS = ("exact_match", "has_changed_f1", "content_match")


# ---------------------------------------------------------------------------
# records -> per-item arrays
# ---------------------------------------------------------------------------

def encode(records):
    """Per-item arrays sufficient to recompute every metric on any subset of items.

    The metric definitions are copied from `se_common.eval_run` rather than re-derived,
    including the one wrinkle: `content_match` is a mean over *parseable* records only, so
    it needs a numerator and a denominator that both move when items are resampled.
    (`se_common.paired_delta` treats a missing `pred_content` as the empty string instead;
    that is a different statistic, and this file uses eval_run's.)
    """
    em, cm_ok, cm_hit, tv, pv = [], [], [], [], []
    for r in records:
        em.append(r["generated_text"].replace(" ", "") == r["target_text"].replace(" ", ""))
        parseable = r.get("pred_content") is not None
        cm_ok.append(parseable)
        cm_hit.append(parseable and r["pred_content"].replace(" ", "")
                      == (r.get("true_content") or "").replace(" ", ""))
        tv.append(VERDICTS.index(r["true_verdict"]))
        p = r.get("pred_verdict")
        pv.append(VERDICTS.index(p) if p in VERDICTS else 2)   # 2 = unparseable
    return {"em": np.array(em, bool), "cm_ok": np.array(cm_ok, bool),
            "cm_hit": np.array(cm_hit, bool),
            "tv": np.array(tv, np.int64), "pv": np.array(pv, np.int64),
            "target": [r["target_text"] for r in records]}


def metrics_on(enc, idx):
    """exact_match, has_changed_f1, content_match over the items in `idx`.

    macro F1 over {changed, unchanged} exactly as sklearn computes it with
    `labels=["changed","unchanged"], average="macro"` and unparseable folded into a third
    predicted class: a predicted-unparseable is a false negative for its true label and a
    false positive for neither.
    """
    em = float(enc["em"][idx].mean())

    denom = int(enc["cm_ok"][idx].sum())
    cm = float(enc["cm_hit"][idx].sum() / denom) if denom else 0.0

    # 2x3 contingency in one pass: true in {0,1}, pred in {0,1,2}
    cell = np.bincount(enc["tv"][idx] * 3 + enc["pv"][idx], minlength=6).reshape(2, 3)
    f1s = []
    for lab in (0, 1):
        tp = cell[lab, lab]
        fn = cell[lab].sum() - tp
        fp = cell[:, lab].sum() - tp
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return {"exact_match": em, "has_changed_f1": float(np.mean(f1s)), "content_match": cm}


# ---------------------------------------------------------------------------
# the bootstrap
# ---------------------------------------------------------------------------

def paired_bootstrap(pairs, n_boot, rng, resample_seeds):
    """95% CI on the seed-averaged (b - a) delta for each metric.

    `pairs` is a list of (enc_a, enc_b), one per training seed, already matched seed-to-seed
    -- the arms share prompts, ordering and data order by construction (NB03 asserts it), so
    seed k of one arm pairs with seed k of the other.

    One index draw per replicate is shared by both arms and by every seed: that is what makes
    it paired, and it removes the eval-set variance that dominates the independent CI.

    `resample_seeds` additionally draws seeds with replacement. NB03's per-seed sd reached
    0.027 on exact_match, which is larger than several of the deltas being tested, so the
    seed-resampled interval is the one to quote. The item-only interval is reported beside it
    to show how much of the width is training nondeterminism rather than eval sampling.
    """
    n_items = len(pairs[0][0]["em"])
    n_seeds = len(pairs)
    draws = {m: np.empty(n_boot) for m in METRICS}

    for b in range(n_boot):
        idx = rng.integers(0, n_items, n_items)
        which = rng.integers(0, n_seeds, n_seeds) if resample_seeds else range(n_seeds)
        acc = {m: [] for m in METRICS}
        for k in which:
            ea, eb = pairs[k]
            ma, mb = metrics_on(ea, idx), metrics_on(eb, idx)
            for m in METRICS:
                acc[m].append(mb[m] - ma[m])
        for m in METRICS:
            draws[m][b] = np.mean(acc[m])

    out = {}
    full = np.arange(n_items)
    point = {m: float(np.mean([metrics_on(eb, full)[m] - metrics_on(ea, full)[m]
                               for ea, eb in pairs])) for m in METRICS}
    for m in METRICS:
        lo, hi = np.percentile(draws[m], [2.5, 97.5])
        out[m] = {"delta": point[m], "ci95": [float(lo), float(hi)],
                  "se": float(draws[m].std(ddof=1)),
                  "p_two_sided": float(2 * min((draws[m] <= 0).mean(),
                                               (draws[m] >= 0).mean()))}
    return out


def stratified_delta(pairs):
    """Point deltas split by true class, plus each arm's predicted-'changed' base rate.

    The diagnostic, not an inference: if the whole gain sits on one class while the
    predicted-'changed' rate moves with it, the macro-F1 difference is a response-bias
    shift and not evidence about the activation.
    """
    rows = {}
    for lab, name in enumerate(VERDICTS):
        d_em, d_cm = [], []
        for ea, eb in pairs:
            sel = np.where(ea["tv"] == lab)[0]
            d_em.append(metrics_on(eb, sel)["exact_match"] - metrics_on(ea, sel)["exact_match"])
            d_cm.append(metrics_on(eb, sel)["content_match"] - metrics_on(ea, sel)["content_match"])
        rows[name] = {"n_items": int((pairs[0][0]["tv"] == lab).sum()),
                      "d_exact_match": float(np.mean(d_em)),
                      "d_content_match": float(np.mean(d_cm))}
    rows["pred_changed_rate"] = {
        "a": float(np.mean([(ea["pv"] == 0).mean() for ea, _ in pairs])),
        "b": float(np.mean([(eb["pv"] == 0).mean() for _, eb in pairs])),
    }
    rows["unparseable_rate"] = {
        "a": float(np.mean([(ea["pv"] == 2).mean() for ea, _ in pairs])),
        "b": float(np.mean([(eb["pv"] == 2).mean() for _, eb in pairs])),
    }
    return rows


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load(root, n_train, rotation, capacity, init, seed, explainer=None):
    d = C.run_dir("patching", rotation, capacity, init, n_train,
                  explainer=explainer or C.EXPLAINER_MODEL_ID, seed=seed, root=root)
    p = f"{d}/eval_records.json"
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def collect(root, n_train, arm_a, arm_b, seeds):
    """Matched (enc_a, enc_b) per seed, skipping seeds where either arm is missing.

    Asserts the two arms scored the same held-out items in the same order. Without that the
    pairing is silently wrong and every interval below is too tight, so it is a hard failure
    rather than a warning.
    """
    pairs, used = [], []
    for s in seeds:
        ra, rb = load(root, n_train, *arm_a, s), load(root, n_train, *arm_b, s)
        if ra is None or rb is None:
            continue
        n = min(len(ra), len(rb))
        ea, eb = encode(ra[:n]), encode(rb[:n])
        assert ea["target"] == eb["target"], (
            f"arms disagree on the eval items at N={n_train}, seed={s} -- the pairing is "
            "invalid; rebuild the eval split")
        pairs.append((ea, eb))
        used.append(s)
    return pairs, used


# ---------------------------------------------------------------------------

PAIRS = {
    # name: (arm_a, arm_b) as (rotation, capacity, init); the delta reported is b - a
    "Cfull": (("identity", "Cfull", "identity"), ("Q", "Cfull", "identity")),
    "C0": (("identity", "C0", "identity"), ("Q", "C0", "identity")),
    "Cfull-rand": (("identity", "Cfull-rand", "orthogonal"), ("Q", "Cfull-rand", "orthogonal")),
}


def audit_eval_sets(root):
    """Fingerprint every run's eval set and group runs that share one.

    `eval_run` takes the eval split as the *last* EVAL_SIZE rows of the ready dataset
    (se_common.py:831, `select(range(len(dataset) - C.EVAL_SIZE, len(dataset)))`). That
    split is positional, so it moves wholesale when the dataset length changes -- and
    ACT_DATASET_PREFIX went 10_000 -> 20_000 partway through this project. A run cached
    before that rebuild holds a *disjoint* eval set from one cached after it, and
    `run_training`/`eval_run` skip on "already done" rather than noticing.

    Scores from two different groups are not comparable at all: not paired, not even
    measuring the same items. `collect()` catches it for a pair being bootstrapped; this
    catches it for the whole tree, including comparisons made by eye across NB03's tables.

    Prints one line per group with its runs, so a stale group can be deleted and re-run.
    """
    import glob
    from collections import defaultdict
    import hashlib

    groups = defaultdict(list)
    for p in sorted(glob.glob(f"{root}/**/eval_records.json", recursive=True)):
        with open(p) as f:
            recs = json.load(f)
        h = hashlib.sha1("\x00".join(r["target_text"] for r in recs).encode()).hexdigest()[:10]
        groups[(h, len(recs))].append((os.path.relpath(os.path.dirname(p), root),
                                       os.path.getmtime(p)))

    print(f"EVAL-SET AUDIT — {root}")
    print("=" * 96)
    if not groups:
        print("  no eval_records.json found")
        return True
    order = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    total = sum(len(v) for v in groups.values())
    print(f"  {total} runs, {len(groups)} distinct eval set(s)\n")
    import datetime as _dt
    for rank, ((h, n_items), runs) in enumerate(order):
        when = [r[1] for r in runs]
        tag = "MAJORITY" if rank == 0 else "STALE?"
        print(f"  [{tag}] fingerprint {h}  {n_items} items  {len(runs)} runs")
        print(f"           written {_dt.datetime.fromtimestamp(min(when)):%Y-%m-%d %H:%M}"
              f" .. {_dt.datetime.fromtimestamp(max(when)):%Y-%m-%d %H:%M}")
        for d, _ in sorted(runs):
            print(f"             {d}")
        print()
    if len(groups) > 1:
        print("  MORE THAN ONE EVAL SET IS PRESENT. Scores across groups are not comparable --")
        print("  they were computed on disjoint held-out items. Delete the minority group's")
        print("  run directories and re-run those cells; the notebooks will stop skipping them.")
    else:
        print("  Every run shares one eval set. Cross-arm comparisons in the tree are sound.")
    return len(groups) == 1


def self_test(root, n_runs=6, n_resamples=8, rng_seed=0):
    """Check `metrics_on` against sklearn and eval_run's own definitions, on real records.

    Everything this file reports rests on recomputing macro F1 from a resampled index set
    without calling sklearn per replicate (10k replicates x sklearn would be minutes, not
    seconds). That reimplementation is the one place a silent error would be invisible, so
    it is checked on the actual eval records rather than on a fixture -- including on
    resampled subsets, where a mishandled empty class would show up and a full-set check
    would not.
    """
    import glob
    from sklearn.metrics import f1_score

    paths = sorted(glob.glob(f"{root}/**/eval_records.json", recursive=True))[:n_runs]
    if not paths:
        print(f"SELF-TEST — no eval_records.json under {root}")
        return False
    rng = np.random.default_rng(rng_seed)
    worst = 0.0
    for p in paths:
        with open(p) as f:
            recs = json.load(f)
        enc = encode(recs)
        for trial in range(n_resamples):
            idx = (np.arange(len(recs)) if trial == 0
                   else rng.integers(0, len(recs), len(recs)))
            sub = [recs[i] for i in idx]
            ref_f1 = f1_score([r["true_verdict"] for r in sub],
                              [r["pred_verdict"] or "unparseable" for r in sub],
                              labels=list(VERDICTS), average="macro")
            ref_em = float(np.mean([r["generated_text"].replace(" ", "")
                                    == r["target_text"].replace(" ", "") for r in sub]))
            par = [r for r in sub if r.get("pred_content") is not None]
            ref_cm = float(np.mean([r["pred_content"].replace(" ", "")
                                    == (r.get("true_content") or "").replace(" ", "")
                                    for r in par])) if par else 0.0
            got = metrics_on(enc, idx)
            worst = max(worst, abs(got["has_changed_f1"] - ref_f1),
                        abs(got["exact_match"] - ref_em),
                        abs(got["content_match"] - ref_cm))
    print(f"SELF-TEST — {len(paths)} runs x {n_resamples} resamples")
    print(f"  max |metrics_on - sklearn/eval_run| = {worst:.3e}")
    ok = worst < 1e-12
    print("  PASS — the reimplementation is exact" if ok else "  FAIL — do not trust the CIs")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", default="Cfull", choices=sorted(PAIRS))
    ap.add_argument("--root", default=None, help="runs tree (default: se_config.RUNS_DIR)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--rng-seed", type=int, default=20260816)
    ap.add_argument("--n-values", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=None, help="JSON output (default: REPORTS_DIR/bootstrap_paired_<pair>.json)")
    ap.add_argument("--self-test", action="store_true",
                    help="verify metrics_on against sklearn on real records, then exit")
    ap.add_argument("--audit", action="store_true",
                    help="group every run in the tree by its eval set fingerprint, then exit")
    args = ap.parse_args()

    root = args.root or C.RUNS_DIR
    if args.self_test:
        sys.exit(0 if self_test(root) else 1)
    if args.audit:
        sys.exit(0 if audit_eval_sets(root) else 1)
    n_values = args.n_values or C.N_TRAIN_VALUES
    arm_a, arm_b = PAIRS[args.pair]
    rng = np.random.default_rng(args.rng_seed)
    report = {"pair": args.pair, "arm_a": arm_a, "arm_b": arm_b, "root": root,
              "n_boot": args.n_boot, "rng_seed": args.rng_seed, "by_n": {}}

    print(f"PAIRED ITEM BOOTSTRAP — delta = ({arm_b[0]}/{arm_b[1]}) - ({arm_a[0]}/{arm_a[1]})")
    print(f"runs tree: {root}   replicates: {args.n_boot}")
    print("=" * 96)

    # --- the null band, from NB04 1b -------------------------------------------------
    # Oracle/R-Q and C0/R-id hand the model the same vector. Any delta is noise, and its
    # spread is the smallest difference anything below is entitled to call real.
    null = None
    n_check = min(C.N_TRAIN_LADDER)
    np_pairs, np_seeds = collect(root, n_check, ("identity", "C0", "identity"),
                                 ("Q", "Oracle", "oracle"), [C.SEED])
    if np_pairs:
        null = paired_bootstrap(np_pairs, args.n_boot, np.random.default_rng(args.rng_seed),
                                resample_seeds=False)
        report["null_from_nb04_1b"] = {"n_train": n_check, "seeds": np_seeds, **null}
        print(f"\nNULL BAND — NB04 §1b, Oracle/R-Q vs C0/R-id at N={n_check} "
              "(algebraically identical runs)")
        print("-" * 96)
        for m in METRICS:
            r = null[m]
            print(f"  {m:>16}: delta {r['delta']:+.4f}   95% [{r['ci95'][0]:+.4f}, "
                  f"{r['ci95'][1]:+.4f}]   half-width {(r['ci95'][1]-r['ci95'][0])/2:.4f}")
        print("  A delta inside this band is retraining noise, whatever its own CI says.")
    else:
        print("\nNULL BAND — not found (run NB04 §1b first); deltas are reported against 0 only.")

    # --- the sweep -------------------------------------------------------------------
    signs = {m: [] for m in METRICS}
    for n in n_values:
        # Analysis reads the seeds that are ON DISK, not the grid the config intended. `seeds_for`
        # encodes a *plan* -- and `MULTI_SEED_N` excludes 16,384 (se_config.py:113) while three seeds
        # for it sit in the tree, so asking `seeds_for` silently dropped two of them AND, because
        # `resample_seeds=len(pairs)>1`, gave that one N an items-only interval while every other N
        # got items+seeds. `collect` already skips seeds it cannot find, so asking for all of them is
        # strictly safer here and adds no GPU work.
        pairs, seeds = collect(root, n, arm_a, arm_b, C.SEEDS)
        if not pairs:
            continue
        multi = len(pairs) > 1
        items = paired_bootstrap(pairs, args.n_boot, rng, resample_seeds=False)
        both = paired_bootstrap(pairs, args.n_boot, rng, resample_seeds=True) if multi else None
        strat = stratified_delta(pairs)
        report["by_n"][str(n)] = {"seeds": seeds, "n_items": len(pairs[0][0]["em"]),
                                  "items_only": items, "items_and_seeds": both,
                                  "stratified": strat}

        print(f"\nN={n}  ({len(seeds)} seed{'s' if multi else ''}: {seeds}, "
              f"{len(pairs[0][0]['em'])} items)")
        print("-" * 96)
        print(f"  {'metric':>16}  {'delta':>8}  {'95% CI (items)':>22}  "
              f"{'95% CI (items+seeds)':>24}  {'vs null':>10}")
        for m in METRICS:
            r = items[m]
            signs[m].append(np.sign(r["delta"]))
            bs = (f"[{both[m]['ci95'][0]:+.4f}, {both[m]['ci95'][1]:+.4f}]"
                  if both else "single seed")
            if null:
                nlo, nhi = null[m]["ci95"]
                verdict = "inside" if nlo <= r["delta"] <= nhi else "OUTSIDE"
            else:
                verdict = "-"
            print(f"  {m:>16}  {r['delta']:+8.4f}  "
                  f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  {bs:>24}  {verdict:>10}")

        print(f"  by true class:  changed (n={strat['changed']['n_items']}) "
              f"d_em {strat['changed']['d_exact_match']:+.4f} | "
              f"unchanged (n={strat['unchanged']['n_items']}) "
              f"d_em {strat['unchanged']['d_exact_match']:+.4f}")
        pc, up = strat["pred_changed_rate"], strat["unparseable_rate"]
        print(f"  predicted-'changed' rate: {pc['a']:.4f} -> {pc['b']:.4f} "
              f"({pc['b']-pc['a']:+.4f})   unparseable: {up['a']:.4f} -> {up['b']:.4f}")

    # --- direction consistency -------------------------------------------------------
    # Each N is a separate training run, so the signs are independent under the null even
    # though the N values share an eval set. One-sided binomial on the modal direction.
    if report["by_n"]:
        from math import comb
        print("\n" + "=" * 96)
        print("SIGN CONSISTENCY ACROSS N (one-sided binomial, p=0.5 under the null)")
        for m in METRICS:
            s = [x for x in signs[m] if x != 0]
            k, n_s = int(sum(x > 0 for x in s)), len(s)
            hi = max(k, n_s - k)
            p = sum(comb(n_s, i) for i in range(hi, n_s + 1)) / 2 ** n_s
            print(f"  {m:>16}: {k}/{n_s} positive   p = {p:.4f}")
        print("  Note: the three metrics are computed from the same generations and are not "
              "independent of\n  each other -- these are three views of one experiment, not "
              "three experiments.")
        report["sign_test"] = {m: [float(x) for x in signs[m]] for m in METRICS}

    out = args.out or f"{C.REPORTS_DIR}/bootstrap_paired_{args.pair}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
