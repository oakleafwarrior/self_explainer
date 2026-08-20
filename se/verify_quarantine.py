"""Verify the bf16-map quarantine: nothing stale left behind, nothing sound moved away.

Path names are not the criterion -- the saved weights are. `run_training` and `eval_run`
default to resume=True, so any stale run left in the tree is silently reused and the
re-run becomes a no-op. That is the failure this catches.

A run is safe to KEEP only if its adapter either

    carries no input_map tensors at all          -- C0, which has no map by construction
    carries a map that provably never moved      -- the v = 0 floor arm, whose map has
                                                    zero gradient and so is still exactly
                                                    the identity it started at

Everything else had a non-identity bfloat16 matrix in the injection path -- Oracle
included, even though it trains nothing -- and belongs in quarantine.

Usage:  python se/verify_quarantine.py [quarantine_dir]
        (default: <SE_ROOT>/quarantine_bf16_map)
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import se_config as C


def runs_under(root):
    """Every directory holding a saved adapter, i.e. every finished training run."""
    if not os.path.isdir(root):
        return
    for dirpath, _, filenames in os.walk(root):
        if "adapter_model.safetensors" in filenames or "adapter_model.bin" in filenames:
            yield dirpath


def map_keys(run_dir):
    """Names of the input-map tensors in this run's adapter, without loading them."""
    st = f"{run_dir}/adapter_model.safetensors"
    if os.path.exists(st):
        from safetensors import safe_open
        with safe_open(st, framework="pt") as f:
            return [k for k in f.keys() if "input_map" in k and k.endswith("weight")]
    sd = torch.load(f"{run_dir}/adapter_model.bin", map_location="cpu", weights_only=True)
    return [k for k in sd if "input_map" in k and k.endswith("weight")]


def arm(run_dir):
    parts = {s.split("_", 1)[0]: s.split("_", 1)[1]
             for s in run_dir.split(os.sep) if s.split("_", 1)[0] in ("rot", "cap")}
    return parts.get("rot"), parts.get("cap")


def map_is_pristine(run_dir, keys):
    """True only if every map matrix is still exactly the identity it was initialized at."""
    st = f"{run_dir}/adapter_model.safetensors"
    if os.path.exists(st):
        from safetensors.torch import load_file
        sd = load_file(st)
    else:
        sd = torch.load(f"{run_dir}/adapter_model.bin", map_location="cpu", weights_only=True)
    for k in keys:
        W = sd[k].float()
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            return False
        off = W - torch.diag(W.diagonal())
        if not torch.equal(W.diagonal(), torch.ones(W.shape[0])) or off.abs().max() != 0:
            return False
    return True


def safe_to_keep(run_dir):
    """(ok, reason). The two exemptions, checked against the weights rather than the path."""
    keys = map_keys(run_dir)
    if not keys:
        return True, "no input_map in the adapter"
    rot, _ = arm(run_dir)
    if rot == "zerovec" and map_is_pristine(run_dir, keys):
        return True, "v=0 floor arm, map still exactly identity"
    return False, f"{len(keys)} trained map tensors"


def main():
    quarantine = (sys.argv[1] if len(sys.argv) > 1
                  else f"{os.path.dirname(C.RUNS_DIR)}/quarantine_bf16_map")
    print(f"runs       : {C.RUNS_DIR}")
    print(f"quarantine : {quarantine}"
          + ("" if os.path.isdir(quarantine) else "   (does not exist)"))

    kept, moved, bad_kept, bad_moved = [], [], [], []
    for d in sorted(runs_under(f"{C.RUNS_DIR}/patching")):
        kept.append(d)
        ok, why = safe_to_keep(d)
        if not ok:
            bad_kept.append((os.path.relpath(d, C.RUNS_DIR), why))
    for d in sorted(runs_under(f"{quarantine}/patching")):
        moved.append(d)
        ok, why = safe_to_keep(d)
        if ok:
            bad_moved.append((os.path.relpath(d, quarantine), why))

    by_arm = {}
    for d in kept:
        by_arm.setdefault(arm(d), [0, 0])[0] += 1
    for d in moved:
        by_arm.setdefault(arm(d), [0, 0])[1] += 1
    print(f"\n{'rotation':<10} {'capacity':<13} {'in runs':>8} {'quarantined':>12}")
    for (rot, cap), (k, m) in sorted(by_arm.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
        print(f"{rot or '?':<10} {cap or '?':<13} {k:>8} {m:>12}")
    print(f"{'TOTAL':<24} {len(kept):>8} {len(moved):>12}")

    print("\nSTALE — still in runs/, has a trained map, will be silently resumed:")
    for rel, why in bad_kept:
        print(f"  !! {rel}   ({why})")
    print("  none" if not bad_kept else f"  -> {len(bad_kept)} still to move")

    print("\nMOVED UNNECESSARILY — no map, or a map that never trained:")
    for rel, why in bad_moved:
        print(f"  !! {rel}   ({why})")
    print("  none" if not bad_moved else f"  -> {len(bad_moved)} to move back")

    ok = not bad_kept and not bad_moved
    print("\n" + ("QUARANTINE IS ACCURATE — every remaining run is one the fp32 change "
                  "does not alter." if ok else
                  "QUARANTINE NEEDS FIXING — see the two lists above."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
