"""Did the input map actually train, or did bfloat16 freeze it?

Run this on the pod against the runs you already paid for. It reads saved adapters only --
no model is loaded, no GPU is used, nothing is written.

The question it settles
-----------------------
`run_training` used to do `input_map.to(dtype=compute_dtype())`, i.e. bfloat16. bfloat16's
spacing at 1.0 is 2^-8 = 3.9e-3, and AdamW at lr=1e-4 produces updates of ~1e-4, so a weight
sitting at 1.0 is rounded straight back to 1.0 on every step and never moves. `Cfull` starts
at the identity, and under R-Q its target is Q^T, whose diagonal at d=4096 is ~0.0125. If the
map really ran in bfloat16, the R-Q arm could never have travelled that distance and the
R-id vs R-Q gap is partly a dtype artifact rather than a fact about basis.

PEFT complicates the prediction: `get_peft_model(..., autocast_adapter_dtype=True)` (the
default) upcasts adapter weights to float32 -- but `cast_adapter_dtype` skips anything that
is not a `BaseTunerLayer`, and `ModulesToSaveWrapper` inherits from `AuxiliaryTrainingWrapper`,
not from `BaseTunerLayer`. So the map should have stayed bfloat16. That is a claim about a
specific PEFT version, which is exactly why it is worth measuring instead of arguing.

Reading the output
------------------
The informative arm is rot_Q. Under R-id the identity is already the solution, so a diagonal
that stayed at 1.0 there proves nothing.

  rot_Q, dtype bfloat16, diagonal exactly 1.0    -> frozen. The arm never inverted anything.
                                                    Re-run the Cfull arms.
  rot_Q, dtype float32, diagonal moved off 1.0   -> PEFT upcast it and it trained. Nothing to
                                                    re-run on this account.
"""

import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import se_config as C


def load_adapter(run_dir):
    st = f"{run_dir}/adapter_model.safetensors"
    if os.path.exists(st):
        from safetensors.torch import load_file
        return load_file(st)
    bin_ = f"{run_dir}/adapter_model.bin"
    if os.path.exists(bin_):
        return torch.load(bin_, map_location="cpu", weights_only=True)
    return None


def report(run_dir):
    sd = load_adapter(run_dir)
    if sd is None:
        return None
    keys = [k for k in sd if "input_map" in k and k.endswith("weight")]
    if not keys:
        print(f"  {run_dir}\n    no input_map weights in the adapter "
              f"({len(sd)} keys) — this arm has no trainable map")
        return None

    print(f"  {os.path.relpath(run_dir, C.RUNS_DIR)}")
    verdicts = []
    for k in sorted(keys):
        W = sd[k]
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            continue
        Wf = W.float()
        diag = Wf.diagonal()
        frozen = int((diag == 1.0).sum())
        off = Wf - torch.diag(diag)
        print(f"    {k.split('input_map.')[-1]:<34} dtype {str(W.dtype):<16} "
              f"diag: {frozen}/{len(diag)} still exactly 1.0, "
              f"mean|diag| {diag.abs().mean():.4f}, mean|off| {off.abs().mean():.5f}")
        verdicts.append((W.dtype, frozen == len(diag)))
    return verdicts


def main():
    print("saved input maps, as they came off the GPU")
    print("=" * 78)

    any_found = False
    all_verdicts = {}
    for rot in ("Q", "identity"):
        pattern = C.run_dir("patching", rot, "Cfull", "identity", "*")
        dirs = sorted(glob.glob(pattern)) + sorted(glob.glob(pattern + "/seed_*"))
        dirs = [d for d in dirs if os.path.isdir(d)]
        print(f"\nrotation = {rot} ({C.ROTATION_LABEL[rot]}), capacity = Cfull, "
              f"{len(dirs)} run dir(s)")
        for d in dirs:
            v = report(d)
            if v:
                any_found = True
                all_verdicts.setdefault(rot, []).extend(v)

    if not any_found:
        print("\nNo saved Cfull adapters found under", C.RUNS_DIR)
        print("Check SE_ROOT, or point this at the volume the runs actually landed on.")
        return

    print("\n" + "=" * 78)
    q = all_verdicts.get("Q", [])
    if not q:
        print("No R-Q Cfull adapter found — that is the arm that decides this. Nothing to say.")
        return
    bf16 = [d for d, _ in q if d in (torch.bfloat16, torch.float16)]
    frozen = [f for _, f in q if f]
    if bf16 and frozen:
        print("VERDICT: the R-Q map was stored in a low-precision dtype AND its diagonal is")
        print("still exactly at its initialization. It never moved. The Cfull arms measured a")
        print("dtype constraint, not the basis. Re-run them.")
    elif not bf16:
        print("VERDICT: the map trained in float32 — PEFT upcast it after all. The frozen-")
        print("diagonal argument does not apply and the Cfull arms are sound on this count.")
    else:
        print("VERDICT: low-precision storage but the diagonal DID move. Partial rounding")
        print("rather than a hard freeze; judge it against how far it had to travel")
        print("(Q^T's diagonal at d=4096 is ~0.0125 in absolute value).")


if __name__ == "__main__":
    main()
