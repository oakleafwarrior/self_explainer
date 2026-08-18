"""RunPod environment bootstrap: dependencies, persistence, credentials, GPU.

Colab supplied four things for free that a RunPod pod does not, and each one is a way for
a run to fail late and expensively instead of early and cheaply:

1. **A preinstalled ML stack.** A RunPod image ships torch built against its own CUDA
   driver, and nothing may replace it — a generic wheel pulled in as somebody's transitive
   dependency turns every kernel launch into a runtime error, usually two hours into a
   sweep. `ensure_deps` writes the installed torch version into a pip constraints file, so
   an install that *would* move torch fails loudly instead of succeeding quietly, and it
   installs only what is missing or below the floor these notebooks actually need. It also
   installs `bitsandbytes`, which Colab had and RunPod images do not — `se_common` asks for
   `optim="paged_adamw_8bit"`, so without it every training cell dies at trainer construction.

2. **Persistence.** `/workspace` (a pod's volume) or `/runpod-volume` (serverless) is the
   only filesystem worth writing to. The container disk is typically 20 GB against a 16 GB
   bf16 checkpoint, and it is gone when the pod is terminated. So the HF cache and every
   artifact go on the volume, and `bootstrap` states which path it picked, whether that path
   is a real mount, and how much room is left — rather than discovering both at hour nine.

3. **A secrets store.** `HF_TOKEN` comes from the pod template's environment, or from
   `<volume>/.hf_token`. There is no interactive prompt: a pod that restarts unattended has
   nobody to answer one. Qwen3 and the Transluce datasets are public, so a missing token is
   a warning, not an error.

4. **A GPU you did not choose.** Colab hands you whatever it has; on RunPod the pod is the
   experiment's own parameter, so each notebook declares what it needs and this module
   checks it. Two 8B bf16 copies live at once in NB01; the training notebooks want 80 GB.

Two ordering constraints are load-bearing, and they are why this is a module the notebooks
call in their second cell rather than advice in the README:

- `HF_HOME` must be set **before** `huggingface_hub` is imported — it is read into module
  constants at import time, so a late assignment silently caches 16 GB onto the container
  disk. `configure_caches` says so if it is already too late.
- `CUDA_VISIBLE_DEVICES` must be set **before** torch initializes CUDA.

Nothing here imports torch, transformers, or huggingface_hub at module scope: `se_config`
imports this module for its paths, and `se_config` has to stay importable on a laptop with
no CUDA and no ML stack (that is where `test_rotate.py` runs).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata

# --- what the notebooks need -------------------------------------------------
# Floors, not pins. An image that already satisfies a floor is left alone: its stack is
# matched to its driver, and upgrading it is a larger risk than a stale minor version.
#
#   transformers >= 4.56   `from_pretrained(dtype=...)`, used in every notebook
#   trl          >= 0.20   `SFTConfig(max_length=..., assistant_only_loss=True)`
#   peft         >= 0.14   `modules_to_save=["input_map"]`
#   bitsandbytes           `optim="paged_adamw_8bit"` in se_common.run_training
REQUIREMENTS = {
    "transformers": "4.56.0",
    "trl": "0.20.0",
    "peft": "0.14.0",
    "datasets": "3.0.0",
    "accelerate": "1.0.0",
    "bitsandbytes": "0.43.0",
    "safetensors": "0.4.0",
    "scikit-learn": "1.3.0",
    "pandas": "2.0.0",
    "matplotlib": "3.7.0",
}

# Never required. hf_transfer moves a 16 GB checkpoint several times faster, which is a real
# line item on a pod that bills by the minute.
OPTIONAL = {"hf_transfer": "0.1.6"}

# Where a volume shows up: pods mount at /workspace, serverless workers at /runpod-volume.
VOLUME_CANDIDATES = ("/workspace", "/runpod-volume")

TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")


# ---------------------------------------------------------------------------
# versions
# ---------------------------------------------------------------------------

def _dist_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _below(installed, floor):
    """True if `installed` is older than `floor`, tolerating odd version strings."""
    try:
        from packaging.version import Version

        return Version(installed) < Version(floor)
    except Exception:                                    # noqa: BLE001
        def parts(v):
            out = []
            for chunk in str(v).split("+")[0].split("."):
                digits = "".join(c for c in chunk if c.isdigit())
                out.append(int(digits) if digits else 0)
            return tuple(out)

        return parts(installed) < parts(floor)


def installed_versions():
    """{distribution: version or None} for everything the notebooks import."""
    names = list(REQUIREMENTS) + list(OPTIONAL) + ["torch", "huggingface-hub"]
    return {name: _dist_version(name) for name in names}


def missing_requirements(requirements=None):
    """{name: (installed or None, floor)} for what is absent or too old."""
    requirements = requirements or REQUIREMENTS
    out = {}
    for name, floor in requirements.items():
        have = _dist_version(name)
        if have is None or _below(have, floor):
            out[name] = (have, floor)
    return out


# ---------------------------------------------------------------------------
# dependencies
# ---------------------------------------------------------------------------

def ensure_deps(extra=(), include_optional=True, verbose=True):
    """Install what is missing or too old, and nothing else. Safe to re-run.

    The image's torch is pinned as a pip constraint, so a dependency that wants a different
    torch makes the install *fail* rather than replace a CUDA-matched build with a generic
    wheel. If that refusal ever fires, the fix is a package version that agrees with the
    image — not a relaxed constraint.
    """
    wanted = dict(REQUIREMENTS)
    if include_optional:
        wanted.update(OPTIONAL)
    for spec in extra:
        wanted.setdefault(spec, None)

    todo = {}
    for name, floor in wanted.items():
        have = _dist_version(name)
        if have is None or (floor and _below(have, floor)):
            todo[name] = (have, floor)

    if not todo:
        if verbose:
            stack = ", ".join(f"{n} {_dist_version(n)}" for n in
                              ("torch", "transformers", "peft", "trl", "bitsandbytes")
                              if _dist_version(n))
            print(f"dependencies: already satisfied ({stack})")
        return {}

    specs = [f"{n}>={f}" if f else n for n, (_, f) in todo.items()]
    torch_version = _dist_version("torch")
    cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
    if torch_version:
        # `==2.6.0` matches the local build `2.6.0+cu124`: PEP 440 ignores the local label
        # when the specifier has none, so this pins the wheel without naming its CUDA tag.
        constraints = os.path.join(tempfile.gettempdir(), "se_torch_constraint.txt")
        with open(constraints, "w") as f:
            f.write(f"torch=={torch_version.split('+')[0]}\n")
        cmd += ["-c", constraints]
    if verbose:
        print(f"installing: {' '.join(specs)}")
        for name, (have, floor) in todo.items():
            print(f"  {name:>14}: {have or 'absent'} -> >= {floor}" if floor
                  else f"  {name:>14}: {have or 'absent'}")
    else:
        cmd.append("-q")

    optional_names = set(OPTIONAL)
    required_specs = [s for s in specs if s.split(">=")[0] not in optional_names]
    optional_specs = [s for s in specs if s.split(">=")[0] in optional_names]

    if required_specs:
        subprocess.run(cmd + required_specs, check=True)
    if optional_specs:
        # A missing accelerator must not stop the run; a missing trainer must.
        failed = subprocess.run(cmd + optional_specs, check=False).returncode
        if failed and verbose:
            print(f"  (optional install failed, continuing without {optional_specs})")

    # pip wrote into a site-packages directory whose listing importlib already cached; without
    # this, the versions below can still read as "absent" immediately after a successful install
    import importlib

    importlib.invalidate_caches()

    after = _dist_version("torch")
    if torch_version and after != torch_version:
        print(f"\n!! torch changed {torch_version} -> {after}: the image's CUDA build was "
              f"replaced.\n   Reinstall the image's torch and restart the kernel before "
              f"training — the constraint should have prevented this.")
    elif verbose:
        print("done; torch untouched")

    installed = {name: _dist_version(name) for name in todo}
    if verbose and any(v is None for v in installed.values()):
        print(f"!! still missing after install: "
              f"{[n for n, v in installed.items() if v is None]}")

    stale = [n for n in todo if n.replace("-", "_") in sys.modules]
    if stale:
        print(f"!! {', '.join(stale)} was already imported in this kernel and has just been "
              f"replaced on disk. Restart the kernel before training, or the run uses the old "
              f"code with the new metadata.")
    return installed


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def persistent_root():
    """Where artifacts belong: the pod's volume, or `~` when this is not a pod.

    `SE_PERSIST_ROOT` overrides. On a pod with no network volume attached, `/workspace` is
    still the right answer — it is the container volume, which survives a *restart* but not
    a *terminate* — so this returns it and `storage_report` reports the difference.
    """
    explicit = os.environ.get("SE_PERSIST_ROOT")
    if explicit:
        return explicit.rstrip("/") or "/"
    for path in VOLUME_CANDIDATES:
        if os.path.isdir(path):
            return path
    return os.path.expanduser("~")


def is_runpod():
    return any(key.startswith("RUNPOD_") for key in os.environ)


def pod_id():
    return os.environ.get("RUNPOD_POD_ID")


def storage_report(root=None):
    """Free space at `root`, and whether it is its own mount.

    `os.path.ismount` is the honest test available inside the container: an attached volume
    is a separate mount, a plain directory on the container filesystem is not. It cannot
    tell a network volume from a container volume, so the wording stays hedged.
    """
    root = root or persistent_root()
    os.makedirs(root, exist_ok=True)
    total, used, free = shutil.disk_usage(root)
    return {"root": root, "mount": os.path.ismount(root),
            "free_gb": free / 2 ** 30, "total_gb": total / 2 ** 30}


def configure_caches(root=None, verbose=True):
    """Point the HF caches at the volume — before anything imports huggingface_hub.

    Anything already set in the pod template wins: if you configured `HF_HOME` there, that
    is a deliberate choice and this respects it.
    """
    root = root or persistent_root()
    env = {}
    hf_home = os.environ.get("HF_HOME") or f"{root}/hf_cache"
    env["HF_HOME"] = hf_home
    env["HF_HUB_CACHE"] = os.environ.get("HF_HUB_CACHE") or f"{hf_home}/hub"
    env["HF_DATASETS_CACHE"] = os.environ.get("HF_DATASETS_CACHE") or f"{hf_home}/datasets"
    env["TORCH_HOME"] = os.environ.get("TORCH_HOME") or f"{root}/torch_cache"
    # dataloader workers fork after the tokenizer is built; the warning is pure noise here
    env["TOKENIZERS_PARALLELISM"] = os.environ.get("TOKENIZERS_PARALLELISM", "false")
    if _dist_version("hf_transfer"):
        env["HF_HUB_ENABLE_HF_TRANSFER"] = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "1")

    os.environ.update(env)
    for key in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME"):
        os.makedirs(env[key], exist_ok=True)

    if "huggingface_hub" in sys.modules:
        from huggingface_hub import constants

        if os.path.realpath(constants.HF_HUB_CACHE) != os.path.realpath(env["HF_HUB_CACHE"]):
            print(f"!! huggingface_hub was imported before this cell ran, so it is caching "
                  f"to\n   {constants.HF_HUB_CACHE} instead of {env['HF_HUB_CACHE']}.\n"
                  f"   Restart the kernel and run the setup cells first, or the container "
                  f"disk fills at the first 8B download.")
    elif verbose:
        print(f"hf cache: {hf_home}")
    return env


# ---------------------------------------------------------------------------
# credentials
# ---------------------------------------------------------------------------

def hf_token(root=None):
    """(token, source) from the pod environment, a file on the volume, or a cached login."""
    for var in TOKEN_VARS:
        value = os.environ.get(var)
        if value and value.strip():
            return value.strip(), var
    path = os.path.join(root or persistent_root(), ".hf_token")
    if os.path.isfile(path):
        token = open(path).read().strip()
        if token:
            return token, path
    try:
        from huggingface_hub import get_token

        token = get_token()
        if token:
            return token, "cached login"
    except Exception:                                    # noqa: BLE001 — hub not installed
        pass
    return None, None


def hf_login(root=None, verbose=True):
    """Authenticate if a token is available. Missing is a warning: the models are public."""
    token, source = hf_token(root)
    if token is None:
        if verbose:
            print("hf token: none found — public models only. Set HF_TOKEN in the pod "
                  f"template, or write it to {root or persistent_root()}/.hf_token")
        return {"token": False, "source": None, "user": None}

    os.environ["HF_TOKEN"] = token            # so pip/datasets subprocesses see it too
    user = None
    try:
        from huggingface_hub import login, whoami

        login(token=token, add_to_git_credential=False)
        user = whoami(token=token).get("name")
    except Exception as err:                             # noqa: BLE001 — offline is survivable
        if verbose:
            print(f"hf token: found in {source}, but login failed ({err.__class__.__name__}: "
                  f"{err}). Continuing unauthenticated.")
        return {"token": True, "source": source, "user": None}
    return {"token": True, "source": source, "user": user}


# ---------------------------------------------------------------------------
# hardware
# ---------------------------------------------------------------------------

def gpu_report(min_vram_gb=0, requirement="optional", verbose=True):
    """What the pod gave us, checked against what the notebook asked for.

    A missing GPU where one is required is fatal — the alternative is an OOM or a
    twelve-hour CPU training run. Too *little* VRAM is only a warning: gradient
    checkpointing is already on, and the batch size is adjustable (keeping
    `PER_DEVICE_TRAIN_BATCH_SIZE x GRADIENT_ACCUMULATION_STEPS` fixed at 16, or the arms
    are no longer comparable to each other).
    """
    info = {"count": 0, "names": [], "vram_gb": 0.0, "total_vram_gb": 0.0,
            "bf16": False, "torch": _dist_version("torch"), "cuda": None}
    try:
        import torch
    except ImportError:
        if requirement == "required":
            raise RuntimeError(
                "torch is not installed. Run the dependency cell (se_env.ensure_deps()) "
                "on a RunPod PyTorch pod; this notebook cannot run without a GPU."
            )
        if verbose:
            print("gpu: torch not installed")
        return info

    info["cuda"] = torch.version.cuda
    if torch.cuda.is_available():
        info["count"] = torch.cuda.device_count()
        for i in range(info["count"]):
            props = torch.cuda.get_device_properties(i)
            info["names"].append(props.name)
            info["total_vram_gb"] += props.total_memory / 2 ** 30
            info["vram_gb"] = max(info["vram_gb"], props.total_memory / 2 ** 30)
        info["bf16"] = torch.cuda.is_bf16_supported()

    if verbose:
        if info["count"]:
            names = ", ".join(sorted(set(info["names"])))
            print(f"gpu     : {info['count']} x {names}  |  {info['vram_gb']:.0f} GiB each"
                  f"  |  bf16 {'yes' if info['bf16'] else 'NO'}")
        else:
            print("gpu     : none visible (CPU pod)")

    if requirement == "required" and not info["count"]:
        raise RuntimeError(
            f"This notebook needs a GPU pod ({min_vram_gb} GB recommended) and no CUDA "
            f"device is visible. Deploy a RunPod GPU pod, or set SE_PIN_GPU/"
            f"CUDA_VISIBLE_DEVICES if you hid the devices yourself."
        )

    if info["count"] and min_vram_gb and info["vram_gb"] + 2 < min_vram_gb:
        print(f"!! {info['vram_gb']:.0f} GiB per GPU, {min_vram_gb} GB expected. Training can "
              f"still fit — gradient checkpointing is already on — by halving "
              f"PER_DEVICE_TRAIN_BATCH_SIZE and doubling GRADIENT_ACCUMULATION_STEPS in "
              f"se_config: keep the product at 16, because the effective batch has to stay "
              f"identical across arms or they are no longer comparable. Expect a slower sweep, "
              f"and watch the eval generate for OOM.")
    if info["count"] > 1:
        print(f"!! {info['count']} GPUs visible. device_map=\"auto\" will shard the model "
              f"across all of them, which is correct but makes throughput and memory "
              f"numbers incomparable between runs. Export SE_PIN_GPU=0 before starting "
              f"Jupyter to keep one pod = one device.")
    if info["count"] and not info["bf16"]:
        print("!! bf16 unsupported on this GPU; se_common falls back to fp16, and Appendix C's "
              "invariance tolerances were quoted for bf16. Prefer an Ampere-or-later pod.")
    return info


def pin_gpu(index=None, verbose=True):
    """Restrict the process to one device. Must run before torch initializes CUDA."""
    index = os.environ.get("SE_PIN_GPU") if index is None else index
    if index in (None, ""):
        return None
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        print(f"!! SE_PIN_GPU={index} ignored: CUDA is already initialized in this kernel. "
              f"Restart the kernel to apply it.")
        return None
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    if verbose:
        print(f"pinned to CUDA_VISIBLE_DEVICES={index}")
    return str(index)


# ---------------------------------------------------------------------------
# the entry point the notebooks call
# ---------------------------------------------------------------------------

def bootstrap(repo_dir=None, gpu="optional", min_vram_gb=0, install=False,
              min_free_gb=60, verbose=True):
    """Prepare a pod for one notebook, print what it actually got, and return it.

    The order is not arbitrary: pin the device before torch sees CUDA, point the caches at
    the volume before huggingface_hub is imported, then authenticate, then look at the
    hardware, and only then import `se_config` — whose output paths are derived from the
    volume this function just resolved.

    `gpu` is "required", "optional", or "none", and `min_vram_gb` is what the notebook was
    written for. Both are per-notebook and documented in each notebook's header.
    """
    pin_gpu(verbose=verbose)
    root = persistent_root()
    storage = storage_report(root)
    configure_caches(root, verbose=False)

    if install:
        ensure_deps(verbose=verbose)
    missing = missing_requirements()
    if missing and gpu != "none":
        print("!! missing or outdated dependencies: "
              + ", ".join(f"{n} ({have or 'absent'} < {floor})"
                          for n, (have, floor) in missing.items())
              + "\n   Run the dependency cell above (se_env.ensure_deps()) and re-run this one.")

    auth = hf_login(root, verbose=False)
    versions = installed_versions()
    # report first, enforce after: a "no GPU visible" traceback is easier to act on when the
    # volume, cache, and stack it would have used are printed above it
    gpu_info = gpu_report(min_vram_gb, requirement="optional", verbose=False)

    import se_config as C

    for d in (C.SE_ROOT, C.ROTATION_DIR, C.RUNS_DIR, C.FIGURES_DIR, C.REPORTS_DIR):
        os.makedirs(d, exist_ok=True)

    env = {
        "runpod": is_runpod(), "pod_id": pod_id(),
        "repo_dir": repo_dir, "persist_root": root, "se_root": C.SE_ROOT,
        "hf_home": os.environ["HF_HOME"], "storage": storage,
        "gpu": gpu_info, "has_gpu": gpu_info["count"] > 0,
        "gpu_count": gpu_info["count"], "vram_gb": gpu_info["vram_gb"],
        "versions": versions, "auth": auth,
        "python": ".".join(str(v) for v in sys.version_info[:3]),
    }

    if verbose:
        _print_report(env)

    # provenance: which pod, which stack. NB07 quotes the gate numbers; a reader asking
    # "on what hardware, with which transformers?" should not have to guess.
    try:
        import datetime

        record = dict(env, written_at=datetime.datetime.now().isoformat(timespec="seconds"))
        with open(f"{C.REPORTS_DIR}/environment.json", "w") as f:
            json.dump(record, f, indent=2, default=str)
    except OSError as err:
        print(f"(could not write environment.json: {err})")

    if gpu == "required" and not gpu_info["count"]:
        if not gpu_info["torch"]:
            raise RuntimeError(
                "torch is not installed in this kernel, so there is no GPU to find. Run the "
                "dependency cell (se_env.ensure_deps()) on a RunPod PyTorch pod first."
            )
        raise RuntimeError(
            f"This notebook needs a GPU pod ({min_vram_gb} GB recommended) and no CUDA device "
            f"is visible. Deploy a RunPod GPU pod, or clear SE_PIN_GPU/CUDA_VISIBLE_DEVICES "
            f"if you hid the devices yourself."
        )
    if not auth["token"] and gpu != "none":
        print(f"!! no HF token found. Public checkpoints still work; set HF_TOKEN in the pod "
              f"template or write it to {root}/.hf_token if a download 401s.")
    if storage["free_gb"] < min_free_gb and gpu != "none":
        print(f"!! {storage['free_gb']:.0f} GiB free on {root}. Qwen3-8B is 16 GB, the 4B "
              f"cross-model 8 GB, and each finished run's adapter ~0.2 GB; {min_free_gb} GB "
              f"is the comfortable floor. Resize the volume before starting a sweep.")
    if not storage["mount"] and is_runpod():
        print(f"!! {root} is not a separate mount, so it is container storage: it survives a "
              f"pod restart but not a terminate. Attach a network volume, or copy "
              f"{C.SE_ROOT} off the pod before you stop it.")
    return env


def _print_report(env):
    gpu = env["gpu"]
    stack = " · ".join(
        f"{name} {env['versions'][name]}" for name in
        ("transformers", "peft", "trl", "datasets", "accelerate", "bitsandbytes")
        if env["versions"].get(name)
    )
    where = "RunPod pod " + env["pod_id"] if env["pod_id"] else (
        "RunPod" if env["runpod"] else "not a RunPod pod")
    storage = env["storage"]
    mount = "own mount" if storage["mount"] else "container filesystem"

    print("RUNPOD ENVIRONMENT")
    print("=" * 78)
    print(f"host    : {where}  |  python {env['python']}")
    if gpu["count"]:
        print(f"gpu     : {gpu['count']} x {', '.join(sorted(set(gpu['names'])))}  |  "
              f"{gpu['vram_gb']:.0f} GiB  |  bf16 {'yes' if gpu['bf16'] else 'NO'}")
    else:
        print("gpu     : none visible")
    print(f"torch   : {gpu['torch']}  (CUDA {gpu['cuda']})")
    print(f"stack   : {stack}")
    if env["repo_dir"]:
        print(f"repo    : {env['repo_dir']}")
    print(f"volume  : {storage['root']}  ({mount}, {storage['free_gb']:.0f} GiB free)")
    print(f"hf cache: {env['hf_home']}")
    print(f"outputs : {env['se_root']}")
    auth = env["auth"]
    print(f"hf token: {('from ' + str(auth['source'])) if auth['token'] else 'none'}"
          + (f"  (user {auth['user']})" if auth.get("user") else ""))


if __name__ == "__main__":
    # `python se/se_env.py --install` provisions a fresh pod from a shell, before Jupyter.
    if "--install" in sys.argv:
        ensure_deps()
    bootstrap(repo_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
