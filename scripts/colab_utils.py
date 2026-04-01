from __future__ import annotations

import gc
import os
import sys
import json
import time
from pathlib import Path
from typing import Optional

# Free-tier training config defaults

FREE_TIER_DEFAULTS = {
    "d_model":     128,  
    "d_mlp":       512,  
    "n_heads":       4,
    "d_head":       32,  
    "n_layers":      1,
    "epochs":     3000, 
    "save_every":  200,
    "log_every":    50,
    "lr":          1e-3,
    "weight_decay": 1.0,
}

FREE_TIER_SEEDS = [42, 1, 7]
FREE_TIER_EPOCHS = {
    "add":      3000,
    "sub":      3000,
    "mul":      3000,
    "ring_add": 3000,
    "s3":       2000,   
    "d5":       3000,
    "a4":       4000,   
    "s4":       5000,   
}

def setup_free_tier(
    project_dir: str = "/content/grokking-research",
    drive_folder: str = "grokking_research_v3",
) -> tuple:
    """
    Complete free-tier setup in one call.

    Returns
    -------
    (CKPT_DIR, FIG_DIR, device)  — strings / torch.device
    """
    _ensure_project_on_path(project_dir)
    check_gpu()
    CKPT_DIR, FIG_DIR = _mount_drive_or_local(drive_folder)
    device = _get_device()
    _print_memory_summary()
    enable_keepalive()
    return CKPT_DIR, FIG_DIR, device

def check_gpu(warn_only: bool = False) -> bool:
    """Print GPU info. Returns True if GPU is available."""
    import subprocess
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                        "--format=csv,noheader"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print(f"✅ GPU: {r.stdout.strip()}")
        return True
    msg = ("⚠️  No GPU detected!\n"
           "   → Runtime → Change runtime type → Hardware accelerator → T4 GPU\n"
           "   → Training will be 10–50× slower on CPU.")
    if warn_only:
        print(msg)
        return False
    raise RuntimeError(msg)

def _get_device() -> "torch.device":  
    try:
        import torch
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Device: {dev}")
        return dev
    except ImportError:
        print("⚠️  torch not installed yet — run pip install cell first.")
        return None 

def free_gpu_memory(model=None) -> None:
    """
    Release GPU memory between experiments.
    Call this after each train_experiment() to avoid OOM.
    """
    try:
        import torch
        if model is not None:
            model.cpu()
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        _print_memory_summary()
    except ImportError:
        gc.collect()

def _print_memory_summary() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            used  = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🔋 GPU memory: {used:.1f} GB used / {total:.1f} GB total")
    except Exception:
        pass

def enable_keepalive() -> None:
    """
    Inject a tiny JavaScript snippet that clicks the Colab 'connect' button
    every 60 seconds, preventing the idle-disconnect timeout.

    Only works inside a real Colab notebook (silently skips otherwise).
    """
    try:
        from IPython.display import Javascript, display
        js = """
        if (typeof window._grokKeepAlive === 'undefined') {
            window._grokKeepAlive = setInterval(function() {
                var btn = document.querySelector('colab-connect-button');
                if (btn) { btn.shadowRoot.querySelector('#connect').click(); }
            }, 60000);
            console.log('[grokking] keepalive enabled (60s interval)');
        }
        """
        display(Javascript(js))
        print("⏰ Session keepalive enabled (prevents idle disconnect).")
    except Exception:
        pass  # Not in Colab — skip silently

def disable_keepalive() -> None:
    """Stop the keepalive timer."""
    try:
        from IPython.display import Javascript, display
        display(Javascript("clearInterval(window._grokKeepAlive); delete window._grokKeepAlive;"))
        print("⏰ Session keepalive disabled.")
    except Exception:
        pass

def _mount_drive_or_local(drive_folder: str) -> tuple:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        base     = f"/content/drive/MyDrive/{drive_folder}"
        ckpt_dir = os.path.join(base, "checkpoints")
        fig_dir  = os.path.join(base, "figures")
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(fig_dir,  exist_ok=True)
        print(f"✅ Drive mounted → {base}")
    except Exception as e:
        print(f"ℹ️  Drive not mounted ({e}). Using local /content/checkpoints.")
        ckpt_dir = "/content/checkpoints"
        fig_dir  = "/content/figures"
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(fig_dir,  exist_ok=True)
    return ckpt_dir, fig_dir

def _ensure_project_on_path(project_dir: str) -> None:
    if not os.path.exists(project_dir):
        raise FileNotFoundError(
            f"Project not found at {project_dir}.\n"
            "Run the setup cell in 00_colab_setup.ipynb first."
        )
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    os.chdir(project_dir)
    print(f"📁 Working directory: {os.getcwd()}")

def load_checkpoint(
    op: str,
    p: int,
    seed: int,
    ckpt_dir: str,
    device: Optional[str] = None,
) -> Optional[dict]:
    """
    Load the FINAL checkpoint for (op, p, seed) from ckpt_dir.
    Returns None if no checkpoint exists (training needed).

    Security: uses weights_only=True on PyTorch ≥ 2.6 to prevent arbitrary
    code execution from malicious pickle payloads. Falls back to
    weights_only=False on older versions where the arg is unsupported.
    """
    import torch
    path = Path(ckpt_dir) / f"{op}_p{p}_seed{seed}_FINAL.pt"
    if not path.exists():
        print(f"  No checkpoint for op={op} p={p} seed={seed} — will train.")
        return None

    # Determine weights_only support (added in PyTorch 1.13, safe to pass True
    # when checkpoints only contain tensors + standard Python objects).
    _torch_major_minor = tuple(int(x) for x in torch.__version__.split(".")[:2])
    _weights_only = _torch_major_minor >= (2, 6)

    try:
        ckpt = torch.load(
            path,
            map_location=device or "cpu",
            weights_only=_weights_only,
        )
    except TypeError:
        # weights_only kwarg not available in this torch version
        ckpt = torch.load(path, map_location=device or "cpu")

    print(f"  ✅ Loaded checkpoint: {path.name}  "
          f"(epoch={ckpt.get('epoch')}, grok_epoch={ckpt.get('grok_epoch')})")
    return ckpt

def smart_train(
    op: str,
    p: int = 113,
    seeds: Optional[list] = None,
    epochs: Optional[int] = None,
    ckpt_dir: str = "/content/checkpoints",
    device: Optional[str] = None,
    free_tier: bool = True,
) -> dict:
    """
    Run multi-seed training with automatic checkpoint resume.

    - If a FINAL checkpoint already exists for a seed, it is loaded and
      that seed is skipped.
    - Uses FREE_TIER_DEFAULTS when free_tier=True.

    Returns the same dict as multi_seed_experiment().
    """
    from src.train import train_experiment, multi_seed_experiment
    from src.train import TrainConfig

    if seeds is None:
        seeds = FREE_TIER_SEEDS
    if epochs is None:
        epochs = FREE_TIER_EPOCHS.get(op, 3000)

    overrides = dict(FREE_TIER_DEFAULTS) if free_tier else {}
    overrides["epochs"] = epochs

    runs = []
    for seed in seeds:
        ckpt = load_checkpoint(op, p, seed, ckpt_dir, device)
        if ckpt is not None:
            
            runs.append({
                "cfg":        ckpt.get("cfg", {}),
                "train_acc":  ckpt.get("train_accs", []),
                "test_acc":   ckpt.get("test_accs", []),
                "train_loss": ckpt.get("train_losses", []),
                "test_loss":  ckpt.get("test_losses", []),
                "log_steps":  ckpt.get("log_steps", []),
                "grok_epoch": ckpt.get("grok_epoch"),
                "model":      None,   
                "dataset":    None,
            })
        else:
            seed_overrides = dict(overrides)
            seed_overrides["seed"] = seed
            res = train_experiment(
                op=op, p=p, epochs=epochs,
                save_dir=ckpt_dir, device=device,
                cfg_overrides=seed_overrides,
            )
            runs.append(res)
            free_gpu_memory(res.get("model"))

    # Build aggregate summary — matches multi_seed_experiment() API exactly
    import numpy as np
    log_steps   = runs[0]["log_steps"] or []
    n_steps     = len(log_steps)

    def _pad(arr):
        if not arr:
            return [0.0] * max(n_steps, 1)
        if len(arr) >= n_steps:
            return arr[:n_steps]
        return arr + [arr[-1]] * (n_steps - len(arr))

    train_accs  = np.array([_pad(r["train_acc"])  for r in runs])
    test_accs   = np.array([_pad(r["test_acc"])   for r in runs])
    train_loss  = np.array([_pad(r["train_loss"]) for r in runs])
    test_loss   = np.array([_pad(r["test_loss"])  for r in runs])
    grok_epochs = [r["grok_epoch"] for r in runs if r["grok_epoch"] is not None]

    return {
        "op":               op,
        "p":                p,
        "seeds":            seeds,
        "runs":             runs,
        "log_steps":        log_steps,
        "mean_train_acc":   train_accs.mean(axis=0).tolist(),
        "std_train_acc":    train_accs.std(axis=0).tolist(),
        "mean_test_acc":    test_accs.mean(axis=0).tolist(),
        "std_test_acc":     test_accs.std(axis=0).tolist(),
        "mean_train_loss":  train_loss.mean(axis=0).tolist(),
        "std_train_loss":   train_loss.std(axis=0).tolist(),
        "mean_test_loss":   test_loss.mean(axis=0).tolist(),
        "std_test_loss":    test_loss.std(axis=0).tolist(),
        "all_grok_epochs":  grok_epochs,
        "mean_grok_epoch":  float(np.mean(grok_epochs))  if grok_epochs else None,
        "std_grok_epoch":   float(np.std(grok_epochs))   if len(grok_epochs) > 1 else None,
        "grok_rate":        len(grok_epochs) / len(seeds),
    }

# ---------------------------------------------------------------------------
# 7. Install helper (skip already-installed packages)
# ---------------------------------------------------------------------------

def install_deps(force: bool = False) -> None:
    """
    Install only what's missing.  On Colab, torch/numpy/scipy are pre-installed,
    so we only need transformer_lens, einops, kaleido, and tqdm.
    Pass force=True to reinstall everything.
    """
    import subprocess
    # Packages Colab doesn't pre-install — pinned for reproducibility
    extra = [
        "transformer_lens>=1.14.0,<2.0.0",
        "einops>=0.7.0",
        "kaleido==0.2.1",
        "tqdm>=4.66.0",
        "ipywidgets>=8.1.0",
        "numpy>=1.24.0,<2.0.0",
    ]
    if force:
        pkgs = [
            "torch>=2.1.0",
            "transformer_lens>=1.14.0,<2.0.0",
            "einops>=0.7.0",
            "plotly>=5.18.0",
            "matplotlib>=3.8.0",
            "scipy>=1.11.0",
            "numpy>=1.24.0,<2.0.0",
            "tqdm>=4.66.0",
            "pandas>=2.1.0",
            "kaleido==0.2.1",
            "ipywidgets>=8.1.0",
        ]
        cmd = [sys.executable, "-m", "pip", "install", "-q"] + pkgs
    else:
        cmd = [sys.executable, "-m", "pip", "install", "-q"] + extra

    print("📦 Installing dependencies...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("⚠️  pip stderr:", result.stderr[-500:])
    else:
        print("✅ Dependencies ready.")

# ---------------------------------------------------------------------------
# 9. restore_session — convenience alias called at the top of every section
# ---------------------------------------------------------------------------

def restore_session(
    project_dir: str = "/content/grokking-research",
    drive_folder: str = "grokking_research_v3",
) -> "torch.device":  # type: ignore[name-defined]
    """
    Re-run the essential session setup after a Colab disconnect or restart.

    Sets CKPT_DIR and FIG_DIR as globals in the calling frame so section
    cells can use them without re-running 0-B manually.

    Usage (at the top of every section cell):
        device = restore_session()

    Returns
    -------
    torch.device — the detected compute device (cuda or cpu)
    """
    import builtins, inspect

    _ensure_project_on_path(project_dir)
    ckpt_dir, fig_dir = _mount_drive_or_local(drive_folder)
    device = _get_device()

    # Inject CKPT_DIR and FIG_DIR into the caller's global namespace so
    # subsequent cells in the same kernel session can access them directly.
    frame = inspect.currentframe()
    if frame is not None and frame.f_back is not None:
        caller_globals = frame.f_back.f_globals
        caller_globals["CKPT_DIR"] = ckpt_dir
        caller_globals["FIG_DIR"]  = fig_dir
        caller_globals["device"]   = device

    _print_memory_summary()
    return device

def estimate_runtime(op: str, epochs: int, device_str: str = "cuda") -> str:
    """
    Return a rough runtime estimate string for display before training starts.
    Based on empirical T4 benchmarks.
    """
    # Approx seconds per epoch on T4 for each op (free-tier model size)
    spe = {  
        "add":      0.10,
        "sub":      0.10,
        "mul":      0.10,
        "ring_add": 0.10,
        "s3":       0.02,
        "d5":       0.03,
        "a4":       0.04,
        "s4":       0.08,
    }
    if device_str == "cpu":
        multiplier = 20.0
    else:
        multiplier = 1.0
    s = spe.get(op, 0.10) * epochs * multiplier
    if s < 60:
        return f"~{s:.0f}s"
    elif s < 3600:
        return f"~{s/60:.0f} min"
    else:
        return f"~{s/3600:.1f} hr"

def cleanup_drive(ckpt_dir: str, dry_run: bool = False) -> dict:
    """
    Remove files from ckpt_dir that are no longer needed:
      - Old-style intermediate epoch checkpoints  (*_epoch*.pt)
      - Rolling checkpoints left by a crash       (*_ROLLING.pt)
      - PDF figure duplicates                     (*.pdf  in figures/)
      - Duplicate PNG/PDF pairs                   (keep PNG, remove PDF)

    Parameters
    ----------
    ckpt_dir : path to the checkpoints directory on Drive
    dry_run  : if True, only print what WOULD be deleted (no actual deletion)

    Returns
    -------
    dict with:
        deleted      : list of deleted file paths
        freed_mb     : approximate MB freed
        kept         : list of files kept
    """
    from pathlib import Path

    ckpt_path = Path(ckpt_dir)
    fig_path  = ckpt_path.parent / "figures"

    deleted: list = []
    kept:    list = []
    freed_bytes: int = 0

    def _delete(p: Path, reason: str) -> None:
        size = p.stat().st_size if p.exists() else 0
        if dry_run:
            print(f"  [DRY RUN] would delete ({size//1024:4d} KB) {p.name}  [{reason}]")
        else:
            p.unlink(missing_ok=True)
            print(f"  🗑️  deleted ({size//1024:4d} KB) {p.name}  [{reason}]")
        deleted.append(str(p))
        nonlocal freed_bytes
        freed_bytes += size

    print(f"{'[DRY RUN] ' if dry_run else ''}Scanning {ckpt_path} ...")
    print()

    # 1. Old-style intermediate epoch checkpoints: *_epoch*.pt
    for f in sorted(ckpt_path.glob("*_epoch*.pt")):
        _delete(f, "old-style intermediate checkpoint")

    # 2. Rolling checkpoints left by a crash: *_ROLLING.pt

    for f in sorted(ckpt_path.glob("*_ROLLING.pt")):
        _delete(f, "rolling checkpoint (crash remnant)")

    # 3. PDF figure duplicates — keep PNG, delete PDF
    if fig_path.exists():
        for pdf in sorted(fig_path.glob("*.pdf")):
            _delete(pdf, "PDF figure duplicate (PNG kept)")

    # 4. Keep FINAL checkpoints and JSON files
    for f in sorted(ckpt_path.glob("*.pt")):
        if f.exists():
            kept.append(str(f))

    freed_mb = freed_bytes / 1_000_000

    print()
    if deleted:
        action = "Would free" if dry_run else "Freed"
        print(f"✅ {action} {freed_mb:.1f} MB  ({len(deleted)} files removed)")
    else:
        print("✅ Nothing to clean up — Drive is already tidy.")
    print(f"   Kept {len(kept)} FINAL checkpoint(s).")

    return {"deleted": deleted, "freed_mb": freed_mb, "kept": kept}

def show_drive_usage(ckpt_dir: str) -> None:
    """Print a summary of all files in the checkpoint and figures directories."""
    from pathlib import Path

    ckpt_path = Path(ckpt_dir)
    fig_path  = ckpt_path.parent / "figures"

    total = 0
    print(f"{'File':<52} {'Size':>8}")
    print("-" * 62)

    for directory, label in [(ckpt_path, "checkpoints"), (fig_path, "figures")]:
        if not directory.exists():
            continue
        files = sorted(directory.iterdir(), key=lambda f: -f.stat().st_size)
        dir_total = sum(f.stat().st_size for f in files if f.is_file())
        print(f"\n📁 {label}/  ({dir_total/1e6:.1f} MB)")
        for f in files:
            if f.is_file():
                kb = f.stat().st_size // 1024
                bar = "█" * min(30, kb // 50)
                print(f"  {f.name:<50} {kb:>5} KB  {bar}")
        total += dir_total

    print()
    print(f"Total Drive usage: {total/1e6:.1f} MB")
    if total > 30_000_000:
        print("⚠️  Over 30 MB — run cleanup_drive(CKPT_DIR) to free space.")
    else:
        print("✅ Under 30 MB target.")
