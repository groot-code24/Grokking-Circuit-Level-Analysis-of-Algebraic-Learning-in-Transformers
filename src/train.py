"""
train.py
--------
Training loop for grokking experiments.

Usage (from repo root):
    python src/train.py --op add --p 113 --epochs 5000 --save_dir checkpoints/

Colab usage:
    from src.train import train_experiment, multi_seed_experiment
    results = train_experiment(op="add", p=113, epochs=5000)
    agg     = multi_seed_experiment(op="add", p=113, epochs=5000, seeds=[42,1,2])
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

try:
    # transformer_lens 1.x references transformers.TRANSFORMERS_CACHE, which was
    # removed in transformers >= 4.38.0. Apply a compatibility patch before import.
    import transformers as _transformers_mod
    if not hasattr(_transformers_mod, "TRANSFORMERS_CACHE"):
        import os as _os
        _transformers_mod.TRANSFORMERS_CACHE = _os.path.join(
            _os.path.expanduser("~"), ".cache", "huggingface", "transformers"
        )
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    _TRANSFORMER_LENS_AVAILABLE = True
except Exception:
    _TRANSFORMER_LENS_AVAILABLE = False

from src.datasets import (
    build_loaders,
    make_modular_addition,
    make_modular_multiplication,
    make_modular_subtraction,
    make_ring_addition,
    make_s3_group,
    make_d5_group,
    make_a4_group,
    make_s4_group,
)


# Weight matrix name fragments used for norm tracking during training.
# Only key interpretable matrices are logged to keep memory overhead low.
_KEY_WEIGHT_FRAGMENTS: tuple = (
    "embed", "W_Q", "W_K", "W_V", "W_O", "W_in", "W_out", "unembed",
)


@dataclass
class TrainConfig:
    op: str                     = "add"
    p: int                      = 113
    train_frac: float           = 0.70
    n_layers: int               = 1
    n_heads: int                = 4
    d_model: int                = 128
    d_head: int                 = 32
    d_mlp: int                  = 512
    act_fn: str                 = "relu"
    lr: float                   = 1e-3
    weight_decay: float         = 1.0
    epochs: int                 = 5000
    log_every: int              = 100
    save_every: int             = 500
    save_dir: str               = "checkpoints"
    seed: int                   = 42
    track_representations: bool = False


# ---------------------------------------------------------------------------
# Minimal fallback transformer (used when TransformerLens is unavailable)
# ---------------------------------------------------------------------------

class _Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_head
        self.scale   = d_head ** -0.5
        inner = n_heads * d_head
        self.W_Q = nn.Linear(d_model, inner, bias=False)
        self.W_K = nn.Linear(d_model, inner, bias=False)
        self.W_V = nn.Linear(d_model, inner, bias=False)
        self.W_O = nn.Linear(inner, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        H, D    = self.n_heads, self.d_head

        def split(t: Tensor) -> Tensor:
            return t.view(B, T, H, D).transpose(1, 2)

        Q, K, V = split(self.W_Q(x)), split(self.W_K(x)), split(self.W_V(x))
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out  = (attn @ V).transpose(1, 2).contiguous().view(B, T, H * D)
        return self.W_O(out)


class _MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_mlp, bias=False),
            nn.ReLU(),
            nn.Linear(d_mlp, d_model, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _TransformerBlock(nn.Module):
    """Single transformer block (attention + MLP with pre-norm)."""

    def __init__(self, d_model: int, n_heads: int, d_head: int, d_mlp: int) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = _Attention(d_model, n_heads, d_head)
        self.ln2  = nn.LayerNorm(d_model)
        self.mlp  = _MLP(d_model, d_mlp)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MinimalTransformer(nn.Module):
    """
    n_layers-deep transformer matching the TransformerLens config.
    Used as fallback when TransformerLens is not installed.
    """

    def __init__(self, cfg: TrainConfig, vocab_size: int) -> None:
        super().__init__()
        dm = cfg.d_model
        self.embed   = nn.Embedding(vocab_size, dm)
        self.pos_emb = nn.Embedding(3, dm)
        self.blocks  = nn.ModuleList([
            _TransformerBlock(dm, cfg.n_heads, cfg.d_head, cfg.d_mlp)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f    = nn.LayerNorm(dm)
        self.unembed = nn.Linear(dm, vocab_size, bias=False)

    def forward(self, tokens: Tensor) -> Tensor:
        B, T = tokens.shape
        pos  = torch.arange(T, device=tokens.device).unsqueeze(0)
        x    = self.embed(tokens) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.unembed(x)


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

_OP_MAP = {
    "add":      make_modular_addition,
    "mul":      make_modular_multiplication,
    "sub":      make_modular_subtraction,
    "ring_add": make_ring_addition,
    "s3":       make_s3_group,
    "d5":       make_d5_group,
    "a4":       make_a4_group,
    "s4":       make_s4_group,
}


def _make_dataset(cfg: TrainConfig) -> dict:
    fn = _OP_MAP[cfg.op]
    if cfg.op in ("s3", "d5", "a4", "s4"):
        return fn(train_frac=cfg.train_frac, seed=cfg.seed)
    if cfg.op == "ring_add":
        return fn(n=cfg.p, train_frac=cfg.train_frac, seed=cfg.seed)
    return fn(p=cfg.p, train_frac=cfg.train_frac, seed=cfg.seed)


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_experiment(
    op: str = "add",
    p: int = 113,
    epochs: int = 5000,
    save_dir: Optional[str] = None,
    device: Optional[str] = None,
    cfg_overrides: Optional[dict] = None,
) -> Dict:
    """
    Train a transformer on an algebraic task and return a results dict.

    Returns
    -------
    dict with keys:
        cfg          : TrainConfig (as dict)
        train_loss   : list[float]
        test_loss    : list[float]
        train_acc    : list[float]
        test_acc     : list[float]
        log_steps    : list[int]
        grok_epoch   : int | None   (first epoch where test_acc >= 0.99)
        model        : nn.Module    (trained model)
        dataset      : dict
    """
    cfg = TrainConfig(op=op, p=p, epochs=epochs)
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)

    # Seed after cfg is fully built so cfg_overrides can modify cfg.seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    if save_dir is not None:
        cfg.save_dir = save_dir

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = _make_dataset(cfg)
    vocab_size = dataset["vocab_size"]
    train_loader, test_loader = build_loaders(dataset, batch_size=-1)

    model: nn.Module
    if _TRANSFORMER_LENS_AVAILABLE:
        tl_cfg = HookedTransformerConfig(
            n_layers           = cfg.n_layers,
            n_heads            = cfg.n_heads,
            d_model            = cfg.d_model,
            d_head             = cfg.d_head,
            d_mlp              = cfg.d_mlp,
            n_ctx              = 3,
            d_vocab            = vocab_size,
            act_fn             = cfg.act_fn,
            normalization_type = "LN",
            attn_only          = False,
        )
        model = HookedTransformer(tl_cfg)
    else:
        model = MinimalTransformer(cfg, vocab_size)

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.lr,
        weight_decay = cfg.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    train_losses:        List[float] = []
    test_losses:         List[float] = []
    train_accs:          List[float] = []
    test_accs:           List[float] = []
    log_steps:           List[int]   = []
    weight_norm_history: List[Dict]  = []
    repr_history:        List[Dict]  = []
    grok_epoch:          Optional[int] = None

    save_path = Path(cfg.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    def _evaluate(loader) -> Tuple[float, float]:
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for tokens, labels in loader:
                tokens, labels = tokens.to(device), labels.to(device)
                logits = model(tokens)
                logits_last = logits[:, -1, :]
                loss = criterion(logits_last, labels)
                preds = logits_last.argmax(dim=-1)
                total_loss    += loss.item() * len(labels)
                total_correct += (preds == labels).sum().item()
                total_samples += len(labels)
        return total_loss / total_samples, total_correct / total_samples

    t0 = time.time()
    print(f"\n[train] op={dataset['op_name']}  seed={cfg.seed}  device={device}  epochs={cfg.epochs}")
    print(f"        model params: {sum(param.numel() for param in model.parameters()):,}")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for tokens, labels in train_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens)
            loss   = criterion(logits[:, -1, :], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if epoch % cfg.log_every == 0:
            tr_loss, tr_acc = _evaluate(train_loader)
            te_loss, te_acc = _evaluate(test_loader)
            train_losses.append(tr_loss)
            test_losses.append(te_loss)
            train_accs.append(tr_acc)
            test_accs.append(te_acc)
            log_steps.append(epoch)

            if grok_epoch is None and te_acc >= 0.99:
                grok_epoch = epoch
                print(f"  *** GROKKING at epoch {epoch} — test_acc={te_acc:.4f} ***")

            norm_snapshot: Dict[str, float] = {"epoch": epoch}
            for name, param in model.named_parameters():
                if param.requires_grad and any(
                    frag in name for frag in _KEY_WEIGHT_FRAGMENTS
                ):
                    norm_snapshot[name] = float(param.detach().cpu().norm(2).item())
            weight_norm_history.append(norm_snapshot)

            if cfg.track_representations:
                try:
                    from src.analysis import representation_formation_tracker
                    rft = representation_formation_tracker(model, dataset, cfg.op)
                    repr_history.append({
                        "epoch":              epoch,
                        "formation_score":    rft["formation_score"],
                        "threshold_crossed":  rft["threshold_crossed"],
                        "dominant_component": rft["dominant_component"],
                    })
                except Exception:
                    pass

            elapsed = time.time() - t0
            print(
                f"  epoch {epoch:5d}/{cfg.epochs}  "
                f"tr_loss={tr_loss:.4f}  te_loss={te_loss:.4f}  "
                f"tr_acc={tr_acc:.3f}  te_acc={te_acc:.3f}  "
                f"[{elapsed:.0f}s]"
            )

        if epoch % cfg.save_every == 0:
            # Rolling checkpoint — overwrites the same file each interval to
            # minimise disk usage while preserving crash-resume capability.
            ckpt = {
                "epoch":           epoch,
                "cfg":             asdict(cfg),
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_accs":      train_accs,
                "test_accs":       test_accs,
            }
            ckpt_path = save_path / f"{cfg.op}_p{cfg.p}_seed{cfg.seed}_ROLLING.pt"
            torch.save(ckpt, ckpt_path)

    # Final checkpoint — optimizer state is intentionally omitted to halve file size.
    final_ckpt = {
        "epoch":               cfg.epochs,
        "cfg":                 asdict(cfg),
        "model_state":         model.state_dict(),
        "train_losses":        train_losses,
        "test_losses":         test_losses,
        "train_accs":          train_accs,
        "test_accs":           test_accs,
        "log_steps":           log_steps,
        "grok_epoch":          grok_epoch,
        "weight_norm_history": weight_norm_history,
        "repr_history":        repr_history,
    }
    final_path = save_path / f"{cfg.op}_p{cfg.p}_seed{cfg.seed}_FINAL.pt"
    torch.save(final_ckpt, final_path)
    print(f"\n[train] saved final checkpoint -> {final_path}")

    rolling_path = save_path / f"{cfg.op}_p{cfg.p}_seed{cfg.seed}_ROLLING.pt"
    if rolling_path.exists():
        rolling_path.unlink()
        print(f"[train] deleted rolling checkpoint")

    meta = {
        "op":              cfg.op,
        "op_name":         dataset["op_name"],
        "p":               cfg.p,
        "seed":            cfg.seed,
        "grok_epoch":      grok_epoch,
        "final_train_acc": train_accs[-1] if train_accs else None,
        "final_test_acc":  test_accs[-1]  if test_accs  else None,
    }
    meta_path = save_path / f"{cfg.op}_p{cfg.p}_seed{cfg.seed}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "cfg":                 asdict(cfg),
        "train_loss":          train_losses,
        "test_loss":           test_losses,
        "train_acc":           train_accs,
        "test_acc":            test_accs,
        "log_steps":           log_steps,
        "grok_epoch":          grok_epoch,
        "model":               model,
        "dataset":             dataset,
        "weight_norm_history": weight_norm_history,
        "repr_history":        repr_history,
    }


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def multi_seed_experiment(
    op: str = "add",
    p: int = 113,
    epochs: int = 5000,
    seeds: Optional[List[int]] = None,
    save_dir: Optional[str] = None,
    device: Optional[str] = None,
    cfg_overrides: Optional[dict] = None,
) -> Dict:
    """
    Run train_experiment over multiple seeds and aggregate results.

    Returns
    -------
    dict with:
        runs         : list of individual result dicts (one per seed)
        mean_train_acc, std_train_acc  : arrays over seeds x log_steps
        mean_test_acc,  std_test_acc   : arrays
        mean_grok_epoch, std_grok_epoch, all_grok_epochs : statistics
        log_steps    : common log step axis
    """
    if seeds is None:
        seeds = [42, 1, 7]

    runs = []
    for seed in seeds:
        overrides = dict(cfg_overrides or {})
        overrides["seed"] = seed
        print(f"\n{'='*60}")
        print(f"  Running seed {seed} / {seeds}")
        print(f"{'='*60}")
        res = train_experiment(
            op=op, p=p, epochs=epochs,
            save_dir=save_dir, device=device,
            cfg_overrides=overrides,
        )
        runs.append(res)

    log_steps = runs[0]["log_steps"]
    n_steps   = len(log_steps)

    def _pad(arr, target_len):
        if len(arr) >= target_len:
            return arr[:target_len]
        return arr + [arr[-1]] * (target_len - len(arr))

    train_accs = np.array([_pad(r["train_acc"], n_steps) for r in runs])
    test_accs  = np.array([_pad(r["test_acc"],  n_steps) for r in runs])
    train_loss = np.array([_pad(r["train_loss"], n_steps) for r in runs])
    test_loss  = np.array([_pad(r["test_loss"],  n_steps) for r in runs])

    grok_epochs = [r["grok_epoch"] for r in runs if r["grok_epoch"] is not None]

    summary = {
        "op":              op,
        "p":               p,
        "seeds":           seeds,
        "runs":            runs,
        "log_steps":       log_steps,
        "mean_train_acc":  train_accs.mean(axis=0).tolist(),
        "std_train_acc":   train_accs.std(axis=0).tolist(),
        "mean_test_acc":   test_accs.mean(axis=0).tolist(),
        "std_test_acc":    test_accs.std(axis=0).tolist(),
        "mean_train_loss": train_loss.mean(axis=0).tolist(),
        "std_train_loss":  train_loss.std(axis=0).tolist(),
        "mean_test_loss":  test_loss.mean(axis=0).tolist(),
        "std_test_loss":   test_loss.std(axis=0).tolist(),
        "all_grok_epochs": grok_epochs,
        "mean_grok_epoch": float(np.mean(grok_epochs)) if grok_epochs else None,
        "std_grok_epoch":  float(np.std(grok_epochs))  if len(grok_epochs) > 1 else None,
        "grok_rate":       len(grok_epochs) / len(seeds),
    }

    print(f"\n[multi_seed] op={op}  p={p}  seeds={seeds}")
    print(f"  grok_epochs : {grok_epochs}")
    print(f"  mean        : {summary['mean_grok_epoch']}")
    print(f"  std         : {summary['std_grok_epoch']}")
    print(f"  grok_rate   : {summary['grok_rate']:.0%}")

    if save_dir is not None:
        agg_path = Path(save_dir) / f"{op}_p{p}_multi_seed_agg.json"
        save_dict = {k: v for k, v in summary.items() if k != "runs"}
        with open(agg_path, "w") as f:
            json.dump(save_dict, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# Multi-p ablation
# ---------------------------------------------------------------------------

def multi_p_experiment(
    op: str = "add",
    p_values: Optional[List[int]] = None,
    epochs: int = 5000,
    seeds: Optional[List[int]] = None,
    save_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict:
    """
    Run multi_seed_experiment for each value of p in p_values.
    Used to verify that the complexity-delay law holds across moduli.

    Returns
    -------
    dict mapping p -> aggregate result from multi_seed_experiment
    """
    if p_values is None:
        p_values = [53, 97, 113]
    if seeds is None:
        seeds = [42, 1, 7]

    results = {}
    for p_val in p_values:
        print(f"\n{'#'*60}")
        print(f"  op={op}  p={p_val}")
        print(f"{'#'*60}")
        results[p_val] = multi_seed_experiment(
            op=op, p=p_val, epochs=epochs,
            seeds=seeds, save_dir=save_dir, device=device,
        )
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train grokking transformer")
    parser.add_argument("--op",      type=str,   default="add",
                        choices=list(_OP_MAP.keys()), help="Algebraic operation")
    parser.add_argument("--p",       type=int,   default=113,
                        help="Modulus (or group size for non-abelian ops)")
    parser.add_argument("--epochs",  type=int,   default=5000)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--wd",      type=float, default=1.0,
                        help="Weight decay (AdamW)")
    parser.add_argument("--save_dir",type=str,   default="checkpoints")
    parser.add_argument("--device",  type=str,   default=None)
    parser.add_argument("--seeds",   type=int,   nargs="+", default=[42],
                        help="One or more seeds; multiple runs multi_seed_experiment")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    overrides = {"lr": args.lr, "weight_decay": args.wd}
    if len(args.seeds) == 1:
        train_experiment(
            op=args.op, p=args.p, epochs=args.epochs,
            save_dir=args.save_dir, device=args.device,
            cfg_overrides={**overrides, "seed": args.seeds[0]},
        )
    else:
        multi_seed_experiment(
            op=args.op, p=args.p, epochs=args.epochs,
            seeds=args.seeds, save_dir=args.save_dir, device=args.device,
            cfg_overrides=overrides,
        )
