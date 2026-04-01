#!/usr/bin/env python3
"""
reproduce_paper.py
------------------
End-to-end script to reproduce ALL experiments and figures from the paper
"Grokking Beyond Addition: Circuit-Level Analysis of Algebraic Learning
in Transformers".

Usage:
    # Full reproduction (GPU recommended, ~4-6 hours on T4):
    python scripts/reproduce_paper.py --ckpt_dir checkpoints --fig_dir paper/figures

    # Quick smoke test (CPU, ~10 min): reduced epochs and model size
    python scripts/reproduce_paper.py --smoke_test

    # Skip already-completed experiments (uses existing checkpoints):
    python scripts/reproduce_paper.py --ckpt_dir checkpoints --skip_existing

Exit code 0 = all experiments completed and all figures generated.
Exit code 1 = one or more experiments failed (see error log).
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import torch

from src.train import train_experiment, multi_seed_experiment, TrainConfig
from src.datasets import (
    make_modular_addition, make_modular_multiplication, make_modular_subtraction,
    make_ring_addition, make_s3_group, make_d5_group, make_a4_group, make_s4_group,
    get_complexity_score, get_complexity_score_v2, COMPLEXITY_MEASURES,
)
from src.analysis import (
    fourier_embedding_analysis, discrete_log_embedding_analysis,
    detect_grokking_phases, nonabelian_fourier_analysis,
    complexity_delay_regression, cka_similarity, cka_matrix,
    representation_formation_tracker, grokking_leading_indicator,
    bootstrap_confidence_interval, describe_learned_circuit,
)
from src.visualise import (
    fig_grokking_curves_multiseed, fig_fourier_spectrum, fig_dlog_analysis_panel,
    fig_complexity_delay_errorbar, fig_cka_heatmap, fig_grokking_delay_comparison,
    fig_representation_vs_grokking, fig_weight_norm_trajectory,
    fig_multi_seed_grokking_delay_cdf, fig_circuit_summary_table,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

FULL_CONFIG = {
    "add":      {"op": "add",      "p": 113, "epochs": 5000, "seeds": [42, 1, 7]},
    "sub":      {"op": "sub",      "p": 113, "epochs": 5000, "seeds": [42, 1, 7]},
    "mul":      {"op": "mul",      "p": 113, "epochs": 5000, "seeds": [42, 1, 7]},
    "ring_add": {"op": "ring_add", "p": 100, "epochs": 5000, "seeds": [42, 1, 7]},
    "s3":       {"op": "s3",       "p": 6,   "epochs": 3000, "seeds": [42, 1, 7]},
    "d5":       {"op": "d5",       "p": 10,  "epochs": 5000, "seeds": [42, 1, 7]},
    "a4":       {"op": "a4",       "p": 12,  "epochs": 7000, "seeds": [42, 1, 7]},
    "s4":       {"op": "s4",       "p": 24,  "epochs": 7000, "seeds": [42]},
}

SMOKE_CONFIG = {
    "add":      {"op": "add",  "p": 13, "epochs": 200, "seeds": [42]},
    "mul":      {"op": "mul",  "p": 13, "epochs": 200, "seeds": [42]},
    "s3":       {"op": "s3",   "p": 6,  "epochs": 200, "seeds": [42]},
    "a4":       {"op": "a4",   "p": 12, "epochs": 200, "seeds": [42]},
}

SMOKE_MODEL_OVERRIDES = {
    "d_model": 32, "d_mlp": 128, "n_heads": 2, "d_head": 16,
    "log_every": 50, "save_every": 100,
    "track_representations": True,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def _load_model(op: str, p: int, seed: int, ckpt_dir: Path,
                device: str = "cpu") -> torch.nn.Module:
    """Load a MinimalTransformer from a FINAL checkpoint."""
    from src.train import MinimalTransformer, TrainConfig
    from src.datasets import (
        make_modular_addition, make_modular_multiplication,
        make_modular_subtraction, make_ring_addition,
        make_s3_group, make_d5_group, make_a4_group, make_s4_group,
    )
    _ds_fn = {
        "add": lambda: make_modular_addition(p=p),
        "sub": lambda: make_modular_subtraction(p=p),
        "mul": lambda: make_modular_multiplication(p=p),
        "ring_add": lambda: make_ring_addition(n=p),
        "s3":  make_s3_group, "d5": make_d5_group,
        "a4":  make_a4_group, "s4": make_s4_group,
    }

    ckpt_path = ckpt_dir / f"{op}_p{p}_seed{seed}_FINAL.pt"
    if not ckpt_path.exists():
        return None

    _torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
    _weights_only  = _torch_version >= (2, 6)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=_weights_only)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    cfg   = TrainConfig(**{**vars(TrainConfig()), **ckpt["cfg"]})
    ds    = _ds_fn[op]()
    model = MinimalTransformer(cfg, vocab_size=ds["vocab_size"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    return model

def _ckpt_exists(op: str, p: int, seed: int, ckpt_dir: Path) -> bool:
    return (ckpt_dir / f"{op}_p{p}_seed{seed}_FINAL.pt").exists()

def _log(msg: str, log_fh=None):
    print(msg)
    if log_fh:
        log_fh.write(msg + "\n")
        log_fh.flush()

# ---------------------------------------------------------------------------
# Phase 1: Train all experiments
# ---------------------------------------------------------------------------

def run_training(config: dict, ckpt_dir: Path, skip_existing: bool,
                 device: str, log_fh=None) -> dict:
    """Run multi-seed training for all operations."""
    results = {}
    for name, cfg_dict in config.items():
        op, p, epochs, seeds = (cfg_dict["op"], cfg_dict["p"],
                                 cfg_dict["epochs"], cfg_dict["seeds"])

        seeds_to_run = [s for s in seeds
                        if not (skip_existing and _ckpt_exists(op, p, s, ckpt_dir))]
        if not seeds_to_run:
            _log(f"  [SKIP] {name}: all checkpoints already exist", log_fh)
            results[name] = {"op": op, "p": p, "status": "skipped"}
            continue

        _log(f"\n{'='*60}", log_fh)
        _log(f"  Training: op={op}  p={p}  epochs={epochs}  seeds={seeds_to_run}",
             log_fh)
        t0 = time.time()
        try:
            overrides = cfg_dict.get("overrides", {})
            overrides["track_representations"] = True
            agg = multi_seed_experiment(
                op=op, p=p, epochs=epochs, seeds=seeds_to_run,
                save_dir=str(ckpt_dir), device=device,
                cfg_overrides=overrides,
            )
            elapsed = time.time() - t0
            _log(f"  ✅ {name}: grok_rate={agg['grok_rate']:.0%}  "
                 f"mean_grok={agg['mean_grok_epoch']}  [{elapsed:.0f}s]", log_fh)
            results[name] = {**agg, "op": op, "p": p, "status": "done"}
        except Exception as e:
            elapsed = time.time() - t0
            _log(f"  ❌ {name}: FAILED after {elapsed:.0f}s — {e}", log_fh)
            _log(traceback.format_exc(), log_fh)
            results[name] = {"op": op, "p": p, "status": "error", "error": str(e)}

    return results

# ---------------------------------------------------------------------------
# Phase 2: Analysis
# ---------------------------------------------------------------------------

def run_analysis(config: dict, results: dict, ckpt_dir: Path,
                 device: str, log_fh=None) -> dict:
    """Run all post-training analysis on saved models."""
    analysis = {}

    # Load models
    models, datasets_map = {}, {}
    for name, cfg_dict in config.items():
        op, p = cfg_dict["op"], cfg_dict["p"]
        seed   = cfg_dict["seeds"][0]
        from src.datasets import (
            make_modular_addition, make_modular_multiplication,
            make_modular_subtraction, make_ring_addition,
            make_s3_group, make_d5_group, make_a4_group, make_s4_group,
        )
        _ds_fns = {
            "add": lambda: make_modular_addition(p=p),
            "sub": lambda: make_modular_subtraction(p=p),
            "mul": lambda: make_modular_multiplication(p=p),
            "ring_add": lambda: make_ring_addition(n=p),
            "s3": make_s3_group, "d5": make_d5_group,
            "a4": make_a4_group, "s4": make_s4_group,
        }
        model = _load_model(op, p, seed, ckpt_dir, device)
        if model is not None:
            models[op]      = model
            datasets_map[op] = _ds_fns[op]()
            _log(f"  [model] loaded {op}", log_fh)

    # Complexity-delay regression (v1 and v2)
    op_results_for_reg = {
        op: res for op, res in {
            cfg["op"]: results.get(name, {})
            for name, cfg in config.items()
        }.items()
        if results.get(list(k for k, v in config.items() if v["op"] == op)[0],
                       {}).get("mean_grok_epoch") is not None
    }
    # Simpler build
    reg_input = {}
    for name, cfg_dict in config.items():
        op  = cfg_dict["op"]
        res = results.get(name, {})
        if res.get("mean_grok_epoch") is not None:
            reg_input[op] = res

    if len(reg_input) >= 3:
        try:
            reg_v1 = complexity_delay_regression(reg_input)
            from src.datasets import get_complexity_score_v2 as _cv2
            reg_v2 = complexity_delay_regression(
                reg_input,
                complexity_scores={op: _cv2(op) for op in reg_input},
            )
            _log(f"\n  Complexity-delay (v1): ρ={reg_v1.get('spearman_r','?'):.3f}  "
                 f"R²={reg_v1.get('r_squared','?'):.3f}  "
                 f"verdict={reg_v1.get('verdict','?')}", log_fh)
            _log(f"  Complexity-delay (v2): ρ={reg_v2.get('spearman_r','?'):.3f}  "
                 f"R²={reg_v2.get('r_squared','?'):.3f}  "
                 f"verdict={reg_v2.get('verdict','?')}", log_fh)
            analysis["regression_v1"] = reg_v1
            analysis["regression_v2"] = reg_v2
        except Exception as e:
            _log(f"  ⚠️  Regression failed: {e}", log_fh)

    # Circuit descriptions
    circuit_descs = {}
    for op, model in models.items():
        try:
            p   = config[next(k for k, v in config.items() if v["op"] == op)]["p"]
            cd  = describe_learned_circuit(model, datasets_map[op], p, op)
            circuit_descs[op] = cd
            _log(f"  [circuit] {op}: {cd['representation_type']}  "
                 f"evidence={cd['evidence_strength']}", log_fh)
        except Exception as e:
            _log(f"  ⚠️  Circuit desc failed for {op}: {e}", log_fh)

    analysis["circuit_descriptions"] = circuit_descs
    analysis["models"]               = models
    analysis["datasets"]             = datasets_map
    return analysis

# ---------------------------------------------------------------------------
# Phase 3: Generate figures
# ---------------------------------------------------------------------------

def run_figures(config: dict, results: dict, analysis: dict,
                fig_dir: Path, ckpt_dir: Path, log_fh=None):
    """Generate all paper figures."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    def _save(name: str, fig):
        if fig is not None:
            fig.savefig(fig_dir / f"{name}.png", bbox_inches="tight", dpi=150)
            fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
            plt.close(fig)
            generated.append(name)
            _log(f"  [figure] {name}.{{png,pdf}}", log_fh)

    # Multi-seed grokking curves
    for name, cfg_dict in config.items():
        op  = cfg_dict["op"]
        res = results.get(name, {})
        if res.get("mean_grok_epoch") is not None:
            try:
                fig = fig_grokking_curves_multiseed(res, save_dir=None)
                _save(f"fig1_grokking_{op}", fig)
            except Exception as e:
                _log(f"  ⚠️  Grokking curve {op}: {e}", log_fh)

    # Complexity-delay error bar
    reg_v1 = analysis.get("regression_v1")
    if reg_v1 and "error" not in reg_v1:
        try:
            op_agg = {
                cfg["op"]: results[name]
                for name, cfg in config.items()
                if results.get(name, {}).get("mean_grok_epoch") is not None
            }
            fig = fig_complexity_delay_errorbar(op_agg, save_dir=None)
            _save("fig_complexity_delay", fig)
        except Exception as e:
            _log(f"  ⚠️  Complexity-delay figure: {e}", log_fh)

    # Grokking CDF
    try:
        op_agg = {
            cfg["op"]: results[name]
            for name, cfg in config.items()
            if results.get(name, {}).get("all_grok_epochs")
        }
        if op_agg:
            fig = fig_multi_seed_grokking_delay_cdf(op_agg, save_dir=None)
            _save("fig_grokking_cdf", fig)
    except Exception as e:
        _log(f"  ⚠️  Grokking CDF: {e}", log_fh)

    # CKA heatmap
    models = analysis.get("models", {})
    if len(models) >= 2:
        try:
            p_ref = 24  # vocab size upper bound for CKA
            mat, labels = cka_matrix(models, p=min(p_ref, min(
                COMPLEXITY_MEASURES[op]["group_order"]
                if not COMPLEXITY_MEASURES[op]["is_abelian"]
                else 10
                for op in models
            )))
            fig = fig_cka_heatmap(mat, labels, save_dir=None)
            _save("fig_cka_heatmap", fig)
        except Exception as e:
            _log(f"  ⚠️  CKA heatmap: {e}", log_fh)

    # Circuit summary table
    circuit_descs = analysis.get("circuit_descriptions", {})
    if circuit_descs:
        try:
            fig = fig_circuit_summary_table(circuit_descs, save_dir=None)
            _save("fig_circuit_table", fig)
        except Exception as e:
            _log(f"  ⚠️  Circuit table: {e}", log_fh)

    _log(f"\n  📊 Generated {len(generated)} figures → {fig_dir}", log_fh)
    return generated

# ---------------------------------------------------------------------------
# Phase 4: Print results summary
# ---------------------------------------------------------------------------

def print_summary(config: dict, results: dict, analysis: dict, log_fh=None):
    _log("\n" + "═" * 70, log_fh)
    _log("  RESULTS SUMMARY", log_fh)
    _log("═" * 70, log_fh)
    _log(f"{'Op':12s}{'Grok Epoch':>18s}{'Test Acc':>12s}{'C1':>8s}{'C2':>8s}", log_fh)
    _log("-" * 60, log_fh)
    for name, cfg_dict in config.items():
        op  = cfg_dict["op"]
        res = results.get(name, {})
        ge  = res.get("mean_grok_epoch")
        sd  = res.get("std_grok_epoch")
        ta  = res.get("runs", [{}])[-1].get("cfg", {})
        ge_str = f"{int(ge)} ± {int(sd or 0)}" if ge else "N/A"
        c1  = get_complexity_score(op)
        c2  = get_complexity_score_v2(op)
        _log(f"{op:12s}{ge_str:>18s}{'—':>12s}{c1:>8.1f}{c2:>8.2f}", log_fh)
    _log("-" * 60, log_fh)
    reg = analysis.get("regression_v1", {})
    if "spearman_r" in reg:
        _log(f"\n  Complexity-Delay Law (v1): Spearman ρ = {reg['spearman_r']:.3f}  "
             f"R² = {reg['r_squared']:.3f}  verdict = {reg['verdict']}", log_fh)
    reg2 = analysis.get("regression_v2", {})
    if "spearman_r" in reg2:
        _log(f"  Complexity-Delay Law (v2): Spearman ρ = {reg2['spearman_r']:.3f}  "
             f"R² = {reg2['r_squared']:.3f}  verdict = {reg2['verdict']}", log_fh)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Reproduce all paper experiments.")
    parser.add_argument("--ckpt_dir",  default="checkpoints")
    parser.add_argument("--fig_dir",   default="paper/figures")
    parser.add_argument("--smoke_test",action="store_true",
                        help="Quick test with tiny configs (CPU, ~10 min)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip experiments that already have checkpoints")
    parser.add_argument("--no_figures", action="store_true",
                        help="Skip figure generation")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    fig_dir  = Path(args.fig_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "reproduce_log.txt"

    config   = SMOKE_CONFIG if args.smoke_test else FULL_CONFIG
    if args.smoke_test:
        for name in config:
            config[name]["overrides"] = SMOKE_MODEL_OVERRIDES

    device   = _detect_device()
    print(f"\n🚀  Reproducing paper experiments")
    print(f"   Mode:    {'smoke test' if args.smoke_test else 'full'}")
    print(f"   Device:  {device}")
    print(f"   Ckpts:   {ckpt_dir}")
    print(f"   Figures: {fig_dir}")
    print(f"   Log:     {log_path}\n")

    errors = []
    with open(log_path, "w") as log_fh:
        _log(f"Reproduce run started — device={device}", log_fh)

        # Phase 1: Train
        results = run_training(config, ckpt_dir, args.skip_existing, device, log_fh)

        # Phase 2: Analyse
        analysis = run_analysis(config, results, ckpt_dir, device, log_fh)

        # Phase 3: Figures
        if not args.no_figures:
            run_figures(config, results, analysis, fig_dir, ckpt_dir, log_fh)

        # Phase 4: Summary
        print_summary(config, results, analysis, log_fh)

        # Phase 5: Auto-generate Table 3
        try:
            from scripts.generate_table3 import main as gen_table3
            sys.argv = ["generate_table3.py",
                        "--ckpt_dir", str(ckpt_dir), "--save"]
            gen_table3()
        except Exception as e:
            _log(f"\n  ⚠️  Table 3 generation failed: {e}", log_fh)

        # Check for errors
        for name, res in results.items():
            if res.get("status") == "error":
                errors.append(f"{name}: {res.get('error','unknown')}")

    if errors:
        print(f"\n❌  {len(errors)} experiment(s) failed:")
        for e in errors:
            print(f"   {e}")
        sys.exit(1)
    else:
        print(f"\n✅  All experiments complete. Log: {log_path}")
        sys.exit(0)

if __name__ == "__main__":
    main()
