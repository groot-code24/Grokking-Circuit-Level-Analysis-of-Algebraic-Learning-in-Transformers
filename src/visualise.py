"""
visualise.py
------------
Publication-quality figures for the paper.

New in this version:
  - fig_grokking_curves_multiseed   : mean +/- std shaded band (multi-seed)
  - fig_dlog_analysis_panel         : side-by-side raw vs dlog Fourier + probe acc
  - fig_cka_heatmap                 : CKA similarity matrix across operations
  - fig_complexity_delay_errorbar   : delay law with error bars across seeds
  - All existing figures unchanged but use consistent styling
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
})

_PALETTE = {
    "purple": "#534AB7",
    "teal":   "#1D9E75",
    "amber":  "#BA7517",
    "coral":  "#D85A30",
    "gray":   "#888780",
    "blue":   "#185FA5",
    "pink":   "#993556",
}

_OP_COLORS = {
    "add":      _PALETTE["purple"],
    "sub":      _PALETTE["teal"],
    "mul":      _PALETTE["amber"],
    "ring_add": _PALETTE["coral"],
    "s3":       _PALETTE["gray"],
    "d5":       _PALETTE["blue"],
    "a4":       _PALETTE["pink"],
    "s4":       "#2E8B57",
}

_OP_LABELS = {
    # it crashes tight_layout() on Colab (no LaTeX install). Use \mathrm{mod}
    # which IS supported by matplotlib mathtext out-of-the-box.
    "add":      r"$(a+b)\,\mathrm{mod}\,p$",
    "sub":      r"$(a-b)\,\mathrm{mod}\,p$",
    "mul":      r"$(a\times b)\,\mathrm{mod}\,p$",
    "ring_add": r"$(a+b)\,\mathrm{mod}\,100$",
    "s3":       r"$S_3$  (|G|=6)",
    "d5":       r"$D_5$  (|G|=10)",
    "a4":       r"$A_4$  (|G|=12)",
    "s4":       r"$S_4$  (|G|=24)",
}

def _savefig(fig: plt.Figure, path: Path, name: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    # (~200 KB PNG vs ~150 KB PDF per figure = ~350 KB saved per figure × 16 figs = ~5 MB).
    # For paper submission, re-enable PDF by changing "png" to ("png", "pdf") below.
    fig.savefig(path / f"{name}.png", bbox_inches="tight", dpi=150)

# ---------------------------------------------------------------------------
# Figure 1 — Single-run grokking curves
# ---------------------------------------------------------------------------

def fig_grokking_curves(
    results: Dict,
    save_dir: str = "paper/figures",
    filename: str = "fig1_grokking_curves",
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    steps      = results["log_steps"]
    op_sym     = results["cfg"]["op"]
    color      = _OP_COLORS.get(op_sym, _PALETTE["purple"])
    op_label   = _OP_LABELS.get(op_sym, op_sym)
    grok_epoch = results.get("grok_epoch")

    ax = axes[0]
    ax.plot(steps, results["train_acc"], color=color, label="Train", linewidth=1.8)
    ax.plot(steps, results["test_acc"],  color=color, label="Test",  linewidth=1.8, linestyle="--")
    if grok_epoch is not None:
        ax.axvline(grok_epoch, color="#E24B4A", linewidth=1.2, linestyle=":", label=f"Grok @ {grok_epoch}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.05, 1.08); ax.set_title(f"Accuracy — {op_label}"); ax.legend(loc="upper left")

    ax = axes[1]
    ax.plot(steps, results["train_loss"], color=color, label="Train", linewidth=1.8)
    ax.plot(steps, results["test_loss"],  color=color, label="Test",  linewidth=1.8, linestyle="--")
    if grok_epoch is not None:
        ax.axvline(grok_epoch, color="#E24B4A", linewidth=1.2, linestyle=":")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"Loss — {op_label}"); ax.legend(loc="upper right")

    fig.tight_layout()
    _savefig(fig, Path(save_dir), filename)
    return fig

# ---------------------------------------------------------------------------
# Figure 1b — Multi-seed grokking curves with shaded std band  (NEW)
# ---------------------------------------------------------------------------

def fig_grokking_curves_multiseed(
    agg: Dict,
    save_dir: str = "paper/figures",
    filename: str = "fig1b_grokking_multiseed",
) -> plt.Figure:
    """
    Plot mean test accuracy +/- 1 std across seeds.

    Parameters
    ----------
    agg : output of train.multi_seed_experiment() or
          analysis.aggregate_multi_seed()
    """
    fig, ax = plt.subplots(figsize=(7, 3.5))
    steps     = np.array(agg["log_steps"])
    mean_tr   = np.array(agg["mean_train_acc"])
    std_tr    = np.array(agg["std_train_acc"])
    mean_te   = np.array(agg["mean_test_acc"])
    std_te    = np.array(agg["std_test_acc"])
    op_sym    = agg.get("op", "add")
    color     = _OP_COLORS.get(op_sym, _PALETTE["purple"])
    op_label  = _OP_LABELS.get(op_sym, op_sym)
    n_seeds   = len(agg.get("seeds", agg.get("all_grok_epochs", [])))
    mean_grok = agg.get("mean_grok_epoch")
    std_grok  = agg.get("std_grok_epoch")

    ax.plot(steps, mean_tr, color=color, label="Train (mean)", linewidth=1.8)
    ax.fill_between(steps, mean_tr - std_tr, mean_tr + std_tr, color=color, alpha=0.15)
    ax.plot(steps, mean_te, color=color, label="Test (mean)",  linewidth=1.8, linestyle="--")
    ax.fill_between(steps, mean_te - std_te, mean_te + std_te, color=color, alpha=0.15)

    if mean_grok is not None:
        lbl = f"Grok @ {mean_grok:.0f}"
        if std_grok is not None:
            lbl += f" ± {std_grok:.0f}"
        ax.axvline(mean_grok, color="#E24B4A", linewidth=1.2, linestyle=":", label=lbl)
        if std_grok is not None and std_grok > 0:
            ax.axvspan(mean_grok - std_grok, mean_grok + std_grok, color="#E24B4A", alpha=0.08)

    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title(f"Grokking ({n_seeds} seeds, mean ± 1 std) — {op_label}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _savefig(fig, Path(save_dir), filename)
    return fig

# ---------------------------------------------------------------------------
# Figure 2 — Multi-operation comparison
# ---------------------------------------------------------------------------

def fig_grokking_comparison(
    all_results: List[Dict],
    save_dir: str = "paper/figures",
    filename: str = "fig2_grokking_comparison",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    for res in all_results:
        op  = res["cfg"]["op"]
        col = _OP_COLORS.get(op, _PALETTE["gray"])
        lbl = _OP_LABELS.get(op, op)
        ax.plot(res["log_steps"], res["test_acc"], color=col, linewidth=1.8, label=lbl)
        gk = res.get("grok_epoch")
        if gk is not None:
            ax.axvline(gk, color=col, linewidth=0.8, linestyle=":", alpha=0.7)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Test accuracy")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("Grokking across algebraic operations")
    ax.legend(loc="upper left", framealpha=0.8)
    fig.tight_layout()
    _savefig(fig, Path(save_dir), filename)
    return fig

# ---------------------------------------------------------------------------
# Figure 3 — Fourier embedding spectrum
# ---------------------------------------------------------------------------

def fig_fourier_spectrum(
    fourier_result: Dict,
    op_sym: Optional[str] = None,
    save_dir: str = "paper/figures",
    filename: str = "fig3_fourier_spectrum",
    *,
    op_name: Optional[str] = None,
) -> plt.Figure:

    if op_sym is None and op_name is not None:
        _display = _OP_LABELS.get(op_name, op_name)
    elif op_sym is not None:
        _display = _OP_LABELS.get(op_sym, op_sym)
    else:
        _display = ""

    freqs = fourier_result["frequencies"]
    norms = fourier_result["fourier_norms"]
    top5  = set(fourier_result["top_freqs"].tolist())
    conc  = fourier_result.get("concentration", None)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    colors = [_PALETTE["purple"] if k in top5 else _PALETTE["gray"] for k in freqs]
    ax.bar(freqs, norms, color=colors, width=0.8, alpha=0.85)
    ax.set_xlabel("Fourier frequency $k$")
    ax.set_ylabel(r"$\|\mathbf{W}_E^\top \mathbf{f}_k\|$")
    title = f"Fourier spectrum — {_display}" if _display else "Fourier spectrum"
    if conc is not None:
        title += f"\n(top-5 concentration = {conc:.2%})"
    ax.set_title(title)
    fig.tight_layout()
    _savefig(fig, Path(save_dir), filename)
    return fig

# ---------------------------------------------------------------------------
# Figure 4 — Discrete-log analysis panel  (NEW — quantitative)
# ---------------------------------------------------------------------------

def fig_dlog_analysis_panel(
    dlog_result: Dict,
    save_dir: str = "paper/figures",
    filename: str = "fig4_dlog_analysis_panel",
) -> plt.Figure:
    """
    3-panel figure:
      (a) Raw Fourier spectrum (no dlog re-indexing)
      (b) Fourier spectrum after dlog re-indexing
      (c) Bar chart: concentration scores + probe accuracy
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    # Left: raw (no dlog)
    fa_raw  = dlog_result.get("fourier_norms_raw", None)
    freqs_r = dlog_result.get("frequencies_raw",   None)
    if fa_raw is not None and freqs_r is not None:
        axes[0].bar(freqs_r, fa_raw, color=_PALETTE["gray"], width=0.8, alpha=0.85)
        conc_raw = dlog_result.get("concentration_raw", 0)
        axes[0].set_title(f"Raw Fourier (conc.={conc_raw:.2%})")
        axes[0].set_xlabel("Frequency $k$")
        axes[0].set_ylabel("Projection norm")
    else:
        axes[0].text(0.5, 0.5, "raw spectrum\nnot available",
                     ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title("Raw Fourier")

    # Middle: dlog re-indexed
    freqs = dlog_result["frequencies"]
    norms = dlog_result["fourier_norms_dlog"]
    top5  = set(dlog_result["top_freqs_dlog"].tolist())
    colors = [_PALETTE["amber"] if k in top5 else _PALETTE["gray"] for k in freqs]
    axes[1].bar(freqs, norms, color=colors, width=0.8, alpha=0.85)
    conc_dlog = dlog_result.get("concentration_dlog", 0)
    g         = dlog_result["primitive_root"]
    axes[1].set_title(f"dlog-reindexed Fourier (conc.={conc_dlog:.2%})\ng={g}")
    axes[1].set_xlabel(r"Frequency $k$ in $\mathbb{Z}/(p{-}1)\mathbb{Z}$")
    axes[1].set_ylabel("Projection norm")

    # Right: metric comparison bar chart
    labels = ["Raw\nconcentration", "dlog\nconcentration", "dlog\nprobe acc."]
    values = [
        dlog_result.get("concentration_raw",  0.0),
        dlog_result.get("concentration_dlog", 0.0),
        dlog_result.get("dlog_probe_acc",     0.0),
    ]
    bar_colors = [_PALETTE["gray"], _PALETTE["amber"], _PALETTE["purple"]]
    bars = axes[2].bar(labels, values, color=bar_colors, alpha=0.85)
    axes[2].set_ylim(0, 1.1)
    axes[2].axhline(0.9, color="black", linewidth=0.7, linestyle="--", label="0.9 threshold")
    axes[2].set_title("Hypothesis test metrics")
    axes[2].set_ylabel("Score (0–1)")
    axes[2].legend(fontsize=8)
    for bar, val in zip(bars, values):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.2f}", ha="center", va="bottom", fontsize=9
        )
    fig.suptitle(
        r"Discrete-log representation hypothesis — $(a\times b)\,\mathrm{mod}\,p$",
        y=1.02, fontsize=12
    )
    fig.tight_layout()
    _savefig(fig, Path(save_dir), filename)
    return fig

# ---------------------------------------------------------------------------
# Old fig4 kept for backward compat
# ---------------------------------------------------------------------------

def fig_dlog_spectrum(
    dlog_result: Dict,
    save_dir: str = "paper/figures",
    filename: str = "fig4_dlog_spectrum",
) -> plt.Figure:
    freqs = dlog_result["frequencies"]
    norms = dlog_result["fourier_norms_dlog"]
    top5  = set(dlog_result["top_freqs_dlog"].tolist())
    g     = dlog_result["primitive_root"]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    colors = [_PALETTE["amber"] if k in top5 else _PALETTE["gray"] for k in freqs]
    ax.bar(freqs, norms, color=colors, width=0.8, alpha=0.85)
    ax.set_xlabel(r"Frequency $k$ in $\mathbb{Z}/(p{-}1)\mathbb{Z}$")
    ax.set_ylabel("Projection norm after dlog re-indexing")
    ax.set_title(rf"Discrete-log Fourier spectrum  [primitive root $g={g}$]")
    fig.tight_layout()
    _savefig(fig, Path(save_dir), filename)
    return fig

# ---------------------------------------------------------------------------
# Figure 5 — Logit attribution
# ---------------------------------------------------------------------------

def fig_logit_attribution(
    attribution: Dict,
    op_sym: str,
    save_dir: str = "paper/figures",
    filename: str = "fig5_logit_attribution",
) -> plt.Figure:
    head_contribs = attribution["head_contributions"]
    mlp_contrib   = attribution["mlp_contribution"]
    embed_contrib = attribution["embed_contribution"]
    is_exact      = attribution.get("is_exact", False)

    n_heads = len(head_contribs)
    labels  = [f"Head {h}" for h in range(n_heads)] + ["MLP", "Embed"]
    values  = list(head_contribs) + [mlp_contrib, embed_contrib]
    colors  = [_PALETTE["purple"]] * n_heads + [_PALETTE["teal"]] + [_PALETTE["gray"]]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(labels, values, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Mean logit contribution (correct answer)")
    title = f"Logit attribution — {_OP_LABELS.get(op_sym, op_sym)}"
    if not is_exact:
        title += "\n(heuristic estimate — TransformerLens not used)"
    ax.set_title(title)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _savefig(fig, Path(save_dir), filename)
    return fig

# ---------------------------------------------------------------------------
# Figure 6 — Activation patching
# ---------------------------------------------------------------------------

def fig_activation_patching(
    importance: np.ndarray,
    op_sym: str,
    save_dir: str = "paper/figures",
    filename: str = "fig6_activation_patching",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar([f"Head {h}" for h in range(len(importance))],
           importance, color=_PALETTE["purple"], alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Mean logit drop when head is patched")
    ax.set_title(f"Causal importance — {_OP_LABELS.get(op_sym, op_sym)}")
    fig.tight_layout()
    _savefig(fig, Path(save_dir), filename)
    return fig

# ---------------------------------------------------------------------------
# Figure 7 — Attention patterns
# ---------------------------------------------------------------------------

def fig_attention_patterns(
    patterns: np.ndarray,
    op_sym: str,
    save_dir: str = "paper/figures",
    filename: str = "fig7_attention_patterns",
) -> plt.Figure:
    n_heads    = patterns.shape[0]
    pos_labels = ["a", "b", "="]

    fig, axes = plt.subplots(1, n_heads, figsize=(2.5 * n_heads, 2.8))
    if n_heads == 1:
        axes = [axes]

    for h, ax in enumerate(axes):
        im = ax.imshow(patterns[h], vmin=0, vmax=1, cmap="Purples", aspect="auto")
        ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(pos_labels); ax.set_yticklabels(pos_labels)
        ax.set_xlabel("Key position")
        if h == 0:
            ax.set_ylabel("Query position")
        ax.set_title(f"Head {h}")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{patterns[h, i, j]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if patterns[h, i, j] > 0.6 else "black")

    fig.suptitle(f"Attention patterns — {_OP_LABELS.get(op_sym, op_sym)}", y=1.02)
    fig.tight_layout()
    _savefig(fig, Path(save_dir), filename)
    return fig

# ---------------------------------------------------------------------------
# Figure 8 — Grokking delay comparison (single seed)
# ---------------------------------------------------------------------------

def fig_grokking_delay_comparison(
    all_results: List[Dict],
    save_dir: str = "paper/figures",
    filename: str = "fig8_grokking_delay",
) -> plt.Figure:
    complexity_order = ["add", "sub", "ring_add", "mul", "s3", "d5", "a4"]
    labels, delays, colors = [], [], []

    for op in complexity_order:
        for res in all_results:
            if res["cfg"]["op"] == op:
                delay = res.get("grok_epoch")
                labels.append(_OP_LABELS.get(op, op))
                delays.append(delay if delay is not None else 0)
                colors.append(_OP_COLORS.get(op, _PALETTE["gray"]))

    fig, ax = plt.subplots(figsize=(9, 3.5))
    bars = ax.bar(labels, delays, color=colors, alpha=0.85)
    ax.set_ylabel("Grokking epoch (test acc >= 99%)")
    ax.set_title("Grokking delay vs. algebraic complexity")
    ax.tick_params(axis="x", labelsize=9)
    for bar, val in zip(bars, delays):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                    str(int(val)), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    _savefig(fig, Path(save_dir), filename)
    return fig

# ---------------------------------------------------------------------------
# Figure 9 — Complexity–delay law with error bars  (NEW)
# ---------------------------------------------------------------------------

def fig_complexity_delay_errorbar(
    agg_results: Dict[str, Dict],
    save_dir: str = "paper/figures",
    filename: str = "fig9_delay_errorbar",
) -> plt.Figure:
    """
    Plot grokking epoch mean +/- std per operation, ordered by complexity.

    Parameters
    ----------
    agg_results : dict mapping op_symbol -> aggregate dict
                  (output of multi_seed_experiment or aggregate_multi_seed)
    """
    complexity_order = ["add", "sub", "ring_add", "mul", "d5", "a4", "s3"]
    ops_present = [op for op in complexity_order if op in agg_results]

    means, stds, labels, colors = [], [], [], []
    for op in ops_present:
        agg = agg_results[op]
        m = agg.get("mean_grok_epoch")
        s = agg.get("std_grok_epoch") or 0.0
        means.append(m if m is not None else 0)
        stds.append(s)
        labels.append(_OP_LABELS.get(op, op))
        colors.append(_OP_COLORS.get(op, _PALETTE["gray"]))

    x = np.arange(len(ops_present))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x, means, color=colors, alpha=0.75, width=0.6)
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black",
                elinewidth=1.5, capsize=5, capthick=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Grokking epoch (mean ± std across seeds)")
    ax.set_title("Complexity–delay law with statistical error bars")

    for xi, (m, s) in enumerate(zip(means, stds)):
        if m > 0:
            label = f"{m:.0f}" + (f"\n±{s:.0f}" if s > 0 else "")
            ax.text(xi, m + s + 30, label, ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    _savefig(fig, Path(save_dir), filename)
    return fig

# ---------------------------------------------------------------------------
# Figure 10 — CKA similarity heatmap  (NEW)
# ---------------------------------------------------------------------------

def fig_cka_heatmap(
    cka_mat: np.ndarray,
    labels: List[str],
    save_dir: str = "paper/figures",
    filename: str = "fig10_cka_heatmap",
) -> plt.Figure:
    """
    Visualise pairwise CKA similarity between embedding representations
    of models trained on different operations.
    """
    n = len(labels)
    fig, ax = plt.subplots(figsize=(0.9 * n + 1.5, 0.9 * n + 1))
    im = ax.imshow(cka_mat, vmin=0, vmax=1, cmap="Purples", aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    display_labels = [_OP_LABELS.get(l, l) for l in labels]
    ax.set_xticklabels(display_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(display_labels, fontsize=9)
    plt.colorbar(im, ax=ax, label="CKA similarity")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cka_mat[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if cka_mat[i, j] > 0.6 else "black")
    ax.set_title("CKA similarity of token embeddings across operations\n"
                 "(1 = identical representations, 0 = orthogonal)")
    fig.tight_layout()
    _savefig(fig, Path(save_dir), filename)
    return fig

# ---------------------------------------------------------------------------
# Figure 11 — Multi-p delay ablation  (NEW)
# ---------------------------------------------------------------------------

def fig_multi_p_delay(
    multi_p_results: Dict[int, Dict],
    op: str = "add",
    save_dir: str = "paper/figures",
    filename: str = "fig11_multi_p_delay",
) -> plt.Figure:
    """
    Show grokking epoch vs modulus p for a fixed operation.
    Demonstrates that the delay law is stable across different primes.
    """
    p_vals = sorted(multi_p_results.keys())
    means  = [multi_p_results[p].get("mean_grok_epoch") or 0 for p in p_vals]
    stds   = [multi_p_results[p].get("std_grok_epoch") or 0  for p in p_vals]
    color  = _OP_COLORS.get(op, _PALETTE["purple"])

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.errorbar(p_vals, means, yerr=stds, fmt="o-", color=color,
                linewidth=1.8, markersize=6, capsize=5, capthick=1.5)
    ax.set_xlabel("Modulus $p$"); ax.set_ylabel("Grokking epoch (mean ± std)")
    ax.set_title(f"Grokking delay vs. modulus — {_OP_LABELS.get(op, op)}")
    ax.set_xticks(p_vals)
    fig.tight_layout()
    _savefig(fig, Path(save_dir), filename)
    return fig

# ===========================================================================
# ===========================================================================

# ---------------------------------------------------------------------------
# Figure 12 — Representation formation vs grokking (leading indicator)
# ---------------------------------------------------------------------------

def fig_representation_vs_grokking(
    formation_scores: List[Tuple[int, float]],
    test_accs: List[float],
    log_steps: List[int],
    grok_epoch: Optional[int] = None,
    transition_epoch: Optional[int] = None,
    op_name: str = "",
    save_dir: str = "paper/figures",
    filename: str = "fig12_repr_vs_grokking",
) -> "plt.Figure":
    """
    Dual-axis plot showing representation formation score alongside test
    accuracy over training. The representation transition typically PRECEDES
    the grokking epoch, serving as a mechanistic leading indicator.

    Left axis  (blue):  formation_score (Fourier concentration or Peter-Weyl
                        dominant irrep fraction)
    Right axis (orange): test accuracy
    Red dashed:  grokking epoch
    Blue dotted: representation transition epoch
    """
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()

    # Formation score
    if formation_scores:
        fs_epochs = [e for e, _ in formation_scores]
        fs_vals   = [s for _, s in formation_scores]
        ax1.plot(fs_epochs, fs_vals, color=_PALETTE["blue"], linewidth=1.8,
                 label="Representation formation score", alpha=0.9)
        ax1.axhline(0.50, color=_PALETTE["blue"], linewidth=0.8,
                    linestyle=":", alpha=0.5, label="Threshold (0.50)")

    # Test accuracy
    if test_accs and log_steps:
        ax2.plot(log_steps, test_accs, color=_PALETTE["amber"], linewidth=1.8,
                 linestyle="--", label="Test accuracy", alpha=0.9)

    # Epoch markers
    if grok_epoch is not None:
        ax1.axvline(grok_epoch, color="#E24B4A", linewidth=1.5,
                    linestyle="--", label=f"Grokking @ {grok_epoch}", zorder=5)
    if transition_epoch is not None:
        ax1.axvline(transition_epoch, color=_PALETTE["blue"], linewidth=1.5,
                    linestyle=":", label=f"Repr. transition @ {transition_epoch}", zorder=5)
        if grok_epoch is not None and grok_epoch > 0:
            lead = grok_epoch - transition_epoch
            ax1.annotate(f"Lead: {lead} ep.",
                         xy=(transition_epoch, 0.05),
                         xytext=(transition_epoch + (grok_epoch - transition_epoch) / 3, 0.12),
                         arrowprops=dict(arrowstyle="->", color="gray"),
                         fontsize=8, color="gray")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Formation score", color=_PALETTE["blue"])
    ax2.set_ylabel("Test accuracy", color=_PALETTE["amber"])
    ax1.set_ylim(-0.05, 1.10)
    ax2.set_ylim(-0.05, 1.10)
    ax1.tick_params(axis="y", labelcolor=_PALETTE["blue"])
    ax2.tick_params(axis="y", labelcolor=_PALETTE["amber"])

    title = f"Representation Formation vs Grokking — {op_name}" if op_name else \
            "Representation Formation vs Grokking"
    ax1.set_title(title)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
               fontsize=8, framealpha=0.8)

    fig.tight_layout()
    if save_dir is not None:
        _savefig(fig, Path(save_dir), filename)
    return fig

# ---------------------------------------------------------------------------
# Figure 13 — Weight norm trajectories
# ---------------------------------------------------------------------------

def fig_weight_norm_trajectory(
    weight_norm_history: List[Dict],
    grok_epoch: Optional[int] = None,
    op_name: str = "",
    save_dir: str = "paper/figures",
    filename: str = "fig13_weight_norms",
) -> "plt.Figure":
    """
    Multi-line plot of key weight matrix L2 norms over training epochs.

    Shows W_E (embedding), W_in (MLP input), W_O (attention output) alongside
    any other logged weight matrices. A vertical line marks the grokking epoch.

    The typical pattern: attention weights (W_Q/K/V/O) stabilise BEFORE
    grokking, while embedding (W_E) and MLP weights continue changing through
    the grokking transition.
    """
    if not weight_norm_history:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.text(0.5, 0.5, "No weight norm history available",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    epochs  = [h["epoch"] for h in weight_norm_history if "epoch" in h]
    all_keys = [k for k in weight_norm_history[0].keys() if k != "epoch"]

    # Colour map: embed=purple, W_O=teal, W_in/W_out=amber, W_Q/K/V=blue tones
    _key_colors = {
        "embed": _PALETTE["purple"],
        "W_O":   _PALETTE["teal"],
        "W_in":  _PALETTE["amber"],
        "W_out": _PALETTE["coral"],
        "W_Q":   _PALETTE["blue"],
        "W_K":   _PALETTE["gray"],
        "W_V":   _PALETTE["pink"],
        "unembed": "#4A7C59",
    }

    fig, ax = plt.subplots(figsize=(9, 4))

    for key in sorted(all_keys):
        vals = [h.get(key, float("nan")) for h in weight_norm_history if "epoch" in h]
        color = next((c for frag, c in _key_colors.items() if frag in key),
                     "#888888")
        # Shorten key name for legend
        short = key.split(".")[-1] if "." in key else key
        ax.plot(epochs, vals, label=short, color=color, linewidth=1.4, alpha=0.85)

    if grok_epoch is not None:
        ax.axvline(grok_epoch, color="#E24B4A", linewidth=1.5,
                   linestyle="--", label=f"Grokking @ {grok_epoch}", zorder=5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 norm")
    title = f"Weight Norm Trajectories — {op_name}" if op_name else \
            "Weight Norm Trajectories"
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.8)
    fig.tight_layout()
    if save_dir is not None:
        _savefig(fig, Path(save_dir), filename)
    return fig

# ---------------------------------------------------------------------------
# Figure 14 — Ablation rank-order (3-panel)
# ---------------------------------------------------------------------------

def fig_ablation_rank_order(
    ablation_results: Dict,
    save_dir: str = "paper/figures",
    filename: str = "fig14_ablation_rank",
) -> "plt.Figure":
    """
    3-panel figure showing Spearman ρ scatter for each ablation condition.

    ablation_results : output of controlled_complexity_ablation()
                       plus an 'op_results_by_condition' key holding the raw data.

    Each panel: scatter complexity_score vs mean_grok_epoch with regression
    line, 95% CI shading, and Spearman ρ annotation.
    """
    try:
        from scipy import stats as _sp_stats
        from src.datasets import get_complexity_score
    except ImportError:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "scipy / datasets not available",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    conditions = ablation_results.get("conditions", [])
    op_by_cond = ablation_results.get("op_results_by_condition", {})
    n_conds    = max(len(conditions), 1)

    fig, axes = plt.subplots(1, n_conds, figsize=(5 * n_conds, 4.5), sharey=False)
    if n_conds == 1:
        axes = [axes]

    for ax, cond in zip(axes, conditions):
        cond_data = op_by_cond.get(cond, {})
        xs, ys, labels = [], [], []
        for op, res in cond_data.items():
            ge = res.get("mean_grok_epoch")
            if ge is None:
                continue
            try:
                xs.append(get_complexity_score(op))
                ys.append(float(ge))
                labels.append(op)
            except ValueError:
                continue

        if len(xs) < 2:
            ax.text(0.5, 0.5, f"Insufficient data\nfor '{cond}'",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(cond)
            continue

        xs_a, ys_a = np.array(xs), np.array(ys)
        sp_r = ablation_results.get("spearman_by_cond", {}).get(cond, (float("nan"),))[0]

        # Scatter
        for x_i, y_i, lbl in zip(xs, ys, labels):
            color = _OP_COLORS.get(lbl, _PALETTE["gray"])
            ax.scatter(x_i, y_i, color=color, s=70, zorder=3)
            ax.annotate(lbl, (x_i, y_i), textcoords="offset points",
                        xytext=(4, 3), fontsize=7)

        # Regression line + CI band
        if len(xs) >= 3:
            slope, intercept, *_ = _sp_stats.linregress(xs_a, ys_a)
            x_fit = np.linspace(xs_a.min(), xs_a.max(), 100)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, color=_PALETTE["purple"], linewidth=1.4)

            # Bootstrap CI band
            boots = []
            rng = np.random.default_rng(0)
            for _ in range(500):
                idx = rng.choice(len(xs), size=len(xs), replace=True)
                s_b, i_b, *_ = _sp_stats.linregress(xs_a[idx], ys_a[idx])
                boots.append(s_b * x_fit + i_b)
            lo = np.percentile(boots, 2.5, axis=0)
            hi = np.percentile(boots, 97.5, axis=0)
            ax.fill_between(x_fit, lo, hi, alpha=0.15, color=_PALETTE["purple"])

        ax.set_xlabel("Complexity score C(G)")
        ax.set_ylabel("Mean grokking epoch")
        ax.set_title(f"{cond}\nSpearman ρ = {sp_r:.2f}")

    fig.suptitle("Complexity-Delay Law: Controlled Ablations", fontsize=12, y=1.02)
    fig.tight_layout()
    if save_dir is not None:
        _savefig(fig, Path(save_dir), filename)
    return fig

# ---------------------------------------------------------------------------
# Figure 15 — Circuit summary visual table
# ---------------------------------------------------------------------------

def fig_circuit_summary_table(
    circuit_descriptions: Dict[str, Dict],
    save_dir: str = "paper/figures",
    filename: str = "fig15_circuit_table",
) -> "plt.Figure":
    """
    Matplotlib table showing describe_learned_circuit() results for all ops.

    Rows: 8 operations (colour-coded: blue=abelian, red=non-abelian)
    Columns: Repr. Type | Key Freqs / Dominant Irrep | MLP Role | Evidence
    """
    rows      = []
    row_colors = []
    ABELIAN_OPS = {"add", "sub", "mul", "ring_add"}

    for op in ["add", "sub", "mul", "ring_add", "s3", "d5", "a4", "s4"]:
        if op not in circuit_descriptions:
            continue
        cd = circuit_descriptions[op]
        freqs_or_irrep = (
            ", ".join(str(f) for f in cd.get("key_frequencies", [])[:4])
            if cd.get("key_frequencies")
            else cd.get("dominant_irrep", "—")
        )
        rows.append([
            _OP_LABELS.get(op, op),
            cd.get("representation_type", "—").replace("_", " "),
            freqs_or_irrep,
            cd.get("mlp_role", "—").replace("_", " "),
            cd.get("evidence_strength", "—"),
        ])
        row_colors.append(
            ["#D6E4F7"] * 5 if op in ABELIAN_OPS else ["#FBDDE0"] * 5
        )

    if not rows:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No circuit descriptions available",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    col_labels = ["Operation", "Repr. Type", "Key Freqs / Irrep", "MLP Role", "Evidence"]
    fig, ax = plt.subplots(figsize=(13, 0.55 * len(rows) + 1.5))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.5)

    # Apply row colours
    for row_idx, colors in enumerate(row_colors, start=1):
        for col_idx, c in enumerate(colors):
            tbl[row_idx, col_idx].set_facecolor(c)

    # Header style
    for col_idx in range(len(col_labels)):
        tbl[0, col_idx].set_facecolor("#4A4A8A")
        tbl[0, col_idx].set_text_props(color="white", fontweight="bold")

    ax.set_title("Learned Circuit Descriptions by Operation\n"
                 "(Blue = abelian, Red = non-abelian)",
                 fontsize=11, pad=12)
    fig.tight_layout()
    if save_dir is not None:
        _savefig(fig, Path(save_dir), filename)
    return fig

# ---------------------------------------------------------------------------
# Figure 16 — Grokking delay CDF across seeds and operations
# ---------------------------------------------------------------------------

def fig_multi_seed_grokking_delay_cdf(
    op_agg_results: Dict[str, Dict],
    save_dir: str = "paper/figures",
    filename: str = "fig16_grokking_cdf",
) -> "plt.Figure":
    """
    Cumulative distribution function of grokking epochs across seeds, one
    line per operation. Shows spread and variance more naturally than
    mean ± std bars, and allows direct visual comparison of delay distributions.

    op_agg_results : dict mapping op_symbol -> aggregate result dict
                     (from multi_seed_experiment or smart_train)
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    max_epoch = 0

    plotted = False
    for op, agg in op_agg_results.items():
        grok_epochs = agg.get("all_grok_epochs", [])
        if not grok_epochs:
            continue
        grok_epochs = sorted(grok_epochs)
        max_epoch   = max(max_epoch, max(grok_epochs))
        n           = len(grok_epochs)
        color       = _OP_COLORS.get(op, _PALETTE["gray"])
        label       = _OP_LABELS.get(op, op)

        # Build CDF: at each grokked epoch, fraction of seeds that have grokked
        x_cdf = [0] + grok_epochs + [grok_epochs[-1] * 1.05]
        y_cdf = [0.0] + [(i + 1) / n for i in range(n)] + [(n) / n]
        ax.step(x_cdf, y_cdf, where="post", color=color, linewidth=1.8,
                label=label)
        ax.scatter(grok_epochs, [(i + 1) / n for i in range(n)],
                   color=color, s=30, zorder=3)
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "No grokking data available",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    ax.set_xlabel("Grokking epoch")
    ax.set_ylabel("Fraction of seeds grokked")
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlim(left=0)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title("Grokking Epoch CDF across Seeds and Operations")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    if save_dir is not None:
        _savefig(fig, Path(save_dir), filename)
    return fig
