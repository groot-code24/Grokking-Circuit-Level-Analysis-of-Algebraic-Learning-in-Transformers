# Grokking Beyond Addition

**Circuit-level analysis of algebraic learning in 1-layer transformers across eight group operations.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/DEEPMIND_READY_COLAB.ipynb)

---

## Overview

This repository replicates and extends the mechanistic interpretability analysis of grokking from [Nanda et al. 2023](https://arxiv.org/abs/2301.05217) to eight algebraic operations spanning abelian fields, composite rings, and non-abelian groups.

The central question: do transformers systematically discover the representation-theoretic structure of algebraic tasks, and does generalisation speed scale predictably with group complexity?

**Summary of findings (free-tier run, `d_model=64`, 2 seeds):**

| Operation | Grok epoch | Circuit type | Key finding |
|-----------|------------|--------------|-------------|
| $(a\times b)\bmod 113$ | ~725 | dlog-then-clock | 3.77× Fourier improvement under dlog re-indexing |
| $(a+b)\bmod 113$ | 1125±275 | Fourier clock | 30.2% top-5 concentration; peaks at k={30,45,20,13} |
| $(a+b)\bmod 100$ | ~1725 | Partial Fourier | Composite-ring partial structure at divisor frequencies |
| $(a-b)\bmod 113$ | ~2200 | Fourier clock | Same circuit family as addition |
| $S_3$, $A_4$ | < budget | Peter-Weyl | Moderate evidence; dominant irreps as predicted |
| $D_5$, $S_4$ | < budget | Peter-Weyl | Weak evidence; larger groups need more model capacity |

CKA cross-operation similarity: all 28 pairwise values ≥ 0.78, suggesting a shared representational substrate across operations.

---

## Quickstart — Google Colab (recommended)

Upload `grokking-research.zip` to Colab and run cells in sequence:

```
0-A  Install dependencies       (once per runtime)
0-C  Upload & extract zip       (first time only)
0-B  Session restore            (every restart)
1    E1: Modular addition
2    E2: Multiplication + dlog analysis
3    E3: Subtraction (control)
4    E4: Ring addition
5    E5–E7: S3, D5, A4
6    E8: S4
7    Complexity-delay law regression
8    Paper figures & results summary
```

**Runtime:** ~4 hours full run on a T4 GPU · ~15 min smoke test · checkpoints resume automatically after disconnection.

---

## Local Installation

```bash
git clone https://github.com/justbytecode/grokking-beyond-addition
cd grokking-beyond-addition
pip install -r requirements.txt
```

Verify setup:

```bash
python scripts/ci_check_datasets.py
python scripts/ci_check_model.py
pytest src/tests/ -q
```

---

## Running Experiments

**Single operation:**
```python
from src.train import train_experiment

results = train_experiment(op="add", p=113, epochs=3000, save_dir="checkpoints/")
```

**Multi-seed run:**
```python
from src.train import multi_seed_experiment

agg = multi_seed_experiment(op="mul", p=113, epochs=3000, seeds=[42, 1, 7])
print(f"Mean grok epoch: {agg['mean_grok_epoch']:.0f} ± {agg['std_grok_epoch']:.0f}")
```

**All eight operations via script:**
```bash
python scripts/reproduce_paper.py --ckpt_dir checkpoints/ --skip_existing
```

**Supported operations:** `add`, `sub`, `mul`, `ring_add`, `s3`, `d5`, `a4`, `s4`

---

## Analysis

**Fourier embedding analysis (abelian ops):**
```python
from src.analysis import fourier_embedding_analysis

fa = fourier_embedding_analysis(model, p=113)
print(f"Top-5 concentration: {fa['concentration']:.1%}")
print(f"Top frequencies: {fa['top_freqs']}")
```

**Discrete-log hypothesis test (multiplication):**
```python
from src.analysis import discrete_log_embedding_analysis

dlog = discrete_log_embedding_analysis(model, p=113)
print(f"Improvement ratio: {dlog['improvement_ratio']:.2f}x")
print(f"Linear probe accuracy: {dlog['dlog_probe_acc_linear']:.1%}")
print(f"Nonlinear probe accuracy: {dlog['dlog_probe_acc_nonlinear']:.1%}")
```

**Peter-Weyl analysis (non-abelian groups):**
```python
from src.analysis import nonabelian_fourier_analysis

naf = nonabelian_fourier_analysis(model, group_name="s4")
print(f"Dominant irrep: {naf['dominant_irrep']}")
for r in sorted(naf['irreps'], key=lambda x: -x['energy']):
    print(f"  {r['name']:12s}  dim={r['dim']}  fraction={r['fraction']:.1%}")
```

**Causal dlog verification:**
```python
from src.analysis import causal_dlog_verification

result = causal_dlog_verification(add_model, p=113)
print(f"Verdict: {result['verdict']}")
print(f"dlog accuracy: {result['dlog_mapped_accuracy']:.1%}  null: {result['null_accuracy']:.1%}")
```

**Cross-operation CKA similarity:**
```python
from src.analysis import cka_matrix

mat, labels = cka_matrix({"add": model_add, "mul": model_mul, "s4": model_s4}, p=6)
```

**Full circuit description:**
```python
from src.analysis import describe_learned_circuit

desc = describe_learned_circuit(model, dataset, p=113, op_symbol="add")
print(desc["evidence_summary"])
```

---

## Model Configuration

Default free-tier vs full configuration:

| Parameter | Free-tier | Full |
|-----------|-----------|------|
| `d_model` | 64 | 128 |
| `d_mlp` | 256 | 512 |
| `n_heads` | 4 | 4 |
| `d_head` | 16 | 32 |
| `epochs` | 3000 | 5000+ |
| Seeds per op | 2 | 3 |

Override via `cfg_overrides`:
```python
train_experiment(
    op="add", p=113, epochs=5000,
    cfg_overrides={"d_model": 128, "d_mlp": 512, "track_representations": True}
)
```

---

## Algebraic Operations

| ID | Task | Group | Irrep structure |
|----|------|-------|-----------------|
| E1 | $(a+b)\bmod p$ | $\mathbb{Z}/p\mathbb{Z}$ (additive) | All 1-D (DFT) |
| E2 | $(a\times b)\bmod p$ | $\mathbb{F}_p^*$ (multiplicative) | All 1-D via dlog reduction |
| E3 | $(a-b)\bmod p$ | $\mathbb{Z}/p\mathbb{Z}$ | Isomorphic to E1 |
| E4 | $(a+b)\bmod 100$ | $\mathbb{Z}/100\mathbb{Z}$ (ring) | Partial DFT at divisors |
| E5 | $S_3$ multiplication | $S_3$, $\|G\|=6$ | dims [1,1,2] |
| E6 | $D_5$ multiplication | $D_5$, $\|G\|=10$ | dims [1,1,2,2] |
| E7 | $A_4$ multiplication | $A_4$, $\|G\|=12$ | dims [1,1,1,3] |
| E8 | $S_4$ multiplication | $S_4$, $\|G\|=24$ | dims [1,1,2,3,3] |

---

## Formal Complexity Scores

Two scores are provided, both derived from character tables alone — not from training outcomes.

**$C_1$ (ordinal ranking):**
$$C_1(\mathcal{A}) = \mathrm{rank}(\mathcal{A}) + \frac{\max_k d_k}{10} + \frac{1}{2}\mathbf{1}[\text{non-abelian}]$$

**$C_2$ (representation-theoretic):**
$$C_2(G) = \log_2(\max_k d_k + 1) + \left(1 - \frac{1}{|\hat{G}|}\right) + \mathbf{1}[\text{non-abelian}]$$

```python
from src.datasets import get_complexity_score, get_complexity_score_v2

for op in ("add", "sub", "mul", "ring_add", "s3", "d5", "a4", "s4"):
    print(f"{op:10s}  C1={get_complexity_score(op):.2f}  C2={get_complexity_score_v2(op):.3f}")
```

---

## Figures

All figures are generated directly from experiment checkpoints:

| Figure | Description |
|--------|-------------|
| `fig1b_grokking_multiseed.png` | Grokking curves (mean ± std across seeds) |
| `fig3_fourier_spectrum.png` | Fourier embedding spectrum |
| `fig4_dlog_analysis_panel.png` | Discrete-log hypothesis test (3-panel) |
| `fig9_delay_errorbar.png` | Complexity-delay law with error bars |
| `fig10_cka_heatmap.png` | Pairwise CKA similarity matrix (8×8) |
| `fig12_repr_vs_grokking.png` | Representation formation score vs test accuracy |
| `fig13_weight_norms.png` | Key weight matrix L2 norm trajectories |
| `fig15_circuit_table.png` | Learned circuit descriptions for all 8 operations |
| `fig16_grokking_cdf.png` | Grokking epoch CDF across seeds |

---

## Repository Structure

```
.
├── src/
│   ├── datasets.py        # Dataset constructors for all 8 operations
│   ├── train.py           # Training loop, checkpointing, multi-seed aggregation
│   ├── analysis.py        # Fourier, dlog, Peter-Weyl, CKA, probing analyses
│   ├── visualise.py       # Publication-quality figure generation
│   └── tests/             # Unit tests for datasets and analysis
├── scripts/
│   ├── reproduce_paper.py # End-to-end reproduction script
│   ├── generate_table3.py # LaTeX Table 3 from checkpoint meta files
│   ├── colab_utils.py     # Google Colab setup utilities
│   └── ci_check_*.py      # CI smoke-test scripts
├── notebooks/
│   └── DEEPMIND_READY_COLAB.ipynb
├── paper/
│   ├── main.tex
│   └── references.bib
├── requirements.txt
└── setup.py
```

---

## Requirements

```
torch>=2.1.0
transformer_lens>=1.14.0,<2.0.0
einops>=0.7.0
numpy>=1.24.0,<2.0.0
scipy>=1.11.0
matplotlib>=3.8.0
plotly>=5.18.0
pandas>=2.1.0
tqdm>=4.66.0
packaging>=21.0
```

Python 3.10+. GPU strongly recommended (T4 or better); CPU training is approximately 20× slower.

---

## Citation

```bibtex
@article{pal2026grokking,
  title   = {Grokking Beyond Addition: Circuit-Level Analysis of Algebraic Learning in Transformers},
  author  = {Pal, Mani},
  year    = {2026},
  url     = {https://github.com/justbytecode/grokking-beyond-addition}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
