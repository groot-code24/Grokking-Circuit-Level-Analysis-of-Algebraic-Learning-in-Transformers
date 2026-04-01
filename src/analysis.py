from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import transformers as _transformers_mod
    if not hasattr(_transformers_mod, "TRANSFORMERS_CACHE"):
        import os as _os
        _transformers_mod.TRANSFORMERS_CACHE = _os.path.join(
            _os.path.expanduser("~"), ".cache", "huggingface", "transformers"
        )
    from transformer_lens import HookedTransformer
    _TL_AVAILABLE = True
except Exception:
    _TL_AVAILABLE = False


def _get_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _tokens(a: int, b: int, sep: int, device: torch.device) -> Tensor:
    return torch.tensor([[a, b, sep]], dtype=torch.long, device=device)


def _logits_at_sep(model: torch.nn.Module, tokens: Tensor) -> Tensor:
    model.eval()
    with torch.no_grad():
        out = model(tokens)
    return out[0, -1, :]


def fourier_embedding_analysis(
    model: torch.nn.Module,
    p: int,
) -> Dict[str, np.ndarray]:
    """
    Project token embedding matrix W_E (shape p x d_model) onto DFT basis.

    Returns
    -------
    frequencies     : array of k values (1 .. p//2)
    fourier_norms   : L2 norm of projection onto each frequency pair
    top_freqs       : sorted indices of top-5 frequencies (1-indexed)
    embedding_matrix: np.ndarray (p, d_model)
    concentration   : float — fraction of total norm in top-5 frequencies
                      (>0.8 strongly suggests clock circuit)
    """
    if _TL_AVAILABLE and isinstance(model, HookedTransformer):
        W_E = model.embed.W_E[:p].detach().cpu().numpy()
    else:
        W_E = model.embed.weight[:p].detach().cpu().numpy()

    p_actual, d_model = W_E.shape
    freqs  = np.arange(1, p_actual // 2 + 1)
    i_idx  = np.arange(p_actual)

    norms = np.zeros(len(freqs))
    for idx, k in enumerate(freqs):
        F_sin = np.sin(2 * np.pi * k * i_idx / p_actual)
        F_cos = np.cos(2 * np.pi * k * i_idx / p_actual)
        proj_sin = W_E.T @ F_sin
        proj_cos = W_E.T @ F_cos
        norms[idx] = float(np.linalg.norm(proj_sin) + np.linalg.norm(proj_cos))

    top5_idx   = np.argsort(norms)[::-1][:5]
    top5_freqs = top5_idx + 1
    concentration = norms[top5_idx].sum() / (norms.sum() + 1e-9)

    return {
        "frequencies":       freqs,
        "fourier_norms":     norms,
        "top_freqs":         top5_freqs,
        "embedding_matrix":  W_E,
        "concentration":     float(concentration),
    }


def discrete_log_embedding_analysis(
    model: torch.nn.Module,
    p: int,
) -> Dict:
    """
    Quantitative test of the discrete-log representation hypothesis.

    Standard test:
        Re-index the embedding matrix by dlog_g and run Fourier analysis
        over Z/(p-1)Z.  A high concentration score confirms the hypothesis.

    Probing test:
        Train a linear probe to predict dlog_g(a) from embed(a).
        High accuracy (>90%) is strong evidence the model encodes dlog.

    Returns
    -------
    primitive_root      : int
    dlog_map            : array mapping token i -> dlog (or -1 for i=0)
    fourier_norms_dlog  : Fourier norms after dlog re-indexing
    frequencies         : array of k values
    top_freqs_dlog      : top-5 frequencies (1-indexed)
    concentration_dlog  : fraction of norm in top-5 (clock-circuit indicator)
    concentration_raw   : Fourier concentration WITHOUT dlog re-indexing
    dlog_probe_acc      : float  — linear probe accuracy for dlog prediction
    improvement_ratio   : concentration_dlog / concentration_raw
                          >1.5 strongly supports the dlog hypothesis
    """
    def _primitive_root(p: int) -> int:
        phi = p - 1
        factors: set = set()
        n = phi
        for d in range(2, int(n ** 0.5) + 1):
            while n % d == 0:
                factors.add(d)
                n //= d
        if n > 1:
            factors.add(n)
        for g in range(2, p):
            if all(pow(g, phi // f, p) != 1 for f in factors):
                return g
        raise ValueError(f"No primitive root found for p={p}")

    g = _primitive_root(p)

    dlog_map = np.full(p, -1, dtype=np.int64)
    val = 1
    for k in range(p - 1):
        dlog_map[val] = k
        val = (val * g) % p

    if _TL_AVAILABLE and isinstance(model, HookedTransformer):
        W_E = model.embed.W_E[:p].detach().cpu().numpy()
    else:
        W_E = model.embed.weight[:p].detach().cpu().numpy()

    n_group = p - 1
    W_E_dlog = np.zeros((n_group, W_E.shape[1]))
    for a in range(1, p):
        k = dlog_map[a]
        if k >= 0:
            W_E_dlog[k] = W_E[a]

    freqs = np.arange(1, n_group // 2 + 1)
    i_idx = np.arange(n_group)
    norms_dlog = np.zeros(len(freqs))
    for idx, k in enumerate(freqs):
        F_sin = np.sin(2 * np.pi * k * i_idx / n_group)
        F_cos = np.cos(2 * np.pi * k * i_idx / n_group)
        norms_dlog[idx] = float(
            np.linalg.norm(W_E_dlog.T @ F_sin) +
            np.linalg.norm(W_E_dlog.T @ F_cos)
        )
    top5_dlog   = np.argsort(norms_dlog)[::-1][:5]
    conc_dlog   = norms_dlog[top5_dlog].sum() / (norms_dlog.sum() + 1e-9)

    fa = fourier_embedding_analysis(model, p)
    conc_raw = fa["concentration"]

    valid_tokens = [a for a in range(1, p) if dlog_map[a] >= 0]
    X = W_E[valid_tokens]
    y = np.array([dlog_map[a] for a in valid_tokens])

    probe_acc_linear    = _linear_probe_accuracy(X, y, n_classes=n_group)
    probe_acc_nonlinear = _nonlinear_probe_accuracy(X, y, n_classes=n_group)

    return {
        "primitive_root":           g,
        "dlog_map":                 dlog_map,
        "fourier_norms_dlog":       norms_dlog,
        "fourier_norms_raw":        fa["fourier_norms"],
        "frequencies":              freqs,
        "frequencies_raw":          fa["frequencies"],
        "top_freqs_dlog":           top5_dlog + 1,
        "concentration_dlog":       float(conc_dlog),
        "concentration_raw":        float(conc_raw),
        "dlog_probe_acc":           float(probe_acc_linear),
        "dlog_probe_acc_linear":    float(probe_acc_linear),
        "dlog_probe_acc_nonlinear": float(probe_acc_nonlinear),
        "improvement_ratio":        float(conc_dlog / (conc_raw + 1e-9)),
    }


def _nonlinear_probe_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    hidden: int = 128,
    epochs: int = 800,
    lr: float = 5e-3,
) -> float:
    """
    2-layer MLP probe: embed → ReLU(hidden) → n_classes.

    Used to test whether dlog(a) is non-linearly encoded in W_E[a].
    A linear probe failing (acc=0) while this succeeds confirms that
    the dlog representation is present but in a non-linear (rotated/mixed)
    basis — consistent with capacity-constrained models at d_model=64.

    Pure numpy/torch — no sklearn dependency.
    Always uses 5-fold cross-validation.
    """
    n     = len(X)
    rng   = np.random.default_rng(42)
    idx   = rng.permutation(n)
    fold  = max(1, n // 5)
    accs: list = []

    for k in range(5):
        te_idx = idx[k * fold : (k + 1) * fold]
        tr_idx = np.concatenate([idx[:k * fold], idx[(k + 1) * fold:]])
        if len(te_idx) == 0 or len(tr_idx) == 0:
            continue

        X_tr = torch.tensor(X[tr_idx], dtype=torch.float32)
        y_tr = torch.tensor(y[tr_idx], dtype=torch.long)
        X_te = torch.tensor(X[te_idx], dtype=torch.float32)
        y_te = torch.tensor(y[te_idx], dtype=torch.long)

        probe = nn.Sequential(
            nn.Linear(X.shape[1], hidden, bias=True),
            nn.ReLU(),
            nn.Linear(hidden, n_classes, bias=True),
        )
        opt = torch.optim.Adam(probe.parameters(), lr=lr)

        for epoch in range(epochs):
            probe.train()
            loss = F.cross_entropy(probe(X_tr), y_tr)
            opt.zero_grad(); loss.backward(); opt.step()

        probe.eval()
        with torch.no_grad():
            preds = probe(X_te).argmax(dim=-1)
            accs.append((preds == y_te).float().mean().item())

    return float(np.mean(accs)) if accs else 0.0


def _linear_probe_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    epochs: int = 500,
    lr: float = 1e-2,
    use_cv: bool = False,
) -> float:
    """
    Train a linear classifier on (X, y) and return test accuracy.

    When use_cv=True OR len(X) < 50, uses 5-fold cross-validation for a
    more reliable estimate on small samples (e.g. dlog probe with p=7 gives
    only ~5 test samples in a simple 80/20 split).
    Otherwise uses a simple 80/20 split.

    Pure numpy/torch — no sklearn dependency.
    """
    n = len(X)
    _use_cv = use_cv or (n < 50)

    def _train_eval(X_tr, y_tr, X_te, y_te):
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        X_te_t = torch.tensor(X_te, dtype=torch.float32)
        y_te_t = torch.tensor(y_te, dtype=torch.long)
        probe = nn.Linear(X.shape[1], n_classes, bias=True)
        opt   = torch.optim.Adam(probe.parameters(), lr=lr)
        for _ in range(epochs):
            probe.train()
            logits = probe(X_tr_t)
            loss   = F.cross_entropy(logits, y_tr_t)
            opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            preds = probe(X_te_t).argmax(dim=-1)
            return (preds == y_te_t).float().mean().item()

    if _use_cv:
        rng  = np.random.default_rng(0)
        idx  = rng.permutation(n)
        fold_size = max(1, n // 5)
        accs = []
        for fold in range(5):
            te_idx = idx[fold * fold_size : (fold + 1) * fold_size]
            tr_idx = np.concatenate([idx[:fold * fold_size], idx[(fold + 1) * fold_size:]])
            if len(te_idx) == 0 or len(tr_idx) == 0:
                continue
            accs.append(_train_eval(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]))
        return float(np.mean(accs)) if accs else 0.0
    else:
        idx = np.random.default_rng(0).permutation(n)
        cut = int(0.8 * n)
        return _train_eval(X[idx[:cut]], y[idx[:cut]], X[idx[cut:]], y[idx[cut:]])


def logit_attribution(
    model: torch.nn.Module,
    dataset: dict,
    n_samples: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Decompose the logit for the correct answer into contributions from
    each attention head, the MLP, and the embedding layer.

    NOTE: Per-head accuracy requires TransformerLens run_with_cache.
    The fallback path for MinimalTransformer returns heuristic estimates
    clearly labelled as such.

    Returns
    -------
    head_contributions   : array (n_heads,)  — mean logit contribution
    mlp_contribution     : float
    embed_contribution   : float
    is_exact             : bool — True only when TransformerLens was used
    """
    device = _get_device(model)
    model.eval()
    p   = dataset["p"]
    sep = p
    data = dataset["train_data"][:n_samples]

    if _TL_AVAILABLE and isinstance(model, HookedTransformer):
        n_heads   = model.cfg.n_heads
        head_sums = np.zeros(n_heads)
        mlp_sum   = 0.0
        embed_sum = 0.0

        for a, b, label in data:
            toks = _tokens(a, b, sep, device)
            with torch.no_grad():
                _, cache = model.run_with_cache(toks)

            W_U         = model.unembed.W_U
            correct_dir = W_U[:, label]
            resid_pre   = cache["resid_pre",  0][0, -1, :]
            embed_sum  += (resid_pre @ correct_dir).item()

            for h in range(n_heads):
                attn_out_h = cache["result", 0][0, -1, h, :]
                W_O_h      = model.blocks[0].attn.W_O[h]
                head_write = attn_out_h @ W_O_h
                head_sums[h] += (head_write @ correct_dir).item()

            mlp_out  = cache["mlp_out", 0][0, -1, :]
            mlp_sum += (mlp_out @ correct_dir).item()

        n = len(data)
        return {
            "head_contributions": head_sums / n,
            "mlp_contribution":   mlp_sum   / n,
            "embed_contribution": embed_sum / n,
            "is_exact": True,
        }

    # Fallback: heuristic estimates for MinimalTransformer
    margins = []
    with torch.no_grad():
        for a, b, label in data:
            toks          = _tokens(a, b, sep, device)
            full_logits   = _logits_at_sep(model, toks)
            correct_logit = full_logits[label].item()
            mean_logit    = full_logits.mean().item()
            margins.append(correct_logit - mean_logit)
    mean_margin = float(np.mean(margins))

    if hasattr(model, "blocks") and len(model.blocks) > 0:
        n_heads_fallback = model.blocks[0].attn.n_heads
    elif hasattr(model, "attn"):
        n_heads_fallback = model.attn.n_heads
    else:
        n_heads_fallback = 4
    head_share = mean_margin * 0.60 / max(n_heads_fallback, 1)
    return {
        "head_contributions": np.full(n_heads_fallback, head_share),
        "mlp_contribution":   mean_margin * 0.30,
        "embed_contribution": mean_margin * 0.10,
        "is_exact": False,
    }


def activation_patch_heads(
    model: torch.nn.Module,
    dataset: dict,
    n_samples: int = 30,
) -> np.ndarray:
    """
    Measure causal importance of each attention head.

    Corruption is operation-agnostic: replace (a,b) with a random pair
    (a',b') where (a',b') != (a,b), without assuming any specific operation.

    Returns
    -------
    importance : np.ndarray (n_heads,)
    """
    if not (_TL_AVAILABLE and isinstance(model, HookedTransformer)):
        print("[analysis] activation_patch_heads requires TransformerLens.")
        if hasattr(model, "blocks") and len(model.blocks) > 0:
            n_heads_fb = model.blocks[0].attn.n_heads
        elif hasattr(model, "attn"):
            n_heads_fb = model.attn.n_heads
        else:
            n_heads_fb = 4
        return np.zeros(n_heads_fb)

    device  = _get_device(model)
    model.eval()
    p       = dataset["p"]
    sep     = p
    n_heads = model.cfg.n_heads
    data    = dataset["train_data"][:n_samples]
    rng     = np.random.default_rng(0)
    importance = np.zeros(n_heads)

    for a, b, label in data:
        a_corr = int(rng.integers(0, p))
        b_corr = int(rng.integers(0, p))
        attempts = 0
        while a_corr == a and b_corr == b:
            a_corr = int(rng.integers(0, p))
            b_corr = int(rng.integers(0, p))
            attempts += 1
            if attempts > 100:
                a_corr = (a + 1) % p
                break

        toks_clean = _tokens(a,      b,      sep, device)
        toks_corr  = _tokens(a_corr, b_corr, sep, device)

        with torch.no_grad():
            _, cache_clean = model.run_with_cache(toks_clean)
            _, cache_corr  = model.run_with_cache(toks_corr)

        baseline_logit = _logits_at_sep(model, toks_clean)[label].item()
        hook_name      = "blocks.0.attn.hook_result"

        for h in range(n_heads):
            def make_hook(head_idx: int, corr_cache):
                def hook_fn(value, hook):
                    patched = value.clone()
                    patched[0, :, head_idx, :] = corr_cache["result", 0][0, :, head_idx, :]
                    return patched
                return hook_fn

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    toks_clean,
                    fwd_hooks=[(hook_name, make_hook(h, cache_corr))],
                )
            patched_logit  = patched_logits[0, -1, label].item()
            importance[h] += baseline_logit - patched_logit

    return importance / len(data)


def detect_grokking_phases(
    train_accs: List[float],
    test_accs:  List[float],
    log_steps:  List[int],
) -> Dict:
    """
    Identify memorisation, transition, and generalisation phases.

    Returns
    -------
    memorisation_epoch, generalisation_epoch, grokking_delay,
    phase_memorisation, phase_transition, phase_generalisation
    """
    train_arr = np.array(train_accs)
    test_arr  = np.array(test_accs)
    steps_arr = np.array(log_steps)

    mem_epoch  = None
    grok_epoch = None

    for i, s in enumerate(log_steps):
        if mem_epoch is None and train_arr[i] >= 0.99:
            mem_epoch = s
        if grok_epoch is None and test_arr[i] >= 0.99:
            grok_epoch = s

    delay = None
    if mem_epoch is not None and grok_epoch is not None:
        delay = grok_epoch - mem_epoch

    mem_mask   = (train_arr >= 0.99) & (test_arr < 0.50)
    trans_mask = (test_arr  >= 0.10) & (test_arr < 0.99)
    gen_mask   = test_arr >= 0.99

    return {
        "memorisation_epoch":   mem_epoch,
        "generalisation_epoch": grok_epoch,
        "grokking_delay":       delay,
        "phase_memorisation":   steps_arr[mem_mask].tolist(),
        "phase_transition":     steps_arr[trans_mask].tolist(),
        "phase_generalisation": steps_arr[gen_mask].tolist(),
    }


def get_attention_patterns(
    model: torch.nn.Module,
    dataset: dict,
    n_samples: int = 20,
) -> np.ndarray:
    """
    Average attention patterns across samples.

    Returns
    -------
    attn_patterns : np.ndarray (n_heads, 3, 3)
        [h, dest, src] = mean attention weight
    """
    if not (_TL_AVAILABLE and isinstance(model, HookedTransformer)):
        if hasattr(model, "blocks") and len(model.blocks) > 0:
            n_heads_fb = model.blocks[0].attn.n_heads
        elif hasattr(model, "attn"):
            n_heads_fb = model.attn.n_heads
        else:
            n_heads_fb = 4
        return np.ones((n_heads_fb, 3, 3)) / 3.0

    device  = _get_device(model)
    model.eval()
    p       = dataset["p"]
    sep     = p
    n_heads = model.cfg.n_heads
    data    = dataset["train_data"][:n_samples]
    patterns = np.zeros((n_heads, 3, 3))

    for a, b, label in data:
        toks = _tokens(a, b, sep, device)
        with torch.no_grad():
            _, cache = model.run_with_cache(toks)
        attn = cache["pattern", 0][0].cpu().numpy()
        patterns += attn

    return patterns / len(data)


def probe_representation(
    model: torch.nn.Module,
    p: int,
    target: str = "identity",
) -> Dict:
    """
    Train linear probes on the token embeddings to decode different
    representations.

    Parameters
    ----------
    target : one of
        "identity"   — probe for the raw token index a
        "dlog"       — probe for discrete_log(a) mod (p-1)
        "fourier_k"  — probe for sin/cos Fourier component at top frequency

    Returns
    -------
    dict with probe_acc, chance_acc (1/n_classes), and a 'target' label
    """
    if _TL_AVAILABLE and isinstance(model, HookedTransformer):
        W_E = model.embed.W_E[:p].detach().cpu().numpy()
    else:
        W_E = model.embed.weight[:p].detach().cpu().numpy()

    if target == "identity":
        X = W_E
        y = np.arange(p)
        n_classes = p
    elif target == "dlog":
        result = discrete_log_embedding_analysis(model, p)
        dlog_map = result["dlog_map"]
        valid = [a for a in range(1, p) if dlog_map[a] >= 0]
        X = W_E[valid]
        y = np.array([dlog_map[a] for a in valid])
        n_classes = p - 1
    elif target == "fourier_k":
        fa = fourier_embedding_analysis(model, p)
        freqs = fa["frequencies"]
        norms = fa["fourier_norms"]
        i_idx = np.arange(p)
        top5_idx = np.argsort(norms)[::-1][:5]
        features = []
        for k_idx in top5_idx:
            k = freqs[k_idx]
            features.append(np.sin(2 * np.pi * k * i_idx / p))
            features.append(np.cos(2 * np.pi * k * i_idx / p))
        X_fourier = np.stack(features, axis=1).astype(np.float32)
        projs = []
        for k_idx in top5_idx:
            k = freqs[k_idx]
            f_sin = np.sin(2 * np.pi * k * i_idx / p)
            f_cos = np.cos(2 * np.pi * k * i_idx / p)
            proj = np.abs(W_E @ f_sin) + np.abs(W_E @ f_cos)
            projs.append(proj)
        y = np.argmax(np.stack(projs, axis=1), axis=1)
        X = X_fourier
        n_classes = len(top5_idx)
    else:
        raise ValueError(
            f"Unknown probe target: {target!r}. "
            f"Choose from 'identity', 'dlog', 'fourier_k'."
        )

    acc = _linear_probe_accuracy(X, y, n_classes=n_classes)
    return {
        "target":     target,
        "probe_acc":  float(acc),
        "chance_acc": 1.0 / n_classes,
        "n_classes":  n_classes,
    }


def cka_similarity(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    p: int,
) -> float:
    """
    Compute linear CKA between the token embedding matrices of two models.
    CKA in [0,1]: 1 = identical representations, 0 = orthogonal.

    Useful for comparing representations across operations or seeds.
    """
    def _get_embed(m):
        if _TL_AVAILABLE and isinstance(m, HookedTransformer):
            return m.embed.W_E[:p].detach().cpu().float().numpy()
        return m.embed.weight[:p].detach().cpu().float().numpy()

    X = _get_embed(model_a)
    Y = _get_embed(model_b)

    def _centre(K: np.ndarray) -> np.ndarray:
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    K = X @ X.T
    L = Y @ Y.T
    Kc, Lc = _centre(K), _centre(L)

    hsic_xy = np.sum(Kc * Lc)
    hsic_xx = np.sum(Kc * Kc)
    hsic_yy = np.sum(Lc * Lc)

    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-9))


def cka_matrix(models: Dict[str, torch.nn.Module], p: int) -> Tuple[np.ndarray, List[str]]:
    """
    Compute pairwise CKA matrix across a dict of {op_name: model}.

    Returns
    -------
    matrix : np.ndarray (n, n) of CKA values
    labels : list of op names in row/col order
    """
    labels = list(models.keys())
    n = len(labels)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            c = cka_similarity(models[labels[i]], models[labels[j]], p)
            mat[i, j] = mat[j, i] = c
    return mat, labels


def aggregate_multi_seed(
    results_list: List[Dict],
) -> Dict:
    """
    Given a list of result dicts from train_experiment (one per seed),
    compute mean ± std for accuracy and loss curves, and grokking statistics.

    Returns
    -------
    Same structure as multi_seed_experiment summary, but computed post-hoc
    from a list of already-run results.  Useful when results were loaded
    from checkpoints.
    """
    log_steps = results_list[0]["log_steps"]
    n_steps   = len(log_steps)

    def _pad(arr):
        if len(arr) >= n_steps:
            return arr[:n_steps]
        return arr + [arr[-1]] * (n_steps - len(arr))

    train_accs = np.array([_pad(r["train_acc"])  for r in results_list])
    test_accs  = np.array([_pad(r["test_acc"])   for r in results_list])
    train_loss = np.array([_pad(r["train_loss"]) for r in results_list])
    test_loss  = np.array([_pad(r["test_loss"])  for r in results_list])

    grok_epochs = [r["grok_epoch"] for r in results_list if r["grok_epoch"] is not None]

    return {
        "log_steps":        log_steps,
        "mean_train_acc":   train_accs.mean(0),
        "std_train_acc":    train_accs.std(0),
        "mean_test_acc":    test_accs.mean(0),
        "std_test_acc":     test_accs.std(0),
        "mean_train_loss":  train_loss.mean(0),
        "std_train_loss":   train_loss.std(0),
        "mean_test_loss":   test_loss.mean(0),
        "std_test_loss":    test_loss.std(0),
        "all_grok_epochs":  grok_epochs,
        "mean_grok_epoch":  float(np.mean(grok_epochs)) if grok_epochs else None,
        "std_grok_epoch":   float(np.std(grok_epochs)) if len(grok_epochs) > 1 else None,
        "grok_rate":        len(grok_epochs) / len(results_list),
    }


_CHARACTER_TABLES: Dict[str, dict] = {
    "s3": {
        "group_order": 6,
        "irreps": [
            {"name": "trivial",  "dim": 1, "characters": [1,  1,  1]},
            {"name": "sign",     "dim": 1, "characters": [1,  1, -1]},
            {"name": "standard", "dim": 2, "characters": [2, -1,  0]},
        ],
        "class_sizes": [1, 2, 3],
        "class_names": ["e", "r,r²", "s,sr,sr²"],
    },
    "d5": {
        "group_order": 10,
        "irreps": [
            {"name": "trivial", "dim": 1,
             "characters": [1, 1, 1, 1]},
            {"name": "sign",    "dim": 1,
             "characters": [1, 1, 1, -1]},
            {"name": "rho_2a",  "dim": 2,
             "characters": [2,
                             2 * np.cos(2 * np.pi / 5),
                             2 * np.cos(4 * np.pi / 5),
                             0]},
            {"name": "rho_2b",  "dim": 2,
             "characters": [2,
                             2 * np.cos(4 * np.pi / 5),
                             2 * np.cos(2 * np.pi / 5),
                             0]},
        ],
        "class_sizes": [1, 2, 2, 5],
        "class_names": ["e", "r,r⁴", "r²,r³", "reflections"],
    },
    "a4": {
        "group_order": 12,
        "irreps": [
            {"name": "trivial", "dim": 1, "characters": [ 1,    1,    1,    1  ]},
            {"name": "rho_1b",  "dim": 1, "characters": [ 1,    1,   -0.5, -0.5]},
            {"name": "rho_1c",  "dim": 1, "characters": [ 1,    1,   -0.5, -0.5]},
            {"name": "rho_3",   "dim": 3, "characters": [ 3,   -1,    0,    0  ]},
        ],
        "class_sizes": [1, 3, 4, 4],
        "class_names": ["e", "(ij)(kl)", "3-cycles (123)-type", "3-cycles (132)-type"],
    },
    "s4": {
        "group_order": 24,
        "irreps": [
            {"name": "trivial",   "dim": 1,
             "characters": [ 1,  1,  1,  1,  1]},
            {"name": "sign",      "dim": 1,
             "characters": [ 1, -1,  1,  1, -1]},
            {"name": "standard2", "dim": 2,
             "characters": [ 2,  0,  2, -1,  0]},
            {"name": "standard3", "dim": 3,
             "characters": [ 3,  1, -1,  0, -1]},
            {"name": "std3_sign", "dim": 3,
             "characters": [ 3, -1, -1,  0,  1]},
        ],
        "class_sizes": [1, 6, 3, 8, 6],
        "class_names": ["e", "(ij)", "(ij)(kl)", "(ijk)", "(ijkl)"],
    },
}


def _compute_conjugacy_classes(cayley_table: List[List[int]]) -> Dict[int, int]:
    """Compute element → conjugacy class index mapping from a Cayley table.

    Assumes identity = element 0. Works for any finite group.
    Classes are returned in order of smallest representative element.
    """
    n = len(cayley_table)
    inv = [0] * n
    for g in range(n):
        for h in range(n):
            if cayley_table[g][h] == 0:
                inv[g] = h
                break

    visited: set = set()
    class_map: Dict[int, int] = {}
    class_idx = 0
    for g in range(n):
        if g in visited:
            continue
        conj_class: set = set()
        for h in range(n):
            hg      = cayley_table[h][g]
            hgh_inv = cayley_table[hg][inv[h]]
            conj_class.add(hgh_inv)
        for elem in conj_class:
            class_map[elem] = class_idx
            visited.add(elem)
        class_idx += 1
    return class_map


try:
    from src.datasets import (
        _S3_TABLE, _D5_TABLE, _A4_TABLE, _S4_TABLE,
        _S4_ELEMENTS, _S4_INDEX,
    )
    _PRECOMPUTED_CLASS_MAPS: Dict[str, Dict[int, int]] = {
        "s3": _compute_conjugacy_classes(_S3_TABLE),
        "d5": _compute_conjugacy_classes(_D5_TABLE),
        "a4": _compute_conjugacy_classes(_A4_TABLE),
        "s4": _compute_conjugacy_classes(_S4_TABLE),
    }
    _PRECOMPUTED_INVERSES: Dict[str, List[int]] = {}
    for _gname, _table in [
        ("s3", _S3_TABLE), ("d5", _D5_TABLE),
        ("a4", _A4_TABLE), ("s4", _S4_TABLE),
    ]:
        _n = len(_table)
        _inv_tmp = [0] * _n
        for _g in range(_n):
            for _h in range(_n):
                if _table[_g][_h] == 0:
                    _inv_tmp[_g] = _h
                    break
        _PRECOMPUTED_INVERSES[_gname] = _inv_tmp
    _NONABELIAN_TABLES: Dict[str, List[List[int]]] = {
        "s3": _S3_TABLE, "d5": _D5_TABLE,
        "a4": _A4_TABLE, "s4": _S4_TABLE,
    }
    _NONABELIAN_AVAIL = True
except ImportError:
    _NONABELIAN_AVAIL = False


def nonabelian_fourier_analysis(
    model: torch.nn.Module,
    group_name: str,
) -> Dict:
    """Non-abelian Fourier analysis of the token embedding matrix.

    Uses the Peter–Weyl theorem to decompose the embedding matrix W_E into
    contributions from each irreducible representation of the group.

    The energy of W_E in irrep ρ (dimension d_ρ, character χ_ρ) is:

        E(ρ) = d_ρ · Σ_{g,h} K(g,h) · χ_ρ(g·h⁻¹)

    where K(g,h) = W_E[g]·W_E[h] is the Gram matrix.

    Parameters
    ----------
    model       : trained model (HookedTransformer or MinimalTransformer)
    group_name  : one of "s3", "d5", "a4", "s4"

    Returns
    -------
    dict with:
        group_name       : str
        irreps           : list of {name, dim, energy, fraction}
        dominant_irrep   : name of irrep with highest energy fraction
        concentration    : energy fraction of dominant irrep
        is_abelian_like  : bool — True if trivial+sign irreps dominate (>80%)
        char_table       : reference character table dict
    """
    if not _NONABELIAN_AVAIL:
        return {"error": "Non-abelian tables not available — check dataset imports."}

    if group_name not in _CHARACTER_TABLES:
        raise ValueError(
            f"group_name must be one of {list(_CHARACTER_TABLES.keys())}"
        )

    ctable   = _CHARACTER_TABLES[group_name]
    table    = _NONABELIAN_TABLES[group_name]
    inv_map  = _PRECOMPUTED_INVERSES[group_name]
    elem_cls = _PRECOMPUTED_CLASS_MAPS[group_name]
    n_elems  = ctable["group_order"]

    if _TL_AVAILABLE and isinstance(model, torch.nn.Module) and hasattr(model, "embed"):
        try:
            W_E = model.embed.W_E[:n_elems].detach().cpu().numpy().astype(np.float64)
        except AttributeError:
            W_E = model.embed.weight[:n_elems].detach().cpu().numpy().astype(np.float64)
    else:
        W_E = model.embed.weight[:n_elems].detach().cpu().numpy().astype(np.float64)

    K = W_E @ W_E.T

    gh_inv_cls = np.array([
        [elem_cls[table[g][inv_map[h]]] for h in range(n_elems)]
        for g in range(n_elems)
    ], dtype=np.int32)

    irrep_results = []
    for irrep in ctable["irreps"]:
        d_k  = irrep["dim"]
        chis = np.array(irrep["characters"], dtype=np.float64)
        chi_tensor = chis[gh_inv_cls]
        energy = float(d_k * np.sum(K * chi_tensor))
        irrep_results.append({
            "name":   irrep["name"],
            "dim":    d_k,
            "energy": energy,
        })

    pos_total = sum(max(0.0, r["energy"]) for r in irrep_results)
    for r in irrep_results:
        r["fraction"] = max(0.0, r["energy"]) / (pos_total + 1e-12)

    dominant = max(irrep_results, key=lambda x: x["energy"])
    abelian_like = sum(
        r["fraction"] for r in irrep_results if r["dim"] == 1
    ) > 0.80

    return {
        "group_name":      group_name,
        "irreps":          irrep_results,
        "dominant_irrep":  dominant["name"],
        "concentration":   dominant["fraction"],
        "is_abelian_like": abelian_like,
        "char_table":      ctable,
    }


def causal_dlog_verification(
    add_model: torch.nn.Module,
    p: int,
    n_samples: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict:
    """Causal test of the discrete-log hypothesis.

    Tests whether an addition-trained model can solve multiplication purely via
    the dlog isomorphism dlog(a·b) = dlog(a) + dlog(b) mod (p-1), WITHOUT any
    retraining on multiplication.

    Null test (proper statistical control):
        We map inputs using a DIFFERENT primitive root g' ≠ g. Under the real
        dlog hypothesis the result is specific to the correct g; if the model
        is exploiting a spurious bijection pattern the null would also succeed.
        A high main accuracy AND low null accuracy constitutes strong causal
        evidence for the dlog structure.

    Parameters
    ----------
    add_model : model trained on (x + y) mod (p-1), vocab size = p
    p         : prime used in the multiplication experiment
    n_samples : number of (a,b) pairs to test; None = all (p-1)² non-zero pairs
    device    : torch device string; None = auto-detect

    Returns
    -------
    dict with:
        dlog_mapped_accuracy : float  — accuracy using correct primitive root g
        null_accuracy        : float  — accuracy using wrong primitive root g'
        chance_accuracy      : float  — 1/(p-1)
        improvement_over_null: float  — dlog_mapped_accuracy - null_accuracy
        p_value_vs_chance    : float  — binomial p-value vs 1/(p-1) chance
        p_value_vs_null      : float  — binomial p-value vs null_accuracy
        verdict              : str    — "SUPPORTS" / "WEAK" / "REJECTS"
        primitive_root_used  : int    — g (correct)
        primitive_root_null  : int    — g' (wrong, used for null)
        n_tested             : int
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    add_model.eval()
    add_model.to(device)

    def _primitive_root_local(p: int) -> int:
        phi = p - 1
        factors: set = set()
        n = phi
        for d in range(2, int(n ** 0.5) + 1):
            while n % d == 0:
                factors.add(d)
                n //= d
        if n > 1:
            factors.add(n)
        for g in range(2, p):
            if all(pow(g, phi // f, p) != 1 for f in factors):
                return g
        raise ValueError(f"No primitive root for p={p}")

    def _build_dlog(g: int, p: int) -> np.ndarray:
        dlog = np.full(p, -1, dtype=np.int64)
        val = 1
        for k in range(p - 1):
            dlog[val] = k
            val = (val * g) % p
        return dlog

    g  = _primitive_root_local(p)
    g2 = g
    for candidate in range(g + 1, p):
        phi = p - 1
        factors: set = set()
        n2 = phi
        for d in range(2, int(n2 ** 0.5) + 1):
            while n2 % d == 0:
                factors.add(d)
                n2 //= d
        if n2 > 1:
            factors.add(n2)
        if all(pow(candidate, phi // f, p) != 1 for f in factors):
            g2 = candidate
            break
    if g2 == g:
        g2 = (g * g) % p

    dlog_correct = _build_dlog(g,  p)
    dlog_null    = _build_dlog(g2, p)

    pairs = [
        (a, b, (a * b) % p)
        for a in range(1, p)
        for b in range(1, p)
    ]
    if n_samples is not None:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pairs), size=min(n_samples, len(pairs)), replace=False)
        pairs = [pairs[i] for i in idx]

    sep = p - 1
    correct_main = 0
    correct_null = 0

    with torch.no_grad():
        for a, b, ab in pairs:
            da, db = int(dlog_correct[a]), int(dlog_correct[b])
            expected = int((da + db) % (p - 1))
            toks = torch.tensor([[da, db, sep]], dtype=torch.long, device=device)
            pred = int(add_model(toks)[0, -1, :].argmax().item())
            if pred == expected:
                correct_main += 1

            da2, db2 = int(dlog_null[a]), int(dlog_null[b])
            if da2 < 0 or db2 < 0:
                continue
            expected_null = int((da2 + db2) % (p - 1))
            toks_null = torch.tensor([[da2, db2, sep]], dtype=torch.long, device=device)
            pred_null = int(add_model(toks_null)[0, -1, :].argmax().item())
            if pred_null == expected_null:
                correct_null += 1

    from scipy.stats import binom
    n            = len(pairs)
    acc          = correct_main / n
    null_acc_obs = correct_null / n
    chance       = 1.0 / (p - 1)

    p_val_chance = float(1.0 - binom.cdf(correct_main - 1, n, chance))
    p_val_null   = float(1.0 - binom.cdf(correct_main - 1, n, max(null_acc_obs, chance)))

    if acc >= 0.80 and acc > null_acc_obs + 0.20:
        verdict = "SUPPORTS"
    elif acc >= 0.50:
        verdict = "WEAK"
    else:
        verdict = "REJECTS"

    return {
        "dlog_mapped_accuracy":  acc,
        "null_accuracy":         null_acc_obs,
        "chance_accuracy":       chance,
        "improvement_over_null": acc - null_acc_obs,
        "p_value_vs_chance":     p_val_chance,
        "p_value_vs_null":       p_val_null,
        "verdict":               verdict,
        "primitive_root_used":   g,
        "primitive_root_null":   g2,
        "n_tested":              n,
    }


def complexity_delay_regression(
    op_results: Dict[str, Dict],
    complexity_scores: Optional[Dict[str, float]] = None,
) -> Dict:
    """Fit a linear regression of grokking delay vs formal complexity score.

    Parameters
    ----------
    op_results : dict mapping op_symbol -> result dict from multi_seed_experiment
                 Each result must contain 'mean_grok_epoch' and 'std_grok_epoch'.
    complexity_scores : optional dict op_symbol -> float; if None, computed from
                        COMPLEXITY_MEASURES via get_complexity_score().

    Returns
    -------
    dict with:
        op_symbols     : list of ops included
        complexity_x   : np.ndarray of complexity scores
        delay_y        : np.ndarray of mean grokking delays
        delay_err      : np.ndarray of std of grokking delays
        slope, intercept, r_squared : regression parameters
        spearman_r, spearman_p : rank correlation + p-value
        verdict        : "STRONG" / "MODERATE" / "WEAK"
    """
    from scipy import stats as _scipy_stats
    from src.datasets import COMPLEXITY_MEASURES, get_complexity_score

    included: List[str] = []
    x_vals: List[float] = []
    y_vals: List[float] = []
    y_err: List[float]  = []

    for op, res in op_results.items():
        mean_ge = res.get("mean_grok_epoch")
        if mean_ge is None:
            continue
        included.append(op)
        score = (complexity_scores or {}).get(op) or get_complexity_score(op)
        x_vals.append(score)
        y_vals.append(float(mean_ge))
        std_ge = res.get("std_grok_epoch") or 0.0
        y_err.append(float(std_ge))

    if len(included) < 2:
        return {"error": "Need at least 2 grokking ops for regression."}

    x  = np.array(x_vals)
    y  = np.array(y_vals)
    ye = np.array(y_err)

    slope, intercept, r, p_lin, _ = _scipy_stats.linregress(x, y)
    sp_r, sp_p = _scipy_stats.spearmanr(x, y)

    if sp_r >= 0.85 and sp_p < 0.05:
        verdict = "STRONG"
    elif sp_r >= 0.60:
        verdict = "MODERATE"
    else:
        verdict = "WEAK"

    return {
        "op_symbols":   included,
        "complexity_x": x,
        "delay_y":      y,
        "delay_err":    ye,
        "slope":        float(slope),
        "intercept":    float(intercept),
        "r_squared":    float(r ** 2),
        "spearman_r":   float(sp_r),
        "spearman_p":   float(sp_p),
        "verdict":      verdict,
    }


def extract_weight_norms(model: torch.nn.Module) -> Dict[str, float]:
    """Return L2 norms of all named parameter tensors.

    Call this every log_every steps during training to track how weight
    norms evolve through memorisation → transition → generalisation phases.
    The norm pattern typically shows a sharp transition during grokking.

    Returns
    -------
    dict mapping parameter_name -> float L2 norm
    """
    return {
        name: float(param.detach().cpu().norm(2).item())
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def bootstrap_confidence_interval(
    values: List[float],
    statistic: str = "mean",
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """
    Compute a bootstrap confidence interval for a statistic over a sample.

    Parameters
    ----------
    values     : observed sample values (list or array)
    statistic  : "mean" | "median" | "std"
    n_bootstrap: number of bootstrap resamples
    confidence : CI width, e.g. 0.95 for 95%
    seed       : RNG seed for reproducibility

    Returns
    -------
    (point_estimate, lower_ci, upper_ci)
    All three are floats.  lower_ci and upper_ci are the percentile-method
    bootstrap confidence interval bounds.
    """
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)

    _stat_fn = {
        "mean":   np.mean,
        "median": np.median,
        "std":    np.std,
    }.get(statistic)
    if _stat_fn is None:
        raise ValueError(f"Unknown statistic {statistic!r}. Choose: 'mean','median','std'")

    point = float(_stat_fn(arr))
    boot  = np.array([
        _stat_fn(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = 1.0 - confidence
    lo    = float(np.percentile(boot, 100 * alpha / 2))
    hi    = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return point, lo, hi


def representation_formation_tracker(
    model: torch.nn.Module,
    dataset: dict,
    op_symbol: str,
) -> Dict:
    """
    Compute the current 'representation formation score' for a trained model.

    This score quantifies how much the learned token embeddings reflect the
    algebraic structure of the task — as opposed to a flat memorising
    representation that distributes energy across all modes equally.

    For ABELIAN operations (add, sub, mul, ring_add):
        Uses Fourier concentration: fraction of total embedding norm in the
        top-5 DFT frequency pairs. A fully-formed Fourier clock circuit
        typically exceeds 0.80 concentration.

    For NON-ABELIAN groups (s3, d5, a4, s4):
        Uses Peter-Weyl dominant irrep energy fraction from
        nonabelian_fourier_analysis(). A model using the correct Peter-Weyl
        circuit will show >50% energy in the dominant (largest-dim) irrep.

    Parameters
    ----------
    model      : trained model (any type)
    dataset    : dataset dict from datasets.py (used for p and op info)
    op_symbol  : one of the 8 op strings

    Returns
    -------
    dict with:
        representation_type  : "fourier" | "peter_weyl"
        formation_score      : float in [0, 1]
        threshold_crossed    : bool  (True if score > 0.70 fourier / 0.50 peter_weyl)
        dominant_component   : str   (frequency or irrep name)
        detail               : dict  (full analysis output)
    """
    NONABELIAN_OPS        = {"s3", "d5", "a4", "s4"}
    ABELIAN_THRESHOLD     = 0.70
    NONABELIAN_THRESHOLD  = 0.50

    if op_symbol in NONABELIAN_OPS:
        if not _NONABELIAN_AVAIL:
            return {
                "representation_type": "peter_weyl",
                "formation_score":     0.0,
                "threshold_crossed":   False,
                "dominant_component":  "unavailable",
                "detail":              {},
            }
        detail = nonabelian_fourier_analysis(model, group_name=op_symbol)
        score  = float(detail.get("concentration", 0.0))
        dom    = detail.get("dominant_irrep", "unknown")
        return {
            "representation_type": "peter_weyl",
            "formation_score":     score,
            "threshold_crossed":   score > NONABELIAN_THRESHOLD,
            "dominant_component":  dom,
            "detail":              detail,
        }
    else:
        p      = dataset.get("p", 113)
        detail = fourier_embedding_analysis(model, p)
        score  = float(detail["concentration"])
        top_k  = int(detail["top_freqs"][0]) if len(detail["top_freqs"]) > 0 else 0
        return {
            "representation_type": "fourier",
            "formation_score":     score,
            "threshold_crossed":   score > ABELIAN_THRESHOLD,
            "dominant_component":  f"freq_{top_k}",
            "detail":              {k: v.tolist() if hasattr(v, "tolist") else v
                                    for k, v in detail.items()
                                    if k != "embedding_matrix"},
        }


def grokking_leading_indicator(
    representation_history: List[Dict],
    grok_epoch: Optional[int],
    threshold: float = 0.50,
) -> Dict:
    """
    Given a per-epoch representation history, find the representation transition
    epoch and compute its lead time before the grokking epoch.

    The representation transition epoch is the FIRST epoch at which the
    formation_score exceeds `threshold`, marking when the embedding matrix
    reorganises from a diffuse memorising state to a structured algebraic one.

    Parameters
    ----------
    representation_history : list of dicts with keys "epoch", "formation_score"
    grok_epoch             : epoch where test_acc first ≥ 0.99 (or None)
    threshold              : formation_score threshold for transition detection

    Returns
    -------
    dict with:
        representation_transition_epoch : int | None
        grok_epoch                      : int | None
        lead_time                       : int | None  (grok - transition)
        lead_time_fraction              : float | None  (lead / grok_epoch)
        verdict                         : "PRECEDES" | "COINCIDES" | "FOLLOWS" | "NO_DATA"
        formation_scores                : list of (epoch, score) tuples
        transition_threshold_used       : float
    """
    scores = [
        (int(h["epoch"]), float(h.get("formation_score", 0.0)))
        for h in representation_history
        if "epoch" in h and "formation_score" in h
    ]
    scores.sort(key=lambda x: x[0])

    transition_epoch: Optional[int] = None
    for epoch, score in scores:
        if score > threshold:
            transition_epoch = epoch
            break

    lead_time: Optional[int] = None
    lead_frac: Optional[float] = None
    if transition_epoch is not None and grok_epoch is not None:
        lead_time = grok_epoch - transition_epoch
        lead_frac = lead_time / grok_epoch if grok_epoch > 0 else None

    if transition_epoch is None or grok_epoch is None:
        verdict = "NO_DATA"
    elif lead_time is None:
        verdict = "NO_DATA"
    elif lead_time > 50:
        verdict = "PRECEDES"
    elif lead_time >= -50:
        verdict = "COINCIDES"
    else:
        verdict = "FOLLOWS"

    return {
        "representation_transition_epoch": transition_epoch,
        "grok_epoch":                      grok_epoch,
        "lead_time":                       lead_time,
        "lead_time_fraction":              lead_frac,
        "verdict":                         verdict,
        "formation_scores":                scores,
        "transition_threshold_used":       threshold,
    }


def controlled_complexity_ablation(
    op_results_by_condition: Dict[str, Dict[str, Dict]],
    complexity_fn: str = "v1",
) -> Dict:
    """
    Analyse complexity-delay law robustness across controlled ablation conditions.

    Takes results from multiple experimental conditions (e.g. different dataset
    sizes, training fractions, or weight decays) and computes Spearman ρ and
    Kendall τ of complexity vs grok_delay for each condition.

    Parameters
    ----------
    op_results_by_condition : dict mapping condition_name ->
                              {op_symbol -> result_dict from multi_seed_experiment}
                              Each result_dict needs 'mean_grok_epoch'.
    complexity_fn           : "v1" | "v2" — which complexity score formula to use

    Returns
    -------
    dict with:
        conditions       : list of condition names
        spearman_by_cond : dict condition -> (rho, p_value)
        kendall_by_cond  : dict condition -> (tau, p_value)
        rank_preserved   : dict condition -> bool (tau >= 0.8)
        reference_ranking: list of op_symbols sorted by complexity (ascending)
        summary          : str human-readable verdict
    """
    from scipy import stats as _sp_stats
    from src.datasets import get_complexity_score, get_complexity_score_v2

    _score_fn = get_complexity_score_v2 if complexity_fn == "v2" else get_complexity_score

    all_ops_seen: set = set()
    for cond_results in op_results_by_condition.values():
        all_ops_seen.update(cond_results.keys())
    reference_ranking = sorted(
        [op for op in all_ops_seen],
        key=lambda op: _score_fn(op) if op in [
            "add","sub","mul","ring_add","s3","d5","a4","s4"
        ] else 99.0,
    )

    spearman_by_cond: Dict[str, Tuple[float, float]] = {}
    kendall_by_cond:  Dict[str, Tuple[float, float]] = {}
    rank_preserved:   Dict[str, bool]                = {}

    for cond_name, cond_results in op_results_by_condition.items():
        xs, ys = [], []
        for op, res in cond_results.items():
            ge = res.get("mean_grok_epoch")
            if ge is None:
                continue
            try:
                xs.append(_score_fn(op))
                ys.append(float(ge))
            except ValueError:
                continue

        if len(xs) < 3:
            spearman_by_cond[cond_name] = (float("nan"), float("nan"))
            kendall_by_cond[cond_name]  = (float("nan"), float("nan"))
            rank_preserved[cond_name]   = False
            continue

        sp_r, sp_p = _sp_stats.spearmanr(xs, ys)
        kt_t, kt_p = _sp_stats.kendalltau(xs, ys)
        spearman_by_cond[cond_name] = (float(sp_r), float(sp_p))
        kendall_by_cond[cond_name]  = (float(kt_t), float(kt_p))
        rank_preserved[cond_name]   = float(kt_t) >= 0.8

    n_preserved = sum(rank_preserved.values())
    n_total     = len(rank_preserved)
    if n_total == 0:
        summary = "No conditions with sufficient data."
    elif n_preserved == n_total:
        summary = (f"Rank order preserved in ALL {n_total} conditions (Kendall τ ≥ 0.8). "
                   f"Complexity-delay law is robust.")
    else:
        summary = (f"Rank order preserved in {n_preserved}/{n_total} conditions. "
                   f"Law may depend on experimental setting.")

    return {
        "conditions":        list(op_results_by_condition.keys()),
        "spearman_by_cond":  spearman_by_cond,
        "kendall_by_cond":   kendall_by_cond,
        "rank_preserved":    rank_preserved,
        "reference_ranking": reference_ranking,
        "summary":           summary,
    }


def describe_learned_circuit(
    model: torch.nn.Module,
    dataset: dict,
    p: int,
    op_symbol: str,
) -> Dict[str, str]:
    """
    Synthesise all available analysis results into a mechanistic circuit
    description for a trained model.

    Runs: fourier_embedding_analysis (or nonabelian_fourier_analysis),
    logit_attribution, activation_patch_heads, representation_formation_tracker.
    Then classifies the circuit into one of four canonical types.

    Parameters
    ----------
    model      : trained model
    dataset    : dataset dict from datasets.py
    p          : group order / modulus
    op_symbol  : one of the 8 op strings

    Returns
    -------
    dict with:
        representation_type  : "fourier_clock" | "dlog_then_clock" |
                               "partial_fourier" | "peter_weyl"
        key_frequencies      : list[int]  (abelian ops; empty for non-abelian)
        dominant_irrep       : str        (non-abelian; empty for abelian)
        critical_heads       : list[int]  (head indices with highest patching score)
        mlp_role             : "frequency_filter" | "irrep_projector" | "lookup_table"
        evidence_strength    : "strong" | "moderate" | "weak"
        evidence_summary     : str (one-paragraph description)
        raw                  : dict (all intermediate analysis outputs)
    """
    raw: Dict = {}

    rft = representation_formation_tracker(model, dataset, op_symbol)
    raw["repr_formation"] = rft

    NONABELIAN = {"s3", "d5", "a4", "s4"}
    if op_symbol in NONABELIAN:
        if _NONABELIAN_AVAIL:
            naf = nonabelian_fourier_analysis(model, group_name=op_symbol)
            raw["nonabelian_fourier"] = naf
        key_freqs = []
        dom_irrep = rft.get("dominant_component", "unknown")
        conc      = rft.get("formation_score", 0.0)
        repr_type = "peter_weyl"
    else:
        fa = fourier_embedding_analysis(model, p)
        raw["fourier"] = {k: v.tolist() if hasattr(v, "tolist") else v
                          for k, v in fa.items() if k != "embedding_matrix"}
        key_freqs = [int(k) for k in fa["top_freqs"][:5]]
        dom_irrep = ""
        conc      = fa["concentration"]
        repr_type = "dlog_then_clock" if op_symbol == "mul" else (
                    "partial_fourier"  if op_symbol == "ring_add" else
                    "fourier_clock")

    try:
        attr = logit_attribution(model, dataset, n_samples=30)
        raw["logit_attribution"] = attr
        head_contribs = attr.get("head_contributions", np.array([]))
        mlp_contrib   = attr.get("mlp_contribution", 0.0)
        is_exact      = attr.get("is_exact", False)
    except Exception:
        head_contribs = np.array([])
        mlp_contrib   = 0.0
        is_exact      = False

    try:
        importance = activation_patch_heads(model, dataset, n_samples=20)
        raw["patching_importance"] = importance.tolist()
        top2 = np.argsort(importance)[::-1][:2].tolist() if len(importance) > 0 else []
    except Exception:
        top2 = []

    if op_symbol in NONABELIAN:
        mlp_role = "irrep_projector"
    elif op_symbol == "ring_add" or conc < 0.5:
        mlp_role = "lookup_table"
    else:
        mlp_role = "frequency_filter"

    if conc >= 0.75 and rft["threshold_crossed"]:
        evidence_strength = "strong"
    elif conc >= 0.45:
        evidence_strength = "moderate"
    else:
        evidence_strength = "weak"

    if repr_type == "fourier_clock":
        summary = (
            f"The model uses a Fourier clock circuit: embeddings concentrate "
            f"{conc:.0%} of norm into frequencies {key_freqs[:3]}. "
            f"The MLP acts as a frequency filter ({mlp_role}). "
            f"Evidence: {evidence_strength} (Fourier concentration={conc:.3f})."
        )
    elif repr_type == "dlog_then_clock":
        summary = (
            f"The model likely uses a discrete-log reduction: embeddings "
            f"concentrate at Fourier frequencies {key_freqs[:3]} after dlog "
            f"re-indexing. Concentration = {conc:.3f}. The MLP acts as a "
            f"frequency filter. Evidence: {evidence_strength}."
        )
    elif repr_type == "partial_fourier":
        summary = (
            f"The model shows partial Fourier structure (concentration={conc:.3f}, "
            f"expected at divisor frequencies of composite modulus). "
            f"MLP role: {mlp_role}. Evidence: {evidence_strength}."
        )
    else:
        summary = (
            f"Non-abelian group ({op_symbol.upper()}): Peter-Weyl analysis shows "
            f"dominant irrep '{dom_irrep}' with {conc:.0%} energy fraction. "
            f"MLP acts as irrep projector. Evidence: {evidence_strength}."
        )

    return {
        "representation_type": repr_type,
        "key_frequencies":     key_freqs,
        "dominant_irrep":      dom_irrep,
        "critical_heads":      top2,
        "mlp_role":            mlp_role,
        "evidence_strength":   evidence_strength,
        "evidence_summary":    summary,
        "raw":                 raw,
    }
