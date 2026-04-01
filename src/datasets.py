from __future__ import annotations

import random
from itertools import product as iter_product
from typing import List, Tuple

import numpy as np


def _is_prime(n: int) -> bool:
    """Miller-Rabin primality check (deterministic for n < 3_215_031_751)."""
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    if n in small_primes:
        return True
    if any(n % p == 0 for p in small_primes):
        return False
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _split(
    pairs: List[Tuple[int, int, int]],
    train_frac: float,
    seed: int,
) -> Tuple[List, List]:
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * train_frac)
    return shuffled[:cut], shuffled[cut:]


# ---------------------------------------------------------------------------
# Experiment E1 — modular addition (a + b) mod p
# ---------------------------------------------------------------------------

def make_modular_addition(
    p: int = 113,
    train_frac: float = 0.7,
    seed: int = 42,
) -> dict:
    """(a + b) mod p  —  baseline experiment."""
    if not _is_prime(p):
        raise ValueError(f"p={p} is not prime. Use a prime for Z/pZ field structure.")
    pairs = [(a, b, (a + b) % p) for a in range(p) for b in range(p)]
    train, test = _split(pairs, train_frac, seed)
    return {
        "train_data": train,
        "test_data": test,
        "vocab_size": p + 1,
        "op_name": f"(a + b) mod {p}",
        "op_symbol": "add",
        "p": p,
    }


# ---------------------------------------------------------------------------
# Experiment E2 — modular multiplication (a x b) mod p
# ---------------------------------------------------------------------------

def make_modular_multiplication(
    p: int = 113,
    train_frac: float = 0.7,
    seed: int = 42,
) -> dict:
    """(a x b) mod p  —  tests discrete-log representation hypothesis."""
    if not _is_prime(p):
        raise ValueError(f"p={p} is not prime.")
    pairs = [(a, b, (a * b) % p) for a in range(p) for b in range(p)]
    train, test = _split(pairs, train_frac, seed)
    return {
        "train_data": train,
        "test_data": test,
        "vocab_size": p + 1,
        "op_name": f"(a x b) mod {p}",
        "op_symbol": "mul",
        "p": p,
    }


# ---------------------------------------------------------------------------
# Experiment E3 — modular subtraction (a - b) mod p
# ---------------------------------------------------------------------------

def make_modular_subtraction(
    p: int = 113,
    train_frac: float = 0.7,
    seed: int = 42,
) -> dict:
    """(a - b) mod p  —  control; expect same Fourier circuit as addition."""
    if not _is_prime(p):
        raise ValueError(f"p={p} is not prime.")
    pairs = [(a, b, (a - b) % p) for a in range(p) for b in range(p)]
    train, test = _split(pairs, train_frac, seed)
    return {
        "train_data": train,
        "test_data": test,
        "vocab_size": p + 1,
        "op_name": f"(a - b) mod {p}",
        "op_symbol": "sub",
        "p": p,
    }


# ---------------------------------------------------------------------------
# Experiment E4 — addition over a composite ring Z/nZ
# ---------------------------------------------------------------------------

def make_ring_addition(
    n: int = 100,
    train_frac: float = 0.7,
    seed: int = 42,
) -> dict:
    """(a + b) mod n  with composite n  —  partial Fourier structure expected."""
    pairs = [(a, b, (a + b) % n) for a in range(n) for b in range(n)]
    train, test = _split(pairs, train_frac, seed)
    return {
        "train_data": train,
        "test_data": test,
        "vocab_size": n + 1,
        "op_name": f"(a + b) mod {n}  [ring, composite]",
        "op_symbol": "ring_add",
        "p": n,
    }


# ---------------------------------------------------------------------------
# Experiment E5 — Symmetric group S3
# ---------------------------------------------------------------------------

_S3_ELEMENTS: List[Tuple[int, ...]] = [
    (0, 1, 2),  # identity
    (1, 2, 0),  # r   (rotation by 120°)
    (2, 0, 1),  # r^2
    (0, 2, 1),  # s   (reflection)
    (2, 1, 0),  # sr
    (1, 0, 2),  # sr^2
]


def _s3_compose(a: int, b: int) -> int:
    perm_a = _S3_ELEMENTS[a]
    perm_b = _S3_ELEMENTS[b]
    composed = tuple(perm_b[perm_a[i]] for i in range(3))
    return _S3_ELEMENTS.index(composed)


_S3_TABLE = [[_s3_compose(a, b) for b in range(6)] for a in range(6)]


def make_s3_group(
    train_frac: float = 0.7,
    seed: int = 42,
) -> dict:
    """S3 group multiplication  —  non-abelian, |S3|=6, reference baseline."""
    n = 6
    pairs = [(a, b, _S3_TABLE[a][b]) for a in range(n) for b in range(n)]
    train, test = _split(pairs, train_frac, seed)
    return {
        "train_data": train,
        "test_data": test,
        "vocab_size": n + 1,
        "op_name": "S3 group multiplication  [non-abelian, |G|=6]",
        "op_symbol": "s3",
        "p": n,
    }


# ---------------------------------------------------------------------------
# Experiment E6 — Dihedral group D5
# ---------------------------------------------------------------------------

def _d5_multiply(i: int, j: int) -> int:
    rot_a, flip_a = i % 5, i // 5
    rot_b, flip_b = j % 5, j // 5
    if flip_a == 0:
        rot_c = (rot_a + rot_b) % 5
        flip_c = flip_b
    else:
        rot_c = (rot_a - rot_b) % 5
        flip_c = 1 - flip_b
    return rot_c + 5 * flip_c


_D5_TABLE = [[_d5_multiply(a, b) for b in range(10)] for a in range(10)]


def make_d5_group(
    train_frac: float = 0.7,
    seed: int = 42,
) -> dict:
    """D5 dihedral group multiplication  —  non-abelian, |D5|=10.

    Significantly larger than S3 (100 pairs vs 36), making the dataset-size
    confounder in S3 far less severe.  D5 has no 1-d complex representations
    over the reals, so standard Fourier analysis cannot apply.
    """
    n = 10
    pairs = [(a, b, _D5_TABLE[a][b]) for a in range(n) for b in range(n)]
    train, test = _split(pairs, train_frac, seed)
    return {
        "train_data": train,
        "test_data": test,
        "vocab_size": n + 1,
        "op_name": "D5 dihedral group multiplication  [non-abelian, |G|=10]",
        "op_symbol": "d5",
        "p": n,
    }


# ---------------------------------------------------------------------------
# Experiment E7 — Alternating group A4
# ---------------------------------------------------------------------------

def _even_perms_4() -> List[Tuple[int, ...]]:
    """Return all 12 even permutations of (0,1,2,3)."""
    from itertools import permutations as _perms

    def _sgn(p: Tuple[int, ...]) -> int:
        """Compute sign of permutation via inversion count."""
        inv = sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j])
        return (-1) ** inv

    return [p for p in _perms(range(4)) if _sgn(p) == 1]


_A4_ELEMENTS = _even_perms_4()
_A4_INDEX = {p: i for i, p in enumerate(_A4_ELEMENTS)}


def _a4_multiply(i: int, j: int) -> int:
    pa, pb = _A4_ELEMENTS[i], _A4_ELEMENTS[j]
    composed = tuple(pb[pa[k]] for k in range(4))
    return _A4_INDEX[composed]


_A4_TABLE = [[_a4_multiply(a, b) for b in range(12)] for a in range(12)]


def make_a4_group(
    train_frac: float = 0.7,
    seed: int = 42,
) -> dict:
    """A4 alternating group multiplication  —  non-abelian, |A4|=12.

    A4 is non-abelian and has no subgroup of order 6 (it famously violates
    the converse of Lagrange's theorem).  With 144 pairs it is large enough
    to be a meaningful test of grokking without trivial memorisation.
    """
    n = 12
    pairs = [(a, b, _A4_TABLE[a][b]) for a in range(n) for b in range(n)]
    train, test = _split(pairs, train_frac, seed)
    return {
        "train_data": train,
        "test_data": test,
        "vocab_size": n + 1,
        "op_name": "A4 alternating group multiplication  [non-abelian, |G|=12]",
        "op_symbol": "a4",
        "p": n,
    }


# ---------------------------------------------------------------------------
# PyTorch DataLoader helper
# ---------------------------------------------------------------------------

class AlgebraicDataset:
    """
    Wraps (a, b, label) triples into tensors with the 3-token format:
        [a, b, SEP]  ->  predict label at SEP position
    Torch is imported lazily so the rest of datasets.py works without it.
    """

    def __init__(self, data: List[Tuple[int, int, int]], sep_token: int) -> None:
        import torch
        self._torch = torch
        self.data = data
        self.sep = sep_token

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        torch = self._torch
        a, b, label = self.data[idx]
        tokens = torch.tensor([a, b, self.sep], dtype=torch.long)
        return tokens, torch.tensor(label, dtype=torch.long)


def build_loaders(
    dataset_dict: dict,
    batch_size: int = -1,
    num_workers: int = 0,
) -> Tuple:
    """
    Return (train_loader, test_loader).
    batch_size=-1 uses full dataset as one batch (recommended for small tasks).
    """
    import torch
    sep_token = dataset_dict["p"]
    train_ds = AlgebraicDataset(dataset_dict["train_data"], sep_token)
    test_ds  = AlgebraicDataset(dataset_dict["test_data"],  sep_token)

    bs_train = len(train_ds) if batch_size == -1 else batch_size
    bs_test  = len(test_ds)  if batch_size == -1 else batch_size

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=bs_train, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=bs_test, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Experiment E8 — Symmetric group S4
# ---------------------------------------------------------------------------

from itertools import permutations as _s4_perms

_S4_ELEMENTS: List[Tuple[int, ...]] = sorted(_s4_perms(range(4)))
_S4_INDEX: dict = {p: i for i, p in enumerate(_S4_ELEMENTS)}


def _s4_multiply(i: int, j: int) -> int:
    pa, pb = _S4_ELEMENTS[i], _S4_ELEMENTS[j]
    composed = tuple(pb[pa[k]] for k in range(4))
    return _S4_INDEX[composed]


_S4_TABLE = [[_s4_multiply(a, b) for b in range(24)] for a in range(24)]


def make_s4_group(
    train_frac: float = 0.7,
    seed: int = 42,
) -> dict:
    """S4 symmetric group multiplication  —  non-abelian, |S4|=24, 576 pairs.

    S4 has five conjugacy classes and five irreducible representations of
    dimensions 1, 1, 2, 3, 3.  With 576 training pairs it removes the
    dataset-size confound present in S3 (36) and D5 (100).
    """
    n = 24
    pairs = [(a, b, _S4_TABLE[a][b]) for a in range(n) for b in range(n)]
    train, test = _split(pairs, train_frac, seed)
    return {
        "train_data": train,
        "test_data": test,
        "vocab_size": n + 1,
        "op_name": "S4 symmetric group multiplication  [non-abelian, |G|=24]",
        "op_symbol": "s4",
        "p": n,
    }


# ---------------------------------------------------------------------------
# Complexity measures and scoring
# ---------------------------------------------------------------------------

COMPLEXITY_MEASURES: dict = {
    "add": {
        "group":            "Z/pZ (additive)",
        "is_abelian":       True,
        "irrep_dims":       [1] * 113,
        "n_irreps":         113,
        "max_irrep_dim":    1,
        "complexity_rank":  1,
        "complexity_label": "Abelian field — prime (additive)",
        "description":      "All irreps 1-D (characters); Fourier clock algorithm.",
    },
    "sub": {
        "group":            "Z/pZ (additive)",
        "is_abelian":       True,
        "irrep_dims":       [1] * 113,
        "n_irreps":         113,
        "max_irrep_dim":    1,
        "complexity_rank":  1,
        "complexity_label": "Abelian field — prime (subtractive)",
        "description":      "Isomorphic to addition via negation; identical circuit.",
    },
    "mul": {
        "group":            "Z/pZ* (multiplicative)",
        "is_abelian":       True,
        "irrep_dims":       [1] * 112,
        "n_irreps":         112,
        "max_irrep_dim":    1,
        "complexity_rank":  2,
        "complexity_label": "Abelian field — multiplicative (dlog reduction)",
        "description":      "Reducible to addition via discrete log; extra layer of indirection.",
    },
    "ring_add": {
        "group":            "Z/nZ (ring, composite n)",
        "is_abelian":       True,
        "irrep_dims":       [1] * 100,
        "n_irreps":         100,
        "max_irrep_dim":    1,
        "complexity_rank":  3,
        "complexity_label": "Abelian ring — composite modulus",
        "description":      "Partial Fourier structure at divisor frequencies only.",
    },
    "s3": {
        "group":            "S3",
        "is_abelian":       False,
        "irrep_dims":       [1, 1, 2],
        "n_irreps":         3,
        "max_irrep_dim":    2,
        "complexity_rank":  4,
        "complexity_label": "Non-abelian (S3, |G|=6)",
        "description":      "Has 2-D irrep; standard Fourier analysis fails.",
    },
    "d5": {
        "group":            "D5",
        "is_abelian":       False,
        "irrep_dims":       [1, 1, 2, 2],
        "n_irreps":         4,
        "max_irrep_dim":    2,
        "complexity_rank":  5,
        "complexity_label": "Non-abelian (D5, |G|=10)",
        "description":      "No 1-D non-trivial real irreps; all non-trivial irreps are 2-D.",
    },
    "a4": {
        "group":            "A4",
        "is_abelian":       False,
        "irrep_dims":       [1, 1, 1, 3],
        "n_irreps":         4,
        "max_irrep_dim":    3,
        "complexity_rank":  6,
        "complexity_label": "Non-abelian (A4, |G|=12)",
        "description":      "Has 3-D irrep; no subgroup of order 6 (Lagrange converse fails).",
    },
    "s4": {
        "group":            "S4",
        "is_abelian":       False,
        "irrep_dims":       [1, 1, 2, 3, 3],
        "n_irreps":         5,
        "max_irrep_dim":    3,
        "complexity_rank":  7,
        "complexity_label": "Non-abelian (S4, |G|=24)",
        "description":      "Richest group tested; 5 conjugacy classes, max irrep dim 3.",
    },
}


def get_complexity_score(op_symbol: str) -> float:
    """Return formal complexity score for an operation (v1 — ad hoc formula).

    Defined as: rank + max_irrep_dim / 10 + (1 if non-abelian else 0) * 0.5
    This gives a continuous ordering consistent with the complexity-delay law.

    Note: COMPLEXITY_MEASURES irrep counts are hardcoded for p=113. The
    complexity score is operation-type specific and safe to use for any prime p.
    """
    m = COMPLEXITY_MEASURES.get(op_symbol)
    if m is None:
        raise ValueError(f"Unknown op_symbol: {op_symbol!r}")
    return (
        m["complexity_rank"]
        + m["max_irrep_dim"] / 10.0
        + (0.5 if not m["is_abelian"] else 0.0)
    )


def get_complexity_score_v2(op_symbol: str) -> float:
    """Return theory-grounded complexity score for an operation (v2).

    Formula: C(G) = log2(max_irrep_dim + 1) + (1 - 1/n_irreps) + (1 if non-abelian else 0)

    Rationale (each term is grounded in representation theory):
      log2(max_irrep_dim + 1):
          Captures the minimal embedding dimension needed to represent the hardest
          irrep. A d-dimensional irrep requires O(d) activation dimensions to store,
          and log2 reflects the information-theoretic cost.
      (1 - 1/n_irreps):
          Measures irrep diversity. A single trivial irrep gives 0; many distinct
          irreps approach 1. Captures how many distinct modes the model must represent.
      Binary non-abelian penalty (0 or 1):
          Accounts for the qualitative transition from Fourier-sufficient (abelian)
          to Peter-Weyl-required (non-abelian) representation.

    Both v1 and v2 should give high Spearman ρ with grok delay.
    """
    m = COMPLEXITY_MEASURES.get(op_symbol)
    if m is None:
        raise ValueError(f"Unknown op_symbol: {op_symbol!r}")
    import math
    max_d    = m["max_irrep_dim"]
    n_irreps = m["n_irreps"]
    abelian  = m["is_abelian"]
    return (
        math.log2(max_d + 1)
        + (1.0 - 1.0 / n_irreps)
        + (0.0 if abelian else 1.0)
    )


if __name__ == "__main__":
    for fn, kwargs in [
        (make_modular_addition,       {"p": 113}),
        (make_modular_multiplication, {"p": 113}),
        (make_modular_subtraction,    {"p": 113}),
        (make_ring_addition,          {"n": 100}),
        (make_s3_group,               {}),
        (make_d5_group,               {}),
        (make_a4_group,               {}),
        (make_s4_group,               {}),
    ]:
        ds = fn(**kwargs)
        total = len(ds["train_data"]) + len(ds["test_data"])
        print(
            f"{ds['op_name']:60s}  "
            f"total={total:6d}  train={len(ds['train_data']):5d}  "
            f"test={len(ds['test_data']):5d}  vocab={ds['vocab_size']}"
        )
    print("\nComplexity scores (v1 ad-hoc | v2 theory-grounded):")
    for op in ("add", "sub", "mul", "ring_add", "s3", "d5", "a4", "s4"):
        s1 = get_complexity_score(op)
        s2 = get_complexity_score_v2(op)
        print(f"  {op:10s}: v1={s1:.2f}  v2={s2:.3f}")
