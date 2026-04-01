
import sys
import os
import importlib.util

_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _ROOT)

def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_ds = _import_module("src.datasets", os.path.join(_ROOT, "src", "datasets.py"))

make_modular_addition       = _ds.make_modular_addition
make_modular_multiplication = _ds.make_modular_multiplication
make_modular_subtraction    = _ds.make_modular_subtraction
make_ring_addition          = _ds.make_ring_addition
make_s3_group               = _ds.make_s3_group
make_d5_group               = _ds.make_d5_group
make_a4_group               = _ds.make_a4_group
make_s4_group               = _ds.make_s4_group  
_is_prime                   = _ds._is_prime
_S3_TABLE                   = _ds._S3_TABLE
_D5_TABLE                   = _ds._D5_TABLE
_A4_TABLE                   = _ds._A4_TABLE
_S4_TABLE                   = _ds._S4_TABLE       

def assert_equal(a, b, msg=""):
    if a != b:
        raise AssertionError(f"{msg}  expected {b!r}, got {a!r}")

def assert_true(cond, msg=""):
    if not cond:
        raise AssertionError(msg)




def test_is_prime():
    primes     = [2, 3, 5, 7, 11, 13, 17, 19, 23, 97, 101, 113]
    composites = [1, 4, 6, 8, 9, 10, 100, 112, 114]
    for p in primes:
        assert_true(_is_prime(p), f"{p} should be prime")
    for n in composites:
        assert_true(not _is_prime(n), f"{n} should not be prime")
    print("  PASS test_is_prime")




def test_s3_table_closure():
    for a in range(6):
        for b in range(6):
            c = _S3_TABLE[a][b]
            assert_true(0 <= c < 6, f"S3[{a}][{b}]={c} out of range")
    print("  PASS test_s3_table_closure")

def test_s3_table_identity():
    for a in range(6):
        assert_equal(_S3_TABLE[0][a], a, f"0*{a}")
        assert_equal(_S3_TABLE[a][0], a, f"{a}*0")
    print("  PASS test_s3_table_identity")

def test_s3_table_inverses():
    for a in range(6):
        assert_true(any(_S3_TABLE[a][b] == 0 for b in range(6)),
                    f"No right inverse for S3 element {a}")
    print("  PASS test_s3_table_inverses")

def test_s3_non_abelian():
    non_comm = [(a, b) for a in range(6) for b in range(6)
                if _S3_TABLE[a][b] != _S3_TABLE[b][a]]
    assert_true(len(non_comm) > 0, "S3 should be non-abelian")
    print(f"  PASS test_s3_non_abelian  ({len(non_comm)} non-commuting pairs)")




def test_d5_table_closure():
    for a in range(10):
        for b in range(10):
            c = _D5_TABLE[a][b]
            assert_true(0 <= c < 10, f"D5[{a}][{b}]={c} out of range")
    print("  PASS test_d5_table_closure")

def test_d5_table_identity():
    for a in range(10):
        assert_equal(_D5_TABLE[0][a], a, f"D5: e*{a}")
        assert_equal(_D5_TABLE[a][0], a, f"D5: {a}*e")
    print("  PASS test_d5_table_identity")

def test_d5_table_inverses():
    for a in range(10):
        assert_true(any(_D5_TABLE[a][b] == 0 for b in range(10)),
                    f"No right inverse for D5 element {a}")
    print("  PASS test_d5_table_inverses")

def test_d5_non_abelian():
    non_comm = [(a, b) for a in range(10) for b in range(10)
                if _D5_TABLE[a][b] != _D5_TABLE[b][a]]
    assert_true(len(non_comm) > 0, "D5 should be non-abelian")
    print(f"  PASS test_d5_non_abelian  ({len(non_comm)} non-commuting pairs)")

def test_d5_associativity():
    """Group axiom: (a*b)*c == a*(b*c) for all a,b,c."""
    import random
    rng = random.Random(0)
    triples = [(rng.randint(0,9), rng.randint(0,9), rng.randint(0,9)) for _ in range(50)]
    for a, b, c in triples:
        lhs = _D5_TABLE[_D5_TABLE[a][b]][c]
        rhs = _D5_TABLE[a][_D5_TABLE[b][c]]
        assert_equal(lhs, rhs, f"D5 not associative: ({a}*{b})*{c} != {a}*({b}*{c})")
    print("  PASS test_d5_associativity")




def test_a4_table_closure():
    for a in range(12):
        for b in range(12):
            c = _A4_TABLE[a][b]
            assert_true(0 <= c < 12, f"A4[{a}][{b}]={c} out of range")
    print("  PASS test_a4_table_closure")

def test_a4_table_identity():
    for a in range(12):
        assert_equal(_A4_TABLE[0][a], a, f"A4: e*{a}")
        assert_equal(_A4_TABLE[a][0], a, f"A4: {a}*e")
    print("  PASS test_a4_table_identity")

def test_a4_table_inverses():
    for a in range(12):
        assert_true(any(_A4_TABLE[a][b] == 0 for b in range(12)),
                    f"No right inverse for A4 element {a}")
    print("  PASS test_a4_table_inverses")

def test_a4_non_abelian():
    non_comm = [(a, b) for a in range(12) for b in range(12)
                if _A4_TABLE[a][b] != _A4_TABLE[b][a]]
    assert_true(len(non_comm) > 0, "A4 should be non-abelian")
    print(f"  PASS test_a4_non_abelian  ({len(non_comm)} non-commuting pairs)")

def test_a4_order():
    assert_equal(len(_ds._A4_ELEMENTS), 12, "A4 should have exactly 12 elements")
    print("  PASS test_a4_order")




def _check_dataset(ds, expected_p):
    assert_true("train_data" in ds,  "missing train_data")
    assert_true("test_data"  in ds,  "missing test_data")
    assert_true("vocab_size" in ds,  "missing vocab_size")
    assert_true("op_name"    in ds,  "missing op_name")
    assert_equal(ds["p"], expected_p, "wrong p")
    p = ds["p"]
    for split in ("train_data", "test_data"):
        for a, b, c in ds[split]:
            assert_true(0 <= a < p,  f"a={a} out of range [0,{p})")
            assert_true(0 <= b < p,  f"b={b} out of range [0,{p})")
            assert_true(0 <= c < p,  f"label={c} out of range [0,{p})")
    train_set = {(a, b) for a, b, _ in ds["train_data"]}
    test_set  = {(a, b) for a, b, _ in ds["test_data"]}
    assert_equal(len(train_set & test_set), 0, "train/test overlap (data leakage)")


def test_modular_addition():
    p = 7
    ds = make_modular_addition(p=p)
    _check_dataset(ds, p)
    lookup = {(a, b): c for a, b, c in ds["train_data"] + ds["test_data"]}
    for a in range(p):
        for b in range(p):
            assert_equal(lookup[(a, b)], (a + b) % p)
    assert_equal(ds["vocab_size"], p + 1)
    print(f"  PASS test_modular_addition  (p={p})")

def test_modular_multiplication():
    p = 7
    ds = make_modular_multiplication(p=p)
    _check_dataset(ds, p)
    lookup = {(a, b): c for a, b, c in ds["train_data"] + ds["test_data"]}
    for a in range(p):
        for b in range(p):
            assert_equal(lookup[(a, b)], (a * b) % p)
    print(f"  PASS test_modular_multiplication  (p={p})")

def test_modular_subtraction():
    p = 7
    ds = make_modular_subtraction(p=p)
    _check_dataset(ds, p)
    lookup = {(a, b): c for a, b, c in ds["train_data"] + ds["test_data"]}
    for a in range(p):
        for b in range(p):
            assert_equal(lookup[(a, b)], (a - b) % p)
    print(f"  PASS test_modular_subtraction  (p={p})")

def test_ring_addition():
    n = 6
    ds = make_ring_addition(n=n)
    _check_dataset(ds, n)
    lookup = {(a, b): c for a, b, c in ds["train_data"] + ds["test_data"]}
    for a in range(n):
        for b in range(n):
            assert_equal(lookup[(a, b)], (a + b) % n)
    print(f"  PASS test_ring_addition  (n={n})")

def test_s3_dataset():
    ds = make_s3_group()
    _check_dataset(ds, 6)
    total = len(ds["train_data"]) + len(ds["test_data"])
    assert_equal(total, 36, "S3 should have 36 pairs")
    print(f"  PASS test_s3_dataset  (36 pairs)")

def test_d5_dataset():
    ds = make_d5_group()
    _check_dataset(ds, 10)
    total = len(ds["train_data"]) + len(ds["test_data"])
    assert_equal(total, 100, "D5 should have 100 pairs")
  
    lookup = {(a, b): c for a, b, c in ds["train_data"] + ds["test_data"]}
    for a in range(10):
        for b in range(10):
            assert_equal(lookup[(a, b)], _D5_TABLE[a][b], f"D5[{a}][{b}] mismatch")
    print(f"  PASS test_d5_dataset  (100 pairs)")

def test_a4_dataset():
    ds = make_a4_group()
    _check_dataset(ds, 12)
    total = len(ds["train_data"]) + len(ds["test_data"])
    assert_equal(total, 144, "A4 should have 144 pairs")
    lookup = {(a, b): c for a, b, c in ds["train_data"] + ds["test_data"]}
    for a in range(12):
        for b in range(12):
            assert_equal(lookup[(a, b)], _A4_TABLE[a][b], f"A4[{a}][{b}] mismatch")
    print(f"  PASS test_a4_dataset  (144 pairs)")

def test_train_fraction():
    p, frac = 11, 0.6
    ds = make_modular_addition(p=p, train_frac=frac)
    expected_train = int(p * p * frac)
    actual_train   = len(ds["train_data"])
    assert_true(abs(actual_train - expected_train) <= 2,
                f"train size {actual_train} far from expected {expected_train}")
    print(f"  PASS test_train_fraction  (expected~{expected_train}, got={actual_train})")

def test_seed_reproducibility():
    ds1 = make_modular_addition(p=7, seed=0)
    ds2 = make_modular_addition(p=7, seed=0)
    assert_equal(ds1["train_data"], ds2["train_data"], "seed not reproducible")
    ds3 = make_modular_addition(p=7, seed=1)
    assert_true(ds1["train_data"] != ds3["train_data"], "different seeds gave same split")
    print("  PASS test_seed_reproducibility")


def test_s4_dataset():
    """FIX v3.0: S4 tests were missing entirely from the test suite."""
    ds = make_s4_group()
    _check_dataset(ds, 24)
    total = len(ds["train_data"]) + len(ds["test_data"])
    assert_equal(total, 576, "S4 should have 576 pairs")
   
    lookup = {(a, b): c for a, b, c in ds["train_data"] + ds["test_data"]}
    for a in range(24):
        for b in range(24):
            assert_equal(lookup[(a, b)], _S4_TABLE[a][b], f"S4[{a}][{b}] mismatch")
    print(f"  PASS test_s4_dataset  (576 pairs)")

def test_s4_table_closure():
    for a in range(24):
        for b in range(24):
            c = _S4_TABLE[a][b]
            assert_true(0 <= c < 24, f"S4[{a}][{b}]={c} out of range")
    print("  PASS test_s4_table_closure")

def test_s4_table_identity():
    for a in range(24):
        assert_equal(_S4_TABLE[0][a], a, f"S4: e*{a}")
        assert_equal(_S4_TABLE[a][0], a, f"S4: {a}*e")
    print("  PASS test_s4_table_identity")

def test_s4_non_abelian():
    non_comm = [(a, b) for a in range(24) for b in range(24)
                if _S4_TABLE[a][b] != _S4_TABLE[b][a]]
    assert_true(len(non_comm) > 0, "S4 should be non-abelian")
    print(f"  PASS test_s4_non_abelian  ({len(non_comm)} non-commuting pairs)")

def test_complexity_scores():
    """FIX v3.0: Complexity scores missing from test suite."""
    from src.datasets import get_complexity_score
    scores = {op: get_complexity_score(op)
              for op in ("add", "sub", "mul", "ring_add", "s3", "d5", "a4", "s4")}
    
    order = ["add", "mul", "ring_add", "s3", "d5", "a4", "s4"]
    for i in range(len(order) - 1):
        op_a, op_b = order[i], order[i + 1]
        assert_true(scores[op_a] < scores[op_b],
                    f"Expected {op_a} complexity < {op_b}: {scores[op_a]} vs {scores[op_b]}")
    print(f"  PASS test_complexity_scores")




def run_all():
    tests = [
        test_is_prime,
        test_s3_table_closure, test_s3_table_identity,
        test_s3_table_inverses, test_s3_non_abelian,
        test_d5_table_closure, test_d5_table_identity,
        test_d5_table_inverses, test_d5_non_abelian, test_d5_associativity,
        test_a4_table_closure, test_a4_table_identity,
        test_a4_table_inverses, test_a4_non_abelian, test_a4_order,
        
        test_s4_table_closure, test_s4_table_identity, test_s4_non_abelian,
        test_modular_addition, test_modular_multiplication,
        test_modular_subtraction, test_ring_addition,
        test_s3_dataset, test_d5_dataset, test_a4_dataset,
        test_s4_dataset,        
        test_train_fraction, test_seed_reproducibility,
        test_complexity_scores,  
    ]
    failures = []
    print(f"\nRunning {len(tests)} tests...\n")
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{'='*55}")
    if failures:
        print(f"FAILED {len(failures)}/{len(tests)}")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        sys.exit(1)
    else:
        print(f"ALL {len(tests)} TESTS PASSED")

if __name__ == "__main__":
    run_all()
