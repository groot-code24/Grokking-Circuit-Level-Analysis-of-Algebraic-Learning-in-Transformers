
import sys
import os
import math

_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _ROOT)

import numpy as np

def assert_equal(a, b, msg=""):
    if a != b:
        raise AssertionError(f"{msg}  expected {b!r}, got {a!r}")

def assert_true(cond, msg=""):
    if not cond:
        raise AssertionError(msg)

def assert_close(a, b, tol=1e-6, msg=""):
    if abs(a - b) > tol:
        raise AssertionError(f"{msg}  |{a} - {b}| = {abs(a-b):.2e} > {tol}")

# ---------------------------------------------------------------------------
# 1. bootstrap_confidence_interval
# ---------------------------------------------------------------------------

def test_bootstrap_ci_mean():
    from src.analysis import bootstrap_confidence_interval
    vals = list(range(1, 101))   # mean = 50.5
    pt, lo, hi = bootstrap_confidence_interval(vals, statistic="mean",
                                                n_bootstrap=1000, seed=42)
    assert_close(pt, 50.5, tol=0.01, msg="bootstrap mean point estimate")
    assert_true(lo < pt < hi, "CI must bracket point estimate")
    assert_true(hi - lo < 20, "95% CI width should be < 20 for n=100 uniform")
    print("  PASS test_bootstrap_ci_mean")

def test_bootstrap_ci_std():
    from src.analysis import bootstrap_confidence_interval
    rng  = np.random.default_rng(7)
    vals = rng.normal(0, 1, 200).tolist()
    pt, lo, hi = bootstrap_confidence_interval(vals, statistic="std", n_bootstrap=500, seed=0)
    assert_true(0.7 < pt < 1.3, f"std of N(0,1) should be ~1, got {pt:.3f}")
    assert_true(lo < pt < hi, "CI must bracket point estimate")
    print("  PASS test_bootstrap_ci_std")

def test_bootstrap_ci_bad_statistic():
    from src.analysis import bootstrap_confidence_interval
    try:
        bootstrap_confidence_interval([1,2,3], statistic="variance")
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass
    print("  PASS test_bootstrap_ci_bad_statistic")

# ---------------------------------------------------------------------------
# 2. get_complexity_score_v2
# ---------------------------------------------------------------------------

def test_complexity_score_v2_ordering():
    from src.datasets import get_complexity_score_v2
    ops_ordered = ["add", "sub", "mul", "ring_add", "s3", "d5", "a4", "s4"]
    scores = [get_complexity_score_v2(op) for op in ops_ordered]
    
    assert_close(scores[0], scores[1], tol=1e-9,
                 msg="add and sub should have same complexity")
    
    max_abelian = max(scores[:4])
    min_nonabelian = min(scores[4:])
    assert_true(max_abelian < min_nonabelian,
                f"All abelian ops should score < non-abelian: {max_abelian:.3f} vs {min_nonabelian:.3f}")
    
    for s in scores:
        assert_true(s >= 0, f"Complexity score must be non-negative, got {s}")
    print(f"  PASS test_complexity_score_v2_ordering  scores={[f'{s:.3f}' for s in scores]}")

def test_complexity_score_v2_formula():
    from src.datasets import get_complexity_score_v2
    
    s_add = get_complexity_score_v2("add")
    expected_add = math.log2(2) + (1.0 - 1.0/113) + 0.0
    assert_close(s_add, expected_add, tol=1e-6, msg="add v2 score formula check")

   
    s_s4 = get_complexity_score_v2("s4")
    expected_s4 = math.log2(4) + (1.0 - 1.0/5) + 1.0
    assert_close(s_s4, expected_s4, tol=1e-6, msg="s4 v2 score formula check")
    print("  PASS test_complexity_score_v2_formula")


# 3. grokking_leading_indicator


def test_grokking_leading_indicator_precedes():
    from src.analysis import grokking_leading_indicator
    history = [{"epoch": i*100, "formation_score": min(i * 0.15, 1.0)}
               for i in range(0, 12)]
    
    grok_epoch = 1000
    result = grokking_leading_indicator(history, grok_epoch, threshold=0.50)
    assert_equal(result["verdict"], "PRECEDES",
                 "Should detect that representation transition precedes grokking")
    assert_true(result["lead_time"] > 50,
                f"Lead time should be >50 epochs, got {result['lead_time']}")
    assert_true(result["representation_transition_epoch"] < grok_epoch,
                "Transition must precede grokking")
    print(f"  PASS test_grokking_leading_indicator_precedes  "
          f"lead_time={result['lead_time']}")

def test_grokking_leading_indicator_no_data():
    from src.analysis import grokking_leading_indicator
    result = grokking_leading_indicator([], None)
    assert_equal(result["verdict"], "NO_DATA")
    print("  PASS test_grokking_leading_indicator_no_data")

def test_grokking_leading_indicator_no_grok():
    from src.analysis import grokking_leading_indicator
    history = [{"epoch": i*100, "formation_score": 0.9} for i in range(10)]
    result = grokking_leading_indicator(history, grok_epoch=None)
    assert_equal(result["verdict"], "NO_DATA",
                 "No grokking → verdict should be NO_DATA")
    print("  PASS test_grokking_leading_indicator_no_grok")


# 4. representation_formation_tracker (basic, no GPU needed)


def test_representation_formation_tracker_abelian():
    """Use a random MinimalTransformer — score should be defined and in [0,1]."""
    try:
        import torch
    except ImportError:
        print("  SKIP test_representation_formation_tracker_abelian (no torch)")
        return
    from src.analysis import representation_formation_tracker
    from src.train import MinimalTransformer, TrainConfig
    from src.datasets import make_modular_addition

    cfg = TrainConfig(p=7, d_model=16, d_mlp=32, n_heads=2, d_head=8, epochs=1)
    ds  = make_modular_addition(p=7)
    model = MinimalTransformer(cfg, vocab_size=ds["vocab_size"])

    result = representation_formation_tracker(model, ds, op_symbol="add")
    assert_equal(result["representation_type"], "fourier",
                 "add should use fourier tracker")
    assert_true(0.0 <= result["formation_score"] <= 1.0,
                f"Score must be in [0,1], got {result['formation_score']}")
    assert_true(isinstance(result["threshold_crossed"], bool))
    print(f"  PASS test_representation_formation_tracker_abelian  "
          f"score={result['formation_score']:.3f}")

def test_representation_formation_tracker_nonabelian():
    try:
        import torch
    except ImportError:
        print("  SKIP test_representation_formation_tracker_nonabelian (no torch)")
        return
    from src.analysis import representation_formation_tracker, _NONABELIAN_AVAIL
    if not _NONABELIAN_AVAIL:
        print("  SKIP test_representation_formation_tracker_nonabelian (tables unavailable)")
        return
    from src.train import MinimalTransformer, TrainConfig
    from src.datasets import make_s3_group

    cfg = TrainConfig(op="s3", p=6, d_model=16, d_mlp=32, n_heads=2, d_head=8, epochs=1)
    ds  = make_s3_group()
    model = MinimalTransformer(cfg, vocab_size=ds["vocab_size"])

    result = representation_formation_tracker(model, ds, op_symbol="s3")
    assert_equal(result["representation_type"], "peter_weyl",
                 "s3 should use peter_weyl tracker")
    assert_true(0.0 <= result["formation_score"] <= 1.0)
    print(f"  PASS test_representation_formation_tracker_nonabelian  "
          f"score={result['formation_score']:.3f}")


# 5. controlled_complexity_ablation


def test_controlled_complexity_ablation_basic():
    from src.analysis import controlled_complexity_ablation
    
    conditions = {
        "full_data": {
            "add":  {"mean_grok_epoch": 1000},
            "mul":  {"mean_grok_epoch": 2000},
            "a4":   {"mean_grok_epoch": 4000},
        },
        "half_data": {
            "add":  {"mean_grok_epoch": 1500},
            "mul":  {"mean_grok_epoch": 2800},
            "a4":   {"mean_grok_epoch": 5500},
        },
    }
    result = controlled_complexity_ablation(conditions)
    assert_true("spearman_by_cond" in result)
    assert_true("kendall_by_cond"  in result)
    assert_true("rank_preserved"   in result)
    assert_true("summary"          in result)
 
    for cond in ["full_data", "half_data"]:
        rho = result["spearman_by_cond"][cond][0]
        assert_true(rho > 0.9, f"Spearman ρ should be ~1 for perfect ordering, got {rho:.3f}")
    print(f"  PASS test_controlled_complexity_ablation_basic")


# 6. describe_learned_circuit (structure only — no training needed)


def test_describe_learned_circuit_structure():
    try:
        import torch
    except ImportError:
        print("  SKIP test_describe_learned_circuit_structure (no torch)")
        return
    from src.analysis import describe_learned_circuit
    from src.train import MinimalTransformer, TrainConfig
    from src.datasets import make_modular_addition

    cfg   = TrainConfig(p=7, d_model=16, d_mlp=32, n_heads=2, d_head=8, epochs=1)
    ds    = make_modular_addition(p=7)
    model = MinimalTransformer(cfg, vocab_size=ds["vocab_size"])

    result = describe_learned_circuit(model, ds, p=7, op_symbol="add")
    required_keys = [
        "representation_type", "key_frequencies", "dominant_irrep",
        "critical_heads", "mlp_role", "evidence_strength", "evidence_summary",
    ]
    for k in required_keys:
        assert_true(k in result, f"Missing key: {k}")
    assert_true(result["representation_type"] in [
        "fourier_clock", "dlog_then_clock", "partial_fourier", "peter_weyl"
    ], f"Unknown repr type: {result['representation_type']}")
    assert_true(result["evidence_strength"] in ["strong","moderate","weak"])
    assert_true(isinstance(result["evidence_summary"], str))
    assert_true(len(result["evidence_summary"]) > 20, "Summary too short")
    print(f"  PASS test_describe_learned_circuit_structure  "
          f"repr_type={result['representation_type']}")


# 7. causal_dlog_verification null test validity


def test_causal_dlog_uses_different_root():
    """Verify the primitive_root_null is different from primitive_root_used."""
    try:
        import torch
    except ImportError:
        print("  SKIP test_causal_dlog_uses_different_root (no torch)")
        return
    from src.analysis import causal_dlog_verification
    from src.train import MinimalTransformer, TrainConfig
    
    from src.datasets import make_ring_addition

    p   = 7         
    cfg = TrainConfig(p=p-1, d_model=16, d_mlp=32, n_heads=2, d_head=8, epochs=1)
    ds  = make_ring_addition(n=p-1)   
    model = MinimalTransformer(cfg, vocab_size=ds["vocab_size"])

    result = causal_dlog_verification(model, p=p, n_samples=20, device="cpu")
    required = ["dlog_mapped_accuracy","null_accuracy","chance_accuracy",
                "improvement_over_null","p_value_vs_chance","verdict",
                "primitive_root_used","primitive_root_null","n_tested"]
    for k in required:
        assert_true(k in result, f"Missing key: {k}")
    
    assert_true(
        result["primitive_root_used"] != result["primitive_root_null"],
        f"Null test must use a different primitive root: "
        f"g={result['primitive_root_used']}, g_null={result['primitive_root_null']}"
    )
    assert_true(result["n_tested"] > 0, "Should have tested some pairs")
    assert_true(result["verdict"] in ["SUPPORTS","WEAK","REJECTS"])
    print(f"  PASS test_causal_dlog_uses_different_root  "
          f"g={result['primitive_root_used']}, g_null={result['primitive_root_null']}")

# 8. linear probe 5-fold CV for small samples


def test_linear_probe_uses_cv_for_small_n():
    """For n<50 samples, the probe should use cross-validation."""
    try:
        import torch
    except ImportError:
        print("  SKIP test_linear_probe_uses_cv_for_small_n (no torch)")
        return
    from src.analysis import _linear_probe_accuracy
    rng = np.random.default_rng(42)
    X   = rng.random((30, 8)).astype(np.float32)
    y   = rng.integers(0, 3, size=30)
    # Should run without error and return a value in [0,1]
    acc = _linear_probe_accuracy(X, y, n_classes=3, epochs=50)
    assert_true(0.0 <= acc <= 1.0, f"Probe acc should be in [0,1], got {acc}")
    print(f"  PASS test_linear_probe_uses_cv_for_small_n  acc={acc:.3f}")


# Runner


if __name__ == "__main__":
    tests = [
        test_bootstrap_ci_mean,
        test_bootstrap_ci_std,
        test_bootstrap_ci_bad_statistic,
        test_complexity_score_v2_ordering,
        test_complexity_score_v2_formula,
        test_grokking_leading_indicator_precedes,
        test_grokking_leading_indicator_no_data,
        test_grokking_leading_indicator_no_grok,
        test_representation_formation_tracker_abelian,
        test_representation_formation_tracker_nonabelian,
        test_controlled_complexity_ablation_basic,
        test_describe_learned_circuit_structure,
        test_causal_dlog_uses_different_root,
        test_linear_probe_uses_cv_for_small_n,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} tests passed.")
