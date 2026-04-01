"""CI smoke-test: import and validate all dataset generators."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.datasets import (
    make_modular_addition, make_modular_multiplication,
    make_modular_subtraction, make_ring_addition,
    make_s3_group, make_d5_group, make_a4_group,
    make_s4_group,   # FIX v3.0: was missing from CI check
)

for fn, kwargs in [
    (make_modular_addition,       {'p': 7}),
    (make_modular_multiplication, {'p': 7}),
    (make_modular_subtraction,    {'p': 7}),
    (make_ring_addition,          {'n': 6}),
    (make_s3_group,               {}),
    (make_d5_group,               {}),
    (make_a4_group,               {}),
    (make_s4_group,               {}),   # FIX v3.0: added
]:
    ds = fn(**kwargs)
    assert len(ds['train_data']) > 0
    assert len(ds['test_data'])  > 0
    p = ds['p']
    for a, b, c in ds['train_data']:
        assert 0 <= c < p, f"Label {c} out of range [0, {p})"
    print(f'  OK: {ds["op_name"]}')

print('All dataset checks passed.')
