"""CI smoke-test: MinimalTransformer forward pass with n_layers=2."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.train import MinimalTransformer, TrainConfig

# Test n_layers=1 (default)
cfg1 = TrainConfig(d_model=32, d_head=8, d_mlp=64, n_heads=2, n_layers=1)
m1 = MinimalTransformer(cfg1, vocab_size=8)
out1 = m1(torch.randint(0, 7, (4, 3)))
assert out1.shape == (4, 3, 8), f"n_layers=1 shape wrong: {out1.shape}"
print(f'  n_layers=1 forward pass OK: {out1.shape}')

# Test n_layers=2 — previously always built only 1 layer (BUG FIX)
cfg2 = TrainConfig(d_model=32, d_head=8, d_mlp=64, n_heads=2, n_layers=2)
m2 = MinimalTransformer(cfg2, vocab_size=8)
assert len(m2.blocks) == 2, f"Expected 2 blocks, got {len(m2.blocks)}"
out2 = m2(torch.randint(0, 7, (4, 3)))
assert out2.shape == (4, 3, 8), f"n_layers=2 shape wrong: {out2.shape}"
print(f'  n_layers=2 forward pass OK: {out2.shape}')

print('MinimalTransformer checks passed.')
