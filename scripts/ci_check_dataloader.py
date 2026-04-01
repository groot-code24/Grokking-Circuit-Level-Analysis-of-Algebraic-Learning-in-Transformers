"""CI smoke-test: DataLoader construction and iteration."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.datasets import make_modular_addition, build_loaders

ds = make_modular_addition(p=7)
train_loader, test_loader = build_loaders(ds)
tokens, labels = next(iter(train_loader))
assert tokens.shape[1] == 3, f"Expected 3 tokens, got {tokens.shape[1]}"
print(f'  DataLoader OK, token shape: {tokens.shape}')
print('DataLoader checks passed.')
