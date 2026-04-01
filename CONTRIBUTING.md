# Contributing

Thank you for your interest in contributing to this project.

## Setting up

```bash
git clone https://github.com/justbytecode/grokking-beyond-addition.git
cd grokking-beyond-addition
pip install -r requirements.txt
```

## Running tests

The CI workflow (`.github/workflows/ci.yml`) runs automatically on every push.
To run locally:

```bash
python -m py_compile src/datasets.py src/train.py src/analysis.py src/visualise.py
python src/datasets.py   
```

## Adding a new algebraic operation

1. Add a `make_<op>()` function in `src/datasets.py` following the existing pattern.
2. Register it in `_OP_MAP` in `src/train.py`.
3. Add its label and colour in `src/visualise.py` (`_OP_LABELS`, `_OP_COLORS`).
4. Create a notebook in `notebooks/` with training + analysis.

## Code style

- Type-annotated Python 3.9+
- Docstrings on all public functions
- No bare `except:` clauses — always catch specific exceptions

## Opening issues

Please include:
- Python version and OS
- Full traceback
- Minimal reproducible example
