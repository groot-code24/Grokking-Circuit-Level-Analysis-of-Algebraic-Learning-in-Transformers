#!/usr/bin/env bash
# Compile the LaTeX paper to PDF.
# Usage: cd paper && bash compile.sh
set -euo pipefail

cd "$(dirname "$0")"

echo "Compiling paper..."
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo "Done — main.pdf"
