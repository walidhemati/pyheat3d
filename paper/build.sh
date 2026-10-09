#!/usr/bin/env bash
# Build paper/paper.pdf. Regenerate figures and results.tex first with figures/make_figures.py.
set -euo pipefail
cd "$(dirname "$0")"
tectonic paper.tex
