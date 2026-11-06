#!/bin/bash
# Regenerate every table and figure of the paper from results/ (the released result files must be in
# results/raw/final/), then build paper/paper.pdf. Analysis only; CPU, a few minutes.
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg JAX_PLATFORMS=cpu
python paper/scripts/rbf_figures.py              # Section 3 figures (needs data/reference_posterior/)
python paper/scripts/section6.py                 # Tables 3-5, 8, 9, 12; Figures 5-7
python paper/scripts/appendix_tables.py          # Tables 6, 7, 10, 11, 13-15
python paper/scripts/temperature_search_plot.py  # Figure 8
python paper/scripts/diagnostics_table.py        # Table 21
if ls results/raw/sweep_* >/dev/null 2>&1; then  # Tables 18-20: the earlier 3-member runs (not released)
    python legacy/results4_scale.py; python legacy/results4_shape.py; python legacy/results4_side.py
else
    echo "results/raw/sweep_* not found: keeping the committed Appendix G tables (results4_*.tex)"
fi
cd paper
pdflatex -interaction=nonstopmode paper.tex >/dev/null
bibtex paper >/dev/null
pdflatex -interaction=nonstopmode paper.tex >/dev/null
pdflatex -interaction=nonstopmode paper.tex >/dev/null
echo "built paper/paper.pdf"
