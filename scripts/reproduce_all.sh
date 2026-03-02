#!/usr/bin/env bash
set -euo pipefail

# One-command reproduction driver.
# Requires:
#   export XOCTANE_DATA_DIR=/path/to/data
# Optional:
#   export XOCTANE_ROOT=/path/to/repo/root (defaults to repo root)

echo "[1/4] Running full multi-WIN/multi-K pipeline (metrics + DSE artifacts)..."
python scripts/01_full_multiwin_multik_pipeline.py

echo "[2/4] Selecting BEST (WIN,K) per platform..."
python scripts/02_best_per_platform.py

echo "[3/4] Generating SHAP artifacts for BEST platform cases..."
python scripts/03_shap_best_platforms.py
python scripts/04_shap_top10_platform.py

echo "[4/4] Generating concordance / stability figures (Fig.9-style etc.)..."
python scripts/07_fig9_percentile_concordance.py
python scripts/05_shap_vs_cpmi_stability_concordance.py

echo "[DONE] X-OCTANE reproduction finished. See Results/ and figures/."
