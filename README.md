# X-OCTANE
## Hybrid CP–MI + SHAP Feature Ranking for Telemetry-Based Anomaly Detection

This repository provides a reproducible experimental pipeline for **X-OCTANE**, 
a hybrid feature ranking framework combining:

- Causal-Preserving Mutual Information (CP–MI)
- SHAP feature attributions

The repository reproduces:

- CP–MI ranking
- SHAP ranking
- Hybrid CP–MI + SHAP ranking (α = 0.5)
- Percentile-rank concordance (CP–MI vs SHAP)
- Jaccard stability analysis
- Top-p% feature retention sweep
- Design Space Exploration (WIN × K × p)
- ROC-AUC and AUC-PR evaluation
- Platform-level WIN/K selection

This repository reflects exactly what is implemented in the codebase.

---

## Quick Start

### 1) Environment

Conda:
```bash
conda env create -f environment.yml
conda activate xoctane
```

Pip:
```bash
pip install -r requirements.txt
```

---

### 2) Dataset Location

Set the dataset root directory using an environment variable:

```bash
export XOCTANE_DATA_DIR=/path/to/TELEMETRY_DATA
```

Example (original development machine path):
```bash
export XOCTANE_DATA_DIR=/Users/hsiaopingni/Desktop/SLM_RAS-main/HW_TELEMETRY_DATA_COLLECTION/TELEMETRY_DATA
```

Do **not** hardcode this path inside scripts. See `docs/DATA.md` for expected folder structure.

---

### 3) Run All Experiments

```bash
python scripts/validate_env.py
bash scripts/reproduce_all.sh
```

Outputs:
- `Results/`
- `FeatureRankOUT/`
- `Explainability_SHAP/`
- `figures/`

---

## Default Configurations

### Setup A (DDR4)
- WIN = 512
- K = 3

### Setup B (DDR5)
- WIN = 1024
- K = 5

### Best (WIN, K) Per Platform Criterion

When multiple (WIN, K) pairs are evaluated (e.g., during DSE), the best (WIN, K) per platform is selected by:

computing AUC-PR for each anomaly on that platform

averaging AUC-PR across anomalies for the platform

choosing the (WIN, K) with the highest mean AUC-PR

DSE sweep:
- WIN ∈ {32, 64, 128, 512, 1024}
- K ∈ {3, 5, 10}
- p ∈ {10, 20, ..., 100}

---

## Documentation

- `docs/REPRODUCIBILITY.md`
- `docs/DATA.md`
- `docs/FIGURE_INDEX.md`
