# Auto-generated from XOCTANE.ipynb (extracted).
# Patched for GitHub reproducibility: uses env vars instead of local absolute paths.
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('XOCTANE_ROOT', str(REPO_ROOT))).resolve()
DATA_DIR_ENV = os.environ.get('XOCTANE_DATA_DIR', '')
if not DATA_DIR_ENV:
    raise SystemExit('Please set XOCTANE_DATA_DIR to the dataset root directory (see README).')
DATA_DIR = Path(DATA_DIR_ENV).resolve()

# Default output directories (created if missing)
RES_DIR = ROOT / 'Results'; RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT / 'figures'; FIG_DIR.mkdir(parents=True, exist_ok=True)

# === Aggregates: subspace-level & PLATFORM×subspace (percentages-only run) ===
# UPDATED for PER-PLATFORM workflow.
#
# Reads:
#   Results/Explainability_SHAP_BestPlatforms/SHAP_vs_CPMI_summary_PLATFORM.csv
#
# Writes:
#   Results/Explainability_SHAP_BestPlatforms/
#     - SHAP_vs_CPMI_aggregate_corr_by_subspace_PLATFORM.csv
#     - SHAP_vs_CPMI_aggregate_corr_by_platform_subspace.csv
#     - SHAP_vs_CPMI_aggregate_jaccard_by_subspace_k_PLATFORM.csv
#     - SHAP_vs_CPMI_aggregate_jaccard_by_subspace_pivot_PLATFORM.csv
#     - SHAP_vs_CPMI_aggregate_jaccard_by_platform_subspace_k.csv
#     - SHAP_vs_CPMI_aggregate_jaccard_by_platform_subspace_pivot.csv
# -------------------------------------------------------------------------------------------

from pathlib import Path
import pandas as pd


BASE_RES = ROOT / "Results"
OUT_DIR  = BASE_RES / "Explainability_SHAP_BestPlatforms"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _find_summary_csv() -> Path:
    fname = "SHAP_vs_CPMI_summary_PLATFORM.csv"
    candidates = [
        OUT_DIR / fname,
        BASE_RES / fname,
        # if someone nested folders accidentally
        OUT_DIR / "Explainability_SHAP_BestPlatforms" / fname,
        BASE_RES / "Explainability_SHAP_BestPlatforms" / fname,
    ]
    for p in candidates:
        if p.exists():
            return p
    hits = list(BASE_RES.rglob(fname))
    if hits:
        return hits[0]
    raise FileNotFoundError(
        f"Could not find {fname}. Tried:\n  - " +
        "\n  - ".join(str(x) for x in candidates) +
        f"\nAlso searched under: {BASE_RES}"
    )

summary_path = _find_summary_csv()
print("[OK] Using summary:", summary_path)

df = pd.read_csv(summary_path)

# Expected per-platform summary schema:
#   setup, win, kfold, subspace, k, jaccard, overlap, rho_rank, rho_prank, rho_score, aligned_features
required = {"setup","win","kfold","subspace","k","jaccard","overlap","rho_rank","rho_prank","rho_score","aligned_features"}
missing = required - set(df.columns)
if missing:
    raise KeyError(f"Summary missing required columns: {sorted(missing)}. Have: {list(df.columns)}")

# Keys for a unique "platform case"
case_keys = ["setup","win","kfold","subspace"]

# --- Correlations aggregated per subspace (one row per platform/subspace) ---
corr_cols = ["rho_rank","rho_prank","rho_score","aligned_features"]

corr_per_case = df[case_keys + corr_cols].drop_duplicates(case_keys).reset_index(drop=True)

# n_cases per subspace (across platforms; typically 2 platforms => n_cases=2)
n_cases = (
    corr_per_case.groupby("subspace")
    .size()
    .rename("n_cases")
    .reset_index()
)

agg_corr_sub = (
    corr_per_case.groupby("subspace")[corr_cols]
    .median(numeric_only=True)
    .reset_index()
)
agg_corr_sub = agg_corr_sub.merge(n_cases, on="subspace", how="left")
agg_corr_sub.to_csv(OUT_DIR / "SHAP_vs_CPMI_aggregate_corr_by_subspace_PLATFORM.csv", index=False)

# Correlations per PLATFORM × subspace
agg_corr_plat_sub = (
    corr_per_case.groupby(["setup","subspace"])[corr_cols]
    .median(numeric_only=True)
    .reset_index()
)
agg_corr_plat_sub.to_csv(OUT_DIR / "SHAP_vs_CPMI_aggregate_corr_by_platform_subspace.csv", index=False)

# --- Jaccard aggregated per subspace at each percentage threshold ---
jacc_sub_k = (
    df.groupby(["subspace","k"])["jaccard"]
      .median()
      .reset_index()
      .sort_values(["subspace","k"])
)
jacc_sub_k.to_csv(OUT_DIR / "SHAP_vs_CPMI_aggregate_jaccard_by_subspace_k_PLATFORM.csv", index=False)

# Pivot for a compact table: subspace × selected thresholds
take_pcts = ["top10%","top25%","top50%","top100%"]
available = [c for c in take_pcts if c in set(jacc_sub_k["k"].astype(str).unique())]

jacc_pivot = (
    jacc_sub_k.pivot(index="subspace", columns="k", values="jaccard")
      .reindex(columns=available)
      .reset_index()
)
jacc_pivot.to_csv(OUT_DIR / "SHAP_vs_CPMI_aggregate_jaccard_by_subspace_pivot_PLATFORM.csv", index=False)

# --- Jaccard per PLATFORM × subspace at each percentage threshold ---
jacc_plat_sub_k = (
    df.groupby(["setup","subspace","k"])["jaccard"]
      .median()
      .reset_index()
      .sort_values(["setup","subspace","k"])
)
jacc_plat_sub_k.to_csv(OUT_DIR / "SHAP_vs_CPMI_aggregate_jaccard_by_platform_subspace_k.csv", index=False)

available2 = [c for c in take_pcts if c in set(jacc_plat_sub_k["k"].astype(str).unique())]
jacc_plat_sub_pivot = (
    jacc_plat_sub_k.pivot(index=["setup","subspace"], columns="k", values="jaccard")
      .reindex(columns=available2)
      .reset_index()
)
jacc_plat_sub_pivot.to_csv(OUT_DIR / "SHAP_vs_CPMI_aggregate_jaccard_by_platform_subspace_pivot.csv", index=False)

print("[OK] Aggregates →",
      OUT_DIR / "SHAP_vs_CPMI_aggregate_corr_by_subspace_PLATFORM.csv", ",",
      OUT_DIR / "SHAP_vs_CPMI_aggregate_corr_by_platform_subspace.csv", ",",
      OUT_DIR / "SHAP_vs_CPMI_aggregate_jaccard_by_subspace_k_PLATFORM.csv", ",",
      OUT_DIR / "SHAP_vs_CPMI_aggregate_jaccard_by_subspace_pivot_PLATFORM.csv", ",",
      OUT_DIR / "SHAP_vs_CPMI_aggregate_jaccard_by_platform_subspace_k.csv", ",",
      OUT_DIR / "SHAP_vs_CPMI_aggregate_jaccard_by_platform_subspace_pivot.csv")