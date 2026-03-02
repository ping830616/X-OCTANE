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

# === Generate BEST_in_DesignSpace_Post_per_platform.csv (PER PLATFORM, not per testcase) ===
# Input:  Results/per_run_metrics_all_PIPELINE.csv  (from your new pipeline)
# Output: Results/BEST_in_DesignSpace_Post_per_platform.csv
#         Results/BEST_in_DesignSpace_Post_per_platform_details.csv
#
# What "per platform" means here:
#   For each platform (DDR4, DDR5), pick ONE (WIN, K) that is best on average across that
#   platform's anomalies, where each anomaly contributes its BEST (method, pct) under that (WIN,K)
#   using PR-first (median AUCPR), tie ROC, tie smaller pct.
#
# Notes:
# - Uses medians across run_id (so it's stable).
# - Uses anomalies set:
#     DDR4: DROOP, RH
#     DDR5: DROOP, SPECTRE
# - Uses method order as stable tie-break: dC_aJ, dC_aM, dE_aJ, dE_aM
# --------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from pathlib import Path

RES_DIR = ROOT / "Results"
PER_RUN = RES_DIR / "per_run_metrics_all_PIPELINE.csv"

if not PER_RUN.exists():
    raise FileNotFoundError(f"Missing pipeline per-run CSV: {PER_RUN}")

df = pd.read_csv(PER_RUN).copy()

ANOMALIES_BY_SETUP = {"DDR4": ["DROOP","RH"], "DDR5": ["DROOP","SPECTRE"]}
METHOD_ORDER = ["dC_aJ", "dC_aM", "dE_aJ", "dE_aM"]

# Defensive grid (match your pipeline)
valid_wins   = {32, 64, 128, 512, 1024}
valid_kfolds = {3, 5, 10}

# Basic cleaning / typing
df = df[df["setup"].isin(["DDR4","DDR5"])].copy()
df = df[df["win"].isin(valid_wins) & df["kfold"].isin(valid_kfolds)].copy()
df = df[df["method"].isin(METHOD_ORDER)].copy()

for c in ["win","kfold","pct","auc_pr","roc_auc"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["setup","anomaly","win","kfold","pct"]).copy()
df["auc_pr"]  = pd.to_numeric(df["auc_pr"], errors="coerce").clip(0.0, 1.0)
df["roc_auc"] = pd.to_numeric(df["roc_auc"], errors="coerce").clip(0.0, 1.0)

# -------------------------------------------------------------------
# 1) Summarize per (setup, anomaly, win, kfold, method, pct) over run_id
# -------------------------------------------------------------------
def q1(x): return float(np.nanpercentile(x, 25))
def q3(x): return float(np.nanpercentile(x, 75))

sum_keys = ["setup","anomaly","win","kfold","method","pct"]
sumdf = (df.groupby(sum_keys, as_index=False)
           .agg(
               auc_pr_median=("auc_pr","median"),
               roc_auc_median=("roc_auc","median"),
               auc_pr_q1=("auc_pr", q1),
               auc_pr_q3=("auc_pr", q3),
               roc_auc_q1=("roc_auc", q1),
               roc_auc_q3=("roc_auc", q3),
               auc_pr_min=("auc_pr","min"),
               auc_pr_max=("auc_pr","max"),
               roc_auc_min=("roc_auc","min"),
               roc_auc_max=("roc_auc","max"),
               n_runs=("run_id","nunique"),
           ))
sumdf["auc_pr_iqr"]  = sumdf["auc_pr_q3"]  - sumdf["auc_pr_q1"]
sumdf["roc_auc_iqr"] = sumdf["roc_auc_q3"] - sumdf["roc_auc_q1"]

sumdf["auc_pr_median_filled"]  = sumdf["auc_pr_median"].fillna(-np.inf)
sumdf["roc_auc_median_filled"] = sumdf["roc_auc_median"].fillna(-np.inf)

# Keep only defined anomalies
def _keep_defined(row):
    setup = str(row["setup"]).upper()
    anom  = str(row["anomaly"]).upper()
    return setup in ANOMALIES_BY_SETUP and anom in [a.upper() for a in ANOMALIES_BY_SETUP[setup]]

sumdf = sumdf[sumdf.apply(_keep_defined, axis=1)].copy()

# -------------------------------------------------------------------
# 2) For each (setup, anomaly, win, kfold, method): pick BEST pct
#     PR-first, ROC-second, smaller pct
# -------------------------------------------------------------------
best_pct = (sumdf.sort_values(
                ["setup","anomaly","win","kfold","method",
                 "auc_pr_median_filled","roc_auc_median_filled","pct"],
                ascending=[True,True,True,True,True, False,False, True]
            )
            .groupby(["setup","anomaly","win","kfold","method"], as_index=False)
            .head(1)
            .rename(columns={"pct":"best_pct_by_median"}))

# -------------------------------------------------------------------
# 3) For each (setup, anomaly, win, kfold): pick BEST method (using its BEST pct)
#     PR-first, ROC-second, stable method order, then smaller pct
# -------------------------------------------------------------------
method_rank = {m:i for i,m in enumerate(METHOD_ORDER)}
best_pct["_mrank"] = best_pct["method"].map(method_rank).fillna(999).astype(int)

best_method = (best_pct.sort_values(
                    ["setup","anomaly","win","kfold",
                     "auc_pr_median_filled","roc_auc_median_filled","_mrank","best_pct_by_median"],
                    ascending=[True,True,True,True, False,False, True, True]
               )
               .groupby(["setup","anomaly","win","kfold"], as_index=False)
               .head(1))

# -------------------------------------------------------------------
# 4) For each platform, pick ONE (win,kfold) by averaging anomaly-wise best
#     Prefer combos that cover more anomalies, then mean AUCPR, mean ROC,
#     then smaller win, smaller kfold
# -------------------------------------------------------------------
platform_rows = []
detail_rows = []

print("\n[SCAN] Choosing ONE (WIN,K) per platform (mean AUCPR across anomalies)")

for setup in ["DDR4","DDR5"]:
    sub = best_method[best_method["setup"].astype(str).str.upper()==setup].copy()
    if sub.empty:
        print(f"  [WARN] No rows for platform={setup}")
        continue

    agg = (sub.groupby(["setup","win","kfold"], as_index=False)
             .agg(mean_auc_pr=("auc_pr_median_filled","mean"),
                  mean_roc_auc=("roc_auc_median_filled","mean"),
                  n_anoms=("anomaly","nunique")))

    agg = agg.sort_values(
        ["n_anoms","mean_auc_pr","mean_roc_auc","win","kfold"],
        ascending=[False, False, False, True, True]
    )
    top = agg.iloc[0]
    win_best = int(top["win"])
    kf_best  = int(top["kfold"])

    platform_rows.append({
        "setup": setup,
        "win": win_best,
        "kfold": kf_best,
        "best_pct_by_median": np.nan,  # not single pct at platform-level; varies by anomaly
        "method": "BEST_PER_ANOMALY",  # varies by anomaly
        "auc_pr_median": float(top["mean_auc_pr"]),
        "roc_auc_median": float(top["mean_roc_auc"]),
        "auc_pr_iqr": np.nan,
        "roc_auc_iqr": np.nan,
        "auc_pr_min": np.nan,
        "auc_pr_max": np.nan,
        "roc_auc_min": np.nan,
        "roc_auc_max": np.nan,
        "num_anomalies_covered": int(top["n_anoms"]),
        "selection_note": "mean across anomalies; each anomaly uses its best (method,pct) under (win,kfold)",
    })

    chosen = sub[(sub["win"]==win_best) & (sub["kfold"]==kf_best)].copy()
    chosen = chosen.sort_values(
        ["auc_pr_median_filled","roc_auc_median_filled","best_pct_by_median"],
        ascending=[False, False, True]
    )

    print(f"  [PLATFORM BEST • {setup}] WIN={win_best} K={kf_best} "
          f"mean AUCPR={float(top['mean_auc_pr']):.3f} mean ROC={float(top['mean_roc_auc']):.3f} "
          f"(anoms covered={int(top['n_anoms'])})")

    for _, r in chosen.iterrows():
        detail_rows.append({
            "setup": setup,
            "anomaly": str(r["anomaly"]),
            "win": win_best,
            "kfold": kf_best,
            "best_pct_by_median": int(r["best_pct_by_median"]),
            "method": str(r["method"]),
            "auc_pr_median": float(r["auc_pr_median"]),
            "roc_auc_median": float(r["roc_auc_median"]),
            "auc_pr_iqr": float(r["auc_pr_iqr"]),
            "roc_auc_iqr": float(r["roc_auc_iqr"]),
            "auc_pr_min": float(r["auc_pr_min"]),
            "auc_pr_max": float(r["auc_pr_max"]),
            "roc_auc_min": float(r["roc_auc_min"]),
            "roc_auc_max": float(r["roc_auc_max"]),
            "n_runs": int(r["n_runs"]),
        })
        print(f"      {setup}/{r['anomaly']}: method={r['method']}  best %={int(r['best_pct_by_median'])}  "
              f"AUCPR_med={float(r['auc_pr_median']):.3f} ROC_med={float(r['roc_auc_median']):.3f}")

# -------------------------------------------------------------------
# 5) Save outputs
# -------------------------------------------------------------------
platform_best = pd.DataFrame(platform_rows).sort_values(["setup"]).reset_index(drop=True)
details = pd.DataFrame(detail_rows).sort_values(["setup","anomaly"]).reset_index(drop=True)

OUT_PLATFORM = RES_DIR / "BEST_in_DesignSpace_Post_per_platform.csv"
OUT_DETAILS  = RES_DIR / "BEST_in_DesignSpace_Post_per_platform_details.csv"

platform_best.to_csv(OUT_PLATFORM, index=False)
details.to_csv(OUT_DETAILS, index=False)

print(f"\n[OK] Wrote per-platform winners → {OUT_PLATFORM}")
print(f"[OK] Wrote per-platform anomaly details → {OUT_DETAILS}")

print("\n[PLATFORM WINNERS]")
print(platform_best[["setup","win","kfold","auc_pr_median","roc_auc_median","num_anomalies_covered"]].to_string(index=False))