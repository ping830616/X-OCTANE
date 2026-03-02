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

# === Cell: Find BEST (win, kfold, pct) from NEW pipeline outputs ===========
import pathlib as pl
import pandas as pd
import numpy as np

RES_DIR = ROOT / "Results"

PER_RUN = RES_DIR / "per_run_metrics_all_PIPELINE.csv"
if not PER_RUN.exists():
    raise FileNotFoundError(f"Missing per-run CSV from pipeline: {PER_RUN}")

df = pd.read_csv(PER_RUN).copy()

# Defensive: keep only the grid you actually run
valid_wins   = {32, 64, 128, 512, 1024}
valid_kfolds = {3, 5, 10}
df = df[df["win"].isin(valid_wins) & df["kfold"].isin(valid_kfolds)].copy()

# Fill NaNs for robust sorting
df["auc_pr_filled"]  = df["auc_pr"].fillna(-np.inf)
df["roc_auc_filled"] = df["roc_auc"].fillna(-np.inf)

# ---------------------------------------------------------------------------
# 0) Build a SUMMARY table from per-run (median/IQR/min/max across run_id)
#    This replaces your old per_workload_summary...csv
# ---------------------------------------------------------------------------
def q1(x): return float(np.nanpercentile(x, 25))
def q3(x): return float(np.nanpercentile(x, 75))

# Summary per (setup, anomaly, win, kfold, method, pct)
sum_keys = ["setup","anomaly","win","kfold","method","pct"]
sumdf = (df.groupby(sum_keys, as_index=False)
           .agg(
               auc_pr_median=("auc_pr", "median"),
               auc_pr_q1=("auc_pr", q1),
               auc_pr_q3=("auc_pr", q3),
               auc_pr_min=("auc_pr", "min"),
               auc_pr_max=("auc_pr", "max"),
               roc_auc_median=("roc_auc", "median"),
               roc_auc_q1=("roc_auc", q1),
               roc_auc_q3=("roc_auc", q3),
               roc_auc_min=("roc_auc", "min"),
               roc_auc_max=("roc_auc", "max"),
               n_runs=("run_id","nunique"),
           ))

sumdf["auc_pr_iqr"]  = sumdf["auc_pr_q3"]  - sumdf["auc_pr_q1"]
sumdf["roc_auc_iqr"] = sumdf["roc_auc_q3"] - sumdf["roc_auc_q1"]

# For sorting
sumdf["auc_pr_median_filled"]  = sumdf["auc_pr_median"].fillna(-np.inf)
sumdf["roc_auc_median_filled"] = sumdf["roc_auc_median"].fillna(-np.inf)

# Save summary (optional but useful)
SUMMARY_OUT = RES_DIR / "per_case_summary_from_PIPELINE.csv"
sumdf.to_csv(SUMMARY_OUT, index=False)
print(f"[OK] Wrote summary derived from pipeline → {SUMMARY_OUT}")

ANOMALIES_BY_SETUP = {"DDR4": ["DROOP","RH"], "DDR5": ["DROOP","SPECTRE"]}
METHOD_ORDER = ["dC_aJ", "dC_aM", "dE_aJ", "dE_aM"]

# ---------------------------------------------------------------------------
# A) Best (win,kfold,pct) per TEST CASE per METHOD
#    (setup, anomaly, method)  across whole sweep
#    Sort: AUCPR_median desc, then ROC_median desc, then smaller pct
# ---------------------------------------------------------------------------
winners_rows = []
print("\n[SCAN-A] Best across all WIN×KF×PCT per (setup, anomaly, method)")
for setup, anomalies in ANOMALIES_BY_SETUP.items():
    for anomaly in anomalies:
        for method in METHOD_ORDER:
            sub = sumdf[
                (sumdf["setup"]==setup) &
                (sumdf["anomaly"]==anomaly) &
                (sumdf["method"]==method)
            ].copy()
            if sub.empty:
                print(f"  [WARN] No rows for {setup}–{anomaly}–{method}")
                continue

            sub = sub.sort_values(
                ["auc_pr_median_filled", "roc_auc_median_filled", "pct"],
                ascending=[False, False, True]
            )
            best = sub.iloc[0]

            winners_rows.append({
                "setup": setup,
                "anomaly": anomaly,
                "method": method,
                "win": int(best["win"]),
                "kfold": int(best["kfold"]),
                "best_pct_by_median": int(best["pct"]),
                "auc_pr_median": float(best["auc_pr_median"]),
                "roc_auc_median": float(best["roc_auc_median"]),
                "auc_pr_iqr": float(best["auc_pr_iqr"]),
                "roc_auc_iqr": float(best["roc_auc_iqr"]),
                "min_auc_pr": float(best["auc_pr_min"]),
                "max_auc_pr": float(best["auc_pr_max"]),
                "min_roc_auc": float(best["roc_auc_min"]),
                "max_roc_auc": float(best["roc_auc_max"]),
                "n_runs": int(best["n_runs"]),
            })

            print(f"  [BEST • {setup}–{anomaly}–{method}] "
                  f"WIN={int(best['win'])}  K={int(best['kfold'])}  %={int(best['pct'])}  "
                  f"AUCPR_med={best['auc_pr_median']:.3f}  ROC_med={best['roc_auc_median']:.3f}")

winners = pd.DataFrame(winners_rows).sort_values(["setup","anomaly","method"])
WINNERS_CSV = RES_DIR / "BEST_in_DesignSpace_Post_per_testcase_PER_METHOD.csv"
winners.to_csv(WINNERS_CSV, index=False)
print(f"\n[OK] Saved per-(setup,anomaly,method) winners → {WINNERS_CSV}")

# ---------------------------------------------------------------------------
# B) For every (setup, anomaly, win, kfold, method), find BEST pct
# ---------------------------------------------------------------------------
best_pct_rows = []
gcols = ["setup","anomaly","win","kfold","method"]
print("\n[SCAN-B] Best PCT per (setup, anomaly, win, kfold, method)")
for keys, g in sumdf.groupby(gcols, dropna=False):
    g = g.sort_values(
        ["auc_pr_median_filled", "roc_auc_median_filled", "pct"],
        ascending=[False, False, True]
    )
    top = g.iloc[0]
    best_pct_rows.append({
        "setup": keys[0], "anomaly": keys[1],
        "win": int(keys[2]), "kfold": int(keys[3]),
        "method": keys[4],
        "best_pct_by_median": int(top["pct"]),
        "auc_pr_median": float(top["auc_pr_median"]),
        "roc_auc_median": float(top["roc_auc_median"]),
        "n_runs": int(top["n_runs"]),
    })
    print(f"  {keys[0]}/{keys[1]}  WIN={int(keys[2])}  K={int(keys[3])}  {keys[4]} "
          f"-> best %={int(top['pct'])}  AUCPR_med={top['auc_pr_median']:.3f}, ROC_med={top['roc_auc_median']:.3f}")

best_pct = pd.DataFrame(best_pct_rows).sort_values(["setup","anomaly","win","kfold","method"])
BEST_PCT_CSV = RES_DIR / "BEST_pct_per_win_kfold_PER_METHOD.csv"
best_pct.to_csv(BEST_PCT_CSV, index=False)
print(f"\n[OK] Saved best pct per (win,kfold,method) → {BEST_PCT_CSV}")

# ---------------------------------------------------------------------------
# C) (Optional) One "headline" BEST per (setup, anomaly) by taking BEST METHOD too
#    Sort: AUCPR_median desc, then ROC_median desc, then smaller pct
# ---------------------------------------------------------------------------
headline_rows = []
print("\n[SCAN-C] Headline BEST per (setup, anomaly) across ALL methods too")
for setup, anomalies in ANOMALIES_BY_SETUP.items():
    for anomaly in anomalies:
        sub = sumdf[(sumdf["setup"]==setup) & (sumdf["anomaly"]==anomaly)].copy()
        if sub.empty:
            print(f"  [WARN] No rows for {setup}–{anomaly}")
            continue
        sub = sub.sort_values(
            ["auc_pr_median_filled", "roc_auc_median_filled", "pct"],
            ascending=[False, False, True]
        )
        best = sub.iloc[0]
        headline_rows.append({
            "setup": setup,
            "anomaly": anomaly,
            "method": str(best["method"]),
            "win": int(best["win"]),
            "kfold": int(best["kfold"]),
            "best_pct_by_median": int(best["pct"]),
            "auc_pr_median": float(best["auc_pr_median"]),
            "roc_auc_median": float(best["roc_auc_median"]),
            "auc_pr_iqr": float(best["auc_pr_iqr"]),
            "roc_auc_iqr": float(best["roc_auc_iqr"]),
            "n_runs": int(best["n_runs"]),
        })
        print(f"  [HEADLINE • {setup}–{anomaly}] "
              f"{best['method']}  WIN={int(best['win'])}  K={int(best['kfold'])}  %={int(best['pct'])}  "
              f"AUCPR_med={best['auc_pr_median']:.3f}  ROC_med={best['roc_auc_median']:.3f}")

headline = pd.DataFrame(headline_rows).sort_values(["setup","anomaly"])
HEADLINE_CSV = RES_DIR / "BEST_in_DesignSpace_Post_HEADLINE_per_testcase.csv"
headline.to_csv(HEADLINE_CSV, index=False)
print(f"\n[OK] Saved headline winners → {HEADLINE_CSV}")