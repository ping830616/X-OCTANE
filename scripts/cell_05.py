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

# ---------------------------------------------------------------------------
# C) Pick ONE (win, kfold) per platform (DDR4, DDR5)  ✅ UPDATED FOR NEW PIPELINE
#    Uses your NEW pipeline outputs:
#      - per_run_metrics_all_PIPELINE.csv  (per-run rows)
#      - or BEST_pct_per_win_kfold_PER_METHOD.csv  (if you already created it)
#
#    Criterion (default):
#      1) For each (setup, anomaly, win, kfold, method): pick BEST pct (PR-first, ROC, smaller pct)
#      2) For each (setup, win, kfold): aggregate across anomalies by taking the BEST method per anomaly
#         then average AUCPR across anomalies (tie avg ROC, smaller win, smaller k)
#
#    Outputs:
#      - Results/BEST_win_kfold_per_platform.csv
#      - Results/BEST_pct_for_chosen_win_kfold_per_platform.csv
# ---------------------------------------------------------------------------

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

# Optional defensive filters (match pipeline grid)
valid_wins   = {32, 64, 128, 512, 1024}
valid_kfolds = {3, 5, 10}
df = df[df["win"].isin(valid_wins) & df["kfold"].isin(valid_kfolds)].copy()
df = df[df["setup"].isin(["DDR4","DDR5"])].copy()
df = df[df["method"].isin(METHOD_ORDER)].copy()

# Fill for robust ranking
df["auc_pr_filled"]  = pd.to_numeric(df["auc_pr"], errors="coerce").fillna(-np.inf)
df["roc_auc_filled"] = pd.to_numeric(df["roc_auc"], errors="coerce").fillna(-np.inf)

# ---------- 0) Summarize per-case at pct level (median across run_id) ----------
def q1(x): return float(np.nanpercentile(x, 25))
def q3(x): return float(np.nanpercentile(x, 75))

sum_keys = ["setup","anomaly","win","kfold","method","pct"]
sumdf = (df.groupby(sum_keys, as_index=False)
           .agg(
               auc_pr_median=("auc_pr", "median"),
               roc_auc_median=("roc_auc", "median"),
               auc_pr_q1=("auc_pr", q1),
               auc_pr_q3=("auc_pr", q3),
               roc_auc_q1=("roc_auc", q1),
               roc_auc_q3=("roc_auc", q3),
               n_runs=("run_id","nunique"),
           ))

sumdf["auc_pr_median"]  = pd.to_numeric(sumdf["auc_pr_median"], errors="coerce").clip(0.0, 1.0)
sumdf["roc_auc_median"] = pd.to_numeric(sumdf["roc_auc_median"], errors="coerce").clip(0.0, 1.0)

sumdf["auc_pr_median_filled"]  = sumdf["auc_pr_median"].fillna(-np.inf)
sumdf["roc_auc_median_filled"] = sumdf["roc_auc_median"].fillna(-np.inf)

# ---------- 1) For each (setup, anomaly, win, kfold, method): choose best pct ----------
# Sort: AUCPR median desc, then ROC median desc, then smaller pct
best_pct = (sumdf.sort_values(
                ["setup","anomaly","win","kfold","method",
                 "auc_pr_median_filled","roc_auc_median_filled","pct"],
                ascending=[True,True,True,True,True, False,False,True]
            )
            .groupby(["setup","anomaly","win","kfold","method"], as_index=False)
            .head(1)
            .rename(columns={"pct":"best_pct_by_median"}))

# Keep only anomalies defined per platform (defensive)
def _keep_defined_anoms(row):
    setup = str(row["setup"]).upper()
    anom  = str(row["anomaly"]).upper()
    return setup in ANOMALIES_BY_SETUP and anom in [a.upper() for a in ANOMALIES_BY_SETUP[setup]]

best_pct = best_pct[best_pct.apply(_keep_defined_anoms, axis=1)].copy()

# Save (optional, helpful for debugging / later steps)
BEST_PCT_PER_METHOD_CSV = RES_DIR / "BEST_pct_per_win_kfold_PER_METHOD_from_PIPELINE.csv"
best_pct.to_csv(BEST_PCT_PER_METHOD_CSV, index=False)
print(f"[OK] Wrote best_pct (per method) derived from pipeline → {BEST_PCT_PER_METHOD_CSV}")

print("\n[SCAN] Best (WIN, K) per platform by mean AUCPR across anomalies")
platform_rows = []
chosen_rows = []

# ---------- 2) Platform selection logic ----------
for setup in ["DDR4", "DDR5"]:
    sub = best_pct[best_pct["setup"].astype(str).str.upper() == setup].copy()
    if sub.empty:
        print(f"  [WARN] No rows for setup={setup} in derived best_pct table")
        continue

    # For each (setup, anomaly, win, kfold), pick BEST METHOD
    # Sort: AUCPR desc, ROC desc, then a stable method preference order
    method_rank = {m:i for i,m in enumerate(METHOD_ORDER)}
    sub["_mrank"] = sub["method"].map(method_rank).fillna(999).astype(int)

    best_method_per_anom = (sub.sort_values(
                                ["setup","anomaly","win","kfold",
                                 "auc_pr_median_filled","roc_auc_median_filled","_mrank",
                                 "best_pct_by_median"],
                                ascending=[True,True,True,True, False,False, True, True]
                            )
                            .groupby(["setup","anomaly","win","kfold"], as_index=False)
                            .head(1))

    # Aggregate per (win, kfold) across anomalies (mean of anomaly-bests)
    agg = (best_method_per_anom.groupby(["setup","win","kfold"], as_index=False)
                           .agg(mean_auc_pr=("auc_pr_median_filled", "mean"),
                                mean_roc_auc=("roc_auc_median_filled", "mean"),
                                n_anoms=("anomaly", "nunique")))

    # Prefer combos that cover more anomalies, then mean AUCPR, mean ROC, smaller win/k
    agg = agg.sort_values(
        ["n_anoms", "mean_auc_pr", "mean_roc_auc", "win", "kfold"],
        ascending=[False, False, False, True, True]
    )
    best_combo = agg.iloc[0]
    win_best = int(best_combo["win"])
    kf_best  = int(best_combo["kfold"])

    platform_rows.append({
        "setup": setup,
        "win": win_best,
        "kfold": kf_best,
        "mean_auc_pr_across_anomalies": float(best_combo["mean_auc_pr"]),
        "mean_roc_auc_across_anomalies": float(best_combo["mean_roc_auc"]),
        "num_anomalies_covered": int(best_combo["n_anoms"]),
        "aggregation_note": "anomaly-wise best method @ best pct, then mean across anomalies",
    })

    # For chosen (win,kfold), list anomaly-wise best pct + best method + metrics
    chosen_detail = best_method_per_anom[
        (best_method_per_anom["win"] == win_best) &
        (best_method_per_anom["kfold"] == kf_best)
    ].copy()

    # Sort for display: higher AUCPR, higher ROC, smaller pct
    chosen_detail = chosen_detail.sort_values(
        ["auc_pr_median_filled", "roc_auc_median_filled", "best_pct_by_median"],
        ascending=[False, False, True]
    )

    print(f"  [PLATFORM BEST • {setup}]  WIN={win_best}  K={kf_best}  "
          f"mean AUCPR={float(best_combo['mean_auc_pr']):.3f}  mean ROC={float(best_combo['mean_roc_auc']):.3f}  "
          f"(across {int(best_combo['n_anoms'])} anomalies)")

    for _, r in chosen_detail.iterrows():
        pct_best = int(r["best_pct_by_median"])
        aucpr = float(r["auc_pr_median"]) if np.isfinite(r["auc_pr_median"]) else float("nan")
        roc   = float(r["roc_auc_median"]) if np.isfinite(r["roc_auc_median"]) else float("nan")
        method = str(r["method"])
        anom = str(r["anomaly"])
        print(f"      {setup}/{anom}: best method={method}  best %={pct_best}  "
              f"AUCPR_med={aucpr:.3f}  ROC_med={roc:.3f}")

        chosen_rows.append({
            "setup": setup,
            "anomaly": anom,
            "win": win_best,
            "kfold": kf_best,
            "best_method": method,
            "best_pct_by_median": pct_best,
            "auc_pr_median": aucpr,
            "roc_auc_median": roc,
            "n_runs": int(r["n_runs"]) if "n_runs" in r else np.nan,
        })

# ---------- 3) Save outputs ----------
platform_best = pd.DataFrame(platform_rows)
PLATFORM_BEST_CSV = RES_DIR / "BEST_win_kfold_per_platform.csv"
platform_best.to_csv(PLATFORM_BEST_CSV, index=False)

chosen_pct = pd.DataFrame(chosen_rows)
CHOSEN_PCT_CSV = RES_DIR / "BEST_pct_for_chosen_win_kfold_per_platform.csv"
chosen_pct.to_csv(CHOSEN_PCT_CSV, index=False)

print(f"\n[OK] Saved platform winners → {PLATFORM_BEST_CSV}")
print(f"[OK] Saved per-anomaly details for chosen (WIN,K) → {CHOSEN_PCT_CSV}")