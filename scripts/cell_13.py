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

# === PER-PLATFORM editor package export (AUTO from per-platform winners) ===
# UPDATED per your current per-platform pipeline.
#
# What this does (per platform):
#   - Uses per-platform winners (auto; no hard-coded WIN/K/PCT):
#       Results/BEST_in_DesignSpace_Post_per_platform.csv
#       Results/BEST_in_DesignSpace_Post_per_platform_details.csv
#   - Exports:
#       1) _editor_package/workloads_performance_platform.csv
#          (platform + anomaly rows with chosen WIN/K, best_pct, best_method and metrics if present)
#       2) _editor_package/sweep_counts_by_subspace_platform.csv
#          (counts vs top-% thresholds for CP-MI ranks at the chosen (WIN,K) per platform)
#          OPTIONAL: also include HYBRID ranks if FeatureRankOUT_HYBRID exists (toggle below)
#       3) Plots:
#          - _editor_package/sweep_feature_counts_per_subspace_<setup>.png
#          - _editor_package/sweep_feature_counts_per_subspace_PLATFORM.png
#
# Inputs:
#   - Results/BEST_in_DesignSpace_Post_per_platform.csv
#   - Results/BEST_in_DesignSpace_Post_per_platform_details.csv
#   - FeatureRankOUT/<setup>_<win>_<kfold>_0_{compute|memory|sensors}.csv
#   - (optional) FeatureRankOUT_HYBRID/<setup>_<win>_<kfold>_0_{compute|memory|sensors}.csv
# ---------------------------------------------------------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# ----------- CONFIG ----------
RES  = Path(globals().get("RES_DIR", ROOT / "Results"))
# normalize if notebook points too deep
if RES.name in ("Explainability_SHAP_BestCases", "Explainability_SHAP_BestPlatforms"):
    RES = RES.parent

BEST_PLATFORM        = RES / "BEST_in_DesignSpace_Post_per_platform.csv"
BEST_PLATFORM_DETAIL = RES / "BEST_in_DesignSpace_Post_per_platform_details.csv"

RANK_DIR = ROOT / "FeatureRankOUT"            # CP-MI baseline ranks
HYB_DIR  = ROOT / "FeatureRankOUT_HYBRID"     # optional hybrid ranks
OUT_DIR  = RES / "_editor_package"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBSPACES = ("compute","memory","sensors")
TOP_PCTS  = list(range(10, 101, 10))

INCLUDE_HYBRID_COUNTS = True   # set False if you only want CP-MI counts

# ---------- Helpers ----------
def read_platform_best(best_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(best_csv)
    df.columns = [c.lower() for c in df.columns]
    # minimal expected columns: setup, win, kfold
    need = {"setup","win","kfold"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"{best_csv} missing required columns: {sorted(miss)}. Have: {list(df.columns)}")
    return df

def read_platform_details(details_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(details_csv)
    df.columns = [c.lower() for c in df.columns]
    if "best_method" in df.columns and "method" not in df.columns:
        df = df.rename(columns={"best_method":"method"})
    if "best_pct_by_median" not in df.columns and "pct" in df.columns:
        df = df.rename(columns={"pct":"best_pct_by_median"})
    need = {"setup","anomaly","win","kfold","best_pct_by_median","method"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"{details_csv} missing required columns: {sorted(miss)}. Have: {list(df.columns)}")
    return df

def pick_score_col(df: pd.DataFrame) -> str | None:
    for c in ("cpmi_score","hybrid_score","shap_mean_abs"):
        if c in df.columns:
            return c
    # fallback: first numeric col after feature if any
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    return nums[0] if nums else (df.columns[1] if df.shape[1] >= 2 else None)

def load_rank_table(rank_dir: Path, setup: str, win: int, kfold: int, subspace: str) -> pd.DataFrame:
    p = rank_dir / f"{setup}_{win}_{kfold}_0_{subspace}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    # feature column may differ by source
    feat_col = (
        "feature" if "feature" in df.columns else
        ("feature_display" if "feature_display" in df.columns else df.columns[0])
    )
    df = df.rename(columns={feat_col: "feature"})
    return df

def top_k_count(n_total: int, pct: int) -> int:
    k = int(math.ceil(n_total * pct / 100.0))
    return max(1, k) if pct > 0 else 0

def enumerate_mean_rank(values: pd.Series) -> float:
    """
    Your prior 'enumerated mean rank 1..5' scheme.
    """
    if values is None or len(values) == 0:
        return np.nan
    v = pd.to_numeric(values, errors="coerce").fillna(0.0)
    r = v.rank(ascending=True, method="average")
    p = (r - 1) / (len(v) - 1) if len(v) > 1 else pd.Series([0.0]*len(v))
    buckets = 5 - (p*5).astype(int).clip(0,4)
    return float(buckets.mean())

# ---------- Load platform winners ----------
if not BEST_PLATFORM.exists():
    raise FileNotFoundError(f"Missing platform BEST CSV: {BEST_PLATFORM}")
if not BEST_PLATFORM_DETAIL.exists():
    raise FileNotFoundError(f"Missing platform BEST details CSV: {BEST_PLATFORM_DETAIL}")

best_plat = read_platform_best(BEST_PLATFORM)
detail    = read_platform_details(BEST_PLATFORM_DETAIL)

# Build PLATFORM_CFG automatically
PLATFORM_CFG = {}
for _, r in best_plat.iterrows():
    setup = str(r["setup"]).strip()
    PLATFORM_CFG[setup] = dict(
        win=int(pd.to_numeric(r["win"], errors="coerce")),
        kfold=int(pd.to_numeric(r["kfold"], errors="coerce")),
    )

if not PLATFORM_CFG:
    raise RuntimeError("No platform configs found in BEST_in_DesignSpace_Post_per_platform.csv")

print("[OK] PLATFORM_CFG (auto):", PLATFORM_CFG)

# ---------- A) workloads_performance_platform.csv ----------
# Use the per-platform details (anomaly-specific best pct & method).
# Include any metric columns that exist (roc/auc/pr/iqr/min/max etc.).
metric_cols = [c for c in detail.columns if any(tok in c for tok in ["auc", "roc", "iqr", "min", "max", "median", "n_runs"])]

perf_rows = []
for setup, cfg in PLATFORM_CFG.items():
    win, kfold = int(cfg["win"]), int(cfg["kfold"])
    sub = detail[
        (detail["setup"].astype(str).str.upper() == setup.upper()) &
        (pd.to_numeric(detail["win"], errors="coerce") == win) &
        (pd.to_numeric(detail["kfold"], errors="coerce") == kfold)
    ].copy()
    if sub.empty:
        continue
    sub["platform_win"] = win
    sub["platform_kfold"] = kfold
    cols_keep = ["setup","anomaly","platform_win","platform_kfold","win","kfold","best_pct_by_median","method"] + metric_cols
    cols_keep = [c for c in cols_keep if c in sub.columns]
    perf_rows.append(sub[cols_keep])

perf_platform = pd.concat(perf_rows, ignore_index=True) if perf_rows else pd.DataFrame()
perf_csv = OUT_DIR / "workloads_performance_platform.csv"
perf_platform.to_csv(perf_csv, index=False)

# ---------- B) sweep_counts_by_subspace_platform.csv ----------
rows = []
for setup, cfg in PLATFORM_CFG.items():
    win, kfold = int(cfg["win"]), int(cfg["kfold"])

    for sub in SUBSPACES:
        # ---- CP-MI counts ----
        df_rank = load_rank_table(RANK_DIR, setup, win, kfold, sub)
        if not df_rank.empty:
            score_col = pick_score_col(df_rank)
            n_total = len(df_rank)
            enum_mean = enumerate_mean_rank(df_rank[score_col]) if score_col and score_col in df_rank.columns else np.nan

            for pct in TOP_PCTS:
                k = top_k_count(n_total, pct)
                rows.append({
                    "setup": setup, "win": win, "kfold": kfold,
                    "rank_source": "CPMI",
                    "subspace": sub, "top_pct": pct,
                    "n_total": n_total, "n_selected": k,
                    "enumerated_mean_rank_1to5": enum_mean
                })
        else:
            print(f"[WARN] Missing CP-MI rank file for {setup} WIN={win} K={kfold} sub={sub}")

        # ---- HYBRID counts (optional) ----
        if INCLUDE_HYBRID_COUNTS:
            df_hyb = load_rank_table(HYB_DIR, setup, win, kfold, sub)
            if not df_hyb.empty:
                score_col_h = pick_score_col(df_hyb)
                n_total_h = len(df_hyb)
                enum_mean_h = enumerate_mean_rank(df_hyb[score_col_h]) if score_col_h and score_col_h in df_hyb.columns else np.nan

                for pct in TOP_PCTS:
                    k = top_k_count(n_total_h, pct)
                    rows.append({
                        "setup": setup, "win": win, "kfold": kfold,
                        "rank_source": "HYBRID",
                        "subspace": sub, "top_pct": pct,
                        "n_total": n_total_h, "n_selected": k,
                        "enumerated_mean_rank_1to5": enum_mean_h
                    })

sweep_platform = pd.DataFrame(rows)
sweep_csv = OUT_DIR / "sweep_counts_by_subspace_platform.csv"
sweep_platform.to_csv(sweep_csv, index=False)

print("Saved:")
print(" -", perf_csv)
print(" -", sweep_csv)

# ---------- C) Plot: top-% vs number of features per subspace (per platform) ----------
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.labelsize": 10,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "lines.linewidth": 2.2,
})

colors = {"compute":"#1f77b4", "memory":"#66b3ff", "sensors":"#ff7f0e"}

# One plot per platform, CPMI only (clean)
for setup, cfg in PLATFORM_CFG.items():
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    d0 = sweep_platform[(sweep_platform["rank_source"]=="CPMI") & (sweep_platform["setup"]==setup)]
    if d0.empty:
        plt.close(fig)
        continue

    for sub in SUBSPACES:
        d = d0[d0["subspace"]==sub]
        if d.empty:
            continue
        g = d.groupby("top_pct")["n_selected"].median().reset_index()
        ax.plot(g["top_pct"], g["n_selected"], marker="o", label=sub.capitalize(), color=colors[sub])

    ax.set_xlabel("Top-% threshold")
    ax.set_ylabel("# features selected")
    ax.set_title(f"Feature count vs top-% threshold (CP-MI) — {setup} (WIN={cfg['win']}, K={cfg['kfold']})")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(rect=[0,0,0.82,1])
    outp = OUT_DIR / f"sweep_feature_counts_per_subspace_{setup}.png"
    fig.savefig(outp, bbox_inches="tight")
    plt.close(fig)
    print(" -", outp)

# Combined plot (two platforms shown by linestyle)
fig, ax = plt.subplots(figsize=(7.8, 4.7))
linestyles = {"DDR4":"-", "DDR5":"--"}

for setup, cfg in PLATFORM_CFG.items():
    d0 = sweep_platform[(sweep_platform["rank_source"]=="CPMI") & (sweep_platform["setup"]==setup)]
    if d0.empty:
        continue
    for sub in SUBSPACES:
        d = d0[d0["subspace"]==sub]
        if d.empty:
            continue
        g = d.groupby("top_pct")["n_selected"].median().reset_index()
        ax.plot(g["top_pct"], g["n_selected"], marker="o",
                label=f"{setup} {sub.capitalize()}",
                color=colors[sub], linestyle=linestyles.get(setup, "-"))

ax.set_xlabel("Top-% threshold")
ax.set_ylabel("# features selected")
ax.set_title("Feature count vs top-% threshold (per subspace, CP-MI) — per platform")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
fig.tight_layout(rect=[0,0,0.78,1])

out_comb = OUT_DIR / "sweep_feature_counts_per_subspace_PLATFORM.png"
fig.savefig(out_comb, bbox_inches="tight")
plt.close(fig)
print(" -", out_comb)