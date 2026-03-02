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

# === PER-PLATFORM enumrank tables (AUTO WIN/K from per-platform winners) ===
# UPDATED so you do NOT hard-code DDR4/DDR5 WIN/K.
#
# Builds ONE enumrank CSV per platform × subspace using the chosen (WIN,K) per platform from:
#   Results/BEST_in_DesignSpace_Post_per_platform.csv
#
# Output (6 files total):
#   OUT_DIR/enumrank__DDR4_PLATFORM_WIN<win>_KF<kfold>__compute.csv
#   OUT_DIR/enumrank__DDR4_PLATFORM_WIN<win>_KF<kfold>__memory.csv
#   OUT_DIR/enumrank__DDR4_PLATFORM_WIN<win>_KF<kfold>__sensors.csv
#   OUT_DIR/enumrank__DDR5_PLATFORM_WIN<win>_KF<kfold>__compute.csv
#   OUT_DIR/enumrank__DDR5_PLATFORM_WIN<win>_KF<kfold>__memory.csv
#   OUT_DIR/enumrank__DDR5_PLATFORM_WIN<win>_KF<kfold>__sensors.csv
#
# Notes:
# - Uses CP-MI rank tables from FeatureRankOUT by (setup,win,kfold,subspace).
# - No anomaly dimension; “PLATFORM” indicates platform-level export.
# --------------------------------------------------------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np

# ---------- paths ----------
RES  = Path(globals().get("RES_DIR", ROOT / "Results"))
# normalize if notebook points too deep
if RES.name in ("Explainability_SHAP_BestCases", "Explainability_SHAP_BestPlatforms"):
    RES = RES.parent

BEST_PLATFORM = RES / "BEST_in_DesignSpace_Post_per_platform.csv"
if not BEST_PLATFORM.exists():
    raise FileNotFoundError(f"Missing per-platform winners CSV: {BEST_PLATFORM}")

RANK_DIR = Path(globals().get("RANK_DIR", ROOT / "FeatureRankOUT"))
OUT_DIR  = Path(globals().get("OUT_DIR", RES / "_editor_package"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBSPACES = ("compute","memory","sensors")

# ---------- helpers ----------
def pick_score_col(df: pd.DataFrame) -> str | None:
    for c in ("cpmi_score","hybrid_score","shap_mean_abs","score"):
        if c in df.columns:
            return c
    feat_col = (
        "feature" if "feature" in df.columns else
        ("feature_display" if "feature_display" in df.columns else df.columns[0])
    )
    nums = [c for c in df.columns if c != feat_col and pd.api.types.is_numeric_dtype(df[c])]
    return nums[0] if nums else None

def load_rank_table(rank_dir: Path, setup: str, win: int, kfold: int, subspace: str) -> pd.DataFrame:
    p = rank_dir / f"{setup}_{win}_{kfold}_0_{subspace}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    feat_col = (
        "feature" if "feature" in df.columns else
        ("feature_display" if "feature_display" in df.columns else df.columns[0])
    )
    df = df.rename(columns={feat_col: "feature"})
    return df

def enum_bucket_per_feature(scores: pd.Series) -> pd.Series:
    """Return a 1..5 bucket per feature (5=top 20%)."""
    s = pd.to_numeric(scores, errors="coerce").fillna(0.0)
    if len(s) <= 1:
        return pd.Series([3]*len(s), index=s.index, dtype=int)  # neutral if trivial
    # percentile rank 0..1 (higher = better)
    r = s.rank(ascending=True, method="average")
    p = (r - 1) / (len(s) - 1)
    # map to 1..5 (5 best)
    return (5 - (p*5).astype(int).clip(0,4)).astype(int)

def write_enumrank_platform_table(setup: str, win: int, kfold: int, subspace: str) -> Path | None:
    df = load_rank_table(RANK_DIR, setup, win, kfold, subspace)
    if df.empty:
        print(f"[SKIP] Missing rank table: {setup} WIN={win} KF={kfold} sub={subspace}")
        return None

    score_col = pick_score_col(df)
    if score_col is None:
        print(f"[SKIP] No score col found in rank table for {setup} WIN={win} KF={kfold} sub={subspace}")
        return None

    tmp = df[["feature", score_col]].copy()
    tmp = tmp.rename(columns={score_col: "score"})
    tmp["score"] = pd.to_numeric(tmp["score"], errors="coerce").fillna(0.0)

    tmp["EnumRank_1to5"] = enum_bucket_per_feature(tmp["score"])

    # Percentile for transparency (0..1)
    if len(tmp) > 1:
        r = tmp["score"].rank(ascending=True, method="average")
        tmp["Percentile_0to1"] = (r - 1) / (len(tmp) - 1)
    else:
        tmp["Percentile_0to1"] = 0.5

    tmp = tmp.sort_values(["EnumRank_1to5","score"], ascending=[False, False]).reset_index(drop=True)

    out_csv = OUT_DIR / f"enumrank__{setup}_PLATFORM_WIN{win}_KF{kfold}__{subspace}.csv"
    tmp.to_csv(out_csv, index=False)
    print("[WROTE]", out_csv)
    return out_csv

# ---------- Load per-platform winners (AUTO WIN/K) ----------
bp = pd.read_csv(BEST_PLATFORM).copy()
bp.columns = [c.lower() for c in bp.columns]
need = {"setup","win","kfold"}
missing = need - set(bp.columns)
if missing:
    raise KeyError(f"{BEST_PLATFORM} missing required columns {sorted(missing)}. Have: {list(bp.columns)}")

# Build PLATFORM_CFG automatically
PLATFORM_CFG = {}
for _, r in bp.iterrows():
    setup = str(r["setup"]).strip()
    win = int(pd.to_numeric(r["win"], errors="coerce"))
    kf  = int(pd.to_numeric(r["kfold"], errors="coerce"))
    PLATFORM_CFG[setup] = dict(win=win, kfold=kf)

print("[OK] PLATFORM_CFG (auto):", PLATFORM_CFG)

# ---------- Run per platform × subspace ----------
for setup, cfg in PLATFORM_CFG.items():
    win, kfold = int(cfg["win"]), int(cfg["kfold"])
    for sub in SUBSPACES:
        write_enumrank_platform_table(setup, win, kfold, sub)