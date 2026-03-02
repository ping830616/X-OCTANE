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

# === PLATFORM-LEVEL SHAP Top-10 plot-only PNGs (ONE per platform) + FULL BLACK BOX + legend ===
# This aggregates SHAP across the platform's anomalies (using the chosen best configs in
# BEST_in_DesignSpace_Post_per_platform_details.csv) and produces ONE Top-10 bar plot per platform.
#
# Inputs:
#   1) Results/BEST_in_DesignSpace_Post_per_platform_details.csv
#   2) Results/Explainability_SHAP_BestPlatforms/
#        SHAP_BESTPLAT_full_<setup>_<anomaly>_WIN<w>_KF<k>_PCT<p>_M<method>.csv
#
# Outputs (3 PNGs total):
#   1) figs/FIG_Top10_SHAP_LEGEND.png
#   2) figs/FIG_Top10_SHAP_PLATFORM_DDR4_PLOT.png
#   3) figs/FIG_Top10_SHAP_PLATFORM_DDR5_PLOT.png
#
# Aggregation rule:
#   - For each platform, load the anomaly-specific SHAP tables (already at each anomaly's best config)
#   - Compute mean SHAP (mean |value|) per feature ACROSS anomalies for that platform:
#       shap_platform(feature) = mean_over_anomalies(shap_mean_abs_anomaly(feature))
#     Missing features in an anomaly contribute 0 for that anomaly.
#   - Subspace label per feature: majority vote across anomalies (tie -> subspace of highest mean SHAP).
#
# Styling:
#   - Full black rectangle frame (all four spines visible & black), like your paper figure
#   - Separate legend PNG
# -------------------------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------ Paths ------------------

DETAILS = RES_DIR / "BEST_in_DesignSpace_Post_per_platform_details.csv"
if not DETAILS.exists():
    raise FileNotFoundError(f"Missing platform details CSV: {DETAILS}")

PLAT_DIR = RES_DIR / "Explainability_SHAP_BestPlatforms"
FIG_DIR  = PLAT_DIR / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ------------------ Colors match paper style ------------------
DOMAIN_COLORS = {"compute": "#1f77b4", "memory": "#ff7f0e", "sensors": "#2ca02c"}  # blue, orange, green
DOMAIN_LABELS = {"compute": "Compute", "memory": "Memory", "sensors": "Sensors"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 13,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
})

def _apply_full_black_box(ax, lw=1.1):
    """Make a full black rectangle frame like the paper (all 4 spines)."""
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(lw)

def save_legend_only(out_png: Path):
    fig, ax = plt.subplots(figsize=(3.0, 1.4), dpi=220)
    ax.axis("off")

    handles, labels = [], []
    for key in ["compute", "memory", "sensors"]:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=DOMAIN_COLORS[key]))
        labels.append(DOMAIN_LABELS[key])

    ax.legend(handles, labels, loc="center", frameon=True, fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("[WROTE]", out_png)

def _load_bestplat_csv(setup: str, anomaly: str, win: int, kfold: int, pct: int, method: str) -> pd.DataFrame:
    p = PLAT_DIR / f"SHAP_BESTPLAT_full_{setup}_{anomaly}_WIN{win}_KF{kfold}_PCT{pct}_M{method}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing best-platform SHAP CSV: {p}")
    df = pd.read_csv(p)
    df.columns = [c.lower() for c in df.columns]
    need = {"feature", "subspace", "shap_mean_abs"}
    if not need.issubset(df.columns):
        raise KeyError(f"{p} missing columns {need - set(df.columns)}. Have: {list(df.columns)}")
    df["feature"] = df["feature"].astype(str)
    df["subspace"] = df["subspace"].astype(str).str.lower()
    df["shap_mean_abs"] = pd.to_numeric(df["shap_mean_abs"], errors="coerce").fillna(0.0)
    return df[["feature","subspace","shap_mean_abs"]].copy()

def _aggregate_platform_shap(cfg_rows: pd.DataFrame) -> pd.DataFrame:
    """
    cfg_rows: rows for one setup (platform), each row includes anomaly/win/kfold/pct/method.
    Returns: aggregated per-feature table with columns:
      feature, subspace, shap_mean_abs_platform
    """
    if cfg_rows.empty:
        return pd.DataFrame(columns=["feature","subspace","shap_mean_abs_platform"])

    dfs = []
    for _, r in cfg_rows.iterrows():
        setup  = str(r["setup"])
        anomaly= str(r["anomaly"])
        win    = int(r["win"])
        kfold  = int(r["kfold"])
        pct    = int(r["best_pct_by_median"])
        method = str(r["method"])
        d = _load_bestplat_csv(setup, anomaly, win, kfold, pct, method)
        d = d.rename(columns={"shap_mean_abs": f"shap_{anomaly}"})
        dfs.append(d)

    # Outer-merge across anomalies on feature; keep subspace columns for voting
    # We merge in a way that preserves all features ever seen across anomalies.
    merged = None
    for d in dfs:
        if merged is None:
            merged = d.copy()
        else:
            # when merging, keep both subspace columns (we'll vote later)
            merged = merged.merge(d, on="feature", how="outer", suffixes=("", "_r"))

            # consolidate/rename subspace columns into a list-friendly set
            # after merge, we may have 'subspace' and 'subspace_r'
            if "subspace_r" in merged.columns:
                # keep both; rename to unique col name
                # we’ll just leave them and handle later
                pass

    if merged is None or merged.empty:
        return pd.DataFrame(columns=["feature","subspace","shap_mean_abs_platform"])

    # Identify all shap columns (one per anomaly)
    shap_cols = [c for c in merged.columns if c.startswith("shap_")]
    if not shap_cols:
        return pd.DataFrame(columns=["feature","subspace","shap_mean_abs_platform"])

    # Fill missing SHAP with 0 (means feature absent in that anomaly)
    merged[shap_cols] = merged[shap_cols].fillna(0.0)

    # Platform aggregate = mean over anomalies (equal weight)
    merged["shap_mean_abs_platform"] = merged[shap_cols].mean(axis=1)

    # Determine subspace via majority vote across available subspace columns
    sub_cols = [c for c in merged.columns if c.startswith("subspace")]
    def vote_subspace(row):
        vals = [str(row[c]).lower() for c in sub_cols if pd.notna(row[c]) and str(row[c]).strip() != ""]
        vals = [v for v in vals if v in ("compute","memory","sensors")]
        if not vals:
            return "compute"
        # majority vote
        counts = {k: vals.count(k) for k in ("compute","memory","sensors")}
        best = max(counts, key=lambda k: counts[k])
        # tie-break: choose the subspace corresponding to the anomaly where this feature has max SHAP
        top_cnt = counts[best]
        ties = [k for k,v in counts.items() if v == top_cnt]
        if len(ties) == 1:
            return best
        # tie: pick subspace of the anomaly with largest shap value among tie subspaces
        # (use first non-empty subspace aligned with max shap col)
        max_col = shap_cols[int(np.argmax([row[c] for c in shap_cols]))]
        # find a subspace column that likely came from same anomaly merge stage:
        # simplest tie-break: pick first subspace that is in ties
        for v in vals:
            if v in ties:
                return v
        return ties[0]

    merged["subspace"] = merged.apply(vote_subspace, axis=1)

    out = merged[["feature","subspace","shap_mean_abs_platform"]].copy()
    out["shap_mean_abs_platform"] = pd.to_numeric(out["shap_mean_abs_platform"], errors="coerce").fillna(0.0)
    return out

def save_platform_plot_only(df_platform: pd.DataFrame, setup: str, out_png: Path, topk: int = 10):
    d = df_platform.sort_values("shap_mean_abs_platform", ascending=False).head(topk).copy()
    if d.empty:
        print("[SKIP] empty platform df for", setup)
        return

    colors = [DOMAIN_COLORS.get(s, "#777777") for s in d["subspace"]]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=220)
    ax.barh(d["feature"], d["shap_mean_abs_platform"], color=colors, edgecolor="none")
    ax.invert_yaxis()

    ax.set_xlabel("SHAP (mean |value|) • platform-avg", fontweight="bold")
    ax.grid(True, axis="both", alpha=0.25)

    _apply_full_black_box(ax, lw=1.1)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("[WROTE]", out_png)

# ------------------ Load per-platform details and build configs ------------------
w = pd.read_csv(DETAILS).copy()
if "best_method" in w.columns and "method" not in w.columns:
    w = w.rename(columns={"best_method":"method"})
if "best_pct_by_median" not in w.columns and "pct" in w.columns:
    w = w.rename(columns={"pct":"best_pct_by_median"})

need = {"setup","anomaly","win","kfold","best_pct_by_median","method"}
missing = need - set(w.columns)
if missing:
    raise KeyError(f"{DETAILS} missing columns {sorted(missing)}. Have: {list(w.columns)}")

# ------------------ 1) Legend PNG (shared) ------------------
legend_png = FIG_DIR / "FIG_Top10_SHAP_LEGEND.png"
save_legend_only(legend_png)

# ------------------ 2) One plot per platform ------------------
for setup in ["DDR4","DDR5"]:
    cfg = w[w["setup"].astype(str).str.upper() == setup].copy()
    if cfg.empty:
        print("[WARN] No config rows for", setup)
        continue

    # Build platform-level aggregated SHAP
    plat = _aggregate_platform_shap(cfg)

    # Save the aggregated table too (useful for checking)
    agg_csv = PLAT_DIR / f"SHAP_PLATFORM_AGG_{setup}.csv"
    plat.sort_values("shap_mean_abs_platform", ascending=False).to_csv(agg_csv, index=False)
    print("[WROTE]", agg_csv)

    out_plot = FIG_DIR / f"FIG_Top10_SHAP_PLATFORM_{setup}_PLOT.png"
    save_platform_plot_only(plat, setup=setup, out_png=out_plot, topk=10)