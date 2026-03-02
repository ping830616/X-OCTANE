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

# === Fig. 9 style (PER PLATFORM): Percentile-rank concordance (CP–MI vs SHAP) ===
# You requested: DDR5 (Platform) WIN=1024 and K=5 ✅
#
# CONSISTENT with revised Table 6:
#   - same _norm_name()
#   - restrict to SHARED feature universe per subspace (intersection)
#   - percentile ranks computed within the shared universe
#
# UPDATED (visual quality):
#   - legend moved OUTSIDE (never blocks points)
#   - smaller points + semi-transparency + white edges (better overlap readability)
#   - diagonal behind points
#   - Spearman ρ shown as in-axes annotation (cleaner than a large title)
#   - optional TOPK per subspace to reduce clutter (default None = keep all)
#
# Inputs:
#   - Results/BEST_in_DesignSpace_Post_per_platform_details.csv
#   - FeatureRankOUT/<setup>_<win>_<kfold>_0_{compute|memory|sensors}.csv
#   - Results/Explainability_SHAP_BestPlatforms/SHAP_BESTPLAT_full_<setup>_<anomaly>_WIN<w>_KF<k>_PCT<p>_M<method>.csv
#
# Outputs:
#   - Results/Explainability_SHAP_BestPlatforms/Fig9_percentile_concordance_DDR4_WIN512_K3.png
#   - Results/Explainability_SHAP_BestPlatforms/Fig9_percentile_concordance_DDR5_WIN1024_K5.png
# --------------------------------------------------------------------------------------------

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple
from scipy.stats import spearmanr

# -------------------- Paths (robust) --------------------
RES_DIR_RAW = Path(globals().get("RES_DIR", ROOT / "Results"))
RES_DIR = RES_DIR_RAW.parent if RES_DIR_RAW.name in ("Explainability_SHAP_BestCases", "Explainability_SHAP_BestPlatforms") else RES_DIR_RAW

OUT_DIR = RES_DIR / "Explainability_SHAP_BestPlatforms"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DETAILS_CSV = RES_DIR / "BEST_in_DesignSpace_Post_per_platform_details.csv"
assert DETAILS_CSV.exists(), f"Missing: {DETAILS_CSV}"

# Rank dirs (CP–MI)
RANK_DIRS = list(globals().get("RANK_DIRS", [
    ROOT / "FeatureRankOUT",
    Path("/Volumes/Untitled") / "FeatureRankOUT",
    Path("/Volumes/Untitled") / "octaneX" / "FeatureRankOUT",
    Path.home() / "Desktop" / "octaneX" / "FeatureRankOUT",
]))

# SHAP dir (same as output dir by your naming)
SHAP_DIR = RES_DIR / "Explainability_SHAP_BestPlatforms"
assert SHAP_DIR.exists(), f"Missing SHAP dir: {SHAP_DIR}"

SUBSPACES = ("compute", "memory", "sensors")

# -------------------- Case configs (forced to your request) --------------------
SETUP_A = dict(setup="DDR4", anomaly="DROOP",   win=512,  kfold=3, tag="DDR4_WIN512_K3")
SETUP_B = dict(setup="DDR5", anomaly="DROOP",   win=1024, kfold=5, tag="DDR5_WIN1024_K5")
# If Fig.9 DDR5 should be SPECTRE instead of DROOP:
# SETUP_B["anomaly"] = "SPECTRE"

# -------------------- Plot controls --------------------
# If DDR5 looks too cluttered, set TOPK_PER_SUBSPACE to 25 (or 20/30).
# None = keep all common features.
TOPK_PER_SUBSPACE: Optional[int] = 25  # e.g., 25

# -------------------- Shared helpers (match Table 6) --------------------
def _norm_name(s: str) -> str:
    s = re.sub(r"\s+", "_", str(s)).lower().replace("%", "pct")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

def _find_rank_file(setup: str, win: int, kfold: int, sub: str) -> Optional[Path]:
    fname = f"{setup}_{win}_{kfold}_0_{sub}.csv"
    for d in RANK_DIRS:
        p = Path(d) / fname
        if p.exists():
            return p
    # fallback: recursive search
    for d in RANK_DIRS:
        d = Path(d)
        if d.exists():
            hits = list(d.rglob(fname))
            if hits:
                return hits[0]
    return None

def _read_cpmi_rank_list(setup: str, win: int, kfold: int, sub: str) -> list[str]:
    p = _find_rank_file(setup, win, kfold, sub)
    if p is None:
        return []
    df = pd.read_csv(p)
    if df.empty:
        return []
    col = "feature" if "feature" in df.columns else df.columns[0]
    return [_norm_name(x) for x in df[col].astype(str).tolist()]

# ---- details: best pct + method (if present for this win/kfold) ----
DETAILS = pd.read_csv(DETAILS_CSV).copy()
DETAILS.columns = [c.lower() for c in DETAILS.columns]
if "best_method" in DETAILS.columns and "method" not in DETAILS.columns:
    DETAILS = DETAILS.rename(columns={"best_method": "method"})
if "best_pct_by_median" not in DETAILS.columns and "pct" in DETAILS.columns:
    DETAILS = DETAILS.rename(columns={"pct": "best_pct_by_median"})

def _best_pct_method_from_details(setup: str, anomaly: str, win: int, kfold: int) -> Optional[Tuple[int, str]]:
    sub = DETAILS[
        (DETAILS["setup"].astype(str).str.upper() == setup.upper()) &
        (DETAILS["anomaly"].astype(str).str.upper() == anomaly.upper()) &
        (pd.to_numeric(DETAILS["win"], errors="coerce") == int(win)) &
        (pd.to_numeric(DETAILS["kfold"], errors="coerce") == int(kfold))
    ].copy()
    if sub.empty:
        return None
    r = sub.iloc[0]
    pct = int(pd.to_numeric(r["best_pct_by_median"], errors="coerce"))
    method = str(r["method"]).strip()
    return pct, method

def _find_shap_file(setup: str, anomaly: str, win: int, kfold: int,
                    pct: Optional[int], method: Optional[str]) -> Optional[Path]:
    # 1) exact (best if pct+method known)
    if pct is not None and method is not None:
        exact = SHAP_DIR / f"SHAP_BESTPLAT_full_{setup}_{anomaly}_WIN{win}_KF{kfold}_PCT{pct}_M{method}.csv"
        if exact.exists():
            return exact
        hits = sorted(SHAP_DIR.glob(
            f"SHAP_BESTPLAT_full_{setup}_{anomaly}_WIN{win}_KF{kfold}_PCT*_M{method}.csv"
        ))
        if hits:
            return hits[0]
    # 2) fallback: any method/pct for that (setup, anomaly, win, kfold)
    hits2 = sorted(SHAP_DIR.glob(
        f"SHAP_BESTPLAT_full_{setup}_{anomaly}_WIN{win}_KF{kfold}_PCT*_M*.csv"
    ))
    return hits2[0] if hits2 else None

def _load_shap_full(setup: str, anomaly: str, win: int, kfold: int) -> Tuple[pd.DataFrame, Optional[int], Optional[str], Optional[Path]]:
    pm = _best_pct_method_from_details(setup, anomaly, win, kfold)
    pct, method = (pm if pm is not None else (None, None))
    p = _find_shap_file(setup, anomaly, win, kfold, pct, method)
    if p is None or not p.exists():
        return pd.DataFrame(), pct, method, None

    df = pd.read_csv(p)
    if df.empty:
        return df, pct, method, p

    df.columns = [c.lower() for c in df.columns]
    if "shap_mean_abs" not in df.columns and "importance" in df.columns:
        df = df.rename(columns={"importance": "shap_mean_abs"})

    need = {"feature", "subspace", "shap_mean_abs"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame(), pct, method, p

    df["feature"] = df["feature"].astype(str).map(_norm_name)
    df["subspace"] = df["subspace"].astype(str).str.lower()
    df["shap_mean_abs"] = pd.to_numeric(df["shap_mean_abs"], errors="coerce").fillna(0.0)
    return df, pct, method, p

def _restrict_to_common(order: list[str], common: set[str]) -> list[str]:
    return [f for f in order if f in common]

def _percentile_rank_from_order(features_in_order: list[str]) -> dict[str, float]:
    n = len(features_in_order)
    if n <= 1:
        return {features_in_order[0]: 1.0} if n == 1 else {}
    # best=1, worst=0
    return {f: (1.0 - (i / (n - 1))) for i, f in enumerate(features_in_order)}

# -------------------- Plot function --------------------
def plot_percentile_concordance_one(setup: str, anomaly: str, win: int, kfold: int, out_png: Path):
    shap, pct_best, method_best, shap_path = _load_shap_full(setup, anomaly, win, kfold)
    if shap.empty:
        print(f"[SKIP] Missing SHAP full for {setup}/{anomaly} WIN={win} KF={kfold}")
        return

    pts = []
    for sub in SUBSPACES:
        cp_order = _read_cpmi_rank_list(setup, win, kfold, sub)
        sh_sub = shap[shap["subspace"] == sub][["feature", "shap_mean_abs"]].copy()
        if (not cp_order) or sh_sub.empty:
            continue

        # SHAP order (desc)
        sh_order = sh_sub.sort_values("shap_mean_abs", ascending=False)["feature"].tolist()

        # shared universe
        common = set(cp_order) & set(sh_order)
        if len(common) < 3:
            continue

        # preserve each method's order, restricted to common
        cp_c = _restrict_to_common(cp_order, common)
        sh_c = _restrict_to_common(sh_order, common)

        # OPTIONAL: reduce clutter by plotting only top-K from each list, then re-intersect
        if TOPK_PER_SUBSPACE is not None:
            cp_c = cp_c[:int(TOPK_PER_SUBSPACE)]
            sh_c = sh_c[:int(TOPK_PER_SUBSPACE)]
            common = set(cp_c) & set(sh_c)
            if len(common) < 3:
                continue
            cp_c = _restrict_to_common(cp_c, common)
            sh_c = _restrict_to_common(sh_c, common)

        # percentile ranks within shared universe
        cp_pr = _percentile_rank_from_order(cp_c)
        sh_pr = _percentile_rank_from_order(sh_c)

        for f in common:
            pts.append({"subspace": sub, "x": float(cp_pr.get(f, 0.0)), "y": float(sh_pr.get(f, 0.0))})

    if not pts:
        print(f"[SKIP] No aligned features for {setup}/{anomaly} WIN={win} KF={kfold}")
        return

    P = pd.DataFrame(pts)
    rho, _ = spearmanr(P["x"].to_numpy(dtype=float), P["y"].to_numpy(dtype=float))
    rho = float(rho) if np.isfinite(rho) else np.nan

    # --- Style (paper-ready) ---
    fig, ax = plt.subplots(figsize=(7.6, 4.3), dpi=160)

    colors  = {"compute": "tab:orange", "memory": "tab:blue", "sensors": "tab:green"}
    markers = {"compute": "o",          "memory": "s",        "sensors": "^"}
    labels  = {"compute": "Compute",    "memory": "Memory",  "sensors": "Sensors"}

    # diagonal behind points
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.4, label="y = x", zorder=1)

    # points
    for sub in ["compute", "memory", "sensors"]:
        Q = P[P["subspace"] == sub]
        if Q.empty:
            continue
        ax.scatter(
            Q["x"], Q["y"],
            s=26, alpha=0.70,
            marker=markers[sub],
            color=colors[sub],
            label=labels[sub],
            edgecolors="white", linewidths=0.4,
            zorder=3
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    ax.set_xlabel("CP-MI percentile rank", fontsize=16)
    ax.set_ylabel("SHAP percentile rank", fontsize=16)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.30)

    ax.tick_params(labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Spearman text (cleaner than big title)
    ax.text(
        0.02, 0.98, f"Spearman ρ = {rho:.2f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=14
    )

    # Legend OUTSIDE (never blocks data)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=12,
        borderaxespad=0.0
    )

    extra = []
    if pct_best is not None: extra.append(f"best%={pct_best}")
    if method_best is not None: extra.append(f"M={method_best}")
    if shap_path is not None: extra.append(shap_path.name)
    extra_txt = " | ".join(extra)

    # leave room for legend on right
    fig.tight_layout(rect=[0, 0, 0.82, 1])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print("[WROTE]", out_png, f"({extra_txt})")

# -------------------- Generate Setup A and Setup B --------------------
outA = OUT_DIR / f"Fig9_percentile_concordance_{SETUP_A['tag']}.png"
outB = OUT_DIR / f"Fig9_percentile_concordance_{SETUP_B['tag']}.png"

plot_percentile_concordance_one(SETUP_A["setup"], SETUP_A["anomaly"], SETUP_A["win"], SETUP_A["kfold"], outA)
plot_percentile_concordance_one(SETUP_B["setup"], SETUP_B["anomaly"], SETUP_B["win"], SETUP_B["kfold"], outB)