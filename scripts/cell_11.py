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

# ==== Cell: Consistency bars (CP-MI selection supported by SHAP) — PER PLATFORM ====
# PER-PLATFORM revision of your "BEST cases" script.
#
# Reads platform winners from:
#   Results/BEST_in_DesignSpace_Post_per_platform_details.csv
#
# Uses SHAP outputs from:
#   Results/Explainability_SHAP_BestPlatforms/SHAP_BESTPLAT_full_<setup>_<anomaly>_WIN<w>_KF<k>_PCT<p>_M<method>.csv
#
# CP-MI rank files under FeatureRankOUT:
#   <setup>_<win>_<kfold>_0_<subspace>.csv  (preferred)
#
# Produces:
#   - ONE plot per PLATFORM (DDR4, DDR5):
#       Results/Explainability_SHAP_BestPlatforms/consistency_v2_platform/
#         CONSISTENCY_BARS_PLATFORM_<setup>_WIN<w>_KF<k>_PLOT.png
#   - Also writes platform-level counts CSV:
#       CONSISTENCY_BARS_PLATFORM_counts.csv
#
# How it works (platform-level):
#   1) For a platform, choose (WIN,KF) from the details file (consistent across anomalies).
#   2) CP-MI selection:
#        - For each anomaly, select top-<pct>% CP-MI features per subspace (from CP-MI rank lists)
#        - Union these anomaly selections (so CP-MI selection reflects the platform's anomaly mix)
#   3) SHAP support:
#        - Load SHAP tables for each anomaly at its best config (method+pct+WIN+KF)
#        - For each anomaly, split features into SHAP-low vs SHAP-high by median SHAP
#        - Count how many CP-MI-selected features fall into SHAP-high vs SHAP-low, per subspace
#        - Sum counts across anomalies (platform totals)
#
# Note:
#   - This is a COUNT-based view (like your original). It does NOT weight by SHAP magnitude.
# -------------------------------------------------------------------------------

import os, re, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Paths (reuse if already defined in your notebook) -----------------------

DETAILS_CSV = RES_DIR / "BEST_in_DesignSpace_Post_per_platform_details.csv"
EXPL_DIR    = RES_DIR / "Explainability_SHAP_BestPlatforms"  # SHAP_BESTPLAT_full_*.csv live here
CONS_DIR    = EXPL_DIR / "consistency_v2_platform"
CONS_DIR.mkdir(parents=True, exist_ok=True)

# CP-MI rank roots (use global RANK_DIRS if already set)
if "RANK_DIRS" in globals():
    RANK_DIRS = list(globals()["RANK_DIRS"])
else:
    RANK_DIRS = [
        ROOT / "FeatureRankOUT",
        Path("/Volumes/Untitled") / "FeatureRankOUT",
        Path("/Volumes/Untitled") / "octaneX" / "FeatureRankOUT",
        Path.home() / "Desktop" / "octaneX" / "FeatureRankOUT",
    ]

SUBSPACES = ("compute","memory","sensors")

# --- Helpers -----------------------------------------------------------------
def _norm(s: str) -> str:
    """Normalize feature names to increase intersection chances."""
    s = re.sub(r"\s+", "_", str(s)).lower()
    s = s.replace("%", "pct")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def _read_csv_safe(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def _read_cpmi_list(setup: str, win: int, kfold: int, sub: str) -> list[str]:
    """
    Load CP-MI FeatureRankOUT list for one subspace. Returns ordered original names.
    Expected filename: <setup>_<win>_<kfold>_0_<sub>.csv
    Falls back to *CPMI*/*CP-MI* patterns.
    """
    fname = f"{setup}_{win}_{kfold}_0_{sub}.csv"
    # 1) direct hits
    for rd in RANK_DIRS:
        p = Path(rd) / fname
        if p.exists():
            df = _read_csv_safe(p)
            if df.empty:
                continue
            col = "feature" if "feature" in df.columns else df.columns[0]
            return df[col].dropna().astype(str).tolist()
    # 2) fallback search
    for rd in RANK_DIRS:
        rd = Path(rd)
        if not rd.exists():
            continue
        hits = list(rd.rglob(f"*{setup}*{sub}*CPMI*.csv")) + list(rd.rglob(f"*{setup}*{sub}*CP-MI*.csv"))
        if hits:
            df = _read_csv_safe(hits[0])
            if df.empty:
                continue
            col = "feature" if "feature" in df.columns else df.columns[0]
            return df[col].dropna().astype(str).tolist()
    return []

def _cpmi_selection_at_pct(setup: str, win: int, kfold: int, pct: int) -> dict:
    """
    Return dict{subspace: [features]}: top-<pct>% per subspace (ceil, min 1 if pct>0).
    """
    out = {}
    for sub in SUBSPACES:
        feats = _read_cpmi_list(setup, win, kfold, sub)
        if feats:
            k = int(np.ceil(len(feats) * (pct / 100.0)))
            if pct > 0 and k == 0:
                k = 1
            out[sub] = feats[:k]
        else:
            out[sub] = []
    return out

def _find_shap_bestplat_full(setup: str, anomaly: str, win: int, kfold: int, pct: int, method: str) -> Path | None:
    """
    Prefer exact BEST file:
      SHAP_BESTPLAT_full_<setup>_<anomaly>_WIN<w>_KF<k>_PCT<p>_M<method>.csv
    Fallback: any PCT for the same (setup, anomaly, win, kfold, method).
    """
    exact = EXPL_DIR / f"SHAP_BESTPLAT_full_{setup}_{anomaly}_WIN{win}_KF{kfold}_PCT{pct}_M{method}.csv"
    if exact.exists():
        return exact
    pat = str(EXPL_DIR / f"SHAP_BESTPLAT_full_{setup}_{anomaly}_WIN{win}_KF{kfold}_PCT*_M{method}.csv")
    alts = sorted(glob.glob(pat))
    return Path(alts[0]) if alts else None

def _split_high_low(shap_df: pd.DataFrame):
    """
    Binary split by median SHAP so both sides are non-empty when >1 feature.
    Returns (low_features, high_features) as pd.Series of feature strings.
    """
    if shap_df.empty:
        return pd.Series([], dtype=str), pd.Series([], dtype=str)
    med = shap_df["shap_mean_abs"].median()
    high_idx = shap_df[shap_df["shap_mean_abs"] >= med]["feature"].astype(str)
    low_idx  = shap_df[shap_df["shap_mean_abs"]  < med]["feature"].astype(str)
    return low_idx, high_idx

def _debug_platform(setup, win, kfold, detail_rows, by_sub_counts):
    print(f"\n[PLATFORM CHECK] {setup}  WIN={win}  K={kfold}")
    print("  anomalies:", ", ".join(sorted(detail_rows["anomaly"].astype(str).unique().tolist())))
    for sub in SUBSPACES:
        lo = by_sub_counts[sub]["low"]
        hi = by_sub_counts[sub]["high"]
        tot = lo + hi
        print(f"  {sub}: SHAP-low={lo}  SHAP-high={hi}  total={tot}")

# --- Main --------------------------------------------------------------------
def plot_cpmi_supported_by_shap_per_platform():
    if not DETAILS_CSV.exists():
        raise FileNotFoundError(f"Missing platform details CSV: {DETAILS_CSV}")

    details = pd.read_csv(DETAILS_CSV).copy()
    details.columns = [c.lower() for c in details.columns]

    # normalize column names
    if "best_method" in details.columns and "method" not in details.columns:
        details = details.rename(columns={"best_method":"method"})
    if "best_pct_by_median" not in details.columns and "pct" in details.columns:
        details = details.rename(columns={"pct":"best_pct_by_median"})

    need = {"setup","anomaly","win","kfold","best_pct_by_median","method"}
    missing = need - set(details.columns)
    if missing:
        raise KeyError(f"Platform details CSV missing columns: {sorted(missing)}. Have: {list(details.columns)}")

    out_rows = []

    for setup in ["DDR4","DDR5"]:
        dplat = details[details["setup"].astype(str).str.upper() == setup].copy()
        if dplat.empty:
            print(f"[WARN] No detail rows for platform={setup}")
            continue

        # WIN/K should be consistent across anomalies for the platform
        win = int(pd.to_numeric(dplat["win"], errors="coerce").dropna().iloc[0])
        kfold = int(pd.to_numeric(dplat["kfold"], errors="coerce").dropna().iloc[0])

        # Platform aggregate counts
        by_sub = {sub: {"low": 0, "high": 0} for sub in SUBSPACES}

        # --- Loop anomalies in this platform ---
        for _, r in dplat.iterrows():
            anomaly = str(r["anomaly"]).strip()
            pct     = int(pd.to_numeric(r["best_pct_by_median"], errors="coerce"))
            method  = str(r["method"]).strip()

            # 1) CP-MI selection for THIS anomaly at its best pct (based on CP-MI lists at platform WIN/K)
            sel = _cpmi_selection_at_pct(setup, win, kfold, pct)

            # Build per-subspace normalized selection sets (for quick membership)
            sel_norm = {sub: set(_norm(f) for f in sel[sub]) for sub in SUBSPACES}

            # 2) Load SHAP best full for THIS anomaly
            shap_csv = _find_shap_bestplat_full(setup, anomaly, win, kfold, pct, method)
            shap_df = pd.DataFrame()
            if shap_csv is not None and shap_csv.exists():
                shap_df = _read_csv_safe(shap_csv)
                if not shap_df.empty:
                    shap_df.columns = [c.lower() for c in shap_df.columns]
                    if "feature" not in shap_df.columns:
                        shap_df = shap_df.rename(columns={shap_df.columns[0]: "feature"})
                    if "shap_mean_abs" not in shap_df.columns and "importance" in shap_df.columns:
                        shap_df = shap_df.rename(columns={"importance": "shap_mean_abs"})
                    if "subspace" not in shap_df.columns:
                        # if missing, we can't do subspace-wise counts reliably
                        shap_df["subspace"] = "compute"
                    shap_df["feature"] = shap_df["feature"].astype(str)
                    shap_df["subspace"] = shap_df["subspace"].astype(str).str.lower()
                    shap_df["shap_mean_abs"] = pd.to_numeric(shap_df["shap_mean_abs"], errors="coerce").fillna(0.0)

            # If missing SHAP, skip this anomaly gracefully
            if shap_df.empty:
                print(f"[WARN] Missing/empty SHAP for {setup}/{anomaly} (WIN={win} K={kfold} PCT={pct} M={method})")
                continue

            # 3) SHAP split (low/high by median)
            low_f, high_f = _split_high_low(shap_df)
            low_norm  = set(low_f.map(_norm))
            high_norm = set(high_f.map(_norm))

            # 4) Count CP-MI-selected features that fall into SHAP-low vs SHAP-high, per subspace
            #    We do membership by normalized name (robust)
            for sub in SUBSPACES:
                if not sel_norm[sub]:
                    continue
                # A feature is "supported by SHAP-high" if it appears in high_norm
                hi = len(sel_norm[sub] & high_norm)
                # Count "low" as those selected but not in high; we can intersect with low_norm explicitly
                lo = len(sel_norm[sub] & low_norm)
                # If some selected features are in neither (e.g., not present in SHAP file), ignore them
                by_sub[sub]["high"] += hi
                by_sub[sub]["low"]  += lo

        # --- Plot ONE grouped-bar figure per platform ---
        cats  = list(SUBSPACES)
        lows  = [by_sub[s]["low"]  for s in cats]
        highs = [by_sub[s]["high"] for s in cats]

        fig = plt.figure(figsize=(12, 6))
        x = np.arange(len(cats)); w = 0.35
        plt.bar(x - w/2, lows,  width=w, label="SHAP-low")
        plt.bar(x + w/2, highs, width=w, label="SHAP-high")
        plt.xticks(x, cats)
        plt.ylabel("Count of CP-MI-selected features")
        plt.title(f"{setup} • WIN={win} K={kfold}\nCP-MI selection supported by SHAP (summed across anomalies)")
        plt.legend()
        plt.tight_layout()

        outp = CONS_DIR / f"CONSISTENCY_BARS_PLATFORM_{setup}_WIN{win}_KF{kfold}_PLOT.png"
        fig.savefig(outp, dpi=220)
        plt.close(fig)
        print(f"[OK] Plot saved: {outp}")

        # record counts for CSV
        for sub in SUBSPACES:
            out_rows.append({
                "setup": setup, "win": win, "kfold": kfold,
                "subspace": sub,
                "count_shap_low": int(by_sub[sub]["low"]),
                "count_shap_high": int(by_sub[sub]["high"]),
                "count_total_counted": int(by_sub[sub]["low"] + by_sub[sub]["high"]),
                "note": "counts are summed across platform anomalies; only features present in SHAP file contribute",
            })

        _debug_platform(setup, win, kfold, dplat, by_sub)

    # Save platform-level counts
    if out_rows:
        out_df = pd.DataFrame(out_rows)
        out_csv = CONS_DIR / "CONSISTENCY_BARS_PLATFORM_counts.csv"
        out_df.to_csv(out_csv, index=False)
        print(f"\n[OK] Counts CSV → {out_csv}")
    else:
        print("\n[WARN] No platform consistency counts produced (check SHAP + details CSV).")

# Run
plot_cpmi_supported_by_shap_per_platform()