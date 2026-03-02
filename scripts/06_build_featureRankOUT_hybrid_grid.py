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

# === JUPYTER: Build FeatureRankOUT_HYBRID PER PLATFORM — ACROSS ALL (WIN,KF) GRID ===
# UPDATED per your request:
#   ✅ "building hybrid ranking across WIN and F like CP-MI ranking"
#
# What this does:
#   - Instead of hard-coding (WIN,KF) per platform, it scans ALL (WINS × KFOLDS) available,
#     exactly like your CP-MI rank grid exists.
#   - For each platform (DDR4, DDR5) and for each (win,kfold):
#       1) Load CP-MI ranks from FeatureRankOUT/<setup>_<win>_<kfold>_0_{sub}.csv
#       2) Pool SHAP across that platform's anomalies at that SAME (win,kfold),
#          using the best pct + method per anomaly from:
#             Results/BEST_in_DesignSpace_Post_per_platform_details.csv
#       3) Join CP-MI and pooled SHAP by normalized feature name
#       4) Compute hybrid_score per subspace and write:
#             FeatureRankOUT_HYBRID/<setup>_<win>_<kfold>_0_{compute|memory|sensors}.csv
#       5) Save the joined table for audit:
#             Results/Explainability_SHAP_BestPlatforms/cmp_cpmi_vs_shap/PLATFORM_<setup>_WIN<w>_KF<k>__SHAP_vs_CPMI_joined.csv
#
# Inputs:
#   - Results/BEST_in_DesignSpace_Post_per_platform_details.csv
#   - FeatureRankOUT/<setup>_<win>_<kfold>_0_{compute|memory|sensors}.csv  (CP-MI ranks)
#   - Results/Explainability_SHAP_BestPlatforms/SHAP_BESTPLAT_full_<setup>_<anomaly>_WIN<w>_KF<k>_PCT<p>_M<method>.csv (SHAP)
#
# Output:
#   - FeatureRankOUT_HYBRID/<setup>_<win>_<kfold>_0_{compute|memory|sensors}.csv
#
# Notes:
#   - If SHAP for a given (win,kfold) isn't available (no SHAP files), it will SKIP that combo.
#   - If CP-MI ranks missing for a given (win,kfold), it will SKIP that combo.
# --------------------------------------------------------------------------------------------

import glob, re
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd

SUBSPACES = ("compute", "memory", "sensors")

# ------------------ Normalization ------------------
def _norm_name(s: str) -> str:
    s = re.sub(r"\s+", "_", str(s)).lower()
    s = s.replace("%", "pct")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def _read_csv(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

# ------------------ Locate CP-MI rank CSVs ------------------
def _find_cpmi_csvs(root: Path, setup: str, win: int, kfold: int, rank_dirs=None) -> Dict[str, Optional[Path]]:
    out = {s: None for s in SUBSPACES}
    search_roots = rank_dirs or [
        root / "FeatureRankOUT",
        ROOT / "FeatureRankOUT",
        ROOT / "FeatureRankOUT",
        Path.home() / "Desktop" / "octaneX" / "FeatureRankOUT",
    ]
    for sub in SUBSPACES:
        fname = f"{setup}_{win}_{kfold}_0_{sub}.csv"
        for rd in search_roots:
            p = Path(rd) / fname
            if p.exists():
                out[sub] = p
                break
        if out[sub] is None:
            for rd in search_roots:
                rd = Path(rd)
                if not rd.exists():
                    continue
                hits = list(rd.rglob(f"*{setup}*{sub}*CPMI*.csv")) + list(rd.rglob(f"*{setup}*{sub}*CP-MI*.csv"))
                if hits:
                    out[sub] = Path(hits[0])
                    break
    return out

def _read_cpmi_rank(root: Path, setup: str, win: int, kfold: int, rank_dirs=None) -> pd.DataFrame:
    """
    Return a long table:
      feature_cpmi, subspace_cpmi, cpmi_score, cpmi_rank, feature_norm
    """
    paths = _find_cpmi_csvs(root, setup, win, kfold, rank_dirs=rank_dirs)
    rows = []
    for sub, p in paths.items():
        if p is None:
            continue
        df = _read_csv(p)
        if df.empty:
            continue

        cols_lower = {c.lower(): c for c in df.columns}
        fcol = cols_lower.get("feature") or list(df.columns)[0]

        # Prefer a numeric score column if present
        num_cols = [c for c in df.columns if c != fcol and pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            s_col = num_cols[0]
            scores = pd.to_numeric(df[s_col], errors="coerce").fillna(0.0).values
        else:
            # fallback: descending pseudo-scores by position
            scores = np.linspace(1.0, 0.0, num=len(df), endpoint=False)

        tmp = pd.DataFrame({
            "feature_cpmi": df[fcol].astype(str).values,
            "subspace_cpmi": sub,
            "cpmi_score": scores
        })
        tmp["cpmi_rank"] = pd.Series(scores).rank(ascending=False, method="dense").astype(int)
        tmp["feature_norm"] = tmp["feature_cpmi"].map(_norm_name)
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["feature_cpmi","subspace_cpmi","cpmi_score","cpmi_rank","feature_norm"])
    return pd.concat(rows, ignore_index=True)

# ------------------ SHAP BESTPLAT full reader ------------------
def _find_shap_bestplat(expl_dir: Path, setup: str, anomaly: str, win: int, kfold: int, pct: int, method: str) -> Optional[Path]:
    exact = expl_dir / f"SHAP_BESTPLAT_full_{setup}_{anomaly}_WIN{win}_KF{kfold}_PCT{pct}_M{method}.csv"
    if exact.exists():
        return exact
    # fallback: any pct for same (setup,anomaly,win,kfold,method)
    pat = str(expl_dir / f"SHAP_BESTPLAT_full_{setup}_{anomaly}_WIN{win}_KF{kfold}_PCT*_M{method}.csv")
    hits = sorted(glob.glob(pat))
    return Path(hits[0]) if hits else None

def _read_shap_bestplat(expl_dir: Path, setup: str, anomaly: str, win: int, kfold: int, pct: int, method: str) -> pd.DataFrame:
    """
    Returns long table:
      feature_shap, subspace_shap, shap_mean_abs, shap_rank, feature_norm
    """
    p = _find_shap_bestplat(expl_dir, setup, anomaly, win, kfold, pct, method)
    if p is None:
        return pd.DataFrame(columns=["feature_shap","subspace_shap","shap_mean_abs","shap_rank","feature_norm"])
    df = _read_csv(p)
    if df.empty:
        return pd.DataFrame(columns=["feature_shap","subspace_shap","shap_mean_abs","shap_rank","feature_norm"])

    cols_lower = {c.lower(): c for c in df.columns}
    fcol = cols_lower.get("feature") or list(df.columns)[0]
    scol = cols_lower.get("shap_mean_abs") or cols_lower.get("importance")
    subc = cols_lower.get("subspace")

    out = pd.DataFrame({"feature_shap": df[fcol].astype(str)})
    out["subspace_shap"] = df[subc].astype(str).str.lower() if subc else "compute"
    out["shap_mean_abs"] = pd.to_numeric(df[scol], errors="coerce").fillna(0.0) if scol else 0.0
    out["feature_norm"]  = out["feature_shap"].map(_norm_name)

    # If multiple rows map to same normalized feature/subspace, keep MAX SHAP
    out = (out.groupby(["feature_norm","subspace_shap"], as_index=False)["shap_mean_abs"]
              .max()
              .merge(out.drop_duplicates(["feature_norm","subspace_shap"])[["feature_norm","subspace_shap","feature_shap"]],
                     on=["feature_norm","subspace_shap"], how="left"))

    out["shap_rank"] = out.groupby("subspace_shap")["shap_mean_abs"].rank(ascending=False, method="dense").astype(int)
    return out[["feature_shap","subspace_shap","shap_mean_abs","shap_rank","feature_norm"]]

# ------------------ Pool SHAP across platform anomalies for a given (win,kfold) ------------------
def _pool_shap_for_platform(details_df: pd.DataFrame, expl_dir: Path, setup: str, win: int, kfold: int) -> pd.DataFrame:
    """
    Pool SHAP across all anomalies for this (setup,win,kfold):
      - Use best_pct_by_median + method per anomaly from BEST_in_DesignSpace_Post_per_platform_details.csv
      - Load SHAP_BESTPLAT_full_* for each anomaly (if exists)
      - Pool by (feature_norm, subspace) using MAX shap_mean_abs across anomalies
    """
    mask = (
        (details_df["setup"].astype(str).str.upper() == setup.upper()) &
        (pd.to_numeric(details_df["win"], errors="coerce") == int(win)) &
        (pd.to_numeric(details_df["kfold"], errors="coerce") == int(kfold))
    )
    sub = details_df.loc[mask].copy()
    if sub.empty:
        return pd.DataFrame(columns=["feature_shap","subspace_shap","shap_mean_abs","shap_rank","feature_norm"])

    shap_parts = []
    for _, r in sub.iterrows():
        anomaly = str(r["anomaly"]).strip().upper()
        pct = int(pd.to_numeric(r["best_pct_by_median"], errors="coerce"))
        method = str(r["method"]).strip()
        s = _read_shap_bestplat(expl_dir, setup, anomaly, win, kfold, pct, method)
        if not s.empty:
            s = s.copy()
            s["anomaly"] = anomaly
            shap_parts.append(s)

    if not shap_parts:
        return pd.DataFrame(columns=["feature_shap","subspace_shap","shap_mean_abs","shap_rank","feature_norm"])

    shap_all = pd.concat(shap_parts, ignore_index=True)

    pooled = (shap_all.groupby(["feature_norm","subspace_shap"], as_index=False)["shap_mean_abs"]
                    .max()
                    .merge(shap_all.drop_duplicates(["feature_norm","subspace_shap"])[["feature_norm","subspace_shap","feature_shap"]],
                           on=["feature_norm","subspace_shap"], how="left"))

    pooled["shap_rank"] = pooled.groupby("subspace_shap")["shap_mean_abs"].rank(ascending=False, method="dense").astype(int)
    return pooled[["feature_shap","subspace_shap","shap_mean_abs","shap_rank","feature_norm"]]

# ------------------ Hybrid scoring + writer ------------------
def _percentile(series: pd.Series) -> pd.Series:
    x = pd.Series(series).astype(float).fillna(0.0)
    if len(x) <= 1:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    r = x.rank(ascending=True, method="average")
    return (r - 1) / (len(x) - 1)

def _write_hybrid_from_joined(root: Path, joined_df: pd.DataFrame, setup: str, win: int, kfold: int,
                              alpha=0.60, beta=0.40, lam=0.05, q=0.80):
    """
    Writes one hybrid CSV per subspace:
      FeatureRankOUT_HYBRID/<setup>_<win>_<kfold>_0_<sub>.csv
    """
    hyb_dir = root / "FeatureRankOUT_HYBRID"
    hyb_dir.mkdir(parents=True, exist_ok=True)

    for sub in ("compute", "memory", "sensors"):
        sub_df = joined_df[joined_df["subspace"] == sub].copy()
        if sub_df.empty:
            continue

        sub_df["cpmi_score"] = pd.to_numeric(sub_df.get("cpmi_score", 0.0), errors="coerce").fillna(0.0)
        sub_df["shap_mean_abs"] = pd.to_numeric(sub_df.get("shap_mean_abs", 0.0), errors="coerce").fillna(0.0)

        sub_df["r_cpmi"] = _percentile(sub_df["cpmi_score"])
        sub_df["r_shap"] = _percentile(sub_df["shap_mean_abs"])

        thr = np.nanpercentile(sub_df["r_shap"], q * 100.0) if len(sub_df) else 1.0
        sub_df["bonus"] = (sub_df["r_shap"] >= thr).astype(float) * lam

        sub_df["hybrid_score"] = alpha * sub_df["r_cpmi"] + beta * sub_df["r_shap"] + sub_df["bonus"]
        sub_df["hybrid_rank"]  = sub_df["hybrid_score"].rank(ascending=False, method="dense").astype(int)

        out = hyb_dir / f"{setup}_{win}_{kfold}_0_{sub}.csv"
        sub_df[["feature_display","hybrid_score","hybrid_rank","cpmi_score","shap_mean_abs"]].to_csv(out, index=False)
        print(f"[OK] Hybrid written → {out}")

# ------------------ Discover available (win,kfold) combos ------------------
def _discover_grid_from_cpmi(root: Path, setup: str, rank_dirs=None) -> List[Tuple[int,int]]:
    """
    Scan FeatureRankOUT files for this setup to discover all (win,kfold) combos present.
    Looks for: <setup>_<win>_<kfold>_0_<sub>.csv
    """
    search_roots = rank_dirs or [
        root / "FeatureRankOUT",
        ROOT / "FeatureRankOUT",
        ROOT / "FeatureRankOUT",
        Path.home() / "Desktop" / "octaneX" / "FeatureRankOUT",
    ]
    combos = set()
    pat = re.compile(rf"^{re.escape(setup)}_(\d+)_(\d+)_0_(compute|memory|sensors)\.csv$", re.I)

    for rd in search_roots:
        rd = Path(rd)
        if not rd.exists():
            continue
        for p in rd.glob(f"{setup}_*_*_0_*.csv"):
            m = pat.match(p.name)
            if not m:
                continue
            win = int(m.group(1))
            kf  = int(m.group(2))
            combos.add((win, kf))

    return sorted(combos, key=lambda x: (x[0], x[1]))

# ------------------ Main driver ------------------
def build_hybrid_rankouts_per_platform_across_grid(
    root_path: str,
    alpha=0.60, beta=0.40, lam=0.05, q=0.80,
    rank_dirs=None,
):
    root = Path(root_path).expanduser().resolve()
    res_dir  = root / "Results"
    expl_dir = res_dir / "Explainability_SHAP_BestPlatforms"
    details_path = res_dir / "BEST_in_DesignSpace_Post_per_platform_details.csv"

    if not details_path.exists():
        raise FileNotFoundError(f"Missing per-platform details CSV: {details_path}")

    details = pd.read_csv(details_path)
    details.columns = [c.lower() for c in details.columns]
    if "best_method" in details.columns and "method" not in details.columns:
        details = details.rename(columns={"best_method":"method"})
    if "best_pct_by_median" not in details.columns and "pct" in details.columns:
        details = details.rename(columns={"pct":"best_pct_by_median"})

    need = {"setup","anomaly","win","kfold","best_pct_by_median","method"}
    miss = need - set(details.columns)
    if miss:
        raise KeyError(f"Details CSV missing required columns: {sorted(miss)}")

    out_join_dir = expl_dir / "cmp_cpmi_vs_shap"
    out_join_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    for setup in ["DDR4", "DDR5"]:
        print(f"\n[PLATFORM] {setup}: discovering CP-MI grid ...")
        combos = _discover_grid_from_cpmi(root, setup, rank_dirs=rank_dirs)
        if not combos:
            print(f"[WARN] No CP-MI rank files found for {setup}. Skipping.")
            continue

        print(f"[INFO] {setup}: found {len(combos)} (WIN,KF) combos from CP-MI ranks.")

        for win, kfold in combos:
            # 1) load CP-MI ranks for this (win,kfold)
            cpmi = _read_cpmi_rank(root, setup, win, kfold, rank_dirs=rank_dirs)
            if cpmi.empty:
                skipped += 1
                continue

            # 2) pool SHAP for this (win,kfold) using per-platform details (best pct/method per anomaly)
            shap = _pool_shap_for_platform(details, expl_dir, setup, win, kfold)
            if shap.empty:
                # no SHAP available for this combo -> skip
                skipped += 1
                continue

            # 3) join
            joined = pd.merge(
                cpmi.drop_duplicates(["feature_norm","subspace_cpmi"]),
                shap.drop_duplicates(["feature_norm","subspace_shap"]),
                on="feature_norm",
                how="outer",
                suffixes=("_cpmi", "_shap")
            )

            joined["feature_display"] = joined["feature_cpmi"].fillna(joined["feature_shap"])
            joined["subspace"] = joined["subspace_cpmi"].fillna(joined["subspace_shap"])

            joined["cpmi_rank"] = pd.to_numeric(joined.get("cpmi_rank"), errors="coerce")
            joined["shap_rank"] = pd.to_numeric(joined.get("shap_rank"), errors="coerce")
            joined["cpmi_score"] = pd.to_numeric(joined.get("cpmi_score"), errors="coerce").fillna(0.0)
            joined["shap_mean_abs"] = pd.to_numeric(joined.get("shap_mean_abs"), errors="coerce").fillna(0.0)

            joined["delta_rank"]  = joined["cpmi_rank"].fillna(np.inf) - joined["shap_rank"].fillna(np.inf)
            joined["delta_score"] = joined["shap_mean_abs"] - joined["cpmi_score"]

            joined = joined.sort_values(["subspace", "shap_rank", "cpmi_rank"], na_position="last")

            out_csv = out_join_dir / f"PLATFORM_{setup}_WIN{win}_KF{kfold}__SHAP_vs_CPMI_joined.csv"
            joined.to_csv(out_csv, index=False)
            print(f"[OK] Joined saved → {out_csv.name}")

            # 4) write hybrid rankouts per subspace
            _write_hybrid_from_joined(
                root, joined, setup, win, kfold,
                alpha=alpha, beta=beta, lam=lam, q=q
            )

            processed += 1

    print(f"\n[DONE] Hybrid rankouts built for {processed} (platform,WIN,KF) combos. Skipped={skipped}.")
    print(f"       Output folder: {root/'FeatureRankOUT_HYBRID'}")

# -------------------- RUN HERE --------------------
build_hybrid_rankouts_per_platform_across_grid(
    root_path=str(ROOT),
    alpha=0.60, beta=0.40, lam=0.05, q=0.80,
    rank_dirs=None,   # optionally pass your own RANK_DIRS list
)