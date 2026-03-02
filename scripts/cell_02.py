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

# Outputs: FeatureRankOUT/DDR{4|5}_{W}_{K}_0_{compute|memory|sensors}.csv  (feature,score)

import re, hashlib, warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------ PATHS ------------------
OUT_DIR  = ROOT / "FeatureRankOUT"; OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------ CONFIG ------------------
WINDOW_SIZES: List[int] = [32, 64, 128, 256, 512, 1024]
OVERLAP: float          = 0.5        # 0.0.. <1.0 ; stride = int(W*(1-OVERLAP))
KFOLDS_SET: List[int]   = [3, 5, 10, 20]
SEED                    = 42
META                    = ["label","setup","run_id"]

# Sampling inside each window -> one scalar per signal per window (no means)
SAMPLING: str           = "center"   # {"center", "first", "last", "offset:<int>"}

# ------------------ File discovery (benign RAW only) ------------------
def is_benign_path(p: Path) -> bool:
    s = str(p).lower()
    if "benign" in s: 
        return True
    bad = ["attack","anom","fault","inject","trojan","mal","rh","droop","spectre","trrespass"]
    return not any(b in s for b in bad)

def detect_setup_from_path(p: Path) -> Optional[str]:
    s = str(p).lower()
    if "ddr4" in s: return "DDR4"
    if "ddr5" in s: return "DDR5"
    return None

def iter_raw_csvs(root: Path):
    for p in root.rglob("*.csv"):
        yield p

def read_csv_clean(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df

def mk_run_id(path: Path) -> str:
    return f"run_{hashlib.md5(str(path).encode('utf-8')).hexdigest()[:10]}"

def unify_benign_by_setup(data_dir: Path) -> Dict[str, List[Tuple[Path, pd.DataFrame]]]:
    buckets = {"DDR4": [], "DDR5": []}
    for p in iter_raw_csvs(data_dir):
        if not is_benign_path(p): 
            continue
        setup = detect_setup_from_path(p)
        if setup is None:
            continue
        try:
            df = read_csv_clean(p)
            if len(df) > 0:
                buckets[setup].append((p, df))
        except Exception as e:
            print(f"[WARN] failed reading {p}: {e}")
    print(f"[DISCOVER] benign RAW files → DDR4: {len(buckets['DDR4'])}, DDR5: {len(buckets['DDR5'])}")
    return buckets

# ------------------ Subspace mapping ------------------
PATS = {
    "memory": [
        r"ddr|dram|mem(ory)?\b", r"\bL1(\b|_)|L2(\b|_)|L3(\b|_)", r"(L[123].*(HIT|MISS|MPI))\b",
        r"cache|fill|evict|wb|rd|wr|load|store", r"bandwidth|bw|throughput|qdepth|queue",
        r"lat(ency)?|stall.*mem|tCCD|tRCD|tRP|tCL|page|row|col",
    ],
    "sensors": [
        r"temp|thermal|hot",
        r"volt|vdd|vcore|vin|vout|cpu[_\- ]?volt|cpu[_\- ]?vdd|vdd[_\- ]?cpu|core[_\- ]?volt|core[_\- ]?voltage",
        r"power|watt|energy|joule",
        r"fan|throttle|current|amps?",
    ],
    "compute": [
        r"\bIPC\b|\bPhysIPC\b|\bEXEC\b|\bissue|retire|dispatch\b",
        r"\bINST\b|INSTnom%|branch|mispred|\balu\b|\barith\b|\blogic\b",
        r"C0res%|C1res%|C6res%|C7res%|CFREQ|AFREQ|ACYC|CYC|TIME|clk|cycle|freq|util",
        r"\bcore|cpu|sm|warp|shader\b",
    ],
}
def subspace_of_feature(base_feat: str) -> str:
    b = str(base_feat)
    if any(re.search(p, b, flags=re.I) for p in PATS["memory"]):  return "memory"
    if any(re.search(p, b, flags=re.I) for p in PATS["sensors"]): return "sensors"
    return "compute"

# ------------------ RAW → overlapped windows → z-norm → sampling ------------------
def telemetry_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]

def _stride(win: int, overlap: float) -> int:
    step = max(1, int(round(win * (1.0 - overlap))))
    return max(1, min(step, win))  # never 0; allow step==win for non-overlap

def _pick_index(start: int, win: int) -> int:
    if SAMPLING == "first":
        return start
    if SAMPLING == "last":
        return start + win - 1
    if SAMPLING.startswith("offset:"):
        try:
            off = int(SAMPLING.split(":",1)[1])
        except Exception:
            off = win // 2
        off = np.clip(off, 0, win-1)
        return start + off
    # default = "center"
    return start + (win // 2)

def build_window_samples(df: pd.DataFrame, win: int, overlap: float,
                         setup: str, run_id: str) -> pd.DataFrame:
    """
    Produce one scalar per signal per window by sampling a single time index inside each window
    after global z-normalization per signal (no within-window mean).
    """
    cols = telemetry_cols(df)
    if not cols:
        return pd.DataFrame()

    X = df[cols].astype(float)
    # Global z-norm per signal (over full run)
    X = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=0) + 1e-9)
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    n = len(X)
    step = _stride(win, overlap)
    rows = []
    for start in range(0, n - win + 1, step):
        idx = _pick_index(start, win)
        sample = X.iloc[idx:idx+1].copy()
        sample["setup"] = setup
        sample["run_id"] = run_id
        sample["label"] = "BENIGN"
        rows.append(sample)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def build_all_window_samples(pairs: List[Tuple[Path, pd.DataFrame]],
                             setup: str, win: int, overlap: float) -> pd.DataFrame:
    out = []
    for p, df in pairs:
        rid = mk_run_id(p)
        part = build_window_samples(df, win=win, overlap=overlap, setup=setup, run_id=rid)
        if not part.empty:
            out.append(part)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

# ------------------ PC1 vs PC1 (MI) ------------------
def _zscore_df(X: pd.DataFrame) -> pd.DataFrame:
    Z = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=0) + 1e-9)
    return Z.replace([np.inf, -np.inf], 0.0).fillna(0.0)

def _quantile_bin(x: np.ndarray, q: int = 16) -> np.ndarray:
    x = np.asarray(x, float)
    mask = np.isfinite(x)
    if mask.sum() == 0:
        return np.full_like(x, -1, dtype=int)
    edges = np.quantile(x[mask], np.linspace(0,1,q+1))
    edges[0]  -= 1e-9; edges[-1] += 1e-9
    out = np.full_like(x, -1, dtype=int)
    out[mask] = np.digitize(x[mask], edges[1:-1], right=False)
    return out

def _discrete_mi(a: np.ndarray, b: np.ndarray) -> float:
    mask = (a >= 0) & (b >= 0)
    if mask.sum() == 0:
        return 0.0
    A = a[mask].astype(int); B = b[mask].astype(int)
    na, nb = A.max()+1, B.max()+1
    idx = (A * nb + B)
    binc = np.bincount(idx, minlength=na*nb).astype(float).reshape(na, nb)
    joint = binc / binc.sum()
    pa = joint.sum(axis=1, keepdims=True); pb = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = joint / (pa @ pb)
        mi = np.nansum(joint * np.log(ratio + 1e-12))
    return float(max(mi, 0.0))

def pc1_vs_pc1_mi_scores(df_windows: pd.DataFrame, qbins: int = 16) -> pd.Series:
    """
    For each feature i:
      - per-signal PC1 = z-scored column Xi (samples = windows)
      - others PC1     = PCA(X_{-i}) first PC scores
      - score_i        = MI(qbin(Xi), qbin(PC1_{others}))
    """
    feat_cols = [c for c in df_windows.columns if c not in META and pd.api.types.is_numeric_dtype(df_windows[c])]
    if not feat_cols:
        return pd.Series(dtype=float)

    X = _zscore_df(df_windows[feat_cols].astype(float))
    names = list(X.columns)
    X_arr = X.values
    n, d = X_arr.shape
    scores = []

    for i in range(d):
        if d == 1:
            scores.append(0.0); continue
        # Per-signal PC1 (1-D) = standardized Xi
        xi = X_arr[:, i]
        # Others PC1
        others = np.delete(X_arr, i, axis=1)
        # If degenerate, fallback to mean of others (still no window mean used anywhere; it's across windows)
        try:
            pca = PCA(n_components=1, svd_solver="auto", random_state=SEED)
            pc1_scores = pca.fit_transform(others).ravel()
        except Exception:
            pc1_scores = others.mean(axis=1)

        ai = _quantile_bin(xi, q=qbins)
        bi = _quantile_bin(pc1_scores, q=qbins)
        scores.append(_discrete_mi(ai, bi))

    s = pd.Series(scores, index=names, name="score").sort_values(ascending=False)
    return s

def _normalize_01(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    lo, hi = float(s.min()), float(s.max())
    if not np.isfinite(lo) or not np.isfinite(hi):
        return s.fillna(0.0)
    if hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)

def split_and_save(scores: pd.Series, setup: str, win: int, kfold: int):
    s_norm = _normalize_01(scores)
    buckets = {"compute": [], "memory": [], "sensors": []}
    for feat, sc in s_norm.items():
        buckets[subspace_of_feature(feat)].append((feat, sc))
    for sub, pairs in buckets.items():
        if not pairs:
            continue
        df_sub = (
            pd.DataFrame(pairs, columns=["feature","score"])
              .sort_values("score", ascending=False, kind="mergesort")
        )
        out_path = OUT_DIR / f"{setup}_{win}_{kfold}_0_{sub}.csv"
        df_sub.to_csv(out_path, index=False)
        print(f"[WRITE] {out_path}")

# ------------------ MAIN ------------------
if __name__ == "__main__":
    buckets = unify_benign_by_setup(DATA_DIR)

    for setup in ["DDR4","DDR5"]:
        pairs = buckets.get(setup, [])
        if not pairs:
            print(f"[WARN] No benign RAW {setup} files found. Skipping.")
            continue

        for W in WINDOW_SIZES:
            df_win = build_all_window_samples(pairs, setup=setup, win=W, overlap=OVERLAP)
            if df_win.empty:
                print(f"[SKIP] {setup} win={W}: no window samples produced.")
                continue

            # type guard for meta
            for m in META:
                if m in df_win.columns:
                    df_win[m] = df_win[m].astype(str)

            for K in KFOLDS_SET:
                # Prefer grouping by run_id to avoid leakage
                if "run_id" in df_win.columns and df_win["run_id"].nunique() >= K:
                    splitter = GroupKFold(n_splits=K)
                    splits = splitter.split(df_win, groups=df_win["run_id"].values)
                else:
                    splitter = KFold(n_splits=K, shuffle=True, random_state=SEED)
                    splits = splitter.split(df_win)

                fold_scores = []
                for tr_idx, _ in splits:
                    df_tr = df_win.iloc[tr_idx]
                    s = pc1_vs_pc1_mi_scores(df_tr, qbins=16)
                    fold_scores.append(s)

                if not fold_scores:
                    print(f"[SKIP] {setup} win={W} k={K}: no folds produced.")
                    continue

                M = pd.concat(fold_scores, axis=1).fillna(0.0)
                mean_s = M.mean(axis=1).sort_values(ascending=False)
                split_and_save(mean_s, setup=setup, win=W, kfold=K)

    print("\n[DONE] RAW PC1_vs_PC1 (MI) K-fold rankings written.")