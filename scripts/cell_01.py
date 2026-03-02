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

# ==== Cell 1: shared helpers ====
import re, hashlib, numpy as np, pandas as pd

def iter_raw_csvs(root: Path):
    for p in root.rglob("*.csv"): yield p

def read_csv_clean(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]

def detect_setup_from_path(p: Path):
    s = str(p).lower()
    return "DDR4" if "ddr4" in s else ("DDR5" if "ddr5" in s else None)

def is_benign_path(p: Path) -> bool:
    s = str(p).lower()
    if "benign" in s: return True
    bad = ["attack","anom","fault","inject","trojan","mal","rh","droop","spectre","trrespass"]
    return not any(b in s for b in bad)

def is_anomaly_path(p: Path, anomaly: str) -> bool:
    return (anomaly.lower() in str(p).lower()) and (not is_benign_path(p))

def collect_raw_pairs_by_setup(data_dir: Path, which: str, anomaly: str|None=None):
    out = {"DDR4": [], "DDR5": []}
    for p in iter_raw_csvs(data_dir):
        setup = detect_setup_from_path(p)
        if setup is None: continue
        if which == "benign":
            if not is_benign_path(p): continue
        else:
            if anomaly is None or not is_anomaly_path(p, anomaly): continue
        try:
            out[setup].append((p, read_csv_clean(p)))
        except Exception as e:
            print(f"[WARN] read failed {p}: {e}")
    return out

def mk_run_id(path: Path) -> str:
    return f"run_{hashlib.md5(str(path).encode('utf-8')).hexdigest()[:10]}"

def telemetry_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META and df[c].dtype.kind in "fcbiu"]

def window_collapse_means(df: pd.DataFrame, win: int, setup: str, run_id: str, label: str) -> pd.DataFrame:
    cols = telemetry_cols(df)
    if not cols: return pd.DataFrame()
    rows = []
    for start in range(0, len(df) - win + 1, win):
        means = df.iloc[start:start+win][cols].astype(float).mean(axis=0, numeric_only=True)
        row = means.to_frame().T
        row["setup"] = setup; row["run_id"] = run_id; row["label"] = label
        rows.append(row)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def build_windowed_raw_means(pairs, setup: str, win: int, label: str) -> pd.DataFrame:
    out = []
    for p, df in pairs:
        agg = window_collapse_means(df, win=win, setup=setup, run_id=mk_run_id(p), label=label)
        if not agg.empty: out.append(agg)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

# robust scaling (winsor + z) fit on BENIGN
def robust_scale_train(Xb_np: np.ndarray, winsor=(2.0, 98.0)):
    Q1, Q2 = np.percentile(Xb_np, winsor[0], axis=0), np.percentile(Xb_np, winsor[1], axis=0)
    mu = np.clip(Xb_np, Q1, Q2).mean(axis=0)
    sd = np.clip(Xb_np, Q1, Q2).std(axis=0, ddof=0) + 1e-9
    return mu, sd, Q1, Q2

def apply_robust_scale(X: pd.DataFrame, mu, sd, Q1, Q2):
    Z = (np.clip(X.values, Q1, Q2) - mu) / sd
    return pd.DataFrame(Z, columns=X.columns, index=X.index)

# subspace map
PATS = {
    "memory":  [r"ddr|dram|mem(ory)?\b", r"\bL1(\b|_)|L2(\b|_)|L3(\b|_)", r"(L[123].*(HIT|MISS|MPI))\b",
               r"cache|fill|evict|wb|rd|wr|load|store", r"bandwidth|bw|throughput|qdepth|queue",
               r"lat(ency)?|stall.*mem|tCCD|tRCD|tRP|tCL|page|row|col"],
    "sensors": [r"temp|thermal|hot",
                r"volt|vdd|vcore|vin|vout|cpu[_\- ]?volt|cpu[_\- ]?vdd|vdd[_\- ]?cpu|core[_\- ]?volt|core[_\- ]?voltage",
                r"power|watt|energy|joule", r"fan|throttle|current|amps?"],
    "compute": [r"\bIPC\b|\bPhysIPC\b|\bEXEC\b|\bissue|retire|dispatch\b",
                r"\bINST\b|INSTnom%|branch|mispred|\balu\b|\barith\b|\blogic\b",
                r"C0res%|C1res%|C6res%|C7res%|CFREQ|AFREQ|ACYC|CYC|TIME|clk|cycle|freq|util",
                r"\bcore|cpu|sm|warp|shader\b"],
}
def subspace_of_feature(name: str) -> str:
    b = str(name)
    if any(re.search(p, b, flags=re.I) for p in PATS["memory"]):  return "memory"
    if any(re.search(p, b, flags=re.I) for p in PATS["sensors"]): return "sensors"
    return "compute"

# rank loader (MMI or Hybrid)
def read_rank_list(rank_dir: Path, setup: str, win: int, kfold: int, sub: str) -> list[str]:
    p = rank_dir / f"{setup}_{win}_{kfold}_0_{sub}.csv"
    if not p.exists(): return []
    df = pd.read_csv(p)
    col = "feature" if "feature" in df.columns else df.columns[0]
    return df[col].dropna().astype(str).tolist()