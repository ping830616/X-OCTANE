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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===============================================================================================
# FULL MULTI-WIN / MULTI-K PIPELINE (AUTHENTIC)
# + DSE PLOTS (all cases)
# + TABLE_SUBSPACEWEIGHTS (all cases)
#
# Runs:
#   - WINS   = [32, 64, 128, 512, 1024]
#   - KFOLDS = [3, 5, 10]
#   - SETUPS:
#       DDR4 anomalies: DROOP, RH
#       DDR5 anomalies: DROOP, SPECTRE
#   - PCT sweep: 10..100 step 10
#
# Produces:
#   1) Results/per_run_metrics_all_PIPELINE.csv
#   2) DSE plots for every (setup, anomaly, win, kfold):
#        Results/DesignSpace/PaperStyle_ALLRUNS/<setup>/<anomaly>/WIN{win}_KF{kfold}/...
#   3) Weight tables for every (setup, win, kfold):
#        Results/Table_SubspaceWeights/<setup>/WIN{win}_KF{kfold}/...
#
# Key behaviors:
#   - Workload-matched benign pairing by filename suffix (_dft/_oe/_tr etc.)
#   - DROOP: append CPU Voltage + transient features to sensors selection (does NOT override)
#   - Weight selection: PR-first on holdout split over candidate weight table
#   - Scoring: draft-math dE/dC + aM/aJ (with softplus positivity for aJ)
#
# ===============================================================================================

import gc, re, hashlib, warnings, unicodedata
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ===============================================================================================
# 0) PATHS
# ===============================================================================================
RES_DIR  = ROOT / "Results"
RES_DIR.mkdir(parents=True, exist_ok=True)

EXT_RES   = EXT_DRIVE / "octaneX_results"

def _can_write_dir(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
        t = p / ".write_test"
        t.write_text("ok")
        t.unlink()
        return True
    except Exception:
        return False

if _can_write_dir(EXT_RES):
    RES_DIR = EXT_RES
    RES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Using external results dir: {RES_DIR}")
else:
    print(f"[WARN] Cannot write to {EXT_RES}. Using local: {RES_DIR}")

RANK_DIRS = [
    ROOT / "FeatureRankOUT",
    EXT_DRIVE / "FeatureRankOUT",
    EXT_DRIVE / "octaneX" / "FeatureRankOUT",
]

# ===============================================================================================
# 1) RUN CONFIG
# ===============================================================================================
SETUPS = ["DDR4", "DDR5"]
ANOMALIES_BY_SETUP = {"DDR4": ["DROOP", "RH"], "DDR5": ["DROOP", "SPECTRE"]}

WINS   = [32, 64, 128, 512, 1024]
KFOLDS = [3, 5, 10]
PCT_SWEEP = list(range(10, 101, 10))

OVERLAP_RATIO       = 0.50
OVERLAP_RATIO_DROOP = 0.80

ROBUST_WINSOR = (2.0, 98.0)
SEED = 1337

GC_EVERY_N_PCTS   = 3
GC_EVERY_N_GROUPS = 1

META = ["label", "setup", "run_id"]
DROOP_META_COLS = ["droop_center_found", "droop_best_score_z", "droop_frac_ge_thr", "droop_vcols"]

BALANCE_EVAL_FOR_METRICS = True

DROP_LOW_VARIANCE_COLS = True
LOW_VAR_EPS = 1e-10

# DROOP boost (small, optional)
DROOP_BOOST = True
DROOP_BOOST_ALPHA = 0.20
DROOP_TRANSIENT_PAT = re.compile(r"__(drop|range|slope|min|max|std)", re.I)

METHOD_ORDER = ["dC_aJ", "dC_aM", "dE_aJ", "dE_aM"]

# ===============================================================================================
# 2) PLOT CONFIG (draft-style, no overlaps)
# ===============================================================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 13.5,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 20,
})

METHOD_TITLE = {
    "dC_aJ": r"Scoring function $d_C(X)$ with Aggregate $a_J$",
    "dC_aM": r"Scoring function $d_C(X)$ with Aggregate $a_M$",
    "dE_aJ": r"Scoring function $d_E(X)$ with Aggregate $a_J$",
    "dE_aM": r"Scoring function $d_E(X)$ with Aggregate $a_M$",
}

PLOT_YMIN = 0.0
PLOT_YMAX = 1.02
ALPHA_MINMAX = 0.28
ALPHA_IQR    = 0.60
GRID_ALPHA   = 0.22
GRID_LS      = "--"
GRID_LW      = 0.6

FIGSIZE = (13.6, 8.4)
WSPACE  = 0.30
HSPACE  = 0.42
TOP     = 0.88
BOTTOM  = 0.20
LEFT    = 0.075
RIGHT   = 0.985
SUPTITLE_Y = 0.975
LEGEND_Y   = 0.055

BAND_EPS = 0.008
def _ensure_visible_band(lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lo = lo.astype(float).copy()
    hi = hi.astype(float).copy()
    flat = (np.abs(hi - lo) < 1e-12)
    if np.any(flat):
        lo2 = lo - BAND_EPS/2
        hi2 = hi + BAND_EPS/2
        lo[flat] = np.clip(lo2[flat], 0.0, 1.0)
        hi[flat] = np.clip(hi2[flat], 0.0, 1.02)
    return lo, hi

# ===============================================================================================
# 3) WEIGHT CANDIDATES (Table-3 + extras) + LOOKUP
# ===============================================================================================
def build_weight_table() -> pd.DataFrame:
    # Paper Table 3 is (wM,wC,wS) -> store as (wC,wM,wS)
    paper_cases = [
        ("T3_01", "Case 1",  1,    0,    0),
        ("T3_02", "Case 2",  0,    1,    0),
        ("T3_03", "Case 3",  0,    0,    1),
        ("T3_04", "Case 4",  1/3,  1/3,  1/3),
        ("T3_05", "Case 5",  1/4,  1/4,  2/4),
        ("T3_06", "Case 6",  1/5,  1/5,  3/5),
        ("T3_07", "Case 7",  1/6,  1/6,  4/6),
        ("T3_08", "Case 8",  1/8,  2/8,  5/8),
        ("T3_09", "Case 9",  1/8,  1/8,  6/8),
        ("T3_10", "Case 10", 1/10, 1/10, 8/10),
        ("T3_11", "Case 11", 2/3,  1/3,  1/3),
        ("T3_12", "Case 12", 3/4,  1/4,  1/4),
        ("T3_13", "Case 13", 5/8,  2/8,  1/8),
        ("T3_14", "Case 14", 6/8,  1/8,  1/8),
        ("T3_15", "Case 15", 1/20, 1/20, 18/20),
        ("T3_16", "Case 16", 1/40, 1/40, 38/40),
    ]
    rows = []
    for cid, name, wM, wC, wS in paper_cases:
        rows.append({
            "weight_case_id": cid,
            "weight_case_name": name,
            "wC": float(wC),
            "wM": float(wM),
            "wS": float(wS),
            "source": "Table3",
        })

    extras = [
        ("EX_01", "SensorOnly",   0.0, 0.0, 1.0),
        ("EX_02", "Sensor90",     0.05, 0.05, 0.90),
        ("EX_03", "Sensor95",     0.025, 0.025, 0.95),
        ("EX_04", "Sensor98",     0.01, 0.01, 0.98),
    ]
    for cid, name, wC, wM, wS in extras:
        rows.append({
            "weight_case_id": cid,
            "weight_case_name": name,
            "wC": float(wC),
            "wM": float(wM),
            "wS": float(wS),
            "source": "Extra",
        })

    df = pd.DataFrame(rows)
    df["_key"] = df.apply(lambda r: (round(r["wC"], 6), round(r["wM"], 6), round(r["wS"], 6)), axis=1)
    df = df.drop_duplicates("_key").drop(columns=["_key"]).reset_index(drop=True)
    return df

WEIGHT_TABLE = build_weight_table()

def _weight_lookup(w: Tuple[float,float,float]) -> Tuple[str,str,str]:
    wC,wM,wS = (round(float(w[0]),6), round(float(w[1]),6), round(float(w[2]),6))
    m = WEIGHT_TABLE[(WEIGHT_TABLE["wC"].round(6)==wC) &
                     (WEIGHT_TABLE["wM"].round(6)==wM) &
                     (WEIGHT_TABLE["wS"].round(6)==wS)]
    if len(m)>0:
        r = m.iloc[0]
        return str(r["weight_case_id"]), str(r["weight_case_name"]), str(r["source"])
    return ("CUSTOM","Custom","Search")

# ===============================================================================================
# 4) IO + workload parsing
# ===============================================================================================
RUNID2PATH: Dict[str,str] = {}

def detect_setup_from_path(p: Path):
    s = str(p).lower()
    if "ddr4" in s: return "DDR4"
    if "ddr5" in s: return "DDR5"
    return None

def is_benign_path(p: Path) -> bool:
    s = str(p).lower()
    if "benign" in s:
        return True
    bad = ["attack","anom","fault","inject","trojan","mal","rh","droop","spectre","trrespass"]
    return not any(b in s for b in bad)

def is_anomaly_path(p: Path, anomaly: str) -> bool:
    return (anomaly.lower() in str(p).lower()) and (not is_benign_path(p))

def iter_raw_csvs(root: Path):
    for p in root.rglob("*.csv"):
        yield p

def read_csv_clean(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]

def mk_run_id(path: Path) -> str:
    rid = f"run_{hashlib.md5(str(path).encode('utf-8')).hexdigest()[:10]}"
    RUNID2PATH[rid] = str(path)
    return rid

_WORKLOADS = ["dft","dj","dp","gl","gs","ha","ja","mm","ni","oe","pi","sh","tr"]
def workload_from_path(p: Path) -> str:
    stem = (p.stem or "").lower()
    m = re.search(r"_([a-z]{2,3})$", stem)
    if m and m.group(1) in _WORKLOADS:
        return m.group(1).upper()
    for tok in _WORKLOADS:
        if tok in stem:
            return tok.upper()
    return "UNK"

def collect_raw_pairs_by_setup(data_dir: Path, which: str, anomaly: Optional[str] = None):
    out = {"DDR4": [], "DDR5": []}
    for p in iter_raw_csvs(data_dir):
        setup = detect_setup_from_path(p)
        if setup is None:
            continue
        try:
            if which == "benign":
                if not is_benign_path(p):
                    continue
            else:
                if anomaly is None or not is_anomaly_path(p, anomaly):
                    continue
            df = read_csv_clean(p)
            out[setup].append((p, df))
        except Exception as e:
            print(f"[WARN] read failed {p}: {e}")
    return out

def telemetry_cols(df: pd.DataFrame):
    exclude = set(META) | set(DROOP_META_COLS) | {"workload"}
    return [c for c in df.columns if (c not in exclude) and (df[c].dtype.kind in "fcbiu")]

def drop_low_variance_cols(df: pd.DataFrame, cols: List[str], eps: float = 1e-10) -> List[str]:
    v = df[cols].astype(float).var(axis=0, ddof=0)
    keep = v[v > eps].index.tolist()
    return keep

# ===============================================================================================
# 5) Scaling
# ===============================================================================================
def robust_scale_train(Xb_np: np.ndarray, winsor=(2.0, 98.0)):
    Q1, Q2 = np.percentile(Xb_np, winsor[0], axis=0), np.percentile(Xb_np, winsor[1], axis=0)
    Xb_clip = np.clip(Xb_np, Q1, Q2)
    mu = Xb_clip.mean(axis=0)
    sd = Xb_clip.std(axis=0, ddof=0) + 1e-9
    return mu, sd, Q1, Q2

def apply_robust_scale(X: pd.DataFrame, mu, sd, Q1, Q2):
    Xc = np.clip(X.to_numpy(dtype=float), Q1, Q2)
    Z  = (Xc - mu) / sd
    return pd.DataFrame(Z, columns=X.columns, index=X.index)

# ===============================================================================================
# 6) Windowing (adds sensor transient feats incl CPU Voltage)
# ===============================================================================================
SENSOR_PAT = re.compile(r"cpu\s*voltage|volt|vdd|vcore|vin|vout|power|energy|joule|current|amps?|temp|thermal|hot", re.I)

def window_collapse_means(df: pd.DataFrame, win: int, setup: str, run_id: str, label: str,
                          overlap_ratio: float, src_path: str="") -> pd.DataFrame:
    cols_all = telemetry_cols(df)
    if not cols_all:
        return pd.DataFrame()

    sensor_cols = [c for c in cols_all if SENSOR_PAT.search(c)]
    if "CPU Voltage" in cols_all and "CPU Voltage" not in sensor_cols:
        sensor_cols.append("CPU Voltage")

    n = len(df)
    stride = max(1, int(round(win * (1 - overlap_ratio))))
    starts = list(range(0, n - win + 1, stride))
    if not starts:
        return pd.DataFrame()

    rows = []
    for start in starts:
        chunk = df.iloc[start:start+win]
        X = chunk[cols_all].astype(float)
        means = X.mean(axis=0, numeric_only=True)

        tfeat = {}
        if sensor_cols:
            Xs = X[sensor_cols]
            mins  = Xs.min(axis=0)
            maxs  = Xs.max(axis=0)
            stds  = Xs.std(axis=0, ddof=0)
            means_s = Xs.mean(axis=0)
            drops  = (means_s - mins)
            ranges = (maxs - mins)
            slope  = (Xs.iloc[-1] - Xs.iloc[0])
            for c in sensor_cols:
                tfeat[f"{c}__min"]   = float(mins[c])
                tfeat[f"{c}__max"]   = float(maxs[c])
                tfeat[f"{c}__std"]   = float(stds[c])
                tfeat[f"{c}__drop"]  = float(drops[c])
                tfeat[f"{c}__range"] = float(ranges[c])
                tfeat[f"{c}__slope"] = float(slope[c])

        row = pd.concat([means, pd.Series(tfeat)]).to_frame().T
        row["setup"]    = setup
        row["run_id"]   = run_id
        row["label"]    = label
        row["workload"] = workload_from_path(Path(src_path))
        rows.append(row)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def build_windowed_raw_means(pairs, setup: str, win: int, label: str, overlap_ratio: float) -> pd.DataFrame:
    out = []
    for p, df in pairs:
        rid = mk_run_id(p)
        agg = window_collapse_means(df, win=win, setup=setup, run_id=rid, label=label,
                                    overlap_ratio=overlap_ratio, src_path=str(p))
        if not agg.empty:
            out.append(agg)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

# ===============================================================================================
# 7) Rank lists + robust mapping + CPU Voltage family append for DROOP
# ===============================================================================================
def _read_rank_feats(p: Path):
    try:
        df = pd.read_csv(p)
    except Exception:
        return []
    if df.empty:
        return []
    col = "feature" if "feature" in df.columns else df.columns[0]
    return df[col].dropna().astype(str).tolist()

def load_full_rank_lists(setup: str, win: int, kfold: int):
    def _find_file(setup, win, kfold, sub):
        fname = f"{setup}_{win}_{kfold}_0_{sub}.csv"
        for d in RANK_DIRS:
            pp = d / fname
            if pp.exists():
                return pp
        return None
    out = {}
    for sub in ["compute","memory","sensors"]:
        pp = _find_file(setup, win, kfold, sub)
        out[sub] = _read_rank_feats(pp) if pp else []
    return out

def slice_by_percent(full_lists: Dict[str, List[str]], pct: int):
    sel = {}
    frac = pct / 100.0
    for sub in ("compute","memory","sensors"):
        feats = full_lists.get(sub, []) or []
        k = int(np.ceil(len(feats) * frac))
        if frac > 0 and len(feats) > 0:
            k = max(1, k)
        sel[sub] = feats[:min(len(feats), max(0, k))]
    return sel

ALIAS_MAP = {
    "voltage": ["cpu","voltage","volt","vcore","vdd","vin","vout"],
    "power": ["power","energy","joules","current","amps"],
    "temperature": ["temp","thermal","hot"],
    "frequency": ["freq","afreq","cfreq","clock","clk","mhz","ghz"],
    "ipc": ["ipc","physipc"],
    "cache": ["l1","l2","l3","hit","miss","evict","fill","mpi"],
    "bandwidth": ["read","write","bw","bandwidth"],
}

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower()
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else "_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__","_")
    return s.strip("_")

def _tokens(s: str):
    t = _norm(s).replace("_"," ").split()
    exp = []
    for tok in t:
        exp.append(tok)
        for k, aliases in ALIAS_MAP.items():
            if tok in aliases:
                exp.append(k)
    return set(exp)

def build_column_index(cols: List[str]):
    norm2orig = {}
    tok_index = {}
    for c in cols:
        nc = _norm(c)
        norm2orig.setdefault(nc, c)
        tok_index[c] = _tokens(c)
    return norm2orig, tok_index

def map_ranks_to_existing(ranked_list: List[str], norm2orig, tok_index):
    mapped, seen = [], set()
    for r in ranked_list:
        nr = _norm(r)
        cand = norm2orig.get(nr, None)
        if cand and cand not in seen:
            mapped.append(cand); seen.add(cand); continue
        rtoks = _tokens(r)
        best_c, best_j = None, 0.0
        for c, ctoks in tok_index.items():
            inter = len(rtoks & ctoks)
            union = len(rtoks | ctoks) or 1
            j = inter / union
            if j > best_j:
                best_j, best_c = j, c
        if best_c and best_j >= 0.60 and best_c not in seen:
            mapped.append(best_c); seen.add(best_c)
    return mapped

def intersect_selection_with_columns_robust(sel_raw: Dict[str, List[str]], xb_cols: set):
    norm2orig, tok_index = build_column_index(list(xb_cols))
    out = {}
    for sub in ("compute","memory","sensors"):
        ranked = sel_raw.get(sub, []) or []
        mapped = map_ranks_to_existing(ranked, norm2orig, tok_index)
        out[sub] = [c for c in mapped if c in xb_cols]
    return out

def force_cpu_voltage_family(sel: Dict[str, List[str]], xb_cols: set) -> Dict[str, List[str]]:
    out = dict(sel)
    sensors_now = out.get("sensors", []) or []
    must = []
    if "CPU Voltage" in xb_cols:
        must.append("CPU Voltage")
    for suf in ["__min","__max","__std","__drop","__range","__slope"]:
        c = f"CPU Voltage{suf}"
        if c in xb_cols:
            must.append(c)
    out["sensors"] = list(dict.fromkeys(sensors_now + must))
    return out

# ===============================================================================================
# 8) Balanced evaluation helper  ✅ ADDED (fixes NameError)
# ===============================================================================================
def balanced_concat(df_ben: pd.DataFrame, df_anom: pd.DataFrame, seed: int = 1337):
    """
    Create a balanced evaluation set by sampling the same number of windows from BENIGN and ANOM.
    Returns:
      df_eval : concatenated dataframe (benign first, anomaly second)
      y_true  : numpy array labels (0 benign, 1 anomaly) aligned with df_eval rows
    """
    if df_ben is None or df_ben.empty:
        raise ValueError("balanced_concat(): df_ben is empty")
    if df_anom is None or df_anom.empty:
        raise ValueError("balanced_concat(): df_anom is empty")

    rng = np.random.default_rng(int(seed))

    nb = int(len(df_ben))
    na = int(len(df_anom))
    n  = int(min(nb, na))

    # sample without replacement if possible; otherwise sample with replacement
    replace_b = nb < n
    replace_a = na < n

    idx_b = rng.choice(nb, size=n, replace=replace_b)
    idx_a = rng.choice(na, size=n, replace=replace_a)

    ben_s = df_ben.iloc[idx_b].copy().reset_index(drop=True)
    anm_s = df_anom.iloc[idx_a].copy().reset_index(drop=True)

    df_eval = pd.concat([ben_s, anm_s], ignore_index=True)
    y_true  = np.concatenate([np.zeros(len(ben_s), dtype=int),
                              np.ones(len(anm_s), dtype=int)])
    return df_eval, y_true

# ===============================================================================================
# 9) Weight selection helpers
# ===============================================================================================
def split_holdout(y, test_size=0.3, seed=SEED):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx = np.arange(len(y))
    _, val_idx = next(sss.split(idx, y))
    return val_idx

def pick_best_w_per_method_PRfirst(y_true: np.ndarray, scores_by_w: Dict[Tuple[float,float,float], Dict[str,np.ndarray]], val_idx: np.ndarray):
    yy = y_true[val_idx]
    best = {m: (-1.0, -1.0, -1.0, None) for m in ["dE_aM","dE_aJ","dC_aM","dC_aJ"]}
    for w, md in scores_by_w.items():
        wS = float(w[2])
        for m, s in md.items():
            sv = np.asarray(s)[val_idx]
            pr  = float(average_precision_score(yy, sv))
            roc = float(roc_auc_score(yy, sv))
            key = (pr, roc, wS)
            if key > best[m][:3]:
                best[m] = (pr, roc, wS, w)
    return {m: best[m][3] for m in best}

# ===============================================================================================
# 10) Scoring (draft math)
# ===============================================================================================
def _build_refs(Xb_base: pd.DataFrame, sel: Dict[str, List[str]], eps: float = 1e-12):
    refs = {}
    for sub in ("compute","memory","sensors"):
        cols = [c for c in (sel.get(sub) or []) if c in Xb_base.columns]
        if not cols:
            refs[sub] = None
            continue
        X = Xb_base[cols].astype(float).values
        mu = X.mean(axis=0)
        var = X.var(axis=0, ddof=0)
        var = np.where(var <= eps, eps, var)
        w = 1.0 / var
        denomC = float(np.sum((w * mu) ** 2))
        refs[sub] = {"cols": cols, "mu": mu, "w": w, "denomC": max(denomC, eps)}
    return refs

def _score_dE_parts(Xe_base: pd.DataFrame, refs: dict) -> Dict[str, np.ndarray]:
    n = len(Xe_base)
    out = {}
    for sub in ("compute","memory","sensors"):
        r = refs.get(sub)
        if r is None:
            out[sub] = np.zeros(n)
            continue
        X = Xe_base[r["cols"]].astype(float).values
        D = X - r["mu"]
        s = np.sum((D * D) * r["w"], axis=1)
        out[sub] = np.log1p(np.maximum(s, 0.0))
    return out

def _score_dC_parts(Xe_base: pd.DataFrame, refs: dict, eps: float = 1e-12) -> Dict[str, np.ndarray]:
    n = len(Xe_base)
    out = {}
    for sub in ("compute","memory","sensors"):
        r = refs.get(sub)
        if r is None:
            out[sub] = np.zeros(n)
            continue
        X = Xe_base[r["cols"]].astype(float).values
        w = r["w"]
        mu = r["mu"]
        term1 = np.sum(w * (X * X), axis=1)
        dot   = np.sum(X * (w * mu), axis=1)
        term2 = (dot * dot) / max(r["denomC"], eps)
        s = term1 - term2
        out[sub] = np.log1p(np.maximum(s, 0.0))
    return out

def fit_parts_normalizer_on_benign(Xb_base: pd.DataFrame, refs: dict) -> dict:
    eps = 1e-9
    partsE_b = _score_dE_parts(Xb_base, refs)
    partsC_b = _score_dC_parts(Xb_base, refs)
    norm = {"dE": {}, "dC": {}}
    for sub in ("compute","memory","sensors"):
        vE = np.asarray(partsE_b[sub], float)
        vC = np.asarray(partsC_b[sub], float)
        norm["dE"][sub] = (float(np.nanmean(vE)), float(np.nanstd(vE) + eps))
        norm["dC"][sub] = (float(np.nanmean(vC)), float(np.nanstd(vC) + eps))
    return norm

def apply_parts_normalizer(parts: Dict[str, np.ndarray], norm_sub: dict) -> Dict[str, np.ndarray]:
    out = {}
    for sub, v in parts.items():
        mu, sd = norm_sub.get(sub, (0.0, 1.0))
        v = np.asarray(v, float)
        out[sub] = (v - mu) / (sd if sd > 0 else 1.0)
    return out

def _aggregate_aM(parts: Dict[str, np.ndarray], weights: Tuple[float,float,float]) -> np.ndarray:
    wC, wM, wS = map(float, weights)
    den = (wC + wM + wS) if (wC + wM + wS) > 0 else 1.0
    return (wC*parts["compute"] + wM*parts["memory"] + wS*parts["sensors"]) / den

def _aggregate_aJ(parts: Dict[str, np.ndarray], weights: Tuple[float,float,float], eps: float = 1e-12) -> np.ndarray:
    wC, wM, wS = map(float, weights)
    def softplus(x):
        x = np.asarray(x, float)
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
    sC = np.maximum(softplus(parts["compute"]), eps)
    sM = np.maximum(softplus(parts["memory"]),  eps)
    sS = np.maximum(softplus(parts["sensors"]), eps)
    return np.exp(wC*np.log(sC) + wM*np.log(sM) + wS*np.log(sS))

def score_four_methods_benign_norm(Xe_base: pd.DataFrame, refs: dict, weights: Tuple[float,float,float], benign_norm: dict):
    parts_E = apply_parts_normalizer(_score_dE_parts(Xe_base, refs), benign_norm["dE"])
    parts_C = apply_parts_normalizer(_score_dC_parts(Xe_base, refs), benign_norm["dC"])
    return {
        "dE_aM": _aggregate_aM(parts_E, weights),
        "dE_aJ": _aggregate_aJ(parts_E, weights),
        "dC_aM": _aggregate_aM(parts_C, weights),
        "dC_aJ": _aggregate_aJ(parts_C, weights),
    }

def droop_boost_score(Xe_base: pd.DataFrame, sel: Dict[str, List[str]]) -> np.ndarray:
    cols = [c for c in (sel.get("sensors") or []) if ("CPU Voltage" in c and DROOP_TRANSIENT_PAT.search(c))]
    if not cols:
        return np.zeros(len(Xe_base))
    Z = Xe_base[cols].astype(float).to_numpy()
    return np.log1p(np.mean(np.abs(Z), axis=1))

# ===============================================================================================
# 11) DSE plots
# ===============================================================================================
def summarize_for_dse(per_run_df: pd.DataFrame) -> pd.DataFrame:
    keys = ["setup","anomaly","win","kfold","method","pct"]
    def q1(x): return float(np.nanpercentile(x, 25))
    def q3(x): return float(np.nanpercentile(x, 75))
    g = (per_run_df.groupby(keys, as_index=False)
            .agg(
                roc_auc_median=("roc_auc","median"),
                roc_auc_q1=("roc_auc", q1),
                roc_auc_q3=("roc_auc", q3),
                roc_auc_min=("roc_auc","min"),
                roc_auc_max=("roc_auc","max"),
                auc_pr_median=("auc_pr","median"),
                auc_pr_q1=("auc_pr", q1),
                auc_pr_q3=("auc_pr", q3),
                auc_pr_min=("auc_pr","min"),
                auc_pr_max=("auc_pr","max"),
                n_runs=("run_id","nunique"),
            ))
    return g.sort_values(keys).reset_index(drop=True)

def plot_dse_grid(summary_case: pd.DataFrame, metric: str, setup: str, anomaly: str,
                  win: int, kfold: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    x_ticks = sorted(summary_case["pct"].unique().tolist())
    if not x_ticks:
        return

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=False, sharey=False)
    axes = axes.flatten()
    fig.subplots_adjust(left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM, wspace=WSPACE, hspace=HSPACE)

    xlab = "Top-ranked features used (%)"
    ylab = "ROC-AUC" if metric == "roc_auc" else "AUC-PR"

    def _clip01(a): return np.clip(np.asarray(a, float), 0.0, 1.0)

    for i, method in enumerate(METHOD_ORDER):
        ax = axes[i]
        ax.set_title(METHOD_TITLE.get(method, method), pad=7)
        ax.set_xlim(min(x_ticks), max(x_ticks))
        ax.set_ylim(PLOT_YMIN, PLOT_YMAX)
        ax.set_xticks(x_ticks)
        ax.grid(True, alpha=GRID_ALPHA, linestyle=GRID_LS, linewidth=GRID_LW)

        sub = summary_case[summary_case["method"] == method].copy().sort_values("pct")
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        else:
            x = sub["pct"].to_numpy(dtype=float)
            med  = _clip01(sub[f"{metric}_median"].to_numpy(dtype=float))
            q1   = _clip01(sub[f"{metric}_q1"].to_numpy(dtype=float))
            q3   = _clip01(sub[f"{metric}_q3"].to_numpy(dtype=float))
            vmin = _clip01(sub[f"{metric}_min"].to_numpy(dtype=float))
            vmax = _clip01(sub[f"{metric}_max"].to_numpy(dtype=float))

            lo_mm, hi_mm = np.minimum(vmin, vmax), np.maximum(vmin, vmax)
            lo_iq, hi_iq = np.minimum(q1, q3),     np.maximum(q1, q3)
            lo_mm, hi_mm = _ensure_visible_band(lo_mm, hi_mm)
            lo_iq, hi_iq = _ensure_visible_band(lo_iq, hi_iq)

            ax.fill_between(x, lo_mm, hi_mm, alpha=ALPHA_MINMAX, label="min-max", linewidth=0.0)
            ax.fill_between(x, lo_iq, hi_iq, alpha=ALPHA_IQR,   label="IQR",     linewidth=0.0)
            ax.plot(x, med, color="black", marker="o", linewidth=2.0, markersize=5.5, zorder=5, label="Median")

        ax.set_xlabel(xlab, labelpad=6)
        ax.set_ylabel(ylab, labelpad=6)
        ax.tick_params(axis="x", labelbottom=True, pad=3)
        ax.tick_params(axis="y", labelleft=True, pad=3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, LEGEND_Y), ncol=3, frameon=False)
    fig.suptitle(f"{setup}: {anomaly} (WIN={win}, K={kfold})", y=SUPTITLE_Y)

    base = f"DSE_{metric.upper()}_{setup}_{anomaly}_WIN{int(win)}_KF{int(kfold)}"
    fig.savefig(out_dir / f"{base}.png", dpi=600, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_dir / f"{base}.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

def write_all_dse_plots(per_run_df: pd.DataFrame):
    out_root = RES_DIR / "DesignSpace" / "PaperStyle_ALLRUNS"
    out_root.mkdir(parents=True, exist_ok=True)
    dse = summarize_for_dse(per_run_df)
    for (setup, anomaly, win, kfold), sub in dse.groupby(["setup","anomaly","win","kfold"]):
        out_dir = out_root / str(setup) / str(anomaly) / f"WIN{int(win)}_KF{int(kfold)}"
        plot_dse_grid(sub, "roc_auc", str(setup), str(anomaly), int(win), int(kfold), out_dir)
        plot_dse_grid(sub, "auc_pr",  str(setup), str(anomaly), int(win), int(kfold), out_dir)

# ===============================================================================================
# 12) Subspace-weight tables like your Table 13/14 (per setup,win,kfold)
# ===============================================================================================
def build_workload_weight_tables(per_run_df: pd.DataFrame, setup: str, win: int, kfold: int) -> None:
    base_dir = RES_DIR / "Table_SubspaceWeights" / setup / f"WIN{win}_KF{kfold}"
    base_dir.mkdir(parents=True, exist_ok=True)

    rep_method = "dE_aM"  # for paper-style merged table

    for anomaly in ANOMALIES_BY_SETUP[setup]:
        df = per_run_df[
            (per_run_df["setup"]==setup) &
            (per_run_df["anomaly"]==anomaly) &
            (per_run_df["win"]==win) &
            (per_run_df["kfold"]==kfold)
        ].copy()
        if df.empty:
            continue

        df["_wkey"] = df.apply(lambda r: (round(r["wC"],6), round(r["wM"],6), round(r["wS"],6)), axis=1)

        rows = []
        for (wl, method), g in df.groupby(["workload","method"]):
            mode_key = g["_wkey"].value_counts().idxmax()
            wC, wM, wS = mode_key
            wid, wname, wsrc = _weight_lookup((wC,wM,wS))
            rows.append({
                "workload": wl,
                "method": method,
                "chosen_wC": wC, "chosen_wM": wM, "chosen_wS": wS,
                "weight_case_id": wid,
                "weight_case_name": wname,
                "weight_case_source": wsrc,
                "n_rows": int(len(g)),
                "median_roc": float(np.nanmedian(g["roc_auc"])),
                "median_pr": float(np.nanmedian(g["auc_pr"])),
            })

        df_out = pd.DataFrame(rows).sort_values(["workload","method"]).reset_index(drop=True)
        (base_dir / f"Table_SubspaceWeights_{setup}_WIN{win}_KF{kfold}_{anomaly}.csv").write_text(
            df_out.to_csv(index=False)
        )

    droop = "DROOP"
    other = [a for a in ANOMALIES_BY_SETUP[setup] if a != droop]
    other = other[0] if other else ""

    def _pick_weight_for(wl: str, anom: str) -> Optional[Tuple[float,float,float]]:
        df = per_run_df[
            (per_run_df["setup"]==setup) &
            (per_run_df["anomaly"]==anom) &
            (per_run_df["win"]==win) &
            (per_run_df["kfold"]==kfold) &
            (per_run_df["workload"]==wl) &
            (per_run_df["method"]==rep_method)
        ].copy()
        if df.empty:
            return None
        df["_wkey"] = df.apply(lambda r: (round(r["wC"],6), round(r["wM"],6), round(r["wS"],6)), axis=1)
        return df["_wkey"].value_counts().idxmax()

    all_wl = sorted(per_run_df[
        (per_run_df["setup"]==setup) &
        (per_run_df["win"]==win) &
        (per_run_df["kfold"]==kfold)
    ]["workload"].astype(str).unique().tolist())

    def fmt(w):
        if w is None:
            return ""
        return f"({w[0]:g}, {w[1]:g}, {w[2]:g})"

    rows = []
    for wl in all_wl:
        wd = _pick_weight_for(wl, droop)
        wo = _pick_weight_for(wl, other) if other else None
        rows.append({
            "Workload": wl,
            droop: fmt(wd),
            other if other else "Other": fmt(wo),
        })

    paper_out = pd.DataFrame(rows)
    (base_dir / f"Table_SubspaceWeights_{setup}_WIN{win}_KF{kfold}_PAPERSTYLE.csv").write_text(
        paper_out.to_csv(index=False)
    )

# ===============================================================================================
# 13) Pipeline core
# ===============================================================================================
def run_full_pipeline() -> pd.DataFrame:
    all_rows = []
    group_counter = 0

    benign_pairs = collect_raw_pairs_by_setup(DATA_DIR, which="benign")
    anomaly_pairs_cache: Dict[Tuple[str,str], List[Tuple[Path,pd.DataFrame]]] = {}

    def get_anom_pairs(setup: str, anomaly: str):
        key = (setup, anomaly.upper())
        if key not in anomaly_pairs_cache:
            anomaly_pairs_cache[key] = collect_raw_pairs_by_setup(DATA_DIR, which="anomaly", anomaly=anomaly)[setup]
        return anomaly_pairs_cache[key]

    for setup in SETUPS:
        anomalies = ANOMALIES_BY_SETUP.get(setup, [])
        ben_pairs_all = benign_pairs.get(setup, [])
        if not ben_pairs_all:
            print(f"[SKIP] {setup}: no benign RAW files")
            continue

        for win in WINS:
            df_ben_all = build_windowed_raw_means(ben_pairs_all, setup=setup, win=win, label="BENIGN",
                                                  overlap_ratio=OVERLAP_RATIO)
            if df_ben_all.empty:
                print(f"[SKIP] {setup} WIN={win}: benign windowed empty")
                continue

            Xb_cols = telemetry_cols(df_ben_all)
            if DROP_LOW_VARIANCE_COLS:
                Xb_cols = drop_low_variance_cols(df_ben_all, Xb_cols, eps=LOW_VAR_EPS)
            if not Xb_cols:
                print(f"[SKIP] {setup} WIN={win}: no telemetry cols after low-var drop")
                continue

            Xb = df_ben_all[Xb_cols].astype(float)
            mu_s, sd_s, Q1, Q2 = robust_scale_train(Xb.values, winsor=ROBUST_WINSOR)
            Xb_base = apply_robust_scale(Xb, mu_s, sd_s, Q1, Q2)
            xb_cols_set = set(Xb_base.columns)

            ben_by_wl = {}
            for wl in df_ben_all["workload"].astype(str).unique():
                ben_by_wl[wl] = df_ben_all[df_ben_all["workload"].astype(str)==wl].copy()

            for kfold in KFOLDS:
                # write candidates table once per setup/win/kfold
                wdir = RES_DIR / "Table_SubspaceWeights" / setup / f"WIN{win}_KF{kfold}"
                wdir.mkdir(parents=True, exist_ok=True)
                WEIGHT_TABLE.to_csv(wdir / "Table_SubspaceWeights_CANDIDATES.csv", index=False)

                full_lists = load_full_rank_lists(setup, win, kfold)

                for anomaly in anomalies:
                    group_counter += 1
                    print(f"\n[RUN] {setup} {anomaly} WIN={win} KF={kfold} (group {group_counter})")

                    overlap = OVERLAP_RATIO_DROOP if anomaly.upper() == "DROOP" else OVERLAP_RATIO
                    an_pairs_all = get_anom_pairs(setup, anomaly)
                    if not an_pairs_all:
                        print(f"[SKIP] {setup} {anomaly}: no anomaly RAW files")
                        continue

                    df_anom = build_windowed_raw_means(an_pairs_all, setup=setup, win=win, label=anomaly,
                                                       overlap_ratio=overlap)
                    if df_anom.empty:
                        print(f"[SKIP] {setup} {anomaly} WIN={win}: anomaly windowed empty")
                        continue

                    run_ids = sorted(df_anom["run_id"].astype(str).unique())

                    for rid in run_ids:
                        dfa_run = df_anom[df_anom["run_id"].astype(str) == rid].copy()
                        if dfa_run.empty:
                            continue

                        wl = str(dfa_run["workload"].iloc[0]) if "workload" in dfa_run.columns else "UNK"
                        df_ben = ben_by_wl.get(wl, df_ben_all)
                        if df_ben.empty:
                            df_ben = df_ben_all

                        # balanced evaluation ✅ (now defined)
                        df_eval, y_true = balanced_concat(df_ben, dfa_run, seed=SEED)

                        # Use only columns that exist in eval (prevents KeyError if any mismatch)
                        use_cols = [c for c in Xb_cols if c in df_eval.columns]
                        if not use_cols:
                            continue

                        Xe_all  = df_eval[use_cols].astype(float)
                        Xe_base = apply_robust_scale(Xe_all, mu_s, sd_s, Q1, Q2)

                        for idx_p, pct in enumerate(PCT_SWEEP, start=1):
                            sel_raw = slice_by_percent(full_lists, pct)
                            sel = intersect_selection_with_columns_robust(sel_raw, xb_cols_set)

                            if (not sel.get("compute")) or (not sel.get("memory")) or (not sel.get("sensors")):
                                continue

                            if anomaly.upper() == "DROOP":
                                sel = force_cpu_voltage_family(sel, xb_cols_set)

                            refs = _build_refs(Xb_base, sel, eps=1e-12)
                            benign_norm = fit_parts_normalizer_on_benign(Xb_base, refs)

                            val_idx = split_holdout(y_true, test_size=0.3, seed=SEED)

                            scores_by_w = {}
                            for _, wr in WEIGHT_TABLE.iterrows():
                                w = (float(wr["wC"]), float(wr["wM"]), float(wr["wS"]))
                                scores_by_w[w] = score_four_methods_benign_norm(Xe_base, refs, w, benign_norm)

                            best_w_for = pick_best_w_per_method_PRfirst(y_true, scores_by_w, val_idx)

                            for method in METHOD_ORDER:
                                w_star = best_w_for[method]
                                y_score = score_four_methods_benign_norm(Xe_base, refs, w_star, benign_norm)[method]
                                y_score = np.nan_to_num(y_score, nan=0.0, posinf=0.0, neginf=0.0)

                                if DROOP_BOOST and anomaly.upper() == "DROOP":
                                    boost = droop_boost_score(Xe_base, sel)
                                    y_score = (1.0 - DROOP_BOOST_ALPHA) * y_score + DROOP_BOOST_ALPHA * boost

                                roc = float(np.clip(roc_auc_score(y_true, y_score), 0.0, 1.0))
                                pr  = float(np.clip(average_precision_score(y_true, y_score), 0.0, 1.0))

                                wid, wname, wsrc = _weight_lookup(w_star)

                                all_rows.append({
                                    "setup": setup,
                                    "anomaly": anomaly,
                                    "win": int(win),
                                    "kfold": int(kfold),
                                    "pct": int(pct),
                                    "run_id": str(rid),
                                    "workload": str(wl),
                                    "method": method,
                                    "roc_auc": roc,
                                    "auc_pr": pr,
                                    "weight_case_id": wid,
                                    "weight_case_name": wname,
                                    "weight_case_source": wsrc,
                                    "wC": float(w_star[0]),
                                    "wM": float(w_star[1]),
                                    "wS": float(w_star[2]),
                                })

                            if idx_p % GC_EVERY_N_PCTS == 0:
                                gc.collect()

                    if group_counter % GC_EVERY_N_GROUPS == 0:
                        gc.collect()

    per_run_df = pd.DataFrame(all_rows)

    out_all = RES_DIR / "per_run_metrics_all_PIPELINE.csv"
    per_run_df.to_csv(out_all, index=False)
    print(f"\n[OK] ALLRUNS metrics → {out_all}")

    # Write workload weight tables per setup/win/kfold
    for setup in SETUPS:
        for win in WINS:
            for kfold in KFOLDS:
                build_workload_weight_tables(per_run_df, setup=setup, win=win, kfold=kfold)

    # Write all DSE plots
    write_all_dse_plots(per_run_df)

    return per_run_df

# ===============================================================================================
# 14) MAIN
# ===============================================================================================
def main():
    run_full_pipeline()
    print("[DONE] Pipeline + DSE plots + weight tables per platform")

if __name__ == "__main__":
    main()