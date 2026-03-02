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

# === Build LaTeX Table: Median AUC-PR vs top-p (10/30/70/100) for specific (WIN,K) ===
# Uses per-run CSV (source of truth):
#   per_run_metrics_all_PIPELINE.csv
#
# Rule for each cell (setup, anomaly, win, kfold, pct):
#   - compute auc_pr_median across run_id
#   - if multiple methods exist, take the BEST (max) auc_pr_median across methods
#
# Targets (your request):
#   - Setup A (DDR4): WIN=512,  K=3
#   - Setup B (DDR5): WIN=1024, K=5
#
# Output:
#   - prints LaTeX table ready to paste into Overleaf

from pathlib import Path
import pandas as pd
import numpy as np

# ----------------------------
# CONFIG
# ----------------------------
TARGETS = {
    "A": {"setup":"DDR4", "win":512,  "kfold":3, "anomalies":["RH","DROOP"]},       # RH -> TRRespass
    "B": {"setup":"DDR5", "win":1024, "kfold":5, "anomalies":["SPECTRE","DROOP"]},
}
PCTS = [10, 30, 70, 100]

def paper_anomaly_name(anom: str) -> str:
    a = str(anom).strip().upper()
    if a == "RH": return "TRRespass"
    if a == "DROOP": return "Droop"
    if a == "SPECTRE": return "Spectre"
    return a.title()

# ----------------------------
# LOAD per-run CSV (uploaded path first)
# ----------------------------
CANDS = [
    ROOT / "Results" / "per_run_metrics_all_PIPELINE.csv",
    ROOT / "Results" / "per_run_metrics_all_PIPELINE.csv",
    ROOT / "Results" / "per_run_metrics_all_PIPELINE.csv",
]
PER_RUN = next((p for p in CANDS if p.exists()), None)
if PER_RUN is None:
    raise FileNotFoundError("Missing per_run_metrics_all_PIPELINE.csv. Looked for:\n  - " + "\n  - ".join(map(str, CANDS)))

df = pd.read_csv(PER_RUN).copy()
df.columns = [c.lower() for c in df.columns]

need = {"setup","anomaly","win","kfold","pct","run_id","auc_pr","method"}
missing = need - set(df.columns)
if missing:
    raise KeyError(f"per_run_metrics_all_PIPELINE.csv missing columns: {sorted(missing)}. Have: {list(df.columns)}")

# Normalize types
for c in ["win","kfold","pct","auc_pr"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["setup"] = df["setup"].astype(str)
df["anomaly"] = df["anomaly"].astype(str)
df["method"] = df["method"].astype(str)
df["run_id"] = df["run_id"].astype(str)

df = df.dropna(subset=["setup","anomaly","win","kfold","pct","auc_pr","method"]).copy()
df["auc_pr"] = df["auc_pr"].clip(0.0, 1.0)

# ----------------------------
# Compute per-(setup, anomaly, win, kfold, pct, method) median over run_id
# then take max across method for each (setup, anomaly, win, kfold, pct)
# ----------------------------
med_by_method = (
    df.groupby(["setup","anomaly","win","kfold","pct","method"], as_index=False)
      .agg(auc_pr_median=("auc_pr","median"))
)

best_over_method = (
    med_by_method.groupby(["setup","anomaly","win","kfold","pct"], as_index=False)
                .agg(auc_pr_median=("auc_pr_median","max"))
)

# Helper: get cell value
def get_aucpr_median(setup, anomaly, win, kfold, pct):
    sub = best_over_method[
        (best_over_method["setup"].str.upper() == str(setup).upper()) &
        (best_over_method["anomaly"].str.upper() == str(anomaly).upper()) &
        (best_over_method["win"] == int(win)) &
        (best_over_method["kfold"] == int(kfold)) &
        (best_over_method["pct"] == int(pct))
    ]
    if sub.empty:
        return None
    return float(sub["auc_pr_median"].iloc[0])

# ----------------------------
# Build table rows
# ----------------------------
rows = []
for setup_label, cfg in TARGETS.items():
    setup = cfg["setup"]
    win = cfg["win"]
    kfold = cfg["kfold"]
    for anom in cfg["anomalies"]:
        row = {"Setup": setup_label, "Anomaly": paper_anomaly_name(anom)}
        for p in PCTS:
            v = get_aucpr_median(setup, anom, win, kfold, p)
            row[f"{p}%"] = f"{v:.3f}" if v is not None and np.isfinite(v) else "NA"
        rows.append(row)

tab = pd.DataFrame(rows)

# Optional: order anomalies as your paper example
order = ["TRRespass","Droop","Spectre"]
tab["__ord"] = tab["Anomaly"].map(lambda x: order.index(x) if x in order else 999)
tab = tab.sort_values(["Setup","__ord","Anomaly"]).drop(columns=["__ord"]).reset_index(drop=True)

# ----------------------------
# Emit LaTeX (paste into Overleaf)
# ----------------------------
latex_lines = []
latex_lines.append(r"\begin{table}[bp]")
latex_lines.append(r"    \centering")
latex_lines.append(r"    \caption{Median AUC--PR summary vs top $p$ features retained across setups.}")
latex_lines.append(r"    \label{tab:aucpr_summary}")
latex_lines.append(r"    \setlength{\tabcolsep}{6pt}")
latex_lines.append(r"    \renewcommand{\arraystretch}{1.15}")
latex_lines.append(r"    \footnotesize")
latex_lines.append(r"    \begin{tabular}{||c|c|c|c|c|c||}")
latex_lines.append(r"        \hline")
latex_lines.append(r"        \multirow{2}{*}{\textbf{Setup}} &")
latex_lines.append(r"        \multirow{2}{*}{\textbf{Anomaly}} &")
latex_lines.append(r"        \multicolumn{4}{c||}{$p$} \\")
latex_lines.append(r"        \cline{3-6}")
latex_lines.append(r"        & & \textbf{10\%} & \textbf{30\%} & \textbf{70\%} & \textbf{100\%} \\")
latex_lines.append(r"        \hline")

for _, r in tab.iterrows():
    latex_lines.append(
        f"        {r['Setup']} & {r['Anomaly']} & {r['10%']} & {r['30%']} & {r['70%']} & {r['100%']} \\\\"
    )

latex_lines.append(r"        \hline")
latex_lines.append(r"    \end{tabular}")
latex_lines.append(r"\end{table}")

latex = "\n".join(latex_lines)
print(latex)

# Also save to a file in the current session (optional)
out_tex = ROOT / "Results" / "aucpr_summary_table_from_perrun.tex"
out_tex.write_text(latex)
print(f"\n[OK] Saved LaTeX to: {out_tex}")