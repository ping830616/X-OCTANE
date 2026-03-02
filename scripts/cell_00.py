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

# ==== Cell 0: paths & knobs ====
from pathlib import Path

RES_DIR  = ROOT / "Results";            RES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR= RES_DIR / "ROC_PR";          PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MMI_DIR  = ROOT / "FeatureRankOUT"                 # your existing MMI ranks
HYB_DIR  = ROOT / "FeatureRankOUT_HYBRID";         HYB_DIR.mkdir(parents=True, exist_ok=True)
EXPL_DIR = RES_DIR / "Explainability";             EXPL_DIR.mkdir(parents=True, exist_ok=True)
HYBRID_REPORT_DIR = RES_DIR / "Hybrid_Reports";    HYBRID_REPORT_DIR.mkdir(parents=True, exist_ok=True)

SETUPS        = ["DDR4","DDR5"]
WINDOW_SIZES  = [32, 64, 128, 256, 512, 1024]
KFOLDS_SET    = [3, 5, 10]
SUBSPACES     = ["compute","memory","sensors"]
META          = ["label","setup","run_id"]
SEED          = 42