# Data

Set `XOCTANE_DATA_DIR` to point to your dataset root.

Your dataset root should contain per-platform directories (e.g., DDR4/DDR5) and per-anomaly subdirectories
as expected by the extracted notebook code.

Because different labs organize traces differently, adjust the dataset loader section in
`scripts/01_full_multiwin_multik_pipeline.py` if your folders differ.
