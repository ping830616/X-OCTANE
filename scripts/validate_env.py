#!/usr/bin/env python3
import os
from pathlib import Path
import sys

def main():
    data = os.environ.get("XOCTANE_DATA_DIR","")
    if not data:
        print("[ERR] XOCTANE_DATA_DIR is not set.", file=sys.stderr)
        sys.exit(2)
    p=Path(data)
    if not p.exists():
        print(f"[ERR] XOCTANE_DATA_DIR does not exist: {p}", file=sys.stderr)
        sys.exit(2)
    print(f"[OK] XOCTANE_DATA_DIR={p.resolve()}")
    print("[OK] Environment looks good.")
if __name__=="__main__":
    main()
