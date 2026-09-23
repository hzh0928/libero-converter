#!/usr/bin/env python3
"""Verify the integrity of downloaded LIBERO raw HDF5 files per suite."""

from __future__ import annotations

import glob
import os
import sys

import h5py

RAW_ROOT = "/root/dataset/libero_raw"
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]


def main() -> int:
    grand_total = 0
    for suite in SUITES:
        pattern = os.path.join(RAW_ROOT, suite, "*.hdf5")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"{suite}: NO FILES")
            continue
        broken = []
        eps = 0
        for f in files:
            try:
                with h5py.File(f, "r") as h:
                    eps += len(h["data"].keys())
            except Exception as exc:  # noqa: BLE001
                broken.append((os.path.basename(f), str(exc)[:80]))
        grand_total += eps
        status = "OK" if not broken else f"BROKEN={len(broken)}"
        print(f"{suite}: files={len(files)} episodes={eps} {status}")
        for name, err in broken[:5]:
            print(f"    ! {name}: {err}")
    print(f"TOTAL episodes={grand_total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
