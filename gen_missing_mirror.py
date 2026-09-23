#!/usr/bin/env python3
"""Rebuild the libero_90 missing-file list, rewriting HuggingFace URLs to the
hf-mirror.com mirror because huggingface.co is DNS-poisoned on this box."""

from __future__ import annotations

import glob
import os
import sys

RAW_DIR = "/root/dataset/libero_raw/libero_90"
TASK_LIST = "/root/workspace/libero_converter/dl_90.txt"
OUT_LIST = "/root/workspace/libero_converter/dl_90_mirror_missing.txt"
POISONED = "https://huggingface.co"
MIRROR = "https://hf-mirror.com"


def main() -> int:
    have = {os.path.basename(p) for p in glob.glob(os.path.join(RAW_DIR, "*.hdf5"))}

    missing = []
    with open(TASK_LIST) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            url, out = parts[0], parts[1]
            if os.path.basename(out) in have:
                continue
            url = url.replace(POISONED, MIRROR)
            rest = " ".join(parts[2:])
            missing.append(f"{url} {out} {rest}".strip())

    with open(OUT_LIST, "w") as fh:
        fh.write("\n".join(missing) + "\n")

    print(f"have={len(have)} missing={len(missing)} -> {OUT_LIST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
