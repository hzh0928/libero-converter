#!/usr/bin/env python3
"""Chunked parallel downloader for hosts whose egress proxy stalls long-lived
single connections (e.g. the 10.6.2.13 box reaching HuggingFace).

Splits the object into fixed size byte ranges, fetches them concurrently with
curl (several retries per chunk), then concatenates the parts.

Usage: python3 dl_parallel.py <url> <output_path>
"""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

CHUNK = 8 << 20
WORKERS = 16
CHUNK_TIMEOUT = 180
CHUNK_ATTEMPTS = 6


def get_size(url: str) -> int:
    # A ranged request is the most reliable probe: HEAD often returns the
    # length of a redirect/error page rather than the real object.
    probe = subprocess.run(
        ["curl", "-sSL", "-r", "0-0", "-D", "-", "-o", "/dev/null", "-m", "60", url],
        capture_output=True,
        text=True,
    ).stdout
    for line in probe.splitlines():
        if line.lower().startswith("content-range:") and "/" in line:
            try:
                return int(line.rsplit("/", 1)[1].strip())
            except ValueError:
                pass
    head = subprocess.run(
        ["curl", "-sSL", "-I", "-m", "60", url], capture_output=True, text=True
    ).stdout
    for line in head.splitlines():
        if line.lower().startswith("content-length:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return 0


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: dl_parallel.py <url> <output> [size]")
        return 2
    url, out = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    size = int(sys.argv[3]) if len(sys.argv) > 3 else get_size(url)
    print(f"total size: {size}", flush=True)
    if size <= 0:
        print("ERROR: could not determine remote size")
        return 1

    parts_dir = out + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    ranges = [(s, min(s + CHUNK - 1, size - 1)) for s in range(0, size, CHUNK)]
    print(f"chunks: {len(ranges)}", flush=True)

    def work(i: int) -> int:
        a, b = ranges[i]
        part = os.path.join(parts_dir, f"{i:06d}.part")
        expected = b - a + 1
        for _ in range(CHUNK_ATTEMPTS):
            if os.path.exists(part) and os.path.getsize(part) == expected:
                return i
            try:
                subprocess.run(
                    [
                        "curl", "-sSL", "-m", str(CHUNK_TIMEOUT), "--retry", "2",
                        "--connect-timeout", "20",
                        "-r", f"{a}-{b}", "-o", part, url,
                    ],
                    capture_output=True,
                    # curl's own -m timeout can fail to fire if the process
                    # blocks in getaddrinfo() (no async DNS resolver), so
                    # enforce a hard wall-clock timeout here as well and
                    # kill the stuck process if it overruns.
                    timeout=CHUNK_TIMEOUT + 30,
                )
            except subprocess.TimeoutExpired:
                pass
            if os.path.exists(part) and os.path.getsize(part) == expected:
                return i
        raise RuntimeError(f"chunk {i} failed after {CHUNK_ATTEMPTS} attempts")

    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for _ in pool.map(work, range(len(ranges))):
            done += 1
            if done % 8 == 0 or done == len(ranges):
                print(f"chunks done: {done}/{len(ranges)}", flush=True)

    with open(out, "wb") as dst:
        for i in range(len(ranges)):
            with open(os.path.join(parts_dir, f"{i:06d}.part"), "rb") as src:
                dst.write(src.read())

    got = os.path.getsize(out)
    print(f"wrote {out} ({got} bytes)", flush=True)
    if got != size:
        print(f"ERROR: size mismatch {got} != {size}")
        return 1
    subprocess.run(["rm", "-rf", parts_dir])
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
