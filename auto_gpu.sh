#!/usr/bin/env bash
# Wait for a GPU whose EGL offscreen rendering works, then run a render speed
# test (1 task x 1 demo) so we know the throughput before launching the full run.
#
# All GPUs on this host are currently occupied by other users' processes, so
# MuJoCo cannot create an EGL context. This script polls until one frees up.

set -u

WORKDIR=/root/workspace/libero_converter
LOG=$WORKDIR/auto_gpu.log
RAW=/root/dataset/libero_raw
OUT=/root/dataset/_speedtest_auto
SRC=/root/LIBERO_src

cd "$WORKDIR" || exit 1

probe() {
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$1" CUDA_VISIBLE_DEVICES="$1" \
    ./venv/bin/python "$WORKDIR/gpu_probe.py" >/dev/null 2>&1
}

echo "[$(date +%F_%T)] waiting for a GPU with working EGL ..." >>"$LOG"

while true; do
  found=""
  for i in 0 1 2 3; do
    if probe "$i"; then
      found="$i"
      break
    fi
  done

  if [ -n "$found" ]; then
    echo "[$(date +%F_%T)] GPU $found EGL OK -> starting render speed test" >>"$LOG"
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="$found" MUJOCO_EGL_DEVICE_ID="$found" \
      PYTHONPATH="$SRC" \
      ./venv/bin/python convert_libero_fastwam.py \
        --raw-dir "$RAW" \
        --output-dir "$OUT" \
        --suites libero_spatial \
        --max-tasks 1 \
        --max-demos 1 \
        --stage render >>"$LOG" 2>&1
    echo "[$(date +%F_%T)] speed test finished (exit=$?) on GPU $found" >>"$LOG"
    exit 0
  fi

  sleep 120
done
