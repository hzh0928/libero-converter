#!/usr/bin/env bash
# Render libero_90 on the GPUs via EGL.
# MuJoCo needs MUJOCO_EGL_DEVICE_ID: without it its device enumeration fails.
# Devices 0-3 are the four H100s, so shards are distributed round-robin.
set -u

SHARDS="${1:-18}"
GPUS="${2:-4}"
cd /root/workspace/libero_converter

for ((i = 0; i < SHARDS; i++)); do
  DEV=$((i % GPUS))
  nohup env MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$DEV" PYTHONPATH=/root/LIBERO_src \
    ./venv/bin/python \
    convert_libero_fastwam.py \
    --raw-dir /root/dataset/libero_raw \
    --output-dir /root/dataset/libero_converted \
    --suites libero_90 \
    --resolution 256 \
    --stage render \
    --shard-index "$i" \
    --shard-count "$SHARDS" \
    > "r90_gpu_${i}.log" 2>&1 &
done

echo "launched ${SHARDS} EGL render shards across ${GPUS} GPUs"
