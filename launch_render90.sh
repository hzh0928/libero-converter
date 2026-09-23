#!/usr/bin/env bash
# Render libero_90 with OSMesa across several parallel shards.
# EGL is unavailable in this container (EGLError), so fall back to CPU osmesa.
set -u

SHARDS="${1:-18}"
cd /root/workspace/libero_converter

for ((i = 0; i < SHARDS; i++)); do
  nohup env MUJOCO_GL=osmesa PYTHONPATH=/root/LIBERO_src ./venv/bin/python \
    convert_libero_fastwam.py \
    --raw-dir /root/dataset/libero_raw \
    --output-dir /root/dataset/libero_converted \
    --suites libero_90 \
    --resolution 256 \
    --stage render \
    --shard-index "$i" \
    --shard-count "$SHARDS" \
    > "r_libero_90_${i}.log" 2>&1 &
done

echo "launched ${SHARDS} render shards for libero_90"
