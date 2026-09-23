#!/usr/bin/env bash
# Wait for all parallel render shards to finish, then run the LeRobot export
# and validation stages for every suite.
set -u

WORK=/root/workspace/libero_converter
OUT=/root/dataset/libero_converted
RAW=/root/dataset/libero_raw
LOG=$WORK/finish.log
SUITES="libero_spatial libero_object libero_goal libero_10"

cd "$WORK" || exit 1
export MUJOCO_GL=osmesa
export PYTHONPATH=/root/LIBERO_src

echo "[$(date +%F_%T)] waiting for render shards ..." >>"$LOG"
while pgrep -f "convert_[l]ibero" >/dev/null; do
  sleep 30
done
echo "[$(date +%F_%T)] all render shards finished" >>"$LOG"

for suite in $SUITES; do
  echo "[$(date +%F_%T)] lerobot stage: $suite" >>"$LOG"
  ./venv/bin/python convert_libero_fastwam.py \
    --raw-dir "$RAW" \
    --output-dir "$OUT" \
    --suites "$suite" \
    --resolution 256 \
    --stage lerobot >>"$LOG" 2>&1
  echo "[$(date +%F_%T)] lerobot done: $suite (exit=$?)" >>"$LOG"
done

for suite in $SUITES; do
  echo "[$(date +%F_%T)] validate stage: $suite" >>"$LOG"
  ./venv/bin/python convert_libero_fastwam.py \
    --output-dir "$OUT" \
    --suites "$suite" \
    --stage validate >>"$LOG" 2>&1
  echo "[$(date +%F_%T)] validate done: $suite (exit=$?)" >>"$LOG"
done

echo "[$(date +%F_%T)] PIPELINE_COMPLETE" >>"$LOG"
