#!/usr/bin/env bash
# Wait for the libero_90 render shards to finish, then run the LeRobot
# conversion and finally validate the result.
set -u

cd /root/workspace/libero_converter
LOG=/root/workspace/libero_converter/pipeline_libero90.log

log() { echo "[$(date +%T)] $*" >> "$LOG"; }

log "watcher started"

# Wait until no render shard is running any more.
while pgrep -f "convert_libero_fastwam.py.*--stage render" > /dev/null; do
  sleep 60
done
log "all render shards finished"

# Sanity check: every one of the 90 tasks must have a staging file.
STAGING=/root/dataset/libero_converted/.staging_3d/libero_90
N=$(ls -1 "$STAGING"/*.hdf5 2>/dev/null | wc -l)
log "staging task files: $N (expected 90)"
if [ "$N" -ne 90 ]; then
  log "ABORT: expected 90 staging files, found $N"
  exit 1
fi

log "starting lerobot conversion"
env MUJOCO_GL=osmesa PYTHONPATH=/root/LIBERO_src ./venv/bin/python \
  convert_libero_fastwam.py \
  --raw-dir /root/dataset/libero_raw \
  --output-dir /root/dataset/libero_converted \
  --suites libero_90 \
  --resolution 256 \
  --stage lerobot \
  >> /root/workspace/libero_converter/lerobot_libero90.log 2>&1
log "lerobot finished (exit=$?)"

log "starting validate"
env PYTHONPATH=/root/LIBERO_src ./venv/bin/python \
  convert_libero_fastwam.py \
  --output-dir /root/dataset/libero_converted \
  --suites libero_90 \
  --stage validate \
  >> /root/workspace/libero_converter/validate_libero90.log 2>&1
log "validate finished (exit=$?)"

log "PIPELINE_LIBERO90_COMPLETE"
