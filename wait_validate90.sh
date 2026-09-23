#!/usr/bin/env bash
# Wait for the libero_90 LeRobot conversion to finish, then validate it.
set -u

cd /root/workspace/libero_converter
LOG=/root/workspace/libero_converter/pipeline_libero90.log

log() { echo "[$(date +%T)] $*" >> "$LOG"; }

log "validate watcher started"

while pgrep -f "[c]onvert_libero_fastwam.py.*--stage lerobot" > /dev/null; do
  sleep 60
done
log "lerobot conversion finished"

env PYTHONPATH=/root/LIBERO_src ./venv/bin/python \
  convert_libero_fastwam.py \
  --output-dir /root/dataset/libero_converted \
  --suites libero_90 \
  --stage validate \
  >> /root/workspace/libero_converter/validate_libero90.log 2>&1
log "validate finished (exit=$?)"

log "VALIDATE90_COMPLETE"
