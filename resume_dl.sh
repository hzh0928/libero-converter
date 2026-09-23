#!/usr/bin/env bash
# Sequentially download every entry of a task list with dl_parallel.py.
# Usage: resume_dl.sh <task_list>
set -u

LIST="${1:?usage: resume_dl.sh <task_list>}"
cd /root/workspace/libero_converter

while IFS=" " read -r url out size; do
  [ -z "${url:-}" ] && continue
  # Skip files already fully downloaded (covers re-runs of this script).
  if [ -f "$out" ]; then
    echo "[$(date +%T)] SKIP (exists) $out"
    continue
  fi
  echo "[$(date +%T)] START $out"
  ./venv/bin/python3 dl_parallel.py "$url" "$out" "$size"
  echo "[$(date +%T)] END $out (exit=$?)"
done < "$LIST"

echo "RESUME_DL_COMPLETE"
