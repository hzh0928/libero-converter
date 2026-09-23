#!/usr/bin/env bash
# Fetch the NVIDIA user-space GL packages without relying on apt's slow
# single-connection download: grab the .deb URLs with apt, pull them in
# parallel with curl, then install with dpkg.
set -u

export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export DEBIAN_FRONTEND=noninteractive

WORK=/root/workspace/libero_converter
DEBDIR=/tmp/nvdebs
LOG=$WORK/install_nv.log

pkill -9 apt-get 2>/dev/null
sleep 2

echo "[$(date +%T)] resolving deb urls" >>"$LOG"
apt-get install --print-uris -y libnvidia-gl-535 2>/dev/null \
  | grep -oE "'http[^']+'" | tr -d "'" | sort -u > "$WORK/deb_urls.txt"

echo "[$(date +%T)] found $(wc -l < "$WORK/deb_urls.txt") packages" >>"$LOG"

rm -rf "$DEBDIR"
mkdir -p "$DEBDIR"

echo "[$(date +%T)] downloading in parallel" >>"$LOG"
cat "$WORK/deb_urls.txt" | xargs -P 6 -I{} bash -c 'curl -sSL --retry 3 -m 900 -o /tmp/nvdebs/$(basename "{}") "{}"'

echo "[$(date +%T)] downloaded:" >>"$LOG"
ls -la "$DEBDIR" >>"$LOG" 2>&1

echo "[$(date +%T)] installing" >>"$LOG"
dpkg -i "$DEBDIR"/*.deb >>"$LOG" 2>&1
apt-get -f install -y >>"$LOG" 2>&1

echo "[$(date +%T)] done" >>"$LOG"
ls /usr/share/glvnd/egl_vendor.d/ >>"$LOG" 2>&1
ls /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.* >>"$LOG" 2>&1
