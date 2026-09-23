#!/usr/bin/env bash
# Launch the libero_90 pipeline watcher in the background.
# Mirrors launch_render90.sh, whose nohup'd children survive the SSH session.
cd /root/workspace/libero_converter

nohup ./wait_and_convert90.sh > /dev/null 2>&1 &
echo "watcher launched pid=$!"
