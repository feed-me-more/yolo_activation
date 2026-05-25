#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONPATH=.

python3 main.py stream-send \
  --model yolo \
  --source ./british_highway_traffic.mp4 \
  --udp-host 127.0.0.1 \
  --udp-port 9999 \
  --num-packets 40 \
  --udp-loss 0.2 \
  --device cuda \
  --out-dir stream_outputs \
  --frame-timeout-ms 250 \
  --save-video
