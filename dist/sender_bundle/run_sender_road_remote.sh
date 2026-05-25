#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONPATH=.

RECEIVER_IP="192.168.1.23"

python3 main.py stream-send \
  --model yolo \
  --source ./road_trafifc.mp4 \
  --udp-host "${RECEIVER_IP}" \
  --udp-port 9999 \
  --num-packets 40 \
  --udp-loss 0.2 \
  --device cuda \
  --out-dir stream_outputs \
  --frame-timeout-ms 250 \
  --save-video
