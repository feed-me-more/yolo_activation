#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONPATH=.

python3 main.py stream-recv \
  --model yolo \
  --num-packets 40 \
  --corpus-size 96 \
  --device cuda \
  --out-dir stream_outputs \
  --udp-port 9999 \
  --frame-timeout-ms 250
