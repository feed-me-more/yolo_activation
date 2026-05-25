# Receiver Bundle

This bundle is meant for the receiver PC.

It already includes:
- code needed for `main.py stream-recv`
- local `yolov8n.pt`
- prebuilt `stream_outputs/yolo/*` corpus and transform cache

## Setup

```bash
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONPATH=.
```

## Run

```bash
bash run_receiver.sh
```

Default receiver command:

```bash
python3 main.py stream-recv \
  --model yolo \
  --num-packets 40 \
  --corpus-size 96 \
  --device cuda \
  --out-dir stream_outputs \
  --udp-port 9999 \
  --frame-timeout-ms 250
```

Start the sender only after the receiver is fully ready.
