# Sender Bundle

This bundle is meant for the sender PC.

It already includes:
- code needed for `main.py stream-send`
- local `yolov8n.pt`
- both sample traffic videos

## Setup

```bash
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONPATH=.
```

## Local Same-PC Demo

```bash
bash run_sender_road_local.sh
```

or

```bash
bash run_sender_british_local.sh
```

## Remote Receiver Demo

Edit the receiver IP in one of:
- `run_sender_road_remote.sh`
- `run_sender_british_remote.sh`

Then run it.

For non-local targets, simulated loss is automatically disabled by the code.
