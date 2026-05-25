# Split Inference Experiments

This repo contains:
- offline split-inference experiments for `mobilenet_v2`, `vit_b16`, and `yolo`
- ablation runs over packet count / bitmask / random permutation
- a YOLO UDP sender/receiver demo for local or multi-device streaming

## Setup

```bash
cd /home/yaswanth-ram-kumar/ablation_andrej
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONPATH=.
```

Optional:

```bash
export HF_TOKEN="your_hf_token_if_needed"
```

`HF_TOKEN` is only needed for ImageNet/HuggingFace fallback paths. YOLO uses local COCO128-style data handling and local weights if available.

## Main CLI

The main entrypoint is:

```bash
python3 main.py <command> [args]
```

Available commands:
- `build-corpus`
- `transform`
- `align`
- `infer`
- `run`
- `stream-send`
- `stream-recv`

## Common Args

- `--model`: `mobilenet_v2`, `vit_b16`, or `yolo`
- `--num-packets`: number of packets `P`
- `--corpus-size`: corpus size `M`
- `--query-size`: number of evaluation queries
- `--loss-rates`: comma-separated loss rates, for example `0.0,0.2,0.4`
- `--batch-size`: dataloader / corpus build batch size
- `--device`: `cpu`, `cuda`, or `auto`
- `--out-dir`: output directory
- `--seed`: random seed
- `--no-bitmask`: disable random sign masking
- `--no-randperm`: disable global permutation

## Offline Experiments

### 1. Build Corpus

```bash
python3 main.py build-corpus \
  --model mobilenet_v2 \
  --num-packets 40 \
  --corpus-size 5000 \
  --batch-size 8 \
  --device cuda \
  --out-dir outputs_mnv2_p40
```

### 2. Apply Transform

Default is bitmask `ON` and randperm `ON`:

```bash
python3 main.py transform \
  --model mobilenet_v2 \
  --num-packets 40 \
  --out-dir outputs_mnv2_p40
```

Bitmask off:

```bash
python3 main.py transform \
  --model mobilenet_v2 \
  --num-packets 40 \
  --no-bitmask \
  --out-dir outputs_mnv2_p40_nobm
```

Both off:

```bash
python3 main.py transform \
  --model mobilenet_v2 \
  --num-packets 40 \
  --no-bitmask \
  --no-randperm \
  --out-dir outputs_mnv2_p40_plain
```

### 3. Alignment Only

```bash
python3 main.py align \
  --model vit_b16 \
  --num-packets 40 \
  --corpus-size 5000 \
  --query-size 100 \
  --loss-rates 0.0,0.2,0.4 \
  --batch-size 8 \
  --device cuda \
  --out-dir outputs_vit_align_p40
```

### 4. Inference Only

```bash
python3 main.py infer \
  --model mobilenet_v2 \
  --num-packets 40 \
  --corpus-size 5000 \
  --query-size 100 \
  --loss-rates 0.0,0.2,0.4 \
  --batch-size 8 \
  --device cuda \
  --out-dir outputs_mnv2_infer_p40
```

### 5. Run Both Alignment + Inference

MobileNetV2:

```bash
python3 main.py run \
  --model mobilenet_v2 \
  --num-packets 40 \
  --corpus-size 5000 \
  --query-size 100 \
  --loss-rates 0.0,0.2,0.4 \
  --batch-size 8 \
  --device cuda \
  --out-dir outputs_mnv2_p40
```

ViT-B16:

```bash
python3 main.py run \
  --model vit_b16 \
  --num-packets 40 \
  --corpus-size 5000 \
  --query-size 100 \
  --loss-rates 0.0,0.2,0.4 \
  --batch-size 8 \
  --device cuda \
  --out-dir outputs_vit_p40
```

YOLO:

```bash
python3 main.py run \
  --model yolo \
  --num-packets 40 \
  --corpus-size 96 \
  --query-size 32 \
  --loss-rates 0.0,0.2,0.4 \
  --batch-size 1 \
  --device cuda \
  --out-dir outputs_yolo_p40
```

YOLO without bitmask:

```bash
python3 main.py run \
  --model yolo \
  --num-packets 20 \
  --corpus-size 96 \
  --query-size 32 \
  --loss-rates 0.0,0.2,0.4 \
  --batch-size 1 \
  --device cuda \
  --no-bitmask \
  --out-dir outputs_yolo_p20_nobm
```

## Full Ablation Script

Run the large ablation sweep:

```bash
bash /home/yaswanth-ram-kumar/ablation_andrej/run_all_ablations.sh
```

Note:
- the script currently `cd`s into `/home/yaswanth-ram-kumar/ablation/ablation_andrej_run`
- if you want it to run this active repo instead, update that first line

The script now cleans up cached corpus/transform files after each run to avoid filling disk/memory.

## YOLO Image Sanity Check

The current `yolo_inference.py` in this repo is a simple single-image demo. Edit `img_path` inside the file if needed, then run:

```bash
python3 yolo_inference.py
```

It will open a matplotlib figure and print detections for the image.

## YOLO UDP Streaming Demo

This is the two-terminal sender/receiver split-inference demo.

Important:
- for UDP streaming, use `P >= 40`
- `P=20` is too large per datagram for this YOLO activation
- simulated packet loss is only applied for localhost-style runs
- for non-local IPs, the code disables simulated loss and reports rough observed loss from packets received

### Receiver

Start receiver first:

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

Wait until it prints:

```text
Receiver ready. Start the sender now.
```

### Sender on the Same Device

Run on a local video with simulated packet loss:

```bash
python3 main.py stream-send \
  --model yolo \
  --source /home/yaswanth-ram-kumar/ablation_andrej/british_highway_traffic.mp4 \
  --udp-host 127.0.0.1 \
  --udp-port 9999 \
  --num-packets 40 \
  --udp-loss 0.2 \
  --device cuda \
  --out-dir stream_outputs \
  --frame-timeout-ms 250 \
  --save-video
```

Second traffic video:

```bash
python3 main.py stream-send \
  --model yolo \
  --source /home/yaswanth-ram-kumar/ablation_andrej/road_trafifc.mp4 \
  --udp-host 127.0.0.1 \
  --udp-port 9999 \
  --num-packets 40 \
  --udp-loss 0.2 \
  --device cuda \
  --out-dir stream_outputs \
  --frame-timeout-ms 250 \
  --save-video
```

Use webcam:

```bash
python3 main.py stream-send \
  --model yolo \
  --source 0 \
  --udp-host 127.0.0.1 \
  --udp-port 9999 \
  --num-packets 40 \
  --udp-loss 0.1 \
  --device cuda \
  --out-dir stream_outputs \
  --frame-timeout-ms 250
```

### Sender to a Different Device

Receiver machine:

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

Sender machine:

```bash
python3 main.py stream-send \
  --model yolo \
  --source /path/to/video.mp4 \
  --udp-host 192.168.1.23 \
  --udp-port 9999 \
  --num-packets 40 \
  --udp-loss 0.2 \
  --device cuda \
  --out-dir stream_outputs \
  --frame-timeout-ms 250 \
  --save-video
```

Behavior on remote IPs:
- `--udp-loss` is ignored automatically
- real network/channel effects determine what is lost
- the receiver reports rough packet loss from `received / total`

### Streaming Flags

- `--source`: video file path or `0` for webcam
- `--udp-host`: receiver IP
- `--udp-port`: UDP port
- `--udp-loss`: simulated loss rate for localhost runs only
- `--frame-timeout-ms`: receiver assembly timeout for partial frames
- `--max-frames`: useful for short smoke tests
- `--save-video`: save annotated sender output
- `--no-show`: disable OpenCV display windows

Example smoke test:

```bash
python3 main.py stream-send \
  --model yolo \
  --source /home/yaswanth-ram-kumar/ablation_andrej/road_trafifc.mp4 \
  --udp-host 127.0.0.1 \
  --udp-port 9999 \
  --num-packets 40 \
  --udp-loss 0.2 \
  --device cuda \
  --out-dir stream_outputs \
  --frame-timeout-ms 250 \
  --max-frames 20
```

## Outputs

Typical outputs include:
- JSON summaries
- alignment / inference plots
- transformed corpus files
- streaming annotated videos

Examples:
- `outputs_*`
- `complete/*`
- `stream_outputs/*`
