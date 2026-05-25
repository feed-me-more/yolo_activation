"""
corpus/build.py  –  build the raw activation corpus from training data.

Saves:
  <out_dir>/<model>/corpus_raw.npy    shape (M, d)   float32
  <out_dir>/<model>/corpus_labels.npy shape (M,)     int32

No bitmask, no permutation – pure activations.
Transform is applied separately by corpus/transform.py.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import load_model


# ─────────────────────────────────────────────
# ImageNet streaming loader (HuggingFace)
# ─────────────────────────────────────────────

def _get_imagenet_loader(
    split: str,
    n: int,
    batch_size: int,
    image_size: int,
    hf_token: str,
    seed: int = 42,
) -> DataLoader:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    from torchvision import transforms
    from torch.utils.data import IterableDataset
    from PIL import Image

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    class _DS(IterableDataset):
        def __iter__(self):
            ds = load_dataset("imagenet-1k", split=split,
                              streaming=True, token=hf_token)
            count = 0
            for s in ds:
                img = s["image"]
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(np.array(img))
                img = img.convert("RGB")
                yield tf(img), int(s["label"])
                count += 1
                if count >= n:
                    break

    return DataLoader(_DS(), batch_size=batch_size, num_workers=0)


# ─────────────────────────────────────────────
# Main build function
# ─────────────────────────────────────────────

def build_corpus(
    model_name: str,
    num_packets: int,
    device: str,
    corpus_size: int,
    out_dir: str,
    batch_size: int = 32,
    hf_token: str = "",
    image_size: int = 224,
    force_rebuild: bool = False,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (corpus, labels) where corpus shape is (M, d).
    Skips rebuild if files already exist unless force_rebuild=True.
    """
    model_dir = Path(out_dir) / model_name
    corpus_path = model_dir / "corpus_raw.npy"
    labels_path = model_dir / "corpus_labels.npy"
    model_dir.mkdir(parents=True, exist_ok=True)

    if corpus_path.exists() and labels_path.exists() and not force_rebuild:
        print(f"  [corpus] Loading cached corpus from {corpus_path}")
        corpus = np.load(str(corpus_path))
        labels = np.load(str(labels_path))
        print(f"  [corpus] shape={corpus.shape}  labels={labels.shape}")
        return corpus, labels

    print(f"\n{'─'*60}")
    print(f"  Building corpus: {model_name}  M={corpus_size}")
    print(f"{'─'*60}")

    model = load_model(model_name, num_packets, device)
    model.dims.print()

    d = model.dims.activation_d

    corpus = np.zeros((corpus_size, d), dtype=np.float32)
    labels = np.zeros(corpus_size, dtype=np.int32)

    # Dataset: COCO128 for yolo, imagenet for others
    if model_name == "yolo":
        from models.yolo import get_coco128_loader
        loader = get_coco128_loader(
            batch_size=batch_size,
            image_size=640,
            split="train",
            n=corpus_size,
            seed=seed,
        )
        if loader is None:
            raise RuntimeError("COCO128 loader failed")
    else:
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN required for ImageNet. "
                "Set env var HF_TOKEN or pass --hf-token."
            )
        loader = _get_imagenet_loader(
            split="train",
            n=corpus_size,
            batch_size=batch_size,
            image_size=image_size,
            hf_token=hf_token,
            seed=seed,
        )

    idx = 0
    t0  = time.perf_counter()
    for batch_x, batch_y in loader:
        for i in range(len(batch_x)):
            if idx >= corpus_size:
                break
            act = model.encoder(batch_x[i:i+1])    # (d,)
            corpus[idx] = act
            if model_name == "yolo":
                labels[idx] = int(batch_y[i].get("primary_class", -1))
            else:
                labels[idx] = int(batch_y[i])
            idx += 1
        elapsed = time.perf_counter() - t0
        rate    = idx / elapsed if elapsed > 0 else 0
        eta     = (corpus_size - idx) / rate if rate > 0 else 0
        print(f"\r  {idx}/{corpus_size}  ({rate:.1f} samples/s  ETA {eta:.0f}s)", end="", flush=True)
        if idx >= corpus_size:
            break

    corpus = corpus[:idx]
    labels = labels[:idx]
    print(f"\n  [corpus] Done. {idx} samples in {time.perf_counter()-t0:.1f}s")

    np.save(str(corpus_path), corpus)
    np.save(str(labels_path), labels)
    print(f"  [corpus] Saved to {corpus_path}")

    model.offload()
    return corpus, labels
