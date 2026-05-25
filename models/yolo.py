"""
models/yolo.py  –  YOLOv8n split at the backbone/neck boundary.

Encoder : YOLOv8 backbone  →  save layers 4 / 6 / 9  → flatten  → (d,)
Decoder : reconstruct those three feature maps  →  run neck + Detect head

Uses ultralytics YOLOv8n pretrained on COCO.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.base import ModelDimensions, SplitModel


class YOLOv8Split(SplitModel):

    def __init__(self, num_packets: int, device: str):
        super().__init__(num_packets, device)
        self.dev = torch.device(device)

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is not installed.  Run: pip install ultralytics"
            )

        from pathlib import Path
        local_candidates = [
            Path("/home/yaswanth-ram-kumar/LT EDL/yolov8n.pt"),
            Path("yolov8n.pt"),
        ]
        weights_path = next((str(p) for p in local_candidates if p.exists()), "yolov8n.pt")
        yolo = YOLO(weights_path)
        self._model = yolo.model.eval().to(self.dev)

        all_layers = list(self._model.model.children())
        self._layers = nn.ModuleList(all_layers)
        self._backbone_end = 9
        self._split_layers = (4, 6, 9)

        for p in self._model.parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 640, 640, device=self.dev)
            feats = self._forward_backbone(dummy)
        self._feat_shapes = [tuple(feat.shape[1:]) for feat in feats]
        self._feat_sizes = [int(np.prod(shape)) for shape in self._feat_shapes]
        self._feat_offsets = np.cumsum([0, *self._feat_sizes]).astype(np.int64)
        self._d = int(sum(self._feat_sizes))

        self._full_model = self._model

    def _forward_backbone(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = x
        saved: dict[int, torch.Tensor] = {}
        for i in range(self._backbone_end + 1):
            out = self._layers[i](out)
            if i in self._split_layers:
                saved[i] = out
        return tuple(saved[i] for i in self._split_layers)

    def _pack_features(self, feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> np.ndarray:
        chunks = [feat.detach().cpu().numpy().reshape(-1).astype(np.float32) for feat in feats]
        return np.concatenate(chunks, axis=0)

    def _unpack_activation(self, activation: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if activation.shape[0] != self._d:
            raise ValueError(f"Expected activation of length {self._d}, got {activation.shape[0]}")

        feats = []
        for idx, shape in enumerate(self._feat_shapes):
            lo = int(self._feat_offsets[idx])
            hi = int(self._feat_offsets[idx + 1])
            arr = activation[lo:hi]
            tensor = torch.from_numpy(arr).reshape(1, *shape).to(self.dev)
            feats.append(tensor)
        return tuple(feats)

    def encoder(self, x: torch.Tensor) -> np.ndarray:
        x = x.to(self.dev)
        with torch.no_grad():
            feats = self._forward_backbone(x)
        return self._pack_features(feats)

    def decoder(self, activation: np.ndarray) -> torch.Tensor:
        f4, f6, f9 = self._unpack_activation(activation.astype(np.float32, copy=False))
        saved: dict[int, torch.Tensor | tuple] = {4: f4, 6: f6, 9: f9}
        out: torch.Tensor | tuple = f9

        with torch.no_grad():
            for i in range(10, len(self._layers)):
                layer = self._layers[i]
                src = getattr(layer, "f", -1)
                if src == -1:
                    inp = out
                elif isinstance(src, int):
                    inp = saved[src]
                else:
                    inp = [out if j == -1 else saved[j] for j in src]
                out = layer(inp)
                saved[i] = out

        pred = out[0] if isinstance(out, (list, tuple)) else out
        return pred

    def decoder_full(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dev)
        with torch.no_grad():
            out = self._full_model(x)
        return out[0] if isinstance(out, (list, tuple)) else out

    @property
    def dims(self) -> ModelDimensions:
        d_c = int(np.ceil(self._d / self.num_packets))
        return ModelDimensions(
            model_name="yolo",
            input_shape=(3, 640, 640),
            activation_d=self._d,
            num_packets=self.num_packets,
            chunk_size=d_c,
            bytes_per_pkt=d_c * 4,
            split_desc=(
                f"backbone/neck boundary with layers {self._split_layers} "
                f"| feature shapes {self._feat_shapes}"
            ),
        )

    def offload(self) -> None:
        del self._model, self._full_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_coco128_loader(
    batch_size: int = 1,
    image_size: int = 640,
    split: str = "train",
    n: int | None = None,
    seed: int = 42,
):
    """
    COCO128 has one image set; we create a deterministic 75/25 split so
    corpus and query images do not overlap.
    """
    try:
        from ultralytics.data.utils import check_det_dataset
        from torchvision import transforms
        from pathlib import Path
        from PIL import Image

        data = check_det_dataset("coco128.yaml")
        dataset_root = Path(data["train"])
        if dataset_root.is_file():
            dataset_root = dataset_root.parent
        label_root = Path(dataset_root.as_posix().replace("/images/", "/labels/"))

        image_paths = np.array(sorted(dataset_root.glob("*.jpg"))[:128], dtype=object)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(image_paths))
        n_train = max(1, int(0.75 * len(image_paths)))
        if split in ("train", "corpus"):
            chosen = perm[:n_train]
        else:
            chosen = perm[n_train:]
        if n is not None:
            chosen = chosen[: min(n, len(chosen))]
        image_paths = image_paths[chosen].tolist()

        tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        class COCO128Dataset(Dataset):
            def __init__(self, paths, transform):
                self.paths = paths
                self.transform = transform

            def __len__(self): return len(self.paths)

            def __getitem__(self, i):
                path = Path(self.paths[i])
                img = Image.open(path).convert("RGB")
                x = self.transform(img)

                label_path = label_root / f"{path.stem}.txt"
                boxes = []
                classes = []
                if label_path.exists():
                    for line in label_path.read_text().splitlines():
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        cls, xc, yc, w, h = map(float, parts)
                        x1 = (xc - w / 2.0) * image_size
                        y1 = (yc - h / 2.0) * image_size
                        x2 = (xc + w / 2.0) * image_size
                        y2 = (yc + h / 2.0) * image_size
                        boxes.append([x1, y1, x2, y2])
                        classes.append(int(cls))

                target = {
                    "path": str(path),
                    "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                    "classes": np.asarray(classes, dtype=np.int64),
                }
                primary = int(classes[0]) if classes else -1
                return x, {"target": target, "primary_class": primary}

        def _collate(batch):
            xs = torch.stack([b[0] for b in batch], dim=0)
            ys = [b[1] for b in batch]
            return xs, ys

        ds = COCO128Dataset(image_paths, tf)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    except Exception as e:
        print(f"  [yolo] COCO128 loader failed: {e}")
        return None


def postprocess_yolo_predictions(
    pred: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    nc: int = 80,
) -> np.ndarray:
    from ultralytics.utils import nms

    dets = nms.non_max_suppression(
        pred,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        nc=nc,
    )
    if not dets:
        return np.zeros((0, 6), dtype=np.float32)
    det = dets[0]
    if det is None or det.numel() == 0:
        return np.zeros((0, 6), dtype=np.float32)
    return det.detach().cpu().numpy().astype(np.float32, copy=False)
