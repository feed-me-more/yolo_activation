"""models/__init__.py  –  factory for split models."""
from __future__ import annotations

from models.base import SplitModel


def load_model(model_name: str, num_packets: int, device: str) -> SplitModel:
    if model_name == "mobilenet_v2":
        from models.mobilenet import MobileNetV2Split
        return MobileNetV2Split(num_packets, device)
    elif model_name == "vit_b16":
        from models.vit import ViTB16Split
        return ViTB16Split(num_packets, device)
    elif model_name == "yolo":
        from models.yolo import YOLOv8Split
        return YOLOv8Split(num_packets, device)
    else:
        raise ValueError(f"Unknown model: {model_name}")
