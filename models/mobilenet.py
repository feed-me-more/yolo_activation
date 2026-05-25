"""
models/mobilenet.py  –  MobileNetV2 split after features[14].

Encoder : features[0..14]   →  activation (C, H, W) → flatten → (d,)
Decoder : features[15..18] + classifier  →  top-1 logits
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from models.base import ModelDimensions, SplitModel


_SPLIT_IDX = 14          # split after features[14]  (out of 18)


class MobileNetV2Split(SplitModel):

    def __init__(self, num_packets: int, device: str):
        super().__init__(num_packets, device)
        self.dev = torch.device(device)

        weights = models.MobileNet_V2_Weights.DEFAULT
        full = models.mobilenet_v2(weights=weights).eval()

        # Encoder: features[0..14]
        self._encoder = nn.Sequential(*list(full.features.children())[:_SPLIT_IDX + 1]).to(self.dev)

        # Decoder: features[15..18] + adaptive pool + classifier
        tail_features = nn.Sequential(*list(full.features.children())[_SPLIT_IDX + 1:])
        self._decoder = nn.Sequential(
            tail_features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            full.classifier,
        ).to(self.dev)

        # Probe activation shape with a dummy input
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self.dev)
            act   = self._encoder(dummy)            # (1, C, H, W)
        self._act_shape = tuple(act.shape[1:])      # (C, H, W)
        self._d = int(np.prod(self._act_shape))

        for p in self._encoder.parameters(): p.requires_grad_(False)
        for p in self._decoder.parameters(): p.requires_grad_(False)

    # ── interface ─────────────────────────────

    def encoder(self, x: torch.Tensor) -> np.ndarray:
        x = x.to(self.dev)
        with torch.no_grad():
            act = self._encoder(x)                  # (1, C, H, W)
        return act.cpu().numpy().flatten().astype(np.float32)

    def decoder(self, activation: np.ndarray) -> torch.Tensor:
        """activation : (d,) → logits : (num_classes,)"""
        C, H, W = self._act_shape
        t = torch.from_numpy(activation).reshape(1, C, H, W).to(self.dev)
        with torch.no_grad():
            logits = self._decoder(t)               # (1, 1000)
        return logits.cpu().squeeze(0)

    @property
    def dims(self) -> ModelDimensions:
        d_c = self._d // self.num_packets
        return ModelDimensions(
            model_name   = "mobilenet_v2",
            input_shape  = (3, 224, 224),
            activation_d = self._d,
            num_packets  = self.num_packets,
            chunk_size   = d_c,
            bytes_per_pkt= d_c * 4,
            split_desc   = f"after features[{_SPLIT_IDX}]  "
                           f"| activation shape {self._act_shape}",
        )

    def offload(self) -> None:
        del self._encoder, self._decoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
