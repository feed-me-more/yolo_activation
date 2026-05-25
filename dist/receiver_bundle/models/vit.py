"""
models/vit.py  –  ViT-B/16 split after transformer block 9 (of 12).

Encoder : patch-embed + blocks[0..8]  →  token sequence  (197, 768) → flatten
Decoder : blocks[9..11] + norm + head  →  logits
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights

from models.base import ModelDimensions, SplitModel


_SPLIT_BLOCK = 9        # split after encoder block index 8  (0-indexed)


class ViTB16Split(SplitModel):

    def __init__(self, num_packets: int, device: str):
        super().__init__(num_packets, device)
        self.dev = torch.device(device)

        weights = ViT_B_16_Weights.DEFAULT
        vit = models.vit_b_16(weights=weights).eval().to(self.dev)

        # ── encoder pieces ────────────────────
        self._conv_proj    = vit.conv_proj
        self._class_token  = vit.class_token
        self._pos_embed    = vit.encoder.pos_embedding
        self._dropout      = vit.encoder.dropout
        self._enc_blocks   = nn.Sequential(
            *list(vit.encoder.layers.children())[:_SPLIT_BLOCK]
        )

        # ── decoder pieces ────────────────────
        self._dec_blocks   = nn.Sequential(
            *list(vit.encoder.layers.children())[_SPLIT_BLOCK:]
        )
        self._ln            = vit.encoder.ln
        self._heads         = vit.heads

        for mod in [self._conv_proj, self._enc_blocks,
                    self._dec_blocks, self._ln, self._heads]:
            mod.to(self.dev)
            for p in mod.parameters():
                p.requires_grad_(False)

        # Probe activation shape
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self.dev)
            act   = self._run_encoder(dummy)        # (1, 197, 768)
        self._act_shape = tuple(act.shape[1:])      # (197, 768)
        self._d = int(np.prod(self._act_shape))

    # ── helpers ───────────────────────────────

    def _run_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """Returns token sequence  (1, num_tokens, hidden)."""
        x   = self._conv_proj(x)                    # (1, 768, 14, 14)
        n, c, h, w = x.shape
        x   = x.reshape(n, c, h * w).permute(0, 2, 1)   # (1, 196, 768)
        cls = self._class_token.expand(n, -1, -1)         # (1, 1, 768)
        x   = torch.cat([cls, x], dim=1)                  # (1, 197, 768)
        x   = x + self._pos_embed
        x   = self._dropout(x)
        x   = self._enc_blocks(x)                         # (1, 197, 768)
        return x

    # ── interface ─────────────────────────────

    def encoder(self, x: torch.Tensor) -> np.ndarray:
        x = x.to(self.dev)
        with torch.no_grad():
            act = self._run_encoder(x)              # (1, 197, 768)
        return act.cpu().numpy().flatten().astype(np.float32)

    def decoder(self, activation: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(activation).reshape(
            1, *self._act_shape
        ).to(self.dev)                              # (1, 197, 768)
        with torch.no_grad():
            x = self._dec_blocks(t)
            x = self._ln(x)
            logits = self._heads(x[:, 0])           # CLS token → (1, 1000)
        return logits.cpu().squeeze(0)

    @property
    def dims(self) -> ModelDimensions:
        d_c = self._d // self.num_packets
        return ModelDimensions(
            model_name   = "vit_b16",
            input_shape  = (3, 224, 224),
            activation_d = self._d,
            num_packets  = self.num_packets,
            chunk_size   = d_c,
            bytes_per_pkt= d_c * 4,
            split_desc   = f"after transformer block {_SPLIT_BLOCK - 1}  "
                           f"| activation shape {self._act_shape}",
        )

    def offload(self) -> None:
        for attr in ["_conv_proj", "_enc_blocks", "_dec_blocks",
                     "_ln", "_heads"]:
            if hasattr(self, attr):
                delattr(self, attr)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
