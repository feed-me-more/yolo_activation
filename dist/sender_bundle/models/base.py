"""
models/base.py  –  abstract interface every split model must implement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ModelDimensions:
    """Printed as the dimension summary when a model is loaded."""
    model_name:   str
    input_shape:  tuple        # e.g. (3, 224, 224)
    activation_d: int          # total flat activation elements  d
    num_packets:  int          # P
    chunk_size:   int          # d_c = d // P
    bytes_per_pkt: int         # chunk_size * 4  (float32)
    split_desc:   str          # human-readable description of split point

    def print(self) -> None:
        print("\n" + "═" * 60)
        print(f"  Model              : {self.model_name}")
        print(f"  Input shape        : {self.input_shape}")
        print(f"  Split point        : {self.split_desc}")
        print(f"  Activation dim (d) : {self.activation_d:,}")
        print(f"  Num packets  (P)   : {self.num_packets}")
        print(f"  Chunk size  (d_c)  : {self.chunk_size:,}  elements")
        print(f"  Bytes / packet     : {self.bytes_per_pkt:,}  bytes  (float32)")
        print(f"  Total payload      : {self.activation_d * 4:,}  bytes")
        print("═" * 60 + "\n")


class SplitModel(ABC):
    """
    Every split model exposes:
      - encoder(x)  →  flat numpy activation  (d,)
      - decoder(a)  →  output  (logits tensor or bbox list)
      - dims        →  ModelDimensions
    """

    def __init__(self, num_packets: int, device: str):
        self.num_packets = num_packets
        self.device = device

    @abstractmethod
    def encoder(self, x: torch.Tensor) -> np.ndarray:
        """Run head inference.  x : (1, C, H, W).  Returns (d,) float32."""
        ...

    @abstractmethod
    def decoder(self, activation: np.ndarray) -> Any:
        """Run tail inference.  activation : (d,) float32.  Returns model output."""
        ...

    @property
    @abstractmethod
    def dims(self) -> ModelDimensions:
        ...

    def offload(self) -> None:
        """Free GPU memory if applicable."""
        pass
