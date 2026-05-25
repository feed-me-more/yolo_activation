"""
utils/packet.py  –  everything to do with splitting activations into
packets, applying the sign-mask / global-permutation encoding, simulating
packet loss, and reassembling.

All functions are pure numpy / torch; no model code here.
"""
from __future__ import annotations

import numpy as np
import torch
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────
# Transform state  (sign vectors + permutation)
# ─────────────────────────────────────────────

class PacketTransform:
    """
    Holds the shared secret between sender and receiver:
      - perm   : global permutation of the flat activation   (d,)
      - signs  : per-chunk sign vectors                      (P, d_c)

    Either component can be disabled (use_randperm / use_bitmask).
    """

    def __init__(
        self,
        d: int,
        P: int,
        seed: int = 42,
        use_bitmask: bool = True,
        use_randperm: bool = True,
    ):
        self.d = d
        self.P = P
        self.d_c = int(np.ceil(d / P))
        self.d_pad = self.P * self.d_c
        self.use_bitmask = use_bitmask
        self.use_randperm = use_randperm

        rng = np.random.default_rng(seed)

        # Global permutation  (applied before chunking)
        if use_randperm:
            self.perm: np.ndarray = rng.permutation(self.d_pad).astype(np.int64)
            self.inv_perm: np.ndarray = np.argsort(self.perm).astype(np.int64)
        else:
            self.perm = np.arange(self.d_pad, dtype=np.int64)
            self.inv_perm = np.arange(self.d_pad, dtype=np.int64)

        # Sign vectors  shape (P, d_c)
        if use_bitmask:
            raw = rng.integers(0, 2, size=(P, self.d_c), dtype=np.int8)
            self.signs: np.ndarray = (2 * raw - 1).astype(np.float32)  # ±1
        else:
            self.signs = np.ones((P, self.d_c), dtype=np.float32)

    # ── apply to a single flat activation ─────

    def encode(self, a: np.ndarray) -> np.ndarray:
        """
        a : (d,)  flat activation (already L2-normalised per chunk by caller)
        returns : (P, d_c)  masked, chunked activation ready for transmission
        """
        if a.shape[0] != self.d:
            raise ValueError(f"Expected activation of length {self.d}, got {a.shape[0]}")
        if self.d_pad > self.d:
            a = np.pad(a, (0, self.d_pad - self.d))
        a_perm = a[self.perm]
        chunks = a_perm.reshape(self.P, self.d_c)
        return chunks * self.signs                     # elementwise sign mask

    def decode_chunk(self, chunk: np.ndarray, j: int) -> np.ndarray:
        """Undo sign mask for chunk at position j.  chunk : (d_c,)"""
        return chunk * self.signs[j]

    # ── apply to the whole corpus ──────────────

    def encode_corpus(self, corpus: np.ndarray) -> np.ndarray:
        """
        corpus : (M, d)
        returns: (M, P, d_c)
        """
        M = corpus.shape[0]
        if corpus.shape[1] != self.d:
            raise ValueError(f"Expected corpus with dim {self.d}, got {corpus.shape[1]}")
        if self.d_pad > self.d:
            pad = np.zeros((M, self.d_pad - self.d), dtype=corpus.dtype)
            corpus = np.concatenate([corpus, pad], axis=1)
        c_perm = corpus[:, self.perm]
        chunks = c_perm.reshape(M, self.P, self.d_c)
        return chunks * self.signs[None, :, :]         # broadcast sign mask

    def save(self, path) -> None:
        import pathlib
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(path),
            perm=self.perm,
            inv_perm=self.inv_perm,
            signs=self.signs,
            d=np.array(self.d),
            d_pad=np.array(self.d_pad),
            P=np.array(self.P),
            use_bitmask=np.array(self.use_bitmask),
            use_randperm=np.array(self.use_randperm),
        )

    @classmethod
    def load(cls, path) -> "PacketTransform":
        data = np.load(str(path))
        obj = cls.__new__(cls)
        obj.perm       = data["perm"]
        obj.inv_perm   = data["inv_perm"]
        obj.signs      = data["signs"]
        obj.d          = int(data["d"])
        obj.d_pad      = int(data["d_pad"]) if "d_pad" in data else int(data["d"])
        obj.P          = int(data["P"])
        obj.d_c        = int(np.ceil(obj.d_pad / obj.P))
        obj.use_bitmask  = bool(data["use_bitmask"])
        obj.use_randperm = bool(data["use_randperm"])
        return obj


# ─────────────────────────────────────────────
# Chunk normalisation
# ─────────────────────────────────────────────

def normalise_chunks(chunks: np.ndarray) -> np.ndarray:
    """
    chunks : (P, d_c)  or  (M, P, d_c)
    L2-normalise each chunk independently.
    """
    norms = np.linalg.norm(chunks, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return chunks / norms


# ─────────────────────────────────────────────
# Packet loss simulation
# ─────────────────────────────────────────────

def simulate_loss(
    chunks: np.ndarray,         # (P, d_c)
    loss_rate: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate iid packet loss AND random reordering.

    Returns
    -------
    received  : (P-k, d_c)  the surviving chunks in RANDOM arrival order
    true_pos  : (P-k,)      the true position index of each received chunk
    """
    P = chunks.shape[0]
    mask = rng.random(P) >= loss_rate          # True = survived
    surviving_idx = np.where(mask)[0]

    # random reordering of surviving packets
    shuffle = rng.permutation(len(surviving_idx))
    surviving_idx = surviving_idx[shuffle]

    received = chunks[surviving_idx]           # (P-k, d_c)
    return received, surviving_idx             # true positions in arrival order


# ─────────────────────────────────────────────
# Activation → chunks pipeline (end-to-end)
# ─────────────────────────────────────────────

def activation_to_packets(
    activation: np.ndarray,     # (d,)  raw flat activation
    transform: PacketTransform,
) -> np.ndarray:
    """Full encode pipeline: permute → chunk → normalise → sign-mask."""
    encoded = transform.encode(activation)     # (P, d_c)
    return normalise_chunks(encoded)


def reconstruct_activation(
    received: np.ndarray,       # (P-k, d_c) received (masked) chunks
    true_pos: np.ndarray,       # (P-k,)  assigned positions
    corpus_entry: np.ndarray,   # (P, d_c)  nearest corpus entry (already in raw space)
    transform: PacketTransform,
    lost_fill: str = "corpus",  # "zeros" | "corpus"
) -> np.ndarray:
    """
    Reconstruct full activation (P, d_c) in raw (unmasked) space.

    received chunks are in sign-masked space; we unmask them.
    Missing positions are filled from corpus_entry (already unmasked) or zeros.
    """
    P, d_c = transform.P, transform.d_c
    full = np.zeros((P, d_c), dtype=np.float32)

    # Fill from corpus first
    if lost_fill == "corpus":
        full[:] = corpus_entry                 # (P, d_c)  raw corpus activation

    # Overwrite received positions with decoded received chunks
    for i, j in enumerate(true_pos):
        full[j] = transform.decode_chunk(received[i], j)

    return full                                # (P, d_c)
