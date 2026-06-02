"""
corpus/transform.py  –  apply sign-mask and/or global permutation to
an existing raw corpus.

4 combinations controlled by --bitmask and --randperm flags:
  bitmask=F  randperm=F  →  identity  (baseline, plain dot product)
  bitmask=T  randperm=F  →  sign masking only
  bitmask=F  randperm=T  →  global permutation only
  bitmask=T  randperm=T  →  full implicit positional encoding  (our method)

Saves:
  transform_bm{0|1}_rp{0|1}.npz   – PacketTransform parameters
  corpus_bm{0|1}_rp{0|1}.npy      – transformed corpus  (M, P, d_c)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.packet import PacketTransform, normalise_chunks


def apply_transform(
    model_name: str,
    num_packets: int,
    out_dir: str,
    use_bitmask: bool = True,
    use_randperm: bool = True,
    seed: int = 42,
    force_rebuild: bool = False,
) -> tuple[PacketTransform, np.ndarray]:
    """
    Load raw corpus, build PacketTransform, apply it, save results.

    Returns (transform, transformed_corpus) where
    transformed_corpus has shape (M, P, d_c).
    """
    model_dir  = Path(out_dir) / model_name
    raw_path   = model_dir / "corpus_raw.npy"

    # tag         = f"bm{int(use_bitmask)}_rp{int(use_randperm)}"
    tag         = f"P{num_packets}_s{seed}_bm{int(use_bitmask)}_rp{int(use_randperm)}"
    tf_path     = model_dir / f"transform_{tag}.npz"
    corpus_path = model_dir / f"corpus_{tag}.npy"

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw corpus not found at {raw_path}.\n"
            f"Run corpus/build.py first."
        )

    corpus_raw = np.load(str(raw_path))          # (M, d)
    M, d = corpus_raw.shape

    # ── Load or create transform ───────────────
    if tf_path.exists() and not force_rebuild:
        # print(f"  [transform] Loading cached transform from {tf_path}")
        transform = PacketTransform.load(tf_path)

        if transform.P != num_packets:
        raise RuntimeError(
            f"Cached transform at {tf_path} has P={transform.P}, "
            f"but current run requested P={num_packets}. "
            f"Delete the cache or change --num-packets."
        )

    else:
        print(f"  [transform] Building transform  bitmask={use_bitmask}  randperm={use_randperm}")
        transform = PacketTransform(
            d=d,
            P=num_packets,
            seed=seed,
            use_bitmask=use_bitmask,
            use_randperm=use_randperm,
        )
        transform.save(tf_path)
        print(f"  [transform] Saved to {tf_path}")

    # ── Load or create transformed corpus ─────
    if corpus_path.exists() and not force_rebuild:
        print(f"  [transform] Loading cached transformed corpus from {corpus_path}")
        transformed = np.load(str(corpus_path))  # (M, P, d_c)
    else:
        print(f"  [transform] Applying transform to corpus  shape={corpus_raw.shape}")
        transformed = transform.encode_corpus(corpus_raw)   # (M, P, d_c)
        transformed = normalise_chunks(transformed)         # normalise each chunk
        np.save(str(corpus_path), transformed)
        print(f"  [transform] Saved transformed corpus  shape={transformed.shape}")
        print(f"              to {corpus_path}")

    _print_transform_summary(transform, M)
    return transform, transformed


def _print_transform_summary(t: PacketTransform, M: int) -> None:
    print(f"\n  Transform summary:")
    print(f"    d            = {t.d:,}")
    print(f"    P            = {t.P}")
    print(f"    d_c          = {t.d_c:,}")
    print(f"    use_bitmask  = {t.use_bitmask}")
    print(f"    use_randperm = {t.use_randperm}")
    print(f"    corpus M     = {M:,}")
    print(f"    FAISS index  = {M * t.P:,} vectors  of dim {t.d_c}")
