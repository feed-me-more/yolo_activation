"""
alignment/faiss_index.py  –  corpus retrieval for Stage 1.

For CPU runs this uses an exact NumPy matrix multiply.
For CUDA runs this keeps the transformed corpus resident on the GPU and
does exact retrieval with torch matmul there, avoiding per-frame
reconstruction / CPU fallback work.
"""
from __future__ import annotations

import time
from typing import Tuple

import numpy as np


class FaissCorpusIndex:
    """
    Exact chunk-level retrieval over a transformed corpus of shape (M, P, d_c).

    On CUDA:
    - the flattened corpus is normalized once
    - moved to GPU once
    - kept resident there for all future retrieve() calls

    On CPU:
    - retrieval is exact via NumPy matmul
    """

    def __init__(
        self,
        transformed_corpus: np.ndarray,   # (M, P, d_c)
        use_gpu: bool = True,
    ):
        self.M, self.P, self.d_c = transformed_corpus.shape
        self._corpus_entries_np = np.asarray(transformed_corpus, dtype=np.float32)
        self._corpus_flat_np = self._corpus_entries_np.reshape(self.M * self.P, self.d_c).copy()
        self._corpus_flat_np = _normalize_np(self._corpus_flat_np)
        self._use_torch_gpu = False
        self._device = "cpu"

        if use_gpu:
            try:
                import torch
                import torch.nn.functional as F

                if torch.cuda.is_available():
                    self._device = "cuda"
                    self._torch = torch
                    self._F = F
                    corpus_gpu = torch.from_numpy(self._corpus_flat_np).to("cuda", dtype=torch.float16)
                    corpus_gpu = F.normalize(corpus_gpu, dim=1)
                    self._corpus_flat_gpu = corpus_gpu
                    self._corpus_entries_gpu = corpus_gpu.view(self.M, self.P, self.d_c)
                    self._use_torch_gpu = True
                    bytes_fp16 = self.M * self.P * self.d_c * 2
                    print(
                        f"  [align] Exact matcher on CUDA  "
                        f"({self.M * self.P:,} vectors  dim={self.d_c}  "
                        f"resident≈{bytes_fp16 / (1024**2):.1f} MiB fp16)"
                    )
                    return
            except Exception as e:
                print(f"  [align] CUDA matcher unavailable, falling back to CPU exact search: {e}")

        print(f"  [align] Exact matcher on CPU  ({self.M * self.P:,} vectors  dim={self.d_c})")

    def retrieve(
        self,
        received: np.ndarray,             # (n_recv, d_c)
        top_k: int | None = None,
    ) -> Tuple[int, np.ndarray, float]:
        """
        Stage 1 retrieval.

        Returns:
        - m_star: best corpus entry index
        - benefit: (n_recv, P) matrix for Stage 2
        - elapsed_ms
        """
        if self._use_torch_gpu:
            return self._retrieve_cuda(received)
        return self._retrieve_cpu(received)

    def _retrieve_cuda(self, received: np.ndarray) -> Tuple[int, np.ndarray, float]:
        torch = self._torch
        F = self._F

        if received.shape[0] == 0:
            return 0, np.zeros((0, self.P), dtype=np.float32), 0.0

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        q = torch.from_numpy(np.asarray(received, dtype=np.float32)).to("cuda", dtype=self._corpus_flat_gpu.dtype)
        q = F.normalize(q, dim=1)

        # Exact similarities: (n_recv, M*P)
        sim = q @ self._corpus_flat_gpu.T
        sim_3d = sim.view(q.shape[0], self.M, self.P)
        agg_scores = sim_3d.amax(dim=2).sum(dim=0)
        m_star = int(torch.argmax(agg_scores).item())
        benefit = sim_3d[:, m_star, :].float().cpu().numpy()

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        return m_star, benefit, elapsed_ms

    def _retrieve_cpu(self, received: np.ndarray) -> Tuple[int, np.ndarray, float]:
        if received.shape[0] == 0:
            return 0, np.zeros((0, self.P), dtype=np.float32), 0.0

        t0 = time.perf_counter()
        q = _normalize_np(np.asarray(received, dtype=np.float32))
        sim = q @ self._corpus_flat_np.T
        sim_3d = sim.reshape(q.shape[0], self.M, self.P)
        agg_scores = sim_3d.max(axis=2).sum(axis=0)
        m_star = int(np.argmax(agg_scores))
        benefit = sim_3d[:, m_star, :].astype(np.float32, copy=False)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        return m_star, benefit, elapsed_ms

    def get_corpus_entry(self, m: int) -> np.ndarray:
        return np.asarray(self._corpus_entries_np[m], dtype=np.float32).copy()


def build_index(
    transformed_corpus: np.ndarray,
    device: str = "auto",
) -> FaissCorpusIndex:
    use_gpu = False
    if device == "cuda":
        use_gpu = True
    elif device == "auto":
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except Exception:
            use_gpu = False
    return FaissCorpusIndex(transformed_corpus, use_gpu=use_gpu)


def _normalize_np(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms
