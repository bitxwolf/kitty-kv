"""
kitty/layout.py
---------------
Phase 2 — Page-Centric KV Cache Layout Manager.

Manages the pre-allocated HBM memory pools for the quantised Key and Value
cache pages.  The three-tensor Dense-Sparse decomposition is used for the Key
Cache:

    Tensor_2bits       — (max_pages, num_kv_heads, D,       G//4)  uint8
                         Low 2 bits for ALL channels (INT2 for non-boosted;
                         lower 2 bits of INT4 for boosted channels).

    Tensor_High_2bits  — (max_pages, num_kv_heads, D_boost, G//4)  uint8
                         High 2 bits ONLY for the D_boost boosted channels.

    key_metadata       — (max_pages, num_kv_heads, 2, D)    float16
                         Row 0 = per-channel scale S_c
                         Row 1 = per-channel zero-point Z_c

For the Value Cache a uniform 2-bit per-token scheme is used:

    value_pool         — (max_pages, num_kv_heads, G//4, D//4)  uint8
                         Packed INT2 values across the head dimension D.

    value_metadata     — (max_pages, num_kv_heads, G, 2)   float16
                         Per-token scale and zero-point.

A simple free-list allocator tracks which physical pages are in use.

Reference: §3 (Dense-Sparse Layout) & §4.1 of the reference spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .config import KittyConfig


# ---------------------------------------------------------------------------
# Page descriptor
# ---------------------------------------------------------------------------

@dataclass
class Page:
    """A single quantised KV cache page occupying G tokens."""

    physical_idx: int         # index into the global pool tensors
    layer_id: int             # which transformer layer this belongs to
    head_id: int              # which KV head this belongs to (or -1 = all heads)
    seq_id: int               # sequence identifier
    start_token: int          # logical first token stored in this page
    num_tokens: int = 0       # how many tokens have been written (≤ G)

    @property
    def is_full(self) -> bool:
        return self.num_tokens == self.config_G  # set externally after alloc

    def __repr__(self) -> str:
        return (
            f"Page(phys={self.physical_idx}, layer={self.layer_id}, "
            f"head={self.head_id}, seq={self.seq_id}, "
            f"tokens=[{self.start_token}..{self.start_token + self.num_tokens}))"
        )


# ---------------------------------------------------------------------------
# Layout manager
# ---------------------------------------------------------------------------

class PageCentricKVLayoutManager:
    """Pre-allocated HBM pools + free-list allocator for Kitty KV cache pages.

    All tensor pools are allocated once at construction time so that runtime
    quantisation and dequantisation never allocate memory, eliminating CUDA
    memory management overhead during inference.

    Parameters
    ----------
    config:
        Global Kitty configuration.
    boost_idx_uint8:
        Tensor of shape ``(num_layers, num_kv_heads, head_dim)`` dtype uint8.
        Produced by :class:`~kitty.sensitivity.KittySensitivityProfiler`.

    Pools
    -----
    The physical layout per pool::

        Key Baseline (low-bits):
            shape = (max_pages, num_kv_heads, D, G//4)   dtype=uint8
        Key Boost (high-bits):
            shape = (max_pages, num_kv_heads, D_boost, G//4)  dtype=uint8
        Key Metadata (scale & ZP):
            shape = (max_pages, num_kv_heads, 2, D)      dtype=float16
        Value Pool (packed INT2):
            shape = (max_pages, num_kv_heads, G//4, D//4) dtype=uint8
            Note: values are packed 4-per-byte along D, then G is packed 4-per-byte.
            Actual layout: (max_pages, num_kv_heads, G, D//4) for cleaner indexing.
        Value Metadata (scale & ZP):
            shape = (max_pages, num_kv_heads, G, 2)      dtype=float16
    """

    def __init__(
        self,
        config: KittyConfig,
        boost_idx_uint8: torch.Tensor,
    ) -> None:
        self.config = config
        self.boost_idx_uint8 = boost_idx_uint8.to(config.device)  # (L, H, D) uint8

        G = config.q_buffer_size
        D = config.head_dim
        K = config.d_boost
        H = config.num_kv_heads
        P = config.max_pages
        dev = torch.device(config.device)

        if config.dtype == "float16":
            self._torch_dtype = torch.float16
        else:
            self._torch_dtype = torch.bfloat16

        # ---------------------------------------------------------------- Key pools
        self.key_baseline: torch.Tensor = torch.zeros(
            P, H, D, G // 4, dtype=torch.uint8, device=dev
        )
        self.key_boost: torch.Tensor = torch.zeros(
            P, H, K, G // 4, dtype=torch.uint8, device=dev
        )
        self.key_metadata: torch.Tensor = torch.zeros(
            P, H, 2, D, dtype=self._torch_dtype, device=dev
        )

        # -------------------------------------------------------------- Value pools
        # Store G tokens per page.  Each token has D values.
        # Pack 4 INT2 values per byte along the D dimension.
        self.value_pool: torch.Tensor = torch.zeros(
            P, H, G, D // 4, dtype=torch.uint8, device=dev
        )
        self.value_metadata: torch.Tensor = torch.zeros(
            P, H, G, 2, dtype=self._torch_dtype, device=dev
        )

        # ---------------------------------------------------------------- Free list
        # A boolean mask: True = free, False = occupied
        self._free_mask: torch.Tensor = torch.ones(P, dtype=torch.bool, device=dev)
        self._num_free: int = P

        # ---------------------------------------------------------------- Page table
        # Map (seq_id, layer_id, page_number) → physical_idx
        self._page_table: Dict[Tuple[int, int, int], int] = {}
        self._pages: Dict[int, Page] = {}  # physical_idx → Page

    # ------------------------------------------------------------------ allocation

    def allocate_page(self, seq_id: int, layer_id: int, page_number: int) -> Page:
        """Allocate one physical page from the free list.

        Parameters
        ----------
        seq_id, layer_id, page_number:
            Logical coordinates identifying this page in the cache.

        Returns
        -------
        Page
            The newly allocated page with its physical index.

        Raises
        ------
        RuntimeError
            If no free pages are available (pool exhausted).
        """
        if self._num_free == 0:
            raise RuntimeError(
                "KV Cache page pool exhausted! "
                f"All {self.config.max_pages} pages are in use. "
                "Increase config.max_pages or reduce max_seq_len/max_batch_size."
            )

        # Find the first free page
        free_indices = self._free_mask.nonzero(as_tuple=True)[0]
        phys_idx = int(free_indices[0].item())

        self._free_mask[phys_idx] = False
        self._num_free -= 1

        page = Page(
            physical_idx=phys_idx,
            layer_id=layer_id,
            head_id=-1,  # shared across all heads for this layer
            seq_id=seq_id,
            start_token=page_number * self.config.q_buffer_size,
        )
        key = (seq_id, layer_id, page_number)
        self._page_table[key] = phys_idx
        self._pages[phys_idx] = page
        return page

    def free_page(self, seq_id: int, layer_id: int, page_number: int) -> None:
        """Return a page to the free list."""
        key = (seq_id, layer_id, page_number)
        phys_idx = self._page_table.pop(key, None)
        if phys_idx is None:
            return
        self._pages.pop(phys_idx, None)
        self._free_mask[phys_idx] = True
        self._num_free += 1

    def free_sequence(self, seq_id: int) -> None:
        """Release all pages belonging to a given sequence."""
        keys_to_free = [k for k in self._page_table if k[0] == seq_id]
        for key in keys_to_free:
            seq, layer, page_num = key
            self.free_page(seq, layer, page_num)

    def get_physical_idx(self, seq_id: int, layer_id: int, page_number: int) -> int:
        """Look up the physical page index for a logical (seq, layer, page) triple."""
        key = (seq_id, layer_id, page_number)
        if key not in self._page_table:
            raise KeyError(f"Page {key} not found in page table.")
        return self._page_table[key]

    # ---------------------------------------------------------------- key accessors

    def key_baseline_page(self, phys_idx: int) -> torch.Tensor:
        """Return a view of the key baseline tensor for one page.

        Shape: ``(num_kv_heads, head_dim, G//4)`` uint8.
        """
        return self.key_baseline[phys_idx]  # (H, D, G//4)

    def key_boost_page(self, phys_idx: int) -> torch.Tensor:
        """Return a view of the key boost tensor for one page.

        Shape: ``(num_kv_heads, d_boost, G//4)`` uint8.
        """
        return self.key_boost[phys_idx]  # (H, K, G//4)

    def key_meta_page(self, phys_idx: int) -> torch.Tensor:
        """Return a view of the key metadata for one page.

        Shape: ``(num_kv_heads, 2, head_dim)`` float16.
        Row 0 = scale, Row 1 = zero-point.
        """
        return self.key_metadata[phys_idx]  # (H, 2, D)

    # -------------------------------------------------------------- value accessors

    def value_page(self, phys_idx: int) -> torch.Tensor:
        """Return a view of the packed value tensor for one page.

        Shape: ``(num_kv_heads, G, D//4)`` uint8.
        """
        return self.value_pool[phys_idx]  # (H, G, D//4)

    def value_meta_page(self, phys_idx: int) -> torch.Tensor:
        """Return a view of the value metadata for one page.

        Shape: ``(num_kv_heads, G, 2)`` float16.
        Column 0 = per-token scale, Column 1 = per-token zero-point.
        """
        return self.value_metadata[phys_idx]  # (H, G, 2)

    # ---------------------------------------------------------------- page lists

    def get_page_list(self, seq_id: int, layer_id: int) -> List[int]:
        """Return ordered list of physical page indices for a (seq, layer) pair."""
        pages = []
        page_num = 0
        while (seq_id, layer_id, page_num) in self._page_table:
            pages.append(self._page_table[(seq_id, layer_id, page_num)])
            page_num += 1
        return pages

    # ------------------------------------------------------------------ stats

    def memory_stats(self) -> Dict[str, float]:
        """Return approximate memory usage in MB for each pool."""
        def mb(t: torch.Tensor) -> float:
            return t.numel() * t.element_size() / (1024 ** 2)

        return {
            "key_baseline_MB": mb(self.key_baseline),
            "key_boost_MB": mb(self.key_boost),
            "key_metadata_MB": mb(self.key_metadata),
            "value_pool_MB": mb(self.value_pool),
            "value_metadata_MB": mb(self.value_metadata),
            "total_MB": sum([
                mb(self.key_baseline), mb(self.key_boost),
                mb(self.key_metadata), mb(self.value_pool),
                mb(self.value_metadata),
            ]),
            "pages_free": self._num_free,
            "pages_used": self.config.max_pages - self._num_free,
            "pages_total": self.config.max_pages,
        }

    def __repr__(self) -> str:
        stats = self.memory_stats()
        return (
            f"PageCentricKVLayoutManager(\n"
            f"  key_baseline:  {self.key_baseline.shape}  uint8\n"
            f"  key_boost:     {self.key_boost.shape}  uint8\n"
            f"  key_metadata:  {self.key_metadata.shape}  {self.key_metadata.dtype}\n"
            f"  value_pool:    {self.value_pool.shape}  uint8\n"
            f"  value_metadata:{self.value_metadata.shape}  {self.value_metadata.dtype}\n"
            f"  Pages: {stats['pages_used']}/{stats['pages_total']} used  "
            f"({stats['total_MB']:.1f} MB total)\n"
            f")"
        )
