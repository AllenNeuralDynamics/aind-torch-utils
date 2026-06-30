"""Pure-numpy connected-components stitching helpers (no GPU/IO deps).

Used by the chunked hysteresis pass: blocks are labeled independently (on the GPU) and
their per-block labels are made globally unique via offsets; these helpers union labels
that touch across block faces (26-connectivity) and resolve the equivalence classes.
Kept dependency-light so the seam-stitching logic can be unit-tested without a GPU.
"""
from typing import List, Optional, Tuple

import numpy as np


def block_ranges(extent: int, block: int) -> List[Tuple[int, int]]:
    """Return ``(start, stop)`` tiles covering ``[0, extent)`` with step ``block``."""
    return [(s, min(s + block, extent)) for s in range(0, extent, block)]


class UnionFind:
    """Array-backed union-find with vectorized full-compression (numpy only)."""

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        p = self.parent
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:  # path compression
            p[x], x = root, p[x]
        return int(root)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def flatten_roots(self) -> np.ndarray:
        """Return ``root[i]`` for all ``i`` via vectorized pointer-jumping."""
        p = self.parent
        while True:
            gp = p[p]
            if np.array_equal(gp, p):
                break
            p[:] = gp
        return p


def union_faces(
    uf: UnionFind,
    g_a: np.ndarray,
    g_b: np.ndarray,
    shifts: Optional[List[Tuple[int, int]]] = None,
) -> None:
    """Union global labels across a shared block face.

    ``g_a`` and ``g_b`` are the two same-shape, globally-labeled face planes one voxel
    apart along the seam axis (0 = background). ``shifts`` is the list of in-plane
    ``(dy, dx)`` offsets defining connectivity across the seam: the default 3x3
    neighbourhood gives full 26-connectivity (use for foreground), while ``[(0, 0)]``
    gives 6-connectivity (only the aligned voxel; use for the background flood in
    hole-filling, complementary to 26-connected foreground).
    """
    if shifts is None:
        shifts = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    p, q = g_a.shape
    for dy, dx in shifts:
        ay0, ay1 = max(0, -dy), p - max(0, dy)
        ax0, ax1 = max(0, -dx), q - max(0, dx)
        by0, by1 = max(0, dy), p - max(0, -dy)
        bx0, bx1 = max(0, dx), q - max(0, -dx)
        a = g_a[ay0:ay1, ax0:ax1]
        b = g_b[by0:by1, bx0:bx1]
        both = (a > 0) & (b > 0)
        for u, v in zip(a[both].tolist(), b[both].tolist()):
            uf.union(u, v)
