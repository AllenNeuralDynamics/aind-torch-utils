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
        # int32 labels halve the parent array (the dominant cost when there are many
        # components) whenever the label space fits; fall back to int64 past 2^31.
        dtype = np.int32 if n < 2**31 else np.int64
        self.parent = np.arange(n, dtype=dtype)
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


def rescale_points(points: np.ndarray, factor) -> np.ndarray:
    """Center-aligned rescale of ``(N, 3)`` integer voxel coords by per-axis ``factor``.

    Maps coordinates from one pyramid level to another with
    ``out = floor((p + 0.5) * factor - 0.5)`` (the center-alignment convention used by
    the flat-field sampler), where
    ``factor[a] = voxel_size(src_level) / voxel_size(dst_level)`` per axis. Returns
    int64 voxel coords in the destination level.
    """
    f = np.asarray(factor, dtype=np.float64).reshape(1, 3)
    return np.floor((points.astype(np.float64) + 0.5) * f - 0.5).astype(np.int64)


def bucket_points(coords: np.ndarray, ids: np.ndarray, block: int, shape) -> dict:
    """Group voxel coords by block cell -> ``{(cz, cy, cx): (coords, ids)}``.

    Coords outside ``shape`` (Z, Y, X) are dropped. Cell index is ``coords // block``.
    Lets :func:`region_seeds` fetch the seeds in a block(+halo) region without scanning
    the full (billion-row) array.
    """
    coords = np.asarray(coords)
    ids = np.asarray(ids)
    nz, ny, nx = shape
    inb = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < nz)
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < ny)
        & (coords[:, 2] >= 0)
        & (coords[:, 2] < nx)
    )
    coords, ids = coords[inb], ids[inb]
    if coords.shape[0] == 0:
        return {}
    cells = coords // block
    order = np.lexsort((cells[:, 2], cells[:, 1], cells[:, 0]))
    coords, ids, cells = coords[order], ids[order], cells[order]
    change = np.any(cells[1:] != cells[:-1], axis=1)
    starts = np.concatenate(([0], np.nonzero(change)[0] + 1))
    stops = np.concatenate((starts[1:], [len(cells)]))
    out = {}
    for s, e in zip(starts.tolist(), stops.tolist()):
        out[(int(cells[s, 0]), int(cells[s, 1]), int(cells[s, 2]))] = (
            coords[s:e],
            ids[s:e],
        )
    return out


def region_seeds(cell_points: dict, block: int, bbox):
    """Return ``(coords, ids)`` of seeds inside ``bbox`` = ``(z0, z1, y0, y1, x0, x1)``.

    Gathers the block cells overlapping ``bbox`` from a :func:`bucket_points` dict, then
    filters to the exact half-open box.
    """
    z0, z1, y0, y1, x0, x1 = bbox
    cc, ii = [], []
    for cz in range(z0 // block, (z1 - 1) // block + 1):
        for cy in range(y0 // block, (y1 - 1) // block + 1):
            for cx in range(x0 // block, (x1 - 1) // block + 1):
                cell = cell_points.get((cz, cy, cx))
                if cell is not None:
                    cc.append(cell[0])
                    ii.append(cell[1])
    if not cc:
        return np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.uint32)
    coords = np.concatenate(cc)
    ids = np.concatenate(ii)
    m = (
        (coords[:, 0] >= z0)
        & (coords[:, 0] < z1)
        & (coords[:, 1] >= y0)
        & (coords[:, 1] < y1)
        & (coords[:, 2] >= x0)
        & (coords[:, 2] < x1)
    )
    return coords[m], ids[m]


def merge_seed_groups(seed_core: np.ndarray, global_ids: np.ndarray):
    """Group seeds that share an intensity core -> compact object groups.

    ``seed_core[i]`` is the connected-core label at seed ``i`` (0 = the seed is not in a
    core, e.g. dimmer than the merge threshold). ``global_ids[i]`` is seed ``i``'s
    global instance id (its point index + 1). Seeds sharing the same core (>0) merge
    into one object; seeds with core 0 stay separate (each its own group). Returns:

    - ``local_to_group`` ``(k + 1,)`` int64: maps a local seed label (1..k, 0 = bg) to
      a compact group index (1..G, 0 stays 0).
    - ``group_to_global`` ``(G + 1,)`` uint32: maps a group to its representative global
      id (the **smallest** global id in the group; ``[0] = 0``).
    """
    k = int(global_ids.shape[0])
    if k == 0:
        return (
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.uint32),
        )
    keys = np.asarray(seed_core, dtype=np.int64).copy()
    away = keys == 0  # seeds not in any core -> unique negative key each (never merge)
    if away.any():
        keys[away] = -np.arange(1, int(away.sum()) + 1, dtype=np.int64)
    uniq, inv = np.unique(keys, return_inverse=True)  # inv in [0, G)
    local_to_group = np.zeros(k + 1, dtype=np.int64)
    local_to_group[1:] = inv + 1  # groups 1..G
    gids = np.asarray(global_ids, dtype=np.uint32)
    repr_u = np.full(uniq.size, np.iinfo(np.uint32).max, dtype=np.uint32)
    np.minimum.at(repr_u, inv, gids)
    group_to_global = np.zeros(uniq.size + 1, dtype=np.uint32)
    group_to_global[1:] = repr_u
    return local_to_group, group_to_global
