"""Tests for the pure-numpy connected-components stitching helpers."""
import numpy as np

from aind_torch_utils.labeling import UnionFind, block_ranges, union_faces


def test_block_ranges():
    assert block_ranges(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert block_ranges(8, 4) == [(0, 4), (4, 8)]
    assert block_ranges(3, 8) == [(0, 3)]


def test_union_find_basic_and_flatten():
    uf = UnionFind(6)
    uf.union(1, 2)
    uf.union(2, 3)
    uf.union(4, 5)
    assert uf.find(1) == uf.find(3)
    assert uf.find(4) == uf.find(5)
    assert uf.find(1) != uf.find(4)
    roots = uf.flatten_roots()
    # All members of {1,2,3} share one root; {4,5} share another.
    assert roots[1] == roots[2] == roots[3]
    assert roots[4] == roots[5]
    assert roots[1] != roots[4]
    assert roots[0] == 0  # untouched


def test_union_faces_26_connectivity_diagonal():
    # A's foreground voxel at (0,0); B's only foreground at (1,1) -> diagonal across the
    # seam, which is 26-connected, so the two labels must be unioned.
    g_a = np.array([[10, 0], [0, 0]], dtype=np.int64)
    g_b = np.array([[0, 0], [0, 20]], dtype=np.int64)
    uf = UnionFind(30)
    union_faces(uf, g_a, g_b)
    assert uf.find(10) == uf.find(20)


def test_union_faces_no_contact_stays_separate():
    # Foreground voxels are >1 apart in-plane -> not 26-connected across the seam.
    g_a = np.zeros((3, 3), dtype=np.int64)
    g_a[0, 0] = 7
    g_b = np.zeros((3, 3), dtype=np.int64)
    g_b[2, 2] = 8
    uf = UnionFind(10)
    union_faces(uf, g_a, g_b)
    assert uf.find(7) != uf.find(8)


def test_union_faces_6_connectivity():
    # 6-conn (shifts=[(0,0)]): only the aligned voxel unions across the seam.
    # Aligned foreground -> union; diagonal-only foreground -> stays separate.
    g_a = np.zeros((2, 2), dtype=np.int64)
    g_a[0, 0] = 10
    g_b = np.zeros((2, 2), dtype=np.int64)
    g_b[0, 0] = 20  # aligned with A's (0,0)
    uf = UnionFind(30)
    union_faces(uf, g_a, g_b, shifts=[(0, 0)])
    assert uf.find(10) == uf.find(20)

    # Diagonal-only contact must NOT union under 6-connectivity.
    g_a2 = np.zeros((2, 2), dtype=np.int64)
    g_a2[0, 0] = 11
    g_b2 = np.zeros((2, 2), dtype=np.int64)
    g_b2[1, 1] = 21
    uf2 = UnionFind(30)
    union_faces(uf2, g_a2, g_b2, shifts=[(0, 0)])
    assert uf2.find(11) != uf2.find(21)


def test_fill_decision_border_and_size():
    # Mirrors _fill_holes_connect's keep computation: a background component is filled
    # iff its root never touches the volume border AND (no cap or size <= cap).
    # Labels: {1,2} merged enclosed hole (size 30), 3 enclosed hole (size 200),
    # 4 border-connected background (must never fill).
    uf = UnionFind(5)
    uf.union(1, 2)
    border = np.array([False, False, False, False, True])  # label 4 touches border
    sizes = np.array([0, 20, 10, 200, 999], dtype=np.int64)
    roots = uf.flatten_roots()

    root_border = np.zeros(5, dtype=bool)
    np.logical_or.at(root_border, roots, border)
    root_size = np.zeros(5, dtype=np.int64)
    np.add.at(root_size, roots, sizes)

    # No cap: every non-border component fills.
    fill = (~root_border)[roots]
    fill[0] = False
    assert fill[1] and fill[2] and fill[3]
    assert not fill[4]  # border-connected stays background

    # With a size cap of 100: the size-30 hole fills, the size-200 hole stays open.
    fillable = (~root_border) & (root_size <= 100)
    fill_capped = fillable[roots]
    fill_capped[0] = False
    assert fill_capped[1] and fill_capped[2]  # merged hole size 30 <= 100
    assert not fill_capped[3]  # size 200 > 100 left open
    assert not fill_capped[4]


def test_seed_propagation_to_roots():
    # Mirrors _hysteresis_connect's keep computation: a component is kept iff any of
    # its labels is seeded. {1,2,3} merged, only 3 seeded -> all kept; {4} not seeded.
    uf = UnionFind(5)
    uf.union(1, 2)
    uf.union(2, 3)
    seeded = np.array([False, False, False, True, False])  # label 3 seeded
    roots = uf.flatten_roots()
    root_seeded = np.zeros(5, dtype=bool)
    np.logical_or.at(root_seeded, roots, seeded)
    keep = root_seeded[roots]
    keep[0] = False
    assert keep[1] and keep[2] and keep[3]  # whole merged component kept
    assert not keep[4]  # unseeded component dropped
