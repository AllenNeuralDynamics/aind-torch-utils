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
