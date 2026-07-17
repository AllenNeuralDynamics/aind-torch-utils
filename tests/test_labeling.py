"""Tests for the pure-numpy connected-components stitching helpers."""

import numpy as np

from aind_torch_utils.labeling import (
    UnionFind,
    adjacency_merge,
    block_ranges,
    bucket_points,
    flood_seed_union,
    merge_seed_groups,
    region_seeds,
    rescale_points,
    union_faces,
)


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


def test_union_find_uses_int32_parent_for_memory():
    # The parent array (the dominant cost with many components) is int32 while the label
    # space fits in 2^31, halving RAM vs int64; find/union still resolve correctly.
    uf = UnionFind(1000)
    assert uf.parent.dtype == np.int32
    uf.union(10, 20)
    uf.union(20, 30)
    assert uf.find(10) == uf.find(30)
    roots = uf.flatten_roots()
    assert roots.dtype == np.int32
    assert roots[10] == roots[20] == roots[30]


def test_union_faces_int32_faces_stitch_across_seam():
    # Faces are stored as int32 (globalized to int64 before union_faces in the
    # pipeline); union_faces must handle int32 inputs identically.
    g_a = np.array([[10, 0], [0, 0]], dtype=np.int32)
    g_b = np.array([[0, 0], [0, 20]], dtype=np.int32)
    uf = UnionFind(30)
    union_faces(uf, g_a, g_b)  # 26-conn: diagonal across the seam unions
    assert uf.find(10) == uf.find(20)


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


def test_instance_ids_merge_across_seam_and_are_contiguous():
    # Mirrors _instance_connect's ID assignment: seam-merged components share one ID,
    # distinct components get distinct IDs, and IDs are a contiguous 1..N.
    uf = UnionFind(6)  # global labels 1..5
    uf.union(1, 2)  # object A spans blocks -> {1,2,3}
    uf.union(2, 3)
    # labels 4 and 5 are separate objects
    roots = uf.flatten_roots()
    keep = np.ones(6, dtype=bool)
    keep[0] = False  # background
    uniq = np.unique(roots[keep])
    id_of_root = np.zeros(6, dtype=np.uint32)
    id_of_root[uniq] = np.arange(1, uniq.size + 1, dtype=np.uint32)
    label_to_id = id_of_root[roots]
    label_to_id[0] = 0

    assert label_to_id[1] == label_to_id[2] == label_to_id[3]  # merged -> one ID
    assert label_to_id[4] != label_to_id[1]
    assert label_to_id[5] not in (label_to_id[1], label_to_id[4])
    assert set(label_to_id[1:].tolist()) == {1, 2, 3}  # 3 objects, contiguous 1..3
    assert label_to_id[0] == 0
    assert label_to_id.dtype == np.uint32


def test_instance_min_size_drops_small_objects():
    # A = {1,2} (size 70), B = {3} (size 5); min_size 10 drops B (-> 0).
    uf = UnionFind(4)
    uf.union(1, 2)
    roots = uf.flatten_roots()
    sizes = np.array([0, 30, 40, 5], dtype=np.int64)
    root_size = np.zeros(4, dtype=np.int64)
    np.add.at(root_size, roots, sizes)
    keep = np.ones(4, dtype=bool)
    keep[0] = False
    keep &= root_size[roots] >= 10  # min_size
    uniq = np.unique(roots[keep])
    id_of_root = np.zeros(4, dtype=np.uint32)
    id_of_root[uniq] = np.arange(1, uniq.size + 1, dtype=np.uint32)
    label_to_id = id_of_root[roots]
    label_to_id[0] = 0

    assert label_to_id[1] == label_to_id[2] >= 1  # A kept
    assert label_to_id[3] == 0  # B dropped (size 5 < 10)
    assert uniq.size == 1


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


def test_rescale_points_center_aligned():
    # factor 2 (points at a 2x-coarser level): (p+0.5)*2-0.5, floored.
    pts = np.array([[1, 2, 3], [0, 0, 0]], dtype=np.int64)
    out = rescale_points(pts, [2.0, 2.0, 2.0])
    assert out.tolist() == [[2, 4, 6], [0, 0, 0]]
    assert out.dtype == np.int64
    # anisotropic factor per axis
    out2 = rescale_points(np.array([[1, 1, 1]]), [4.0, 2.0, 1.0])
    assert out2.tolist() == [[5, 2, 1]]  # (1.5*4-0.5,1.5*2-0.5,1.5*1-0.5)=(5.5,2.5,1.0)


def test_bucket_points_groups_by_cell_and_drops_out_of_bounds():
    coords = np.array([[0, 0, 0], [0, 0, 5], [10, 10, 10], [100, 0, 0]], dtype=np.int64)
    ids = np.array([1, 2, 3, 4], dtype=np.uint32)
    cells = bucket_points(coords, ids, block=8, shape=(20, 20, 20))
    assert set(cells) == {(0, 0, 0), (1, 1, 1)}  # (100,0,0) dropped (out of bounds)
    c00, i00 = cells[(0, 0, 0)]
    assert sorted(i00.tolist()) == [1, 2]  # both fall in cell (0,0,0)
    _, i11 = cells[(1, 1, 1)]
    assert i11.tolist() == [3]


def test_region_seeds_gathers_and_filters_to_bbox():
    coords = np.array([[0, 0, 0], [0, 0, 5], [10, 10, 10]], dtype=np.int64)
    ids = np.array([1, 2, 3], dtype=np.uint32)
    cells = bucket_points(coords, ids, block=8, shape=(20, 20, 20))
    # bbox exactly around the first cell's region -> the two points at (0,0,0)/(0,0,5).
    c, i = region_seeds(cells, block=8, bbox=(0, 8, 0, 8, 0, 8))
    assert sorted(i.tolist()) == [1, 2]
    # a bbox that includes (10,10,10) only.
    c2, i2 = region_seeds(cells, block=8, bbox=(8, 16, 8, 16, 8, 16))
    assert i2.tolist() == [3]
    # empty region.
    _, i3 = region_seeds(cells, block=8, bbox=(0, 4, 8, 12, 0, 4))
    assert i3.tolist() == []


def test_merge_seed_groups_merges_shared_core_to_min_id():
    # Seeds 0,1 share core 5 -> merge (repr = min global id 10); seed 2 in core 7 alone;
    # seed 3 has core 0 (dimmer than threshold) -> stays its own group.
    seed_core = np.array([5, 5, 7, 0], dtype=np.int64)
    global_ids = np.array([10, 12, 20, 30], dtype=np.uint32)
    local_to_group, group_to_global = merge_seed_groups(seed_core, global_ids)

    assert local_to_group[0] == 0  # background
    # seeds 0 and 1 (local labels 1,2) land in the same group.
    assert local_to_group[1] == local_to_group[2]
    assert local_to_group[3] != local_to_group[1]  # different core
    assert local_to_group[4] != local_to_group[1]  # core-0 singleton
    # groups are compact 1..G.
    assert sorted(set(local_to_group[1:].tolist())) == [1, 2, 3]
    # representative global id = smallest in the group.
    assert group_to_global[local_to_group[1]] == 10  # min(10, 12)
    assert group_to_global[local_to_group[3]] == 20
    assert group_to_global[local_to_group[4]] == 30
    assert group_to_global[0] == 0
    assert group_to_global.dtype == np.uint32


def test_merge_seed_groups_all_core_zero_stay_separate():
    # No seed in a core -> every seed is its own object (identity-like grouping).
    seed_core = np.array([0, 0, 0], dtype=np.int64)
    global_ids = np.array([7, 8, 9], dtype=np.uint32)
    local_to_group, group_to_global = merge_seed_groups(seed_core, global_ids)
    assert sorted(set(local_to_group[1:].tolist())) == [1, 2, 3]  # all distinct
    assert sorted(group_to_global[1:].tolist()) == [7, 8, 9]


def test_merge_seed_groups_empty():
    local_to_group, group_to_global = merge_seed_groups(
        np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.uint32)
    )
    assert local_to_group.tolist() == [0]
    assert group_to_global.tolist() == [0]


def test_flood_seed_union_shallow_merges_deep_splits():
    # Seeds 0 & 1 (both in blob comp 1) are joined by a shallow saddle at level 0.8;
    # seed 2 only connects to them at 0.6 (a deep valley). With frac=0.2 -> floors at
    # si*(1-0.2)=0.72/0.64/0.72, so 0,1 merge (meet at 0.8 >= their floors) but seed 2
    # (floor 0.72) is inactive by the time they all connect at 0.6 -> stays separate.
    si = np.array([0.9, 0.8, 0.9])
    seed_comp = np.array([1, 1, 1])
    levels = np.array([0.9, 0.8, 0.7, 0.6])
    level_labels = np.array(
        [
            [1, 0, 2],  # 0.9: seed1 below level -> 0
            [1, 1, 2],  # 0.8: seeds 0,1 share comp 1; seed 2 separate
            [1, 1, 2],  # 0.7
            [1, 1, 1],  # 0.6: all connected (deep valley)
        ]
    )
    min_level = si * (1 - 0.2)
    key = flood_seed_union(si, seed_comp, level_labels, levels, min_level)
    assert key[0] == key[1]  # shallow saddle -> merged
    assert key[2] != key[0]  # deep valley -> separate


def test_flood_seed_union_max_distance_gate():
    # Three seeds all valley-connected (share a bright ridge at every level, one blob).
    # Seeds 0,1 are 3 voxels apart; seed 2 is 100 away. With max_dist=5 only 0,1 merge;
    # with no distance limit all three merge.
    si = np.array([1.0, 1.0, 1.0])
    seed_comp = np.array([1, 1, 1])
    levels = np.array([1.0, 0.7])
    level_labels = np.array([[1, 1, 1], [1, 1, 1]])
    min_level = si * 0.5
    coords = np.array([[0, 0, 0], [0, 0, 3], [0, 0, 100]], dtype=np.int64)

    gated = flood_seed_union(
        si, seed_comp, level_labels, levels, min_level, coords=coords, max_dist=5.0
    )
    assert gated[0] == gated[1]  # within 5 -> merged
    assert gated[2] != gated[0]  # 100 away -> stays separate

    free = flood_seed_union(si, seed_comp, level_labels, levels, min_level)
    assert free[0] == free[1] == free[2]  # no distance limit -> all merge


def test_adjacency_merge_only_touching_same_group():
    # 4 seeds (labels 1..4). Proposed: {1,2,3} want to merge (key 7), 4 alone (key 9).
    # Adjacency: 1-2 touch, 3 touches nobody in-group, 2-4 touch (different group).
    # Expect: 1 & 2 merge; 3 stays separate (same group but not adjacent); 4 separate.
    group_key = np.array([7, 7, 7, 9])
    edges = np.array([[1, 2], [2, 4]])
    final = adjacency_merge(edges, group_key)
    assert final[0] == final[1]  # 1,2 same group + adjacent -> merged
    assert final[2] != final[0]  # 3 same group but NOT adjacent -> separate
    assert final[3] != final[0]  # 4 adjacent to 2 but different group -> separate
    assert final[3] != final[1]


def test_adjacency_merge_chain_transitive():
    # 1-2 and 2-3 adjacent, all same group -> 1,2,3 merge transitively via the chain.
    group_key = np.array([5, 5, 5])
    edges = np.array([[1, 2], [2, 3]])
    final = adjacency_merge(edges, group_key)
    assert final[0] == final[1] == final[2]


def test_adjacency_merge_no_edges():
    final = adjacency_merge(np.empty((0, 2), dtype=np.int64), np.array([3, 3, 3]))
    assert len(set(final.tolist())) == 3  # no adjacency -> nobody merges


def test_flood_seed_union_same_comp_guard():
    # Seeds 0 & 1 share a superlevel comp label but are in DIFFERENT blob components
    # (a bridge that doesn't exist in the carved mask) -> must NOT union.
    si = np.array([0.9, 0.9])
    seed_comp = np.array([1, 2])  # different blobs
    levels = np.array([0.9, 0.8])
    level_labels = np.array([[1, 1], [1, 1]])  # same superlevel label
    key = flood_seed_union(si, seed_comp, level_labels, levels, si * 0.5)
    assert key[0] != key[1]  # cross-component union blocked
