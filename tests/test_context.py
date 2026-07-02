"""Tests for BlockContext construction."""
from aind_torch_utils.context import BlockContext


def test_from_block_derives_halo_left():
    core = (slice(40, 60), slice(40, 60), slice(40, 60))
    expanded = (slice(32, 68), slice(32, 68), slice(32, 68))

    ctx = BlockContext.from_block(
        block_idx=(1, 1, 1),
        core_bbox=core,
        expanded_bbox=expanded,
        full_shape=(100, 100, 100),
        t_idx=0,
        c_idx=0,
    )

    assert ctx.halo_left == (8, 8, 8)
    assert ctx.core_bbox == core
    assert ctx.expanded_bbox == expanded
    assert ctx.full_shape == (100, 100, 100)


def test_from_block_clipped_border_halo():
    """At a volume border the halo is one-sided; halo_left must reflect it."""
    core = (slice(0, 20), slice(0, 20), slice(0, 20))
    expanded = (slice(0, 28), slice(0, 28), slice(0, 28))

    ctx = BlockContext.from_block(
        block_idx=(0, 0, 0),
        core_bbox=core,
        expanded_bbox=expanded,
        full_shape=(200, 200, 200),
        t_idx=0,
        c_idx=0,
    )

    assert ctx.halo_left == (0, 0, 0)
    assert ctx.expanded_bbox == expanded
