"""Tests for BlockContext construction and writer-side reconstruction."""
from aind_torch_utils.context import BlockContext


class _FakePreds:
    """Just the fields BlockContext.from_preds reads."""

    def __init__(self, block_idx, block_bbox, halo_left, acc_shape):
        self.block_idx = block_idx
        self.block_bbox = block_bbox
        self.halo_left = halo_left
        self.acc_shape = acc_shape


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


def test_from_preds_reconstructs_expanded_bbox():
    core = (slice(40, 60), slice(40, 60), slice(40, 60))
    preds = _FakePreds(
        block_idx=(1, 1, 1),
        block_bbox=core,
        halo_left=(8, 8, 8),
        acc_shape=(36, 36, 36),  # 20 core + 8 halo on each side
    )

    ctx = BlockContext.from_preds(preds, full_shape=(100, 100, 100), t_idx=0, c_idx=0)

    assert ctx.expanded_bbox == (slice(32, 68), slice(32, 68), slice(32, 68))
    assert ctx.core_bbox == core
    assert ctx.halo_left == (8, 8, 8)


def test_from_block_and_from_preds_round_trip():
    """A context built in prep equals the one reconstructed in the writer."""
    core = (slice(40, 60), slice(40, 60), slice(40, 60))
    expanded = (slice(32, 68), slice(32, 68), slice(32, 68))
    full = (100, 100, 100)

    prep_ctx = BlockContext.from_block(
        block_idx=(1, 1, 1),
        core_bbox=core,
        expanded_bbox=expanded,
        full_shape=full,
        t_idx=2,
        c_idx=3,
    )

    acc_shape = tuple(e.stop - e.start for e in expanded)
    preds = _FakePreds((1, 1, 1), core, prep_ctx.halo_left, acc_shape)
    writer_ctx = BlockContext.from_preds(preds, full_shape=full, t_idx=2, c_idx=3)

    assert prep_ctx == writer_ctx


def test_from_preds_handles_clipped_border_halo():
    """At a volume border the halo is one-sided; reconstruction must still match."""
    # Core touches z=0: left halo clipped to 0, right halo 8 -> acc_shape z = 28.
    core = (slice(0, 20), slice(0, 20), slice(0, 20))
    preds = _FakePreds(
        block_idx=(0, 0, 0),
        block_bbox=core,
        halo_left=(0, 0, 0),
        acc_shape=(28, 28, 28),
    )

    ctx = BlockContext.from_preds(preds, full_shape=(200, 200, 200), t_idx=0, c_idx=0)

    assert ctx.expanded_bbox == (slice(0, 28), slice(0, 28), slice(0, 28))
    assert ctx.halo_left == (0, 0, 0)
