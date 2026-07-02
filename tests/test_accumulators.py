"""Tests for the pluggable block accumulators and the default factory."""
import numpy as np

from aind_torch_utils.accumulators import (
    BlockAccumulator,
    LastWriteAccumulator,
    MajorityVoteAccumulator,
    MaxAccumulator,
    SumAccumulator,
    WeightedAverageAccumulator,
    weighted_average_factory,
)

CTX = None  # accumulators do not use ctx


def _patch(values):
    """A (1, 1, N) float32 patch from a flat list."""
    return np.asarray(values, dtype=np.float32).reshape(1, 1, -1)


def test_max_accumulator_takes_elementwise_max_over_overlap():
    acc = MaxAccumulator((1, 1, 4))
    acc.add(_patch([1, 5, 1]), (0, 0, 0), (1, 1, 3))  # x0..2
    acc.add(_patch([9, 2, 9]), (0, 0, 1), (1, 1, 3))  # x1..3
    np.testing.assert_array_equal(acc.finalize().ravel(), [1, 9, 2, 9])


def test_max_accumulator_uncovered_voxels_are_zero():
    acc = MaxAccumulator((1, 1, 4))
    acc.add(_patch([3, 3]), (0, 0, 0), (1, 1, 2))  # only x0,x1 covered
    out = acc.finalize().ravel()
    np.testing.assert_array_equal(out, [3, 3, 0, 0])


def test_last_write_accumulator_last_patch_wins():
    acc = LastWriteAccumulator((1, 1, 4))
    acc.add(_patch([1, 1, 1]), (0, 0, 0), (1, 1, 3))
    acc.add(_patch([7, 7, 7]), (0, 0, 1), (1, 1, 3))
    # Overlap x1,x2 overwritten by the second patch.
    np.testing.assert_array_equal(acc.finalize().ravel(), [1, 7, 7, 7])


def test_sum_accumulator_adds_overlap():
    acc = SumAccumulator((1, 1, 4))
    acc.add(_patch([1, 1, 1]), (0, 0, 0), (1, 1, 3))
    acc.add(_patch([10, 10, 10]), (0, 0, 1), (1, 1, 3))
    np.testing.assert_array_equal(acc.finalize().ravel(), [1, 11, 11, 10])


def test_majority_vote_picks_mode_and_breaks_ties_low():
    acc = MajorityVoteAccumulator((1, 1, 3))
    acc.add(_patch([1, 1, 2]), (0, 0, 0), (1, 1, 3))
    acc.add(_patch([1, 2, 2]), (0, 0, 0), (1, 1, 3))
    # x0: {1:2}; x1: {1:1, 2:1} tie -> smaller label; x2: {2:2}.
    np.testing.assert_array_equal(acc.finalize().ravel(), [1, 1, 2])


def test_majority_vote_empty_block_is_zeros():
    acc = MajorityVoteAccumulator((1, 1, 2))
    np.testing.assert_array_equal(acc.finalize().ravel(), [0, 0])


def test_accumulators_track_count():
    acc = SumAccumulator((1, 1, 2))
    assert acc.count == 0
    acc.add(_patch([1, 1]), (0, 0, 0), (1, 1, 2))
    assert acc.count == 1


def test_weighted_average_factory_builds_default_accumulator():
    factory = weighted_average_factory(
        eps=1e-6, overlap=4, seam_mode="trim", trim_voxels=2, min_blend_weight=0.05
    )
    acc = factory((2, 2, 2), CTX)
    assert isinstance(acc, WeightedAverageAccumulator)
    # a single block-sized patch (no trim at borders) round-trips through finalize
    patch = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    acc.total = 1
    acc.add(patch, (0, 0, 0), (2, 2, 2))
    np.testing.assert_allclose(acc.finalize(), patch)


def test_accumulators_satisfy_protocol():
    for acc in (
        MaxAccumulator((1, 1, 1)),
        LastWriteAccumulator((1, 1, 1)),
        SumAccumulator((1, 1, 1)),
        MajorityVoteAccumulator((1, 1, 1)),
        WeightedAverageAccumulator((1, 1, 1), 1e-6, 0, "trim", 0, 0.05),
    ):
        assert isinstance(acc, BlockAccumulator)
