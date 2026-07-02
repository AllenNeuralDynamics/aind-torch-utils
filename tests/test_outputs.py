"""Tests for OutputSpec defaults and output-domain post-processors."""
import numpy as np
import pytest

from aind_torch_utils.accumulators import weighted_average_factory
from aind_torch_utils.outputs import (
    BlockPostProcessor,
    OutputSpec,
    Threshold,
    ThresholdThenOpen,
)

CTX = None


def _factory():
    return weighted_average_factory(1e-6, 4, "trim", 2, 0.05)


def test_output_spec_defaults():
    spec = OutputSpec(store=object(), accumulator_factory=_factory())
    assert spec.postprocess is None
    assert spec.invert is False


def test_threshold_binarizes():
    block = np.array([[[-1.0, 0.0, 0.5, 2.0]]], dtype=np.float32)
    out = Threshold(thresh=0.0)(block, CTX)
    np.testing.assert_array_equal(out.ravel(), [0, 0, 1, 1])
    assert isinstance(Threshold(), BlockPostProcessor)


def test_threshold_custom_levels():
    block = np.array([[[1.0, 5.0]]], dtype=np.float32)
    out = Threshold(thresh=3.0, above=255.0, below=7.0)(block, CTX)
    np.testing.assert_array_equal(out.ravel(), [7, 255])


def test_threshold_then_open_removes_speckle():
    pytest.importorskip("scipy")
    # A 1-voxel speckle surrounded by background is removed by opening; a solid
    # block survives.
    block = np.zeros((5, 5, 5), dtype=np.float32)
    block[0, 0, 0] = 10.0  # isolated speckle -> opened away
    block[1:4, 1:4, 1:4] = 10.0  # solid cube -> (mostly) survives
    out = ThresholdThenOpen(thresh=0.0, open_iters=1)(block, CTX)
    assert out[0, 0, 0] == 0.0
    assert out[2, 2, 2] == 1.0
