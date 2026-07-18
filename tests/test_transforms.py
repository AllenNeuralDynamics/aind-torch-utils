"""Tests for block preprocessors / invertible transforms.

Besides the behavioral checks, `test_*_matches_legacy_*` lock the arithmetic to the
exact operations PrepWorker/WriterWorker used to inline, so PR2b can swap them in
without changing any existing run's output.
"""
import numpy as np
import pytest

from aind_torch_utils.transforms import (
    Clip,
    GlobalNormalizer,
    IdentityTransform,
    IntensityTransformAdapter,
    PercentileNormalizer,
    Sequential,
    from_config,
    is_invertible,
)

# ctx is unused by every transform here; None proves it.
CTX = None


def _rand_block(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((8, 8, 8), dtype=np.float32) * 1000.0).astype(np.float32)


def test_percentile_normalizer_matches_legacy_forward_and_inverse():
    lower, upper, eps = 0.5, 99.9, 1e-6
    block = _rand_block()

    # Legacy prep math (in place on a float32 copy).
    legacy = block.copy()
    mn, mx = np.percentile(legacy, [lower, upper])
    scale = max(mx - mn, eps)
    legacy -= mn
    legacy /= scale

    norm = PercentileNormalizer(lower, upper, eps)
    out, state = norm.forward(block.copy(), CTX)
    assert np.array_equal(out, legacy)
    assert state == (float(mn), float(mx))

    # Legacy writer denorm.
    dscale = max(state[1] - state[0], eps)
    legacy_inv = (out * np.float32(dscale) + np.float32(state[0])).astype(np.float32)
    assert np.array_equal(norm.inverse(out, state, CTX), legacy_inv)


def test_percentile_normalizer_round_trips():
    block = _rand_block(1)
    norm = PercentileNormalizer(0.0, 100.0)  # full range -> exact affine
    out, state = norm.forward(block.copy(), CTX)
    recovered = norm.inverse(out, state, CTX)
    np.testing.assert_allclose(recovered, block, rtol=1e-4, atol=1e-2)


def test_global_normalizer_matches_legacy():
    lower, upper, eps = 10.0, 900.0, 1e-6
    block = _rand_block(2)

    scale = max(upper - lower, eps)
    legacy = np.clip(block, lower, upper)
    legacy = ((legacy - lower) / scale).astype(np.float32)

    norm = GlobalNormalizer(lower, upper, eps)
    out, state = norm.forward(block.copy(), CTX)
    assert np.array_equal(out, legacy)
    assert state == (lower, upper)
    assert out.dtype == np.float32


def test_identity_transform_is_noop():
    block = _rand_block(3)
    ident = IdentityTransform()
    out, state = ident.forward(block, CTX)
    assert out is block
    assert state is None
    assert np.array_equal(ident.inverse(block, state, CTX), block)


def test_intensity_transform_adapter_delegates_exactly():
    """The adapter adds calling convention only, including no dtype coercion."""

    class SourceTransform:
        def __init__(self):
            self.forward_input = None
            self.inverse_input = None

        def forward(self, array):
            self.forward_input = array
            return (array.astype(np.float32) / 3.0).astype(np.float32)

        def inverse(self, array):
            self.inverse_input = array
            return np.rint(np.clip(array * 3.0, 0, 65535)).astype(np.uint16)

    source = SourceTransform()
    adapter = IntensityTransformAdapter(source)
    block = np.array([-3, 0, 3, 60000], dtype=np.int32)

    expected_forward = source.forward(block)
    out, state = adapter.forward(block, CTX)
    np.testing.assert_array_equal(out, expected_forward)
    assert state is None
    assert source.forward_input is block
    assert adapter.transform is source
    assert adapter.inverse_stage == "after_finalize"

    expected_inverse = source.inverse(out)
    recovered = adapter.inverse(out, state, CTX)
    np.testing.assert_array_equal(recovered, expected_inverse)
    assert recovered.dtype == np.uint16
    assert source.inverse_input is out


@pytest.mark.parametrize("missing", ["forward", "inverse"])
def test_intensity_transform_adapter_validates_required_methods(missing):
    class SourceTransform:
        def forward(self, array):
            return array

        def inverse(self, array):
            return array

    source = SourceTransform()
    setattr(source, missing, None)
    with pytest.raises(TypeError, match=missing):
        IntensityTransformAdapter(source)


def test_intensity_transform_adapter_matches_external_asinh_when_available():
    """Exercise the optional dependency's clipping, rounding, and HDR mapping."""
    try:
        from aind_exaspim_image_compression.machine_learning.transforms import (
            AsinhTransform,
        )
    except ImportError:
        pytest.skip("checkpoint-aware denoise-net transform API is not installed")

    source = AsinhTransform(offset=35.0, scale=32.0)
    adapter = IntensityTransformAdapter(source)
    values = np.array(
        [0, 35, 100, 1000, 10000, 60000, 65535], dtype=np.float32
    )

    expected, state = source.forward(values), None
    actual, actual_state = adapter.forward(values, CTX)
    np.testing.assert_array_equal(actual, expected)
    assert actual_state is state
    assert actual.dtype == expected.dtype == np.float32
    assert np.all(np.diff(actual) > 0)

    expected_inverse = source.inverse(expected)
    actual_inverse = adapter.inverse(actual, actual_state, CTX)
    np.testing.assert_array_equal(actual_inverse, expected_inverse)
    assert actual_inverse.dtype == expected_inverse.dtype == np.uint16
    assert np.all(np.diff(actual_inverse[-4:]) > 0)


def test_clip_is_non_invertible_and_clips():
    block = _rand_block(4)
    clip = Clip(100.0, 200.0)
    out, state = clip.forward(block, CTX)
    assert state is None
    assert not is_invertible(clip)
    assert out.min() >= 100.0 and out.max() <= 200.0


def test_sequential_forward_composes_and_inverse_reverses():
    block = _rand_block(5)
    seq = Sequential([PercentileNormalizer(0.0, 100.0), Clip(0.0, 1.0)])

    out, state = seq.forward(block.copy(), CTX)
    # Composite state is one entry per member (Clip contributes None).
    assert isinstance(state, tuple) and len(state) == 2
    assert state[1] is None
    assert out.min() >= 0.0 and out.max() <= 1.0

    # inverse skips the non-invertible Clip and undoes the normalizer.
    recovered = seq.inverse(out, state, CTX)
    # Values were within [0,1] after normalization (full-range percentiles), so the
    # Clip was a no-op here and the affine inverse recovers the original block.
    np.testing.assert_allclose(recovered, block, rtol=1e-4, atol=1e-2)


def test_sequential_inverse_stage_agreement():
    seq = Sequential([PercentileNormalizer(0.0, 100.0), Clip(0.0, 1.0)])
    assert seq.inverse_stage == "after_finalize"

    class _Weird:
        inverse_stage = "before_accumulate"

        def forward(self, b, c):
            return b, None

        def inverse(self, b, s, c):
            return b

    with pytest.raises(ValueError, match="disagree on inverse_stage"):
        _ = Sequential([PercentileNormalizer(0.0, 100.0), _Weird()]).inverse_stage


@pytest.mark.parametrize(
    "normalize,expected_type",
    [
        ("percentile", PercentileNormalizer),
        ("global", GlobalNormalizer),
        (False, IdentityTransform),
    ],
)
def test_from_config_selects_transform(normalize, expected_type):
    t = from_config(normalize, 0.5, 99.9, 1e-6, clip_norm=False)
    assert isinstance(t, expected_type)


def test_from_config_wraps_clip_in_sequential():
    t = from_config("percentile", 0.5, 99.9, 1e-6, clip_norm=True)
    assert isinstance(t, Sequential)
    assert isinstance(t.steps[0], PercentileNormalizer)
    assert isinstance(t.steps[1], Clip)
    assert (t.steps[1].lo, t.steps[1].hi) == (0.0, 1.0)

    t2 = from_config("global", 10.0, 900.0, 1e-6, clip_norm=(0.1, 0.9))
    assert isinstance(t2, Sequential)
    assert (t2.steps[1].lo, t2.steps[1].hi) == (0.1, 0.9)
