"""Tests for flat-field background correction (pure numpy; no GPU/IO stack)."""
import numpy as np

from aind_torch_utils.correction import (
    BackgroundField,
    apply_flatfield,
    fill_no_data,
    normalize_global,
    sample_background,
    scale_params,
)


def _global_field(shape=(40, 40), period=37.0):
    """A smooth low-frequency 'shading' field over a 2D plane (z is trivial here)."""
    yy, xx = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    field = 100.0 + 50.0 * np.sin(2 * np.pi * xx / period) + 0.5 * yy
    return field.astype(np.float32)


def test_subtract_is_white_tophat_nonnegative():
    block = np.array([[10.0, 5.0], [20.0, 0.0]], dtype=np.float32)
    bg = np.array([[3.0, 9.0], [25.0, 1.0]], dtype=np.float32)

    out = apply_flatfield(block, bg, mode="subtract")

    assert out.dtype == np.float32
    # max(block - bg, 0)
    np.testing.assert_allclose(out, [[7.0, 0.0], [0.0, 0.0]])
    assert (out >= 0).all()


def test_divide_preserves_overall_scale():
    block = np.array([[100.0, 200.0]], dtype=np.float32)
    bg = np.array([[100.0, 200.0]], dtype=np.float32)
    bg_mean = float(bg.mean())  # 150

    out = apply_flatfield(block, bg, mode="divide", bg_mean=bg_mean)

    # block == bg -> ratio 1 -> equals bg_mean everywhere (flat output).
    np.testing.assert_allclose(out, [[150.0, 150.0]])


def test_divide_floors_denominator():
    block = np.array([[5.0]], dtype=np.float32)
    bg = np.array([[0.0]], dtype=np.float32)  # would divide-by-zero without eps floor
    out = apply_flatfield(block, bg, mode="divide", eps=1e-6, bg_mean=1.0)
    assert np.isfinite(out).all()


def test_unknown_mode_raises():
    block = np.zeros((2, 2), dtype=np.float32)
    try:
        apply_flatfield(block, block, mode="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_seam_free_across_overlapping_blocks():
    """Two overlapping crops of a global volume must agree on shared voxels.

    This is the core seam guarantee: because the background is a single global field
    sliced by absolute coordinates and the correction is point-wise, a voxel computed
    inside block A's region equals the same voxel computed inside block B's region.
    """
    Z, Y, X = 3, 40, 40
    rng = np.random.default_rng(0)
    bg2d = _global_field((Y, X))
    background = np.broadcast_to(bg2d, (Z, Y, X)).astype(np.float32)
    volume = (background + 30.0 * rng.standard_normal((Z, Y, X))).astype(np.float32)
    bg_mean = float(background.mean())

    for mode in ("subtract", "divide"):
        # Block A covers x in [0, 24), block B covers x in [16, 40); overlap [16, 24).
        a = apply_flatfield(
            volume[:, :, 0:24], background[:, :, 0:24], mode=mode, bg_mean=bg_mean
        )
        b = apply_flatfield(
            volume[:, :, 16:40], background[:, :, 16:40], mode=mode, bg_mean=bg_mean
        )
        # Shared region in A is columns 16:24 -> a[..., 16:24]; in B -> b[..., 0:8].
        np.testing.assert_array_equal(a[:, :, 16:24], b[:, :, 0:8])


def test_sample_background_reproduces_coarse_at_scale_one():
    # scale=1 (coarse == processing grid) -> sampling returns the field verbatim.
    field = _global_field((8, 8))[None].repeat(4, axis=0).astype(np.float32)
    out = sample_background(field, (1.0, 1.0, 1.0), 0, 4, 0, 8, 0, 8)
    np.testing.assert_allclose(out, field, atol=1e-4)


def test_sample_background_seam_free_across_blocks():
    """Per-block sampling of the coarse field agrees on shared voxels (no seam)."""
    cz, cy, cx = 4, 12, 12
    rng = np.random.default_rng(1)
    field = rng.uniform(50, 200, size=(cz, cy, cx)).astype(np.float32)
    scale = (3.0, 4.0, 4.0)  # processing grid is 12 x 48 x 48

    # Block A: x in [0, 30); block B: x in [20, 48); overlap [20, 30).
    a = sample_background(field, scale, 0, 12, 0, 48, 0, 30)
    b = sample_background(field, scale, 0, 12, 0, 48, 20, 48)
    np.testing.assert_array_equal(a[:, :, 20:30], b[:, :, 0:10])


def test_normalize_global_clips_and_scales():
    vol = np.array([50.0, 90.0, 600.0, 1200.0, 2000.0], dtype=np.float32)
    out = normalize_global(vol, 90.0, 1200.0, eps=1e-6)
    # 90 -> 0, 1200 -> 1; below/above clip to the bounds.
    np.testing.assert_allclose(out[0], 0.0)  # 50 clipped to 90 -> 0
    np.testing.assert_allclose(out[1], 0.0)  # exactly lower
    np.testing.assert_allclose(out[3], 1.0)  # exactly upper
    np.testing.assert_allclose(out[4], 1.0)  # 2000 clipped to 1200 -> 1
    assert (out >= 0).all() and (out <= 1).all()
    np.testing.assert_allclose(out[2], (600.0 - 90.0) / (1200.0 - 90.0), rtol=1e-6)


def test_normalize_global_eps_guards_zero_range():
    vol = np.array([5.0, 5.0], dtype=np.float32)
    out = normalize_global(vol, 5.0, 5.0, eps=1e-6)  # upper == lower
    assert np.isfinite(out).all()


def test_scale_params_scales_sigma_and_open():
    # Coarse tune -> finer target by 4x in every axis.
    sigma, op = scale_params((0.5, 1.0, 1.0), 1, (4.0, 4.0, 4.0))
    assert sigma == (2.0, 4.0, 4.0)
    assert op == 4  # round(1 * mean(4,4))


def test_scale_params_open_rounds_on_yx_only():
    # Z factor must not affect the in-plane open scaling.
    _, op = scale_params((1, 1, 1), 2, (10.0, 1.0, 1.0))
    assert op == 2  # round(2 * mean(1,1))


def test_background_field_mean():
    field = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    bf = BackgroundField(field=field, scale=(2.0, 2.0, 2.0))
    assert bf.mean == float(field.mean())


def test_subtract_flattens_shading_for_uniform_threshold():
    """After subtracting the true background, a constant offset signal is uniform.

    Demonstrates why a single global threshold becomes chunk-invariant: a fixed-contrast
    structure on top of varying shading yields a constant residual everywhere.
    """
    Y, X = 40, 40
    bg = _global_field((Y, X))[None]  # (1, Y, X)
    signal_contrast = 80.0
    volume = bg + signal_contrast  # structure sits 80 above local background everywhere

    out = apply_flatfield(volume, bg, mode="subtract")

    # Residual is the constant contrast everywhere -> one threshold catches it all.
    np.testing.assert_allclose(out, signal_contrast, rtol=0, atol=1e-3)


def _min_filter_1d(a, size=3):
    """A tiny 1-D grey erosion (min filter) with edge padding, for the test below."""
    r = size // 2
    pad = np.pad(a, r, mode="edge")
    out = []
    for i in range(len(a)):
        end = i + size
        out.append(pad[i:end].min())
    return np.array(out)


def test_fill_no_data_fills_with_valid_median():
    v = np.array([0, 0, 10, 20, 30, 40], dtype=np.float32)  # the 0s are no-data
    out = fill_no_data(v, 0.0)
    med = np.median([10, 20, 30, 40])  # 25
    assert out[0] == med and out[1] == med
    assert np.array_equal(out[2:], v[2:])  # valid voxels untouched


def test_fill_no_data_is_noop_when_all_or_none_valid():
    v = np.array([1, 2, 3], dtype=np.float32)
    assert fill_no_data(v, 0.0) is v  # all valid -> same object
    z = np.zeros(3, dtype=np.float32)
    assert fill_no_data(z, 0.0) is z  # none valid -> same object


def test_fill_no_data_prevents_background_underestimate_at_interface():
    # Empty (0) abutting tissue (~100): a grey-opening's erosion drags the tissue-side
    # background to 0 at the interface (-> white top-hat leaves a bright rim). Filling
    # the no-data region with the valid median keeps the estimate flat across it.
    raw = np.array([0, 0, 0, 100, 100, 100, 100], dtype=np.float32)
    filled = fill_no_data(raw, 0.0)
    assert np.allclose(filled, 100.0)  # zeros -> median of the 100s
    assert _min_filter_1d(raw, 3)[3] == 0.0  # tissue voxel next to empty: under-est.
    assert _min_filter_1d(filled, 3)[3] == 100.0  # fixed after fill
