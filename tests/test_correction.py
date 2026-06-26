"""Tests for flat-field background correction (pure numpy; no GPU/IO stack)."""
import numpy as np

from aind_torch_utils.correction import (
    BackgroundField,
    apply_flatfield,
    sample_background,
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
