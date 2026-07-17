"""Pure-numpy per-object metrics + outlier statistics (no GPU/IO deps).

Two halves, both dependency-light so they can be unit-tested without a GPU:

- **Accumulators -> metrics.** Per-object stats over a huge instance-label volume are
  computed by *additive* reductions per global id over disjoint blocks (see
  ``examples/object_metrics.py``): sums of intensity powers, coordinate sums / second
  moments, intensity-weighted coordinate sums, plus min/max and bounding box. Because
  ids are globally unique, summing the per-block partials is exact across block seams.
  :func:`allocate_accumulators` defines the canonical set; :func:`derive_metrics` turns
  finished accumulators into a per-object feature dict (voxel + physical units).

- **Outlier statistics.** :func:`mad_flags` / :func:`iqr_flags` (robust per-feature) and
  :func:`mahalanobis` (multivariate) for flagging outliers / inliers on the table.

Non-additive metrics (true surface area, convex-hull solidity, exact median) are out of
scope; roundness uses moment-based shape (covariance eigenvalues), which IS additive.
"""

from typing import Dict, Tuple

import numpy as np

# Additive sum accumulators (index 0 = background, kept for id alignment).
SUM_KEYS = (
    "sum_i",
    "sum_i2",
    "sum_i3",
    "sum_i4",
    "sum_z",
    "sum_y",
    "sum_x",
    "sum_zz",
    "sum_yy",
    "sum_xx",
    "sum_zy",
    "sum_zx",
    "sum_yx",
    "sum_iz",
    "sum_iy",
    "sum_ix",
    "sum_izz",
    "sum_iyy",
    "sum_ixx",
)
MIN_KEYS = ("min_i", "bz0", "by0", "bx0")  # reduce by np.minimum across blocks
MAX_KEYS = ("max_i", "bz1", "by1", "bx1")  # reduce by np.maximum across blocks

# Higher moments (skew/kurtosis) are recovered from power-sums via central-moment
# expansion, which suffers catastrophic cancellation when the true variance is tiny
# (near-uniform objects) -- the surviving "m3"/"m4" is then float rounding noise divided
# by a near-zero denominator, which explodes to +/-1e8. Only trust them when the spread
# is a non-negligible fraction of the mean and the object has enough voxels. Use float64
# accumulators (not --float32-acc) when skew/kurtosis matter; float32 is far worse here.
REL_STD_EPS = 1e-3  # require std >= REL_STD_EPS * mean before computing skew/kurtosis
MIN_MOMENT_N = 4  # and at least this many voxels


def allocate_accumulators(n: int, sum_dtype=np.float64) -> Dict[str, np.ndarray]:
    """Allocate zero/inf-initialised host accumulators for ids ``0..n-1``.

    ``count`` is int64; the ``SUM_KEYS`` are ``sum_dtype`` (float64 default, float32 to
    halve memory); min/max/bbox keys are float64 initialised to +/-inf so the first
    block's ``np.minimum``/``np.maximum`` reduction wins.
    """
    acc: Dict[str, np.ndarray] = {"count": np.zeros(n, dtype=np.int64)}
    for k in SUM_KEYS:
        acc[k] = np.zeros(n, dtype=sum_dtype)
    for k in MIN_KEYS:
        acc[k] = np.full(n, np.inf, dtype=np.float64)
    for k in MAX_KEYS:
        acc[k] = np.full(n, -np.inf, dtype=np.float64)
    return acc


def eig3x3_sym(cov: np.ndarray) -> np.ndarray:
    """Eigenvalues of stacked symmetric 3x3 matrices, descending. ``cov`` is (M, 3, 3).

    Closed-form (Cardano) so it is vectorised over M without a Python loop. Returns
    ``(M, 3)`` with ``lambda1 >= lambda2 >= lambda3`` (clipped to >= 0).
    """
    a = cov[:, 0, 0]
    b = cov[:, 1, 1]
    c = cov[:, 2, 2]
    d = cov[:, 0, 1]
    e = cov[:, 1, 2]
    f = cov[:, 0, 2]
    q = (a + b + c) / 3.0
    p1 = d * d + e * e + f * f
    p2 = (a - q) ** 2 + (b - q) ** 2 + (c - q) ** 2 + 2.0 * p1
    p = np.sqrt(np.maximum(p2 / 6.0, 0.0))
    safe = p > 0
    pp = np.where(safe, p, 1.0)  # avoid divide-by-zero for isotropic matrices
    # r = det((A - qI)/p) / 2, clipped to [-1, 1]
    aq, bq, cq = a - q, b - q, c - q
    det = aq * (bq * cq - e * e) - d * (d * cq - e * f) + f * (d * e - bq * f)
    r = np.clip(det / (2.0 * pp**3), -1.0, 1.0)
    phi = np.arccos(r) / 3.0
    eig1 = q + 2.0 * pp * np.cos(phi)
    eig3 = q + 2.0 * pp * np.cos(phi + 2.0 * np.pi / 3.0)
    eig2 = 3.0 * q - eig1 - eig3
    lam = np.stack([eig1, eig2, eig3], axis=1)
    lam = np.where(safe[:, None], lam, q[:, None])  # isotropic -> all equal
    lam.sort(axis=1)  # ascending
    return np.maximum(lam[:, ::-1], 0.0)  # descending, non-negative


def _intensity_metrics(acc, ids, n) -> Dict[str, np.ndarray]:
    """Mean/std/CV/skew/kurtosis/min/max/integrated intensity from power sums."""
    si = acc["sum_i"][ids]
    mean = si / n
    m2 = np.maximum(acc["sum_i2"][ids] / n - mean**2, 0.0)
    std = np.sqrt(m2)
    m3 = acc["sum_i3"][ids] / n - 3 * mean * acc["sum_i2"][ids] / n + 2 * mean**3
    m4 = (
        acc["sum_i4"][ids] / n
        - 4 * mean * acc["sum_i3"][ids] / n
        + 6 * mean**2 * acc["sum_i2"][ids] / n
        - 3 * mean**4
    )
    # Only trust skew/kurtosis where the spread is a non-negligible fraction of the mean
    # AND there are enough voxels; otherwise the power-sum cancellation is pure noise.
    reliable = (m2 > (REL_STD_EPS * np.abs(mean)) ** 2) & (n >= MIN_MOMENT_N)
    denom = np.where(reliable, m2, 1.0)
    return {
        "mean_intensity": mean,
        "std_intensity": std,
        "cv_intensity": np.where(mean != 0, std / np.where(mean != 0, mean, 1.0), 0.0),
        "skew_intensity": np.where(reliable, m3 / denom**1.5, 0.0),
        "kurtosis_intensity": np.where(reliable, m4 / denom**2 - 3.0, 0.0),
        "min_intensity": acc["min_i"][ids],
        "max_intensity": acc["max_i"][ids],
        "integrated_intensity": si,
    }


def _shape_metrics(acc, ids, n, scale) -> Dict[str, np.ndarray]:
    """Volume, centroid, bbox, extent, equiv diameter, moment-based shape ratios."""
    sz, sy, sx = scale
    cz, cy, cx = acc["sum_z"][ids] / n, acc["sum_y"][ids] / n, acc["sum_x"][ids] / n
    # Central second moments (voxel), scaled to physical: C_phys[i,j] = s_i s_j C[i,j].
    czz = acc["sum_zz"][ids] / n - cz * cz
    cyy = acc["sum_yy"][ids] / n - cy * cy
    cxx = acc["sum_xx"][ids] / n - cx * cx
    czy = acc["sum_zy"][ids] / n - cz * cy
    czx = acc["sum_zx"][ids] / n - cz * cx
    cyx = acc["sum_yx"][ids] / n - cy * cx
    m = ids.shape[0]
    cov = np.zeros((m, 3, 3), dtype=np.float64)
    cov[:, 0, 0], cov[:, 1, 1], cov[:, 2, 2] = (
        czz * sz * sz,
        cyy * sy * sy,
        cxx * sx * sx,
    )
    cov[:, 0, 1] = cov[:, 1, 0] = czy * sz * sy
    cov[:, 0, 2] = cov[:, 2, 0] = czx * sz * sx
    cov[:, 1, 2] = cov[:, 2, 1] = cyx * sy * sx
    lam = eig3x3_sym(cov)  # descending physical variance eigenvalues (µm^2)
    l1, l2, l3 = lam[:, 0], lam[:, 1], lam[:, 2]
    # Floor the smaller eigenvalues at the single-voxel second moment (variance of a
    # uniform 1-voxel-wide slab = s^2/12) before forming ratios. A degenerate thin/flat
    # object has a near-zero smaller eigenvalue that is really float noise, which would
    # blow the ratios up to ~1e8; flooring caps elongation/flatness at a finite value
    # set by resolution. The raw eigenvalues are still used for axis_*_um below.
    eig_floor = float(min(sz, sy, sx)) ** 2 / 12.0
    l2r = np.maximum(l2, eig_floor)
    l3r = np.maximum(l3, eig_floor)
    l1r = np.maximum(l1, eig_floor)
    dz = acc["bz1"][ids] - acc["bz0"][ids] + 1
    dy = acc["by1"][ids] - acc["by0"][ids] + 1
    dx = acc["bx1"][ids] - acc["bx0"][ids] + 1
    vol_um3 = n * sz * sy * sx
    return {
        "volume_vox": n.astype(np.int64),
        "volume_um3": vol_um3,
        "equiv_diameter_um": (6.0 * vol_um3 / np.pi) ** (1.0 / 3.0),
        "centroid_z_um": cz * sz,
        "centroid_y_um": cy * sy,
        "centroid_x_um": cx * sx,
        "bbox_z0_vox": acc["bz0"][ids].astype(np.int64),
        "bbox_y0_vox": acc["by0"][ids].astype(np.int64),
        "bbox_x0_vox": acc["bx0"][ids].astype(np.int64),
        "bbox_dz_vox": dz.astype(np.int64),
        "bbox_dy_vox": dy.astype(np.int64),
        "bbox_dx_vox": dx.astype(np.int64),
        "extent": n / (dz * dy * dx),  # fill fraction of the bbox
        # equivalent-ellipsoid semi-axes (uniform solid: 2nd moment = a^2 / 5)
        "axis_major_um": np.sqrt(5.0 * l1),
        "axis_mid_um": np.sqrt(5.0 * l2),
        "axis_minor_um": np.sqrt(5.0 * l3),
        "elongation": np.sqrt(l1r / l2r),  # major/mid, >= 1
        "flatness": np.sqrt(l2r / l3r),  # mid/minor, >= 1
        "sphericity": np.sqrt(l3r / l1r),  # (0, 1], 1 = isotropic
    }


def _profile_metrics(acc, ids, n, scale) -> Dict[str, np.ndarray]:
    """Per-axis intensity spread + geometric-to-intensity centroid offset."""
    sz, sy, sx = scale
    si = np.where(acc["sum_i"][ids] > 0, acc["sum_i"][ids], 1.0)
    icz, icy, icx = (
        acc["sum_iz"][ids] / si,
        acc["sum_iy"][ids] / si,
        acc["sum_ix"][ids] / si,
    )
    vz = np.maximum(acc["sum_izz"][ids] / si - icz**2, 0.0)
    vy = np.maximum(acc["sum_iyy"][ids] / si - icy**2, 0.0)
    vx = np.maximum(acc["sum_ixx"][ids] / si - icx**2, 0.0)
    cz, cy, cx = acc["sum_z"][ids] / n, acc["sum_y"][ids] / n, acc["sum_x"][ids] / n
    off = np.sqrt(
        ((cz - icz) * sz) ** 2 + ((cy - icy) * sy) ** 2 + ((cx - icx) * sx) ** 2
    )
    return {
        "int_std_z_um": np.sqrt(vz) * sz,
        "int_std_y_um": np.sqrt(vy) * sy,
        "int_std_x_um": np.sqrt(vx) * sx,
        "centroid_offset_um": off,
    }


def derive_metrics(
    acc: Dict[str, np.ndarray], scale_zyx, shape_zyx
) -> Dict[str, np.ndarray]:
    """Turn finished accumulators into a per-object feature dict (one entry per column).

    ``scale_zyx`` is the physical voxel size (µm) per axis; ``shape_zyx`` the volume
    ``(nz, ny, nx)`` (to flag border objects). Only ids with ``count > 0`` are returned;
    ``out["id"]`` gives their global instance ids.
    """
    ids = np.nonzero(acc["count"] > 0)[0]
    n = acc["count"][ids].astype(np.float64)
    out: Dict[str, np.ndarray] = {"id": ids.astype(np.int64)}
    out.update(_shape_metrics(acc, ids, n, scale_zyx))
    out.update(_intensity_metrics(acc, ids, n))
    out.update(_profile_metrics(acc, ids, n, scale_zyx))
    nz, ny, nx = shape_zyx
    out["touches_border"] = (
        (acc["bz0"][ids] <= 0)
        | (acc["by0"][ids] <= 0)
        | (acc["bx0"][ids] <= 0)
        | (acc["bz1"][ids] >= nz - 1)
        | (acc["by1"][ids] >= ny - 1)
        | (acc["bx1"][ids] >= nx - 1)
    )
    return out


def mad_flags(x: np.ndarray, k: float = 3.5) -> Tuple[np.ndarray, np.ndarray]:
    """Robust per-feature outlier flags via the median absolute deviation.

    Returns ``(flags, robust_z)`` where ``robust_z = |x - median| / (1.4826 * MAD)`` and
    ``flags = robust_z > k``. MAD of 0 (constant feature) -> no flags.
    """
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    if mad <= 0:
        return np.zeros(x.shape, dtype=bool), np.zeros(x.shape, dtype=np.float64)
    z = np.abs(x - med) / mad
    return z > k, z


def iqr_flags(x: np.ndarray, k: float = 1.5) -> np.ndarray:
    """Per-feature outlier flags via Tukey IQR fences (outside q1-k*IQR .. q3+k*IQR)."""
    x = np.asarray(x, dtype=np.float64)
    q1, q3 = np.percentile(x, [25.0, 75.0])
    iqr = q3 - q1
    return (x < q1 - k * iqr) | (x > q3 + k * iqr)


def mahalanobis(features: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distance of each row of ``features`` (M, F) to the centre.

    Robustly standardises each column (median / MAD) before estimating the covariance,
    so a few outliers don't inflate the scale. Uses the pseudo-inverse for degenerate
    covariances. Returned distances can be thresholded (e.g. a chi-square cutoff, or a
    robust cutoff via :func:`mad_flags` on the distances).
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("features must be 2D (n_objects, n_features)")
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0) * 1.4826
    mad = np.where(mad > 0, mad, 1.0)
    z = (x - med) / mad
    cov = np.cov(z, rowvar=False)
    cov = np.atleast_2d(cov)
    inv = np.linalg.pinv(cov)
    centred = z - np.median(z, axis=0)
    return np.einsum("ij,jk,ik->i", centred, inv, centred)
