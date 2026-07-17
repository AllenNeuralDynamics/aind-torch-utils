"""Per-object metrics + statistical analysis for a huge instance-label OME-Zarr.

Companion to ``seeded_segmentation.py``. Two stages:

``extract`` — stream the uint32 label volume and the matching intensity volume in blocks
(reusing ``_prefetch_blocks``), compute **additive per-object reductions on the GPU**
(count, intensity power-sums, coordinate sums / second moments, intensity-weighted
coordinate sums, min/max, bounding box), and accumulate them per global id across blocks
(ids are globally unique, so summing partials is exact across block seams). Then derive
a per-object feature table (:func:`aind_torch_utils.object_stats.derive_metrics`) in
voxel + physical (µm) units and write ``objects.parquet`` / ``.csv``.

``analyze`` — load the table, flag outliers per-feature (robust MAD / IQR) and
multivariate (Mahalanobis), summarise distributions, and render ``report.pdf``
(histograms, correlation heatmap, ranked-outlier + feature-space + spatial pages).
With ``--gallery`` (and the label/intensity specs) it also reads image crops of the top
outliers and writes ``outlier_gallery.pdf`` (montage of the actual objects, mask
outlined) so they can be eyeballed at scale. Objects touching the volume border are
excluded from the fit (their metrics are truncated).

Metrics math + outlier stats live in ``aind_torch_utils.object_stats`` (pure numpy,
unit-tested); this script adds GPU streaming and I/O. Run from ``examples/`` (or with it
on ``PYTHONPATH``) so ``run_gfp_mask_example`` is importable. ``extract`` needs a GPU
(cupy + cupyx); ``analyze`` needs pandas + matplotlib (pyarrow for parquet, optional).

Usage
-----
python examples/object_metrics.py \
    --labels-spec labels_lvl1.json --intensity-spec intensity_lvl1.json \
    --out /scratch/metrics/DATASET --devices cuda:0 --max-id 5000000 --stage all
"""

import argparse
import logging
import os
import re
import sys

import numpy as np

# object_stats is pure numpy (needed by both stages). The GPU/S3 helpers from
# run_gfp_mask_example (which pulls torch/cupy) + tensorstore are imported lazily inside
# the extract functions so `--stage analyze` runs CPU-only (numpy/pandas/matplotlib).
from aind_torch_utils.object_stats import (
    SUM_KEYS,
    allocate_accumulators,
    derive_metrics,
    mahalanobis,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Feature columns used for distributions + outlier detection (scale-comparable scalars).
FEATURE_COLS = (
    "volume_um3",
    "equiv_diameter_um",
    "extent",
    "elongation",
    "flatness",
    "sphericity",
    "mean_intensity",
    "std_intensity",
    "cv_intensity",
    "skew_intensity",
    "integrated_intensity",
    "centroid_offset_um",
    "int_std_z_um",
    "int_std_y_um",
    "int_std_x_um",
)

# Strictly-positive, heavy-tailed features get a log x-axis in the report so the bulk of
# the distribution is visible instead of collapsing into the first linear bin. The rest
# (bounded or roughly symmetric) use a linear x-axis. All panels use a log y (count)
# axis so long tails stay visible.
LOG_X_FEATURES = (
    "volume_um3",
    "equiv_diameter_um",
    "integrated_intensity",
    "std_intensity",
    "cv_intensity",
    "int_std_z_um",
    "int_std_y_um",
    "int_std_x_um",
    "centroid_offset_um",
)


def _spec_level_and_group(spec):
    """Return (bucket, group_path, level) parsed from the spec's kvstore path."""
    from run_gfp_mask_example import _kvstore_bucket_path, _load_spec_dict

    bucket, kv_path, _ = _kvstore_bucket_path(_load_spec_dict(spec))
    kv_path = kv_path.rstrip("/")
    m = re.search(r"/(\d+)$", kv_path)
    if m is None:
        raise ValueError(f"kvstore path must end with a multiscale level: '{kv_path}'")
    return bucket, kv_path[: m.start()], m.group(1)


def _level_scale(spec):
    """Physical voxel size (z, y, x) for the spec's level, else (1, 1, 1)."""
    from run_gfp_mask_example import _read_source_multiscales, _scale_zyx

    bucket, group_path, level = _spec_level_and_group(spec)
    info = _read_source_multiscales(bucket, group_path)
    if info is not None:
        _, datasets, _ = info
        by_path = {d.get("path"): d for d in datasets}
        scale = _scale_zyx(by_path[level]) if level in by_path else None
        if scale is not None:
            return tuple(float(s) for s in scale)
    logger.warning("No scale transform found; using (1, 1, 1) voxel units.")
    return (1.0, 1.0, 1.0)


def _scan_max_id(store, cores, readahead):
    """One streaming pass to find the max label id (fallback when --max-id absent)."""
    import cupy as cp

    from run_gfp_mask_example import _prefetch_blocks

    mx = 0
    for _blk, data in _prefetch_blocks(store, cores, readahead, "scan-max-id"):
        blk_max = int(cp.asarray(data).max())
        mx = max(mx, blk_max)
    logger.info("Scanned max label id = %d", mx)
    return mx


def _accumulate_block(acc, blk, lab_host, int_host, cp, cndi):
    """Add one block's per-object partial reductions into the host accumulators."""
    z0, _, y0, _, x0, _ = blk
    # _prefetch_blocks reads src[0, 0, ...], so blocks are already 3D (dz, dy, dx).
    lab = cp.asarray(lab_host)
    fg = lab > 0
    if not bool(fg.any()):
        return
    inten = cp.asarray(int_host).astype(cp.float64)[fg]
    ids = lab[fg].astype(cp.int64)
    lz, ly, lx = cp.nonzero(fg)
    gz = lz.astype(cp.float64) + z0
    gy = ly.astype(cp.float64) + y0
    gx = lx.astype(cp.float64) + x0
    uid, inv = cp.unique(ids, return_inverse=True)
    k = int(uid.shape[0])
    idx = cp.arange(k)

    def bc(w):
        return cp.bincount(inv, weights=w, minlength=k)

    parts = {
        "sum_i": bc(inten),
        "sum_i2": bc(inten**2),
        "sum_i3": bc(inten**3),
        "sum_i4": bc(inten**4),
        "sum_z": bc(gz),
        "sum_y": bc(gy),
        "sum_x": bc(gx),
        "sum_zz": bc(gz * gz),
        "sum_yy": bc(gy * gy),
        "sum_xx": bc(gx * gx),
        "sum_zy": bc(gz * gy),
        "sum_zx": bc(gz * gx),
        "sum_yx": bc(gy * gx),
        "sum_iz": bc(inten * gz),
        "sum_iy": bc(inten * gy),
        "sum_ix": bc(inten * gx),
        "sum_izz": bc(inten * gz * gz),
        "sum_iyy": bc(inten * gy * gy),
        "sum_ixx": bc(inten * gx * gx),
    }
    uid_h = cp.asnumpy(uid)
    acc["count"][uid_h] += cp.asnumpy(cp.bincount(inv, minlength=k))
    for key in SUM_KEYS:
        acc[key][uid_h] += cp.asnumpy(parts[key])
    # min/max + bbox via labeled reductions, combined across blocks on the host.
    for key, arr, red in (
        ("min_i", inten, cndi.minimum),
        ("bz0", gz, cndi.minimum),
        ("by0", gy, cndi.minimum),
        ("bx0", gx, cndi.minimum),
    ):
        acc[key][uid_h] = np.minimum(acc[key][uid_h], cp.asnumpy(red(arr, inv, idx)))
    for key, arr, red in (
        ("max_i", inten, cndi.maximum),
        ("bz1", gz, cndi.maximum),
        ("by1", gy, cndi.maximum),
        ("bx1", gx, cndi.maximum),
    ):
        acc[key][uid_h] = np.maximum(acc[key][uid_h], cp.asnumpy(red(arr, inv, idx)))


def _save_df(df, out_dir, name, write_csv):
    """Write ``df`` as parquet (primary) and optional CSV; raise if parquet is missing.

    Parquet is the primary format because at 10M-200M objects a CSV is many GB and
    effectively unopenable. CSV is opt-in (``--write-csv``) for small runs only.
    """
    os.makedirs(out_dir, exist_ok=True)
    pq_path = os.path.join(out_dir, f"{name}.parquet")
    try:
        df.to_parquet(pq_path, index=False)
    except Exception as exc:  # pyarrow/fastparquet not installed
        raise SystemExit(
            f"Could not write parquet ({exc}). Install the parquet engine: "
            "`pip install pyarrow` (part of the 'metrics' extra). Use --write-csv only "
            "for small runs (CSV is multi-GB at scale)."
        )
    logger.info("Wrote %d rows -> %s", len(df), pq_path)
    if write_csv:
        csv_path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(csv_path, index=False)
        logger.info("Wrote %s", csv_path)


def _write_table(metrics, out_dir, write_csv):
    """Write the per-object metrics dict to parquet (primary) and optional CSV."""
    import pandas as pd

    df = pd.DataFrame(metrics)
    _save_df(df, out_dir, "objects", write_csv)
    return df


def _device_index(dev):
    """Parse a cupy device index from a 'cuda:N' (or bare 'N') string."""
    return int(dev.split(":")[1]) if ":" in dev else int(dev)


def _extract(args):
    """Stream label+intensity, accumulate per-object reductions, write the table."""
    import cupy as cp
    import cupyx.scipy.ndimage as cndi

    from aind_torch_utils.labeling import block_ranges
    from aind_torch_utils.utils import open_ts_spec
    from run_gfp_mask_example import _prefetch_blocks

    labels = open_ts_spec(args.labels_spec)
    inten = open_ts_spec(args.intensity_spec)
    lshape = tuple(int(s) for s in labels.domain.shape[-3:])
    ishape = tuple(int(s) for s in inten.domain.shape[-3:])
    if lshape != ishape:
        raise ValueError(f"label shape {lshape} != intensity shape {ishape}")
    scale = _level_scale(args.labels_spec)
    logger.info("Volume %s, voxel size (z,y,x)=%s µm", lshape, scale)

    nz, ny, nx = lshape
    cores = [
        (z0, z1, y0, y1, x0, x1)
        for (z0, z1) in block_ranges(nz, args.block)
        for (y0, y1) in block_ranges(ny, args.block)
        for (x0, x1) in block_ranges(nx, args.block)
    ]
    with cp.cuda.Device(_device_index(args.devices[0])):
        max_id = args.max_id or _scan_max_id(labels, cores, args.readahead)
        acc = allocate_accumulators(
            max_id + 1, np.float32 if args.float32_acc else np.float64
        )
        lab_reads = _prefetch_blocks(labels, cores, args.readahead, "metrics/labels")
        int_reads = _prefetch_blocks(inten, cores, args.readahead, "metrics/intensity")
        for (blk, lab_host), (_blk2, int_host) in zip(lab_reads, int_reads):
            _accumulate_block(acc, blk, lab_host, int_host, cp, cndi)
    metrics = derive_metrics(acc, scale, lshape)
    return _write_table(metrics, args.out, args.write_csv)


def _top_frac_flag(values, keep, frac):
    """Flag the top ``frac`` of ``values`` using a threshold from the border-free rows.

    A fixed fraction gives a bounded, predictable outlier count regardless of how heavy
    the tail is -- unlike a fixed robust-z cutoff, which flags a huge, tissue-wide set
    at 10M+ objects. The threshold is the ``1 - frac`` quantile of the kept
    (border-free) values; ties at the threshold are all included.
    """
    ref = values[keep] if keep.any() else values
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.zeros(values.shape, dtype=bool)
    thr = float(np.quantile(ref, 1.0 - frac))
    return np.isfinite(values) & (values >= thr) & (values > 0)


def _outlier_columns(df, cols, frac):
    """Per-feature robust-z + a top-``frac`` outlier flag, fit on border-free rows."""
    keep = ~df["touches_border"].to_numpy()
    any_flag = np.zeros(len(df), dtype=bool)
    for col in cols:
        if col not in df:
            continue
        x = df[col].to_numpy(dtype=np.float64)
        med = np.median(x[keep]) if keep.any() else np.median(x)
        mad = np.median(np.abs(x[keep] - med)) * 1.4826 if keep.any() else 0.0
        z = np.abs(x - med) / mad if mad > 0 else np.zeros_like(x)
        flag = _top_frac_flag(z, keep, frac)  # top-frac by deviation magnitude
        df[f"outlier_{col}"] = flag
        df[f"robust_z_{col}"] = z
        any_flag |= flag
    df["outlier_any_feature"] = any_flag
    return df


# Columns dropped from the Mahalanobis fit because they are near-collinear with a kept
# feature (correlation ~1), which would otherwise triple-count size/brightness and let
# the distance be dominated by object size: integrated_intensity ≈ volume_um3;
# int_std_{y,x} ≈ int_std_z; cv_intensity ≈ std_intensity.
MULTIVARIATE_DROP = (
    "integrated_intensity",
    "int_std_y_um",
    "int_std_x_um",
    "cv_intensity",
)


def _multivariate_outliers(df, cols, frac):
    """Add a Mahalanobis distance + a top-``frac`` multivariate outlier flag.

    Fits on a de-collinearised subset (see :data:`MULTIVARIATE_DROP`) so the distance is
    not dominated by the redundant size/brightness features; flags the top ``frac`` of
    the border-free d^2 distribution.
    """
    present = [c for c in cols if c in df and c not in MULTIVARIATE_DROP]
    # copy=True: to_numpy() may return a read-only view; we mutate columns below.
    x = df[present].to_numpy(dtype=np.float64, copy=True)
    # log1p the strictly-positive, heavy-tailed size/intensity features before fitting.
    for j, c in enumerate(present):
        if c.startswith(("volume", "integrated", "equiv")):
            x[:, j] = np.log1p(np.clip(x[:, j], 0, None))
    d2 = mahalanobis(x)
    df["mahalanobis_d2"] = d2
    keep = ~df["touches_border"].to_numpy()
    df["outlier_multivariate"] = _top_frac_flag(d2, keep, frac)
    return df


def _hist_panel(ax, col, vals):
    """Draw one metric histogram with robust x-limits + log scaling where it helps."""
    ax.set_title(col, fontsize=8)
    ax.tick_params(labelsize=6)
    if vals.size == 0:
        return
    # Robust x-range: clip to [0.5%, 99.5%] so a few extremes can't flatten the bulk.
    lo, hi = np.percentile(vals, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(vals.min()), float(vals.max())
    use_log_x = col in LOG_X_FEATURES and lo > 0 and hi > lo
    if use_log_x:
        bins = np.logspace(np.log10(lo), np.log10(hi), 60)
        ax.set_xscale("log")
    else:
        bins = np.linspace(lo, hi, 60) if hi > lo else 60
    clipped = vals[(vals >= lo) & (vals <= hi)]
    ax.hist(clipped if clipped.size else vals, bins=bins)
    ax.set_yscale("log")  # counts span orders of magnitude; keep the tail visible
    if hi > lo:
        ax.set_xlim(lo, hi)


# Informative, low-collinearity feature pairs for the outlier scatter page
# (name_x, name_y, log_x, log_y).
SCATTER_PAIRS = (
    ("volume_um3", "mean_intensity", True, False),
    ("equiv_diameter_um", "sphericity", True, False),
    ("elongation", "flatness", False, False),
    ("cv_intensity", "skew_intensity", True, False),
)
# Cap on scatter markers per plot. At scale there can be 100k+ flagged outliers; drawing
# each as a vector marker bloats the PDF until viewers can't open it. Subsample (and
# rasterize the dense layers) so the file stays small and openable.
PLOT_SCATTER_MAX = 20000


def _subsample(frame, max_n=PLOT_SCATTER_MAX):
    """Deterministically subsample a DataFrame to <= ``max_n`` rows for plotting."""
    if len(frame) <= max_n:
        return frame
    rng = np.random.default_rng(0)
    return frame.iloc[np.sort(rng.choice(len(frame), size=max_n, replace=False))]


def _page_mahalanobis_spatial(pdf, fit, plt):
    """Ranked multivariate score + a spatial map of where outliers sit."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 5))
    d2 = np.sort(fit["mahalanobis_d2"].to_numpy(dtype=np.float64))[::-1]
    idx = np.linspace(0, d2.size - 1, min(d2.size, 20000)).astype(int)
    axl.plot(idx, np.maximum(d2[idx], 1e-6), lw=0.8)
    axl.set_yscale("log")
    axl.set_xlabel("object rank")
    axl.set_ylabel("Mahalanobis d^2")
    axl.set_title("Ranked multivariate outlier score")
    flagged = fit[fit["outlier_multivariate"]]
    n_flag = len(flagged)
    if n_flag:
        thr = float(flagged["mahalanobis_d2"].min())
        axl.axhline(thr, color="r", ls="--", lw=0.8, label=f"{n_flag} flagged")
        axl.legend(fontsize=7)
    x = fit["centroid_x_um"].to_numpy(dtype=np.float64)
    y = fit["centroid_y_um"].to_numpy(dtype=np.float64)
    axr.hexbin(x, y, gridsize=120, cmap="Greys", bins="log", rasterized=True)
    if n_flag:
        pts = _subsample(flagged)
        sc = axr.scatter(
            pts["centroid_x_um"],
            pts["centroid_y_um"],
            c=pts["centroid_z_um"],
            s=6,
            cmap="autumn",
            edgecolor="k",
            linewidths=0.2,
            rasterized=True,
        )
        cb = fig.colorbar(sc, ax=axr, fraction=0.046)
        cb.set_label("centroid z (µm)", fontsize=7)
    axr.set_xlabel("x (µm)")
    axr.set_ylabel("y (µm)")
    axr.set_title(f"Outlier locations ({n_flag} flagged; grey = all-object density)")
    axr.set_aspect("equal")
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _page_scatter_pairs(pdf, fit, plt):
    """Density scatter of key feature pairs with multivariate outliers overplotted."""
    pairs = [p for p in SCATTER_PAIRS if p[0] in fit and p[1] in fit]
    if not pairs:
        return
    flagged = _subsample(fit[fit["outlier_multivariate"]])
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (cx, cy, logx, logy) in zip(np.ravel(axes), pairs):
        x = fit[cx].to_numpy(dtype=np.float64)
        y = fit[cy].to_numpy(dtype=np.float64)
        good = np.isfinite(x) & np.isfinite(y)
        kw = {}
        if logx and good.any() and (x[good] > 0).all():
            kw["xscale"] = "log"
        if logy and good.any() and (y[good] > 0).all():
            kw["yscale"] = "log"
        ax.hexbin(x[good], y[good], gridsize=80, cmap="Blues", bins="log", **kw)
        if len(flagged):
            ax.scatter(
                flagged[cx],
                flagged[cy],
                s=5,
                c="r",
                alpha=0.5,
                edgecolor="none",
                rasterized=True,
            )
        ax.set_xlabel(cx, fontsize=8)
        ax.set_ylabel(cy, fontsize=8)
        ax.tick_params(labelsize=6)
    n_pairs = len(pairs)
    for ax in np.ravel(axes)[n_pairs:]:
        ax.axis("off")
    fig.suptitle("Feature-space (density = border-free; red = multivariate outliers)")
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# A rate map bin needs at least this many objects before its outlier fraction is trusted
# (avoids a lone flagged object in a sparse bin reading as a 100% hot-spot).
RATE_MIN_COUNT = 20


def _page_outlier_rate(pdf, fit, plt):
    """Spatial outlier *rate* (flagged / total per bin) to reveal clustered anomalies.

    Unlike the scatter map (which just shows where flagged objects are, and is dominated
    by overall object density), the rate normalises by local density, so a genuinely bad
    tile/region stands out above the uniform baseline.
    """
    fl = fit["outlier_any_feature"].to_numpy(dtype=bool)
    baseline = float(fl.mean()) if fl.size else 0.0
    x = fit["centroid_x_um"].to_numpy(dtype=np.float64)
    y = fit["centroid_y_um"].to_numpy(dtype=np.float64)
    z = fit["centroid_z_um"].to_numpy(dtype=np.float64)
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 5))
    ht, xe, ye = np.histogram2d(x, y, bins=80)
    ho, _, _ = np.histogram2d(x[fl], y[fl], bins=[xe, ye])
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(ht >= RATE_MIN_COUNT, ho / ht, np.nan)
    im = axl.imshow(
        rate.T,
        origin="lower",
        extent=[xe[0], xe[-1], ye[0], ye[-1]],
        aspect="auto",
        cmap="magma",
    )
    fig.colorbar(im, ax=axl, fraction=0.046).set_label("outlier rate", fontsize=7)
    axl.set_xlabel("x (µm)")
    axl.set_ylabel("y (µm)")
    axl.set_title(f"Outlier rate x-y (baseline {baseline:.2%})")
    ht_z, ze = np.histogram(z, bins=100)
    ho_z, _ = np.histogram(z[fl], bins=ze)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate_z = np.where(ht_z >= RATE_MIN_COUNT, ho_z / ht_z, np.nan)
    axr.plot(0.5 * (ze[:-1] + ze[1:]), rate_z, lw=0.9)
    axr.axhline(baseline, color="r", ls="--", lw=0.8, label="baseline")
    axr.set_xlabel("z (µm)")
    axr.set_ylabel("outlier rate")
    axr.set_title("Outlier rate vs z")
    axr.legend(fontsize=7)
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _make_report(df, cols, out_dir):
    """Render histograms, correlation heatmap + outlier feature-space pages to a PDF."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    present = [c for c in cols if c in df]
    fit = df[~df["touches_border"]] if "touches_border" in df else df
    pdf_path = os.path.join(out_dir, "report.pdf")
    with PdfPages(pdf_path) as pdf:
        ncol = 3
        nrow = int(np.ceil(len(present) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
        for ax, col in zip(np.ravel(axes), present):
            vals = fit[col].to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            _hist_panel(ax, col, vals)
        flat_axes = np.ravel(axes)
        n_used = len(present)
        for ax in flat_axes[n_used:]:
            ax.axis("off")
        fig.suptitle("Per-metric distributions (border objects excluded)")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        corr = fit[present].corr().to_numpy()
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present, rotation=90, fontsize=6)
        ax.set_yticks(range(len(present)))
        ax.set_yticklabels(present, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title("Feature correlation")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if "mahalanobis_d2" in fit:
            # A failing optional page must not abort the whole PDF (leaving it
            # truncated/unopenable); log and keep the pages already written.
            pages = (
                _page_mahalanobis_spatial,
                _page_scatter_pairs,
                _page_outlier_rate,
            )
            for page in pages:
                try:
                    page(pdf, fit, plt)
                except Exception as exc:
                    logger.warning("report page %s failed: %s", page.__name__, exc)
    logger.info("Wrote %s", pdf_path)
    return pdf_path


def _axis_window(center, extent, margin, max_crop, n):
    """One axis' [lo, hi) crop window around ``center`` (voxels), clamped to [0, n)."""
    span = int(max(1, min(int(extent) + 2 * margin, max_crop)))
    lo = int(round(center - span / 2.0))
    lo = max(0, min(lo, n - span)) if n >= span else 0
    return lo, min(n, lo + span)


def _object_window(row, scale, args, shape):
    """(z0,z1,y0,y1,x0,x1) crop bounds for one object row (bbox origin if present)."""
    nz, ny, nx = shape
    keys = ("bbox_z0_vox", "bbox_y0_vox", "bbox_x0_vox")
    have = all(k in row and np.isfinite(row[k]) for k in keys)

    def center(o_key, d_key, c_key, s):
        d = float(row[d_key])
        return float(row[o_key]) + d / 2.0 if have else float(row[c_key]) / s

    cz = center("bbox_z0_vox", "bbox_dz_vox", "centroid_z_um", scale[0])
    cy = center("bbox_y0_vox", "bbox_dy_vox", "centroid_y_um", scale[1])
    cx = center("bbox_x0_vox", "bbox_dx_vox", "centroid_x_um", scale[2])
    mg, mc = args.gallery_margin, args.gallery_max_crop
    z0, z1 = _axis_window(cz, row["bbox_dz_vox"], mg, mc, nz)
    y0, y1 = _axis_window(cy, row["bbox_dy_vox"], mg, mc, ny)
    x0, x1 = _axis_window(cx, row["bbox_dx_vox"], mg, mc, nx)
    return z0, z1, y0, y1, x0, x1


def _read_crop(labels_store, inten_store, win):
    """Read the label + intensity crop for a window from the 5D (1,1,Z,Y,X) stores."""
    z0, z1, y0, y1, x0, x1 = win
    lab = labels_store[0, 0, z0:z1, y0:y1, x0:x1].read().result()
    inten = inten_store[0, 0, z0:z1, y0:y1, x0:x1].read().result()
    return np.asarray(lab), np.asarray(inten)


def _render_tile(ax, lab, inten, obj_id, title):
    """Draw one object's z-max-projection with its mask outlined and a caption."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=5)
    if inten.size == 0:
        return
    mip = inten.max(axis=0).astype(np.float64)
    lo, hi = np.percentile(mip, [1.0, 99.5])
    ax.imshow(mip, cmap="gray", vmin=lo, vmax=max(hi, lo + 1.0))
    mmask = (lab == obj_id).any(axis=0)
    if mmask.any():
        ax.contour(mmask.astype(float), levels=[0.5], colors="r", linewidths=0.5)


def _gallery_sections(df, cols, topn, per_feature):
    """Build (title, rows, value_column) sections: multivariate top-N + per-feature."""
    fit = df[~df["touches_border"]] if "touches_border" in df else df
    sections = []
    if "mahalanobis_d2" in fit and len(fit):
        mv = fit.nlargest(topn, "mahalanobis_d2")
        title = f"Multivariate top {len(mv)} (Mahalanobis)"
        sections.append((title, mv, "mahalanobis_d2"))
    for c in cols:
        zc = f"robust_z_{c}"
        if zc not in fit:
            continue
        top = fit.nlargest(per_feature, zc)
        top = top[top[zc] > 0]
        if len(top):
            sections.append((f"{c} extremes", top, c))
    return sections


def _render_section(pdf, plt, section, stores, scale, args, shape):
    """Render one montage page: a grid of crops for the section's rows."""
    title, rows, valcol = section
    n = len(rows)
    ncol = 8
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(1.6 * ncol, 1.75 * nrow), squeeze=False
    )
    flat = np.ravel(axes)
    for ax, (_, row) in zip(flat, rows.iterrows()):
        try:
            lab, inten = _read_crop(*stores, _object_window(row, scale, args, shape))
            cap = f"id {int(row['id'])}\n{valcol}={float(row[valcol]):.2g}"
            _render_tile(ax, lab, inten, int(row["id"]), cap)
        except Exception as exc:  # a bad/oversized read must not kill the whole gallery
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, "read err", fontsize=5, ha="center", va="center")
            logger.warning("gallery tile id=%s failed: %s", row.get("id"), exc)
    for ax in flat[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _make_gallery(df, cols, args):
    """Render image crops of the top outliers (multivariate + per-feature) to a PDF."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    from aind_torch_utils.utils import open_ts_spec

    labels_store = open_ts_spec(args.labels_spec)
    inten_store = open_ts_spec(args.intensity_spec)
    shape = tuple(int(s) for s in labels_store.domain.shape[-3:])
    scale = _level_scale(args.labels_spec)
    sections = _gallery_sections(df, cols, args.gallery_topn, args.gallery_per_feature)
    pdf_path = os.path.join(args.out, "outlier_gallery.pdf")
    with PdfPages(pdf_path) as pdf:
        for section in sections:
            _render_section(
                pdf, plt, section, (labels_store, inten_store), scale, args, shape
            )
    logger.info("Wrote %s (%d sections)", pdf_path, len(sections))
    return pdf_path


def _analyze(args):
    """Load the table, flag outliers, summarise distributions, write the report."""
    import pandas as pd

    base = os.path.join(args.out, "objects")
    if os.path.exists(base + ".parquet"):
        df = pd.read_parquet(base + ".parquet")
    else:
        df = pd.read_csv(base + ".csv")
    cols = [c for c in FEATURE_COLS if c in df]
    df = _outlier_columns(df, cols, args.outlier_frac)
    df = _multivariate_outliers(df, cols, args.outlier_frac)

    fit = df[~df["touches_border"]] if "touches_border" in df else df
    summary = fit[cols].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    summary["mad"] = [
        np.median(np.abs(v - np.median(v))) * 1.4826
        for v in (fit[c].to_numpy(dtype=np.float64) for c in cols)
    ]
    summary.to_csv(os.path.join(args.out, "distribution_summary.csv"))
    _make_report(df, cols, args.out)

    n_out = int(df["outlier_any_feature"].sum())
    n_mv = int(df["outlier_multivariate"].sum())
    logger.info(
        "Analyzed %d objects (%d border): top-%.3g%% -> %d per-feature (any), "
        "%d multivariate outliers.",
        len(df),
        int(df["touches_border"].sum()) if "touches_border" in df else 0,
        args.outlier_frac * 100.0,
        n_out,
        n_mv,
    )
    _save_df(df, args.out, "objects_analyzed", args.write_csv)
    if args.gallery:
        if not args.labels_spec or not args.intensity_spec:
            raise SystemExit(
                "--gallery needs --labels-spec and --intensity-spec to read crops."
            )
        _make_gallery(df, cols, args)
    return df


def _parse_args(argv):
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Per-object metrics + statistical analysis for an instance-label "
        "OME-Zarr."
    )
    ap.add_argument(
        "--labels-spec",
        help="TensorStore spec: uint32 labels (required for --stage extract/all).",
    )
    ap.add_argument(
        "--intensity-spec",
        help="TensorStore spec: matching intensity (required for "
        "--stage extract/all).",
    )
    ap.add_argument("--out", required=True, help="Output directory for tables/report.")
    ap.add_argument("--devices", nargs="+", default=["cuda:0"])
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--readahead", type=int, default=8)
    ap.add_argument(
        "--max-id",
        type=int,
        default=0,
        help="Max object id (e.g. #seeds). 0 = scan the volume first (extra pass).",
    )
    ap.add_argument(
        "--float32-acc",
        action="store_true",
        help="Use float32 accumulators (halves RAM; slight precision loss at scale).",
    )
    ap.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write CSV beside parquet (small runs only; CSV is multi-GB).",
    )
    ap.add_argument(
        "--gallery",
        action="store_true",
        help="Render image crops of top outliers (needs the label/intensity specs).",
    )
    ap.add_argument(
        "--gallery-topn",
        type=int,
        default=48,
        help="Number of multivariate (Mahalanobis) outliers in the gallery.",
    )
    ap.add_argument(
        "--gallery-per-feature",
        type=int,
        default=6,
        help="Number of per-feature extreme outliers shown per metric.",
    )
    ap.add_argument(
        "--gallery-margin",
        type=int,
        default=4,
        help="Extra voxels around each object's bbox in its crop.",
    )
    ap.add_argument(
        "--gallery-max-crop",
        type=int,
        default=128,
        help="Cap each crop axis (voxels); large objects are centre-cropped.",
    )
    ap.add_argument(
        "--outlier-frac",
        type=float,
        default=0.0005,
        help="Flag the top fraction of border-free objects as outliers (per feature "
        "and multivariate). Bounded, predictable count. Default 0.0005 = top 0.05%%.",
    )
    ap.add_argument(
        "--stage",
        choices=["extract", "analyze", "all"],
        default="all",
        help="extract (stream->table), analyze (table->flags+report), or all.",
    )
    return ap.parse_args(argv)


def main(argv=None):
    """Run the extract and/or analyze stage."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.stage in ("extract", "all"):
        if not args.labels_spec or not args.intensity_spec:
            raise SystemExit(
                "--labels-spec and --intensity-spec are required for the extract stage."
            )
        _extract(args)
    if args.stage in ("analyze", "all"):
        _analyze(args)


if __name__ == "__main__":
    main()
