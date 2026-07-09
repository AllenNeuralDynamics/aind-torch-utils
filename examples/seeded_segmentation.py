"""Seeded instance segmentation from a points array (marker-controlled watershed).

Standalone companion to ``run_gfp_mask_example.py`` for when you already have a
**point per object** (e.g. billions of bouton centers in a numpy ``(N, 3)`` array).
Instead of thresholding intensity (which fails on dim/uneven objects), it grows each
object from its seed with a per-chunk marker-controlled watershed and writes a
``uint32`` instance-label OME-Zarr (each object's voxels carry its point's row index
+ 1) plus a label-preserving pyramid.

Because objects are small, each block + halo is segmented **independently** — the
seed's array index is its global instance ID, so there is no cross-block merging /
union-find. The flat-field, read-prefetch, and pyramid machinery is reused from
``run_gfp_mask_example`` (which is imported, not modified).

Points are ``(N, 3)`` ``(z, y, x)`` **voxel** coords at ``--seeds-level``; they are
rescaled (center-aligned, anisotropy-aware) to the segmented level from ``--in-spec``.

Usage
-----
python examples/seeded_segmentation.py \
    --in-spec seg_spec_lvl0.json --seeds points.npy --seeds-level 3 \
    --out-bucket my-bucket --out-prefix predictions/boutons/DATASET \
    --params-json params.json --devices cuda:0 \
    --ccl-block 512 --ccl-readahead 16 --watershed-halo 64 \
    --seed-flood-threshold 0.05 --watershed-surface edt

Requires the ``postprocess`` stack (cupy + cucim). Run from ``examples/`` (or with it on
``PYTHONPATH``) so ``run_gfp_mask_example`` is importable.
"""

import argparse
import logging
import re
import sys

import numpy as np
import tensorstore as ts

from aind_torch_utils.config import InferenceConfig  # noqa: E402
from aind_torch_utils.correction import (  # noqa: E402
    apply_flatfield,
    normalize_global,
    sample_background,
)
from aind_torch_utils.labeling import (  # noqa: E402
    block_ranges,
    bucket_points,
    rescale_points,
    region_seeds,
)
from aind_torch_utils.utils import open_ts_spec  # noqa: E402

from run_gfp_mask_example import (  # noqa: E402
    LABEL_DOWNSAMPLE_METHOD,
    _build_mask_pyramid,
    _estimate_background,
    _kvstore_bucket_path,
    _load_spec_dict,
    _normalization_kwargs,
    _prefetch_blocks,
    _read_source_multiscales,
    _scale_zyx,
    _upsample_mask_pyramid,
    _write_output_group_metadata,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _spec_level_and_group(spec):
    """Return (bucket, group_path, level) parsed from the spec's kvstore path."""
    bucket, kv_path, _ = _kvstore_bucket_path(_load_spec_dict(spec))
    kv_path = kv_path.rstrip("/")
    m = re.search(r"/(\d+)$", kv_path)
    if m is None:
        raise ValueError(
            f"kvstore path must end with a multiscale level, got '{kv_path}'"
        )
    return bucket, kv_path[: m.start()], m.group(1)


def _device_index(dev: str) -> int:
    """Parse a cupy device index from a 'cuda:N' (or bare 'N') string."""
    return int(dev.split(":")[1]) if ":" in dev else int(dev)


def _open_uint32_store(bucket, base_path, path, shape, chunks, region):
    """Create (or reset) a uint32 zarr2 label array on S3."""
    return ts.open(
        {
            "driver": "zarr",
            "kvstore": {
                "driver": "s3",
                "bucket": bucket,
                "path": f"{base_path}{path}",
                "aws_region": region,
            },
            "metadata": {
                "shape": list(shape),
                "chunks": list(chunks),
                "dtype": "<u4",  # little-endian uint32 instance IDs
                "dimension_separator": "/",
            },
            "create": True,
            "delete_existing": True,
        }
    ).result()


def _expand(core, halo, nz, ny, nx):
    """Halo-expanded block bbox, clamped to the volume."""
    z0, z1, y0, y1, x0, x1 = core
    return (
        max(z0 - halo, 0),
        min(z1 + halo, nz),
        max(y0 - halo, 0),
        min(y1 + halo, ny),
        max(x0 - halo, 0),
        min(x1 + halo, nx),
    )


def _seeded_watershed(args, base_path, level, in_store, cell_points, bg_field, cfg):
    """Per-chunk marker-controlled watershed from points -> uint32 instance labels.

    For each block + halo: flat-field + normalize the intensity, build a foreground mask
    (permissive threshold OR a sphere around each seed), rasterize the chunk's seeds as
    local markers, run watershed (``-EDT`` or ``-intensity``, masked to foreground), map
    local labels back to the seeds' global IDs, crop to the core, and write. Halo seeds
    give cross-boundary objects the right basin; core-crop => each object written once.
    """
    import cupy as cp
    import cupyx.scipy.ndimage as cndi
    from cucim.skimage.segmentation import watershed

    block, halo, region = args.ccl_block, args.watershed_halo, args.aws_region
    nz, ny, nx = (int(s) for s in in_store.domain.shape[-3:])

    cores = [
        (z0, z1, y0, y1, x0, x1)
        for (z0, z1) in block_ranges(nz, block)
        for (y0, y1) in block_ranges(ny, block)
        for (x0, x1) in block_ranges(nx, block)
    ]
    exps = [_expand(c, halo, nz, ny, nx) for c in cores]

    cz, cy, cx = min(128, nz), min(128, ny), min(128, nx)
    out = _open_uint32_store(
        args.out_bucket,
        base_path,
        f"{level}/",
        (1, 1, nz, ny, nx),
        (1, 1, cz, cy, cx),
        region,
    )

    n_objects = 0
    reads = _prefetch_blocks(in_store, exps, args.ccl_readahead, "seeded")
    for (exp, data), core in zip(reads, cores):
        ez0, ez1, ey0, ey1, ex0, ex1 = exp

        # Flat-field + global-normalize the (expanded) intensity, like PrepWorker.
        vol = data.astype(np.float32, copy=False)
        if cfg.flatfield and bg_field is not None:
            bg = sample_background(
                bg_field.field, bg_field.scale, ez0, ez1, ey0, ey1, ex0, ex1
            )
            vol = apply_flatfield(
                vol, bg, mode=cfg.flatfield_mode, eps=cfg.eps, bg_mean=bg_field.mean
            )
        norm = cp.asarray(
            normalize_global(vol, cfg.norm_lower, cfg.norm_upper, cfg.eps)
        )

        # Seeds in this block+halo -> local markers 1..k, with a local->global ID map.
        coords, ids = region_seeds(cell_points, block, exp)
        k = int(ids.shape[0])
        markers = cp.zeros(norm.shape, dtype=cp.int32)
        local_to_global = np.zeros(k + 1, dtype=np.uint32)
        if k:
            lz = cp.asarray(coords[:, 0] - ez0)
            ly = cp.asarray(coords[:, 1] - ey0)
            lx = cp.asarray(coords[:, 2] - ex0)
            markers[lz, ly, lx] = cp.arange(1, k + 1, dtype=cp.int32)
            local_to_global[1:] = ids
            n_objects += k

        # Foreground: permissive threshold, plus a sphere around each seed so dim
        # objects (below threshold) still have a growable core.
        fg = norm >= args.seed_flood_threshold
        if k and args.seed_sphere_radius > 0:
            fg = fg | cndi.binary_dilation(
                markers > 0, iterations=int(args.seed_sphere_radius)
            )

        surface = (
            -cndi.distance_transform_edt(fg)
            if args.watershed_surface == "edt"
            else -norm
        )
        ws = watershed(surface, markers=markers, mask=fg)  # local labels {0..k}
        labels = cp.asarray(local_to_global)[ws]  # -> global uint32 IDs

        z0, z1, y0, y1, x0, x1 = core
        # Core's local offset within the expanded block, hoisted to simple slice bounds.
        lz0, ly0, lx0 = z0 - ez0, y0 - ey0, x0 - ex0
        lz1, ly1, lx1 = lz0 + (z1 - z0), ly0 + (y1 - y0), lx0 + (x1 - x0)
        crop = labels[lz0:lz1, ly0:ly1, lx0:lx1]
        out[0, 0, z0:z1, y0:y1, x0:x1].write(cp.asnumpy(crop)).result()
        del vol, norm, markers, fg, surface, ws, labels, crop

    logger.info(
        "[seeded] wrote uint32 instance labels to %s%s/ (%d seeded objects)",
        base_path,
        level,
        n_objects,
    )


def _build_label_pyramid(args, base_path, spec, level, start, datasets, source_ms):
    """Coarser levels (+ optional finer) + OME metadata for a uint32 label volume."""
    datasets_from_l = datasets[start:]
    _build_mask_pyramid(
        args.out_bucket,
        base_path,
        datasets_from_l,
        args.aws_region,
        concurrency=args.pyramid_concurrency,
        copy_concurrency=args.pyramid_copy_concurrency,
        reducer=LABEL_DOWNSAMPLE_METHOD,
        dtype="<u4",
    )
    fill_finer = args.fill_finer_levels and start > 0
    if fill_finer:
        _upsample_mask_pyramid(
            args.out_bucket,
            base_path,
            spec,
            datasets[: start + 1],
            args.aws_region,
            concurrency=args.pyramid_concurrency,
            copy_concurrency=args.pyramid_copy_concurrency,
            processes=args.pyramid_processes,
            dtype="<u4",
        )
    meta_datasets = datasets if fill_finer else datasets_from_l
    _write_output_group_metadata(
        args.out_bucket,
        base_path,
        source_ms,
        meta_datasets,
        None,  # labels are not intensity; no omero window
        args.aws_region,
        reducer=LABEL_DOWNSAMPLE_METHOD,
    )


def _parse_args(argv):
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Seeded instance segmentation from a points array (watershed)."
    )
    ap.add_argument(
        "--in-spec",
        required=True,
        help="TensorStore spec; level = trailing digit of kvstore path.",
    )
    ap.add_argument("--seeds", required=True, help="points.npy, (N,3) (z,y,x) voxels.")
    ap.add_argument(
        "--seeds-level",
        type=int,
        required=True,
        help="Pyramid level the points are in (rescaled to the seg level).",
    )
    ap.add_argument("--out-bucket", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--aws-region", default="us-west-2")
    ap.add_argument(
        "--params-json",
        default=None,
        help="JSON with normalize/flat-field params (same keys as the mask "
        "pipeline).",
    )
    ap.add_argument(
        "--devices",
        nargs="+",
        default=["cuda:0"],
        help="cupy device(s); only the first is used.",
    )
    ap.add_argument(
        "--ccl-block",
        type=int,
        default=512,
        help="Block (chunk) size for the per-chunk watershed.",
    )
    ap.add_argument(
        "--ccl-readahead",
        type=int,
        default=8,
        help="Concurrent block reads prefetched (S3-latency hiding).",
    )
    ap.add_argument(
        "--watershed-halo",
        type=int,
        default=64,
        help="Halo around each block; must be >= the largest object "
        "diameter for seam-correct, union-find-free segmentation.",
    )
    ap.add_argument(
        "--seed-flood-threshold",
        type=float,
        default=0.05,
        help="Normalized intensity threshold for the watershed foreground.",
    )
    ap.add_argument(
        "--watershed-surface",
        choices=["edt", "intensity"],
        default="edt",
        help="Flood surface: 'edt' (shape, robust) or 'intensity'.",
    )
    ap.add_argument(
        "--seed-sphere-radius",
        type=int,
        default=-1,
        help="Dilate each seed by this radius into the foreground mask so "
        "dim objects have a growable core. -1 = auto (factor/2 + 1).",
    )
    ap.add_argument(
        "--no-pyramid",
        action="store_true",
        help="Write only the segmented level (no pyramid / OME metadata).",
    )
    ap.add_argument(
        "--fill-finer-levels",
        action="store_true",
        help="Also nearest-upsample to finer levels (0..L-1).",
    )
    ap.add_argument("--pyramid-concurrency", type=int, default=64)
    ap.add_argument("--pyramid-copy-concurrency", type=int, default=16)
    ap.add_argument("--pyramid-processes", type=int, default=1)
    return ap.parse_args(argv)


def main(argv=None):
    """Rescale points, then per-chunk seeded watershed -> uint32 instance OME-Zarr."""
    import cupy as cp

    args = _parse_args(sys.argv[1:] if argv is None else argv)
    spec = args.in_spec

    bucket, group_path, level = _spec_level_and_group(spec)
    in_store = open_ts_spec(spec)
    shape = tuple(int(s) for s in in_store.domain.shape[-3:])
    logger.info("Input s3://%s/%s level %s shape=%s", bucket, group_path, level, shape)

    tile_name = group_path.rstrip("/").rsplit("/", 1)[-1]
    base_path = f"{args.out_prefix.rstrip('/')}/{tile_name}/"

    ms_info = _read_source_multiscales(bucket, group_path)
    if ms_info is None:
        raise ValueError("No OME multiscales metadata on the source group.")
    source_ms, datasets, _omero = ms_info
    by_path = {d.get("path"): d for d in datasets}
    for lvl in (level, str(args.seeds_level)):
        if lvl not in by_path:
            raise ValueError(
                f"Level '{lvl}' not among source datasets {list(by_path)}."
            )
    start = [d.get("path") for d in datasets].index(level)

    # Per-axis rescale factor from seeds-level to seg-level voxels.
    s_seeds = _scale_zyx(by_path[str(args.seeds_level)])
    s_seg = _scale_zyx(by_path[level])
    if s_seeds is None or s_seg is None:
        raise ValueError("Source datasets are missing scale transforms.")
    factor = [s_seeds[a] / s_seg[a] for a in range(3)]
    if args.seed_sphere_radius < 0:
        args.seed_sphere_radius = int(round(max(factor) / 2)) + 1
    logger.info(
        "Points @ level %s -> seg level %s, factor(z,y,x)=%s, seed-sphere r=%d",
        args.seeds_level,
        level,
        tuple(round(f, 3) for f in factor),
        args.seed_sphere_radius,
    )

    # Load points, rescale, index by block cell.
    pts = np.load(args.seeds)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"--seeds must be (N,3) (z,y,x); got shape {pts.shape}.")
    ids = np.arange(1, pts.shape[0] + 1, dtype=np.uint32)  # ID = row index + 1
    coords = rescale_points(pts, factor)
    cell_points = bucket_points(coords, ids, args.ccl_block, shape)
    logger.info(
        "Loaded %d points; %d block cells populated.", pts.shape[0], len(cell_points)
    )

    cfg = InferenceConfig(**_normalization_kwargs(args.params_json))

    with cp.cuda.Device(_device_index(args.devices[0])):
        bg_field = _estimate_background(spec, cfg, shape)
        _seeded_watershed(args, base_path, level, in_store, cell_points, bg_field, cfg)

    if args.no_pyramid:
        logger.info("Done (single level %s). Skipped pyramid (--no-pyramid).", level)
        return
    _build_label_pyramid(args, base_path, spec, level, start, datasets, source_ms)
    logger.info(
        "Done. Instance-label OME-Zarr at s3://%s/%s", args.out_bucket, base_path
    )


if __name__ == "__main__":
    main()
