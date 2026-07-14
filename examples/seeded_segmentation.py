"""Seeded instance segmentation from a points array (GPU EDT nearest-seed labeling).

Standalone companion to ``run_gfp_mask_example.py`` for when you already have a
**point per object** (e.g. billions of bouton centers in a numpy ``(N, 3)`` array).
Instead of one global intensity threshold (which fails on dim/uneven objects), it
optionally Gaussian-denoises the normalized intensity (``--smooth-sigma``, "smooth then
segment"), assigns every voxel to its nearest seed via the Euclidean distance transform
(a fully-GPU stand-in for a ``-EDT`` marker-controlled watershed, since cuCIM has no
watershed), and carves each blob at a fraction of *its own seed's* brightness
(``--seed-relative-threshold``) — so bright and dim blobs hug tightly and the background
between seeds is excluded. Morphological cleanup (``--close-iterations`` /
``--fill-holes`` / ``--dilate-iterations``) makes objects solid and well-formed, and a
size filter (``--min-object-size`` / ``--max-object-size``) drops specks and over-grown
blobs. Optionally, ``--merge-core-threshold`` merges several seeds sharing one bright
blob into a single object (seeds separated by an intensity valley still split). It
writes a ``uint32`` instance-label OME-Zarr (each object's voxels carry its point's row
index + 1) plus a label-preserving pyramid. (``params.json`` supplies only normalize/
flat-field keys; its ``smooth_sigma`` is a model param, ignored here — use
``--smooth-sigma``.)

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
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

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
    merge_seed_groups,
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


def _norm_sigma(values):
    """Normalize --smooth-sigma to a (z, y, x) tuple (broadcast a single value)."""
    if len(values) == 1:
        return tuple(values) * 3
    if len(values) == 3:
        return tuple(values)
    raise SystemExit("--smooth-sigma takes 1 value (isotropic) or 3 (z y x).")


def _size_keep(assigned, k, min_size, max_size, cp):
    """Boolean keep[0..k]: local labels whose voxel count is within [min, max].

    ``assigned`` holds the per-voxel local object label (0 = background). Counts are
    over the (cleaned) block+halo, which is the full object for the owning chunk.
    """
    counts = cp.bincount(assigned.ravel(), minlength=k + 1)
    keep = cp.ones(k + 1, dtype=bool)
    if min_size > 0:
        keep &= counts >= min_size
    if max_size > 0:
        keep &= counts <= max_size
    keep[0] = False  # background never kept
    return keep


def _clean_mask(mask, cndi, close_iters, fill_holes, dilate_iters):
    """Morphological cleanup so objects look solid and well-formed.

    Closing bridges small gaps and smooths edges, hole-filling solidifies interiors,
    and dilation makes objects fuller. Applied to the binary carve before the
    nearest-seed relabel, so touching objects still split at the Voronoi midplane
    (boundaries stay crisp) even if closing/dilation bridges them.
    """
    if close_iters > 0:
        mask = cndi.binary_closing(mask, iterations=int(close_iters), brute_force=True)
    if fill_holes:
        mask = cndi.binary_fill_holes(mask)
    if dilate_iters > 0:
        mask = cndi.binary_dilation(
            mask, iterations=int(dilate_iters), brute_force=True
        )
    return mask


def _prep_block(data, exp, cell_points, block, cfg, bg_field):
    """Host-side flat-field + normalize + seed extraction (runs off the GPU thread).

    Returns ``(norm_host, coords, ids)`` so the GPU consumer only uploads and computes.
    """
    ez0, ez1, ey0, ey1, ex0, ex1 = exp
    vol = data.astype(np.float32, copy=False)
    if cfg.flatfield and bg_field is not None:
        bg = sample_background(
            bg_field.field, bg_field.scale, ez0, ez1, ey0, ey1, ex0, ex1
        )
        vol = apply_flatfield(
            vol, bg, mode=cfg.flatfield_mode, eps=cfg.eps, bg_mean=bg_field.mean
        )
    norm_host = normalize_global(vol, cfg.norm_lower, cfg.norm_upper, cfg.eps)
    coords, ids = region_seeds(cell_points, block, exp)
    return norm_host, coords, ids


def _seed_groups(mask, norm, lz, ly, lx, local_to_global, threshold, cp, cndi):
    """Group seeds sharing a bright intensity core -> (local_to_group, group_to_global).

    A "core" is ``mask & (norm >= threshold)``. Seeds whose cores are connected at that
    (higher) level merge into one object; seeds separated by an intensity valley (or
    dimmer than ``threshold``) stay separate. See :func:`labeling.merge_seed_groups`.
    """
    core_comp, _ = cndi.label(mask & (norm >= threshold))
    seed_core = cp.asnumpy(core_comp[lz, ly, lx])
    return merge_seed_groups(seed_core, local_to_global[1:])


def _segment_block(norm, coords, ids, exp, args, cp, cndi):
    """GPU per-block: markers -> nearest-seed carve -> morphology -> size filter.

    ``norm`` is the (smoothed) normalized intensity already on the GPU. Returns
    ``(labels_uint32, k)`` for the full expanded block; the caller crops to the core.
    cuCIM has no watershed, so each voxel takes its nearest seed via the Euclidean
    feature transform (== the ``-EDT`` watershed for point markers).
    """
    ez0, ey0, ex0 = exp[0], exp[2], exp[4]
    k = int(ids.shape[0])
    labels = cp.zeros(norm.shape, dtype=cp.uint32)
    if not k:
        return labels, 0
    markers = cp.zeros(norm.shape, dtype=cp.int32)
    local_to_global = np.zeros(k + 1, dtype=np.uint32)
    lz = cp.asarray(coords[:, 0] - ez0)
    ly = cp.asarray(coords[:, 1] - ey0)
    lx = cp.asarray(coords[:, 2] - ex0)
    markers[lz, ly, lx] = cp.arange(1, k + 1, dtype=cp.int32)
    local_to_global[1:] = ids
    # return_distances=False -> the indices array is returned alone (no tuple).
    inds = cndi.distance_transform_edt(
        markers == 0, return_distances=False, return_indices=True
    )
    nearest = markers[inds[0], inds[1], inds[2]]  # nearest seed's local label
    seed_int = norm[inds[0], inds[1], inds[2]]  # nearest seed's intensity
    # Carve at a fraction of the nearest seed's brightness + an absolute floor.
    mask = (norm >= args.seed_relative_threshold * seed_int) & (
        norm >= args.seed_flood_threshold
    )
    if args.seed_sphere_radius > 0:  # guarantee a core at each seed
        mask = mask | cndi.binary_dilation(
            markers > 0,
            iterations=int(args.seed_sphere_radius),
            brute_force=True,  # cupyx implements only brute_force here
        )
    # Solidify/smooth so objects look full with clear boundaries.
    mask = _clean_mask(
        mask, cndi, args.close_iterations, args.fill_holes, args.dilate_iterations
    )
    # Label the CARVED mask into connected components.
    comp, _ = cndi.label(mask)
    # A voxel may only take a seed that lies in ITS OWN component: the nearest seed's
    # component must equal the voxel's. This keeps every ID inside one connected
    # component (no leaking across a background gap into a neighbour), still splits
    # touching objects among their own seeds, and drops no-seed blobs (their nearest
    # seed is foreign).
    comp_of_nearest = comp[inds[0], inds[1], inds[2]]
    assigned = cp.where(mask & (comp == comp_of_nearest), nearest, cp.int32(0))
    # Map local seed labels -> compact object "groups": identity (one object per seed),
    # or merged so seeds sharing a bright intensity core become one object.
    if args.merge_core_threshold > 0:
        local_to_group, group_to_global = _seed_groups(
            mask, norm, lz, ly, lx, local_to_global, args.merge_core_threshold, cp, cndi
        )
    else:
        local_to_group = np.arange(k + 1, dtype=np.int64)
        group_to_global = local_to_global
    assigned = cp.asarray(local_to_group)[assigned]  # local label -> group index
    if args.min_object_size > 0 or args.max_object_size > 0:
        n_groups = group_to_global.shape[0] - 1  # size filter counts the whole object
        keep = _size_keep(
            assigned, n_groups, args.min_object_size, args.max_object_size, cp
        )
        assigned = cp.where(keep[assigned], assigned, cp.int32(0))
    return (
        cp.asarray(group_to_global)[assigned],
        k,
    )  # group -> global uint32 (0 stays 0)


def _write_block(out, core, data):
    """Write the cropped label block to its core region (runs in a writer thread)."""
    z0, z1, y0, y1, x0, x1 = core
    out[0, 0, z0:z1, y0:y1, x0:x1].write(data).result()


def _seeded_segment(args, base_path, level, in_store, cell_points, bg_field, cfg):
    """Per-chunk GPU EDT nearest-seed labeling from points -> uint32 instance labels.

    Pipelined to keep the GPU busy: a producer thread does the host flat-field/normalize
    (numpy is CPU-only) for the next block while the GPU processes the current one, and
    a writer pool does the S3 writes in the background. The GPU consumer loop only
    uploads, runs :func:`_segment_block`, and hands the cropped core off to be written.
    """
    import cupy as cp
    import cupyx.scipy.ndimage as cndi

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

    # Producer: prefetched read + host prep -> bounded queue (overlaps the GPU work).
    ready = Queue(maxsize=max(2, args.ccl_readahead))

    def _producer():
        try:
            reads = _prefetch_blocks(in_store, exps, args.ccl_readahead, "seeded")
            for (exp, data), core in zip(reads, cores):
                prepped = _prep_block(data, exp, cell_points, block, cfg, bg_field)
                ready.put((core, exp) + prepped)
        finally:
            ready.put(None)  # sentinel

    prod = threading.Thread(target=_producer, daemon=True)
    prod.start()

    n_writers = max(2, args.ccl_readahead // 2)
    writer = ThreadPoolExecutor(max_workers=n_writers)
    pending = deque()
    n_objects = 0
    while True:
        item = ready.get()
        if item is None:
            break
        core, exp, norm_host, coords, ids = item
        norm = cp.asarray(norm_host)
        if any(s > 0 for s in args.smooth_sigma):
            norm = cndi.gaussian_filter(norm, sigma=tuple(args.smooth_sigma))
        labels, k = _segment_block(norm, coords, ids, exp, args, cp, cndi)
        n_objects += k
        z0, z1, y0, y1, x0, x1 = core
        lz0, ly0, lx0 = z0 - exp[0], y0 - exp[2], x0 - exp[4]
        lz1, ly1, lx1 = lz0 + (z1 - z0), ly0 + (y1 - y0), lx0 + (x1 - x0)
        crop_host = cp.asnumpy(labels[lz0:lz1, ly0:ly1, lx0:lx1])
        del norm, labels
        # Return cached free blocks each block; the EDT feature transform needs a large
        # contiguous allocation and the pool fragments over many blocks.
        cp.get_default_memory_pool().free_all_blocks()
        pending.append(writer.submit(_write_block, out, core, crop_host))
        while len(pending) > 2 * n_writers:  # backpressure: bound in-flight writes
            pending.popleft().result()

    for fut in pending:
        fut.result()
    writer.shutdown()
    prod.join()
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
        description="Seeded instance segmentation from a points array "
        "(GPU EDT nearest-seed labeling)."
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
        help="Block (chunk) size for the per-chunk labeling.",
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
        help="Absolute intensity floor: a voxel must be at least this (normalized) to "
        "be foreground at all, regardless of the relative threshold. Excludes "
        "near-zero background.",
    )
    ap.add_argument(
        "--seed-relative-threshold",
        type=float,
        default=0.5,
        help="Carve each blob at this fraction of ITS NEAREST SEED's intensity: keep a "
        "voxel when norm >= this * seed_intensity. Higher = tighter hug (less "
        "background), lower = grows more. Handles bright and dim blobs uniformly.",
    )
    ap.add_argument(
        "--merge-core-threshold",
        type=float,
        default=0.0,
        help="Merge multiple seeds that share one bright blob into a SINGLE object: "
        "seeds whose cores (norm >= this) stay connected merge; a dip below this "
        "between them (an intensity valley) keeps them split. Set above --seed-flood "
        "and below the blob peak. 0 = off (one object per seed).",
    )
    ap.add_argument(
        "--watershed-surface",
        choices=["edt", "intensity"],
        default="edt",
        help="Kept for compatibility. Only 'edt' (nearest-seed via distance "
        "transform) is implemented; 'intensity' warns and falls back to 'edt' "
        "(cuCIM has no watershed for an intensity surface).",
    )
    ap.add_argument(
        "--seed-sphere-radius",
        type=int,
        default=-1,
        help="Dilate each seed by this radius into the foreground mask so "
        "dim objects have a growable core. -1 = auto (factor/2 + 1).",
    )
    ap.add_argument(
        "--smooth-sigma",
        type=float,
        nargs="+",
        default=[0.0],
        help="Gaussian sigma (voxels) applied to the normalized intensity BEFORE the "
        "carve, to denoise. One value = isotropic; three = (z, y, x). 0 = off. Try "
        "~0.7 (or '0.5 1 1' for anisotropic data). params.json 'smooth_sigma' is "
        "ignored here.",
    )
    ap.add_argument(
        "--close-iterations",
        type=int,
        default=0,
        help="Binary-closing iterations on each object mask: smooths edges and "
        "bridges small gaps so objects are well-formed (0 = off).",
    )
    ap.add_argument(
        "--fill-holes",
        action="store_true",
        help="Fill enclosed holes in each object so masks look solid, not patchy.",
    )
    ap.add_argument(
        "--dilate-iterations",
        type=int,
        default=0,
        help="Dilate each object by this many voxels so masks look fuller. Bounded "
        "by the nearest-seed Voronoi split, so boundaries between objects stay clear "
        "(0 = off).",
    )
    ap.add_argument(
        "--min-object-size",
        type=int,
        default=0,
        help="Drop objects smaller than this many voxels (measured on the final "
        "smoothed+carved+cleaned object, at the seg level). 0 = off.",
    )
    ap.add_argument(
        "--max-object-size",
        type=int,
        default=0,
        help="Drop objects larger than this many voxels (removes over-grown blobs "
        "from bad seeds). 0 = off (no cap).",
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
    """Rescale points, then per-chunk nearest-seed labeling -> uint32 instances."""
    import cupy as cp

    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.watershed_surface == "intensity":
        logger.warning(
            "--watershed-surface intensity is not implemented (cuCIM has no "
            "watershed); falling back to 'edt' (nearest-seed)."
        )
    args.smooth_sigma = _norm_sigma(args.smooth_sigma)
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
    pts = pts[:, :3]
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
        _seeded_segment(args, base_path, level, in_store, cell_points, bg_field, cfg)

    if args.no_pyramid:
        logger.info("Done (single level %s). Skipped pyramid (--no-pyramid).", level)
        return
    _build_label_pyramid(args, base_path, spec, level, start, datasets, source_ms)
    logger.info(
        "Done. Instance-label OME-Zarr at s3://%s/%s", args.out_bucket, base_path
    )


if __name__ == "__main__":
    main()
