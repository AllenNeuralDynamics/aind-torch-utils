"""Seeded instance segmentation from a points array (GPU EDT nearest-seed labeling).

Standalone companion to ``run_gfp_mask_example.py`` for when you already have a
**point per object** (e.g. billions of bouton centers in a numpy ``(N, 3)`` array).
Instead of one global intensity threshold (which fails on dim/uneven objects), it
optionally Gaussian-denoises the normalized intensity (``--smooth-sigma``, "smooth then
segment"), carves each blob at a fraction of *its own seed's* brightness
(``--seed-relative-threshold``) — so bright and dim blobs hug tightly and the background
between seeds is excluded — then assigns voxels to seeds by marker-controlled **geodesic
growth** through the carved mask (each id stays one connected region; cuCIM has no
watershed). ``--grow-min-thickness`` makes splits blobbier (territories set by the thick
cores, split at thin necks) without dropping seeded objects. Morphological cleanup
(``--close-iterations`` /
``--fill-holes`` / ``--dilate-iterations``) makes objects solid and well-formed, and a
size filter (``--min-object-size`` / ``--max-object-size``) drops specks and over-grown
blobs. Optionally, ``--merge-valley-frac`` merges several seeds sharing one blob into a
single object unless a deep intensity valley (relative to each seed's peak) separates
them (and, with ``--merge-max-distance``, only if their centroids are within that many
full-resolution voxels). It writes a ``uint32`` instance-label OME-Zarr (each object's
voxels carry its point's row index + 1) plus a label-preserving pyramid.
(``params.json`` supplies only normalize/flat-field keys; its ``smooth_sigma`` is a
model param, ignored here — use ``--smooth-sigma``.)

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
    adjacency_merge,
    block_ranges,
    bucket_points,
    flood_seed_union,
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


def _core_key(mask, norm, lz, ly, lx, threshold, cp, cndi):
    """Per-seed merge key from a bright intensity core (``mask & norm >= threshold``).

    Seeds sharing a connected core get the same key (candidate merge); seeds with no
    core (dimmer than ``threshold``) get distinct negative keys so they never merge.
    """
    core_comp, _ = cndi.label(mask & (norm >= threshold))
    seed_core = cp.asnumpy(core_comp[lz, ly, lx]).astype(np.int64)
    away = seed_core == 0
    if away.any():
        seed_core[away] = -np.arange(1, int(away.sum()) + 1, dtype=np.int64)
    return seed_core


def _flood_key(mask, norm, comp, lz, ly, lx, args, cp, cndi):
    """Per-seed merge key from superlevel-set flooding (intensity-valley test).

    Two seeds get the same key only if the saddle between them stays above their flood
    floors (shallow valley) and they are within ``--merge-max-distance``. The floor is
    relative to each seed's own intensity (``--merge-valley-frac``) or an absolute drop
    (``--merge-valley-depth``). Distinct keys => no merge. See
    :func:`labeling.flood_seed_union`.
    """
    si = cp.asnumpy(norm[lz, ly, lx])  # seed intensities (the prior)
    scmp = cp.asnumpy(comp[lz, ly, lx])  # seed blob components
    k = si.shape[0]
    # Nothing to merge unless some blob holds >= 2 seeds -> distinct keys (no merge).
    pos = scmp[scmp > 0]
    if pos.size == 0 or int(np.bincount(pos).max()) < 2:
        return np.arange(1, k + 1, dtype=np.int64)
    if args.merge_valley_frac > 0:
        min_level = si * (1.0 - args.merge_valley_frac)
    else:
        min_level = si - args.merge_valley_depth
    lo = max(float(args.seed_flood_threshold), float(min_level.min()))
    levels = np.linspace(float(si.max()), lo, int(args.merge_levels))
    lev_lab = np.empty((levels.shape[0], k), dtype=np.int64)
    for i, lev in enumerate(levels):
        lab, _ = cndi.label(mask & (norm >= float(lev)))
        lev_lab[i] = cp.asnumpy(lab[lz, ly, lx])
    # coords in seg-level voxels -> scale per axis to full-res (level-0) voxels so
    # --merge-max-distance is interpreted at the highest resolution.
    coords = np.stack([cp.asnumpy(lz), cp.asnumpy(ly), cp.asnumpy(lx)], axis=1)
    coords = coords.astype(np.float64) * np.asarray(args.merge_dist_scale)
    return flood_seed_union(
        si, scmp, lev_lab, levels, min_level, coords, args.merge_max_distance
    )


def _cell_adjacency(assigned, cp):
    """Adjacent distinct local-label pairs (6-connectivity) from the labelled cells.

    Returns a host ``(E, 2)`` array of 1-based label pairs whose cells touch; used to
    restrict merging to cells that are actually connected.
    """
    kmax = int(assigned.max())
    if kmax < 1:
        return np.empty((0, 2), dtype=np.int64)
    codes = []
    for x, y in (
        (assigned[:-1], assigned[1:]),
        (assigned[:, :-1], assigned[:, 1:]),
        (assigned[:, :, :-1], assigned[:, :, 1:]),
    ):
        m = (x > 0) & (y > 0) & (x != y)
        if bool(m.any()):
            u, v = x[m].astype(cp.int64), y[m].astype(cp.int64)
            lo, hi = cp.minimum(u, v), cp.maximum(u, v)
            codes.append(lo * (kmax + 1) + hi)
    if not codes:
        return np.empty((0, 2), dtype=np.int64)
    uniq = cp.asnumpy(cp.unique(cp.concatenate(codes)))
    return np.stack([uniq // (kmax + 1), uniq % (kmax + 1)], axis=1)


def _object_groups(
    assigned, mask, norm, comp, lz, ly, lx, local_to_global, args, cp, cndi
):
    """Map local seed labels -> object groups, merging only ADJACENT same-group cells.

    A merge mode proposes which seeds are one object (flood/core key); the proposal is
    intersected with cell adjacency so a merged id is a connected chain of cells
    (no disconnected same-id pieces). Identity (no merge) returns one group per seed.
    """
    if args.merge_valley_frac > 0 or args.merge_valley_depth > 0:
        key = _flood_key(mask, norm, comp, lz, ly, lx, args, cp, cndi)
    elif args.merge_core_threshold > 0:
        key = _core_key(mask, norm, lz, ly, lx, args.merge_core_threshold, cp, cndi)
    else:
        return np.arange(local_to_global.shape[0], dtype=np.int64), local_to_global
    final_key = adjacency_merge(_cell_adjacency(assigned, cp), key)
    return merge_seed_groups(final_key, local_to_global[1:])


def _grow_labels(markers, grow_mask, cp, cndi):
    """Marker-controlled geodesic growth: spread labels through ``grow_mask`` only.

    Repeatedly dilate the labelled region into adjacent unlabelled foreground until
    stable. Because labels only ever spread to neighbouring foreground voxels, every
    labelled voxel stays connected to its seed (no straight-line jump across a gap).
    Ties break by max label id (deterministic); foreground unreachable from any seed
    stays 0.
    """
    labels = markers.copy()
    for _ in range(int(max(labels.shape))):  # safety cap; converges in ~geodesic radius
        grown = cndi.grey_dilation(labels, size=3)
        nxt = cp.where((labels == 0) & grow_mask, grown, labels)
        if bool((nxt == labels).all()):
            break
        labels = nxt
    return labels


def _assign_geodesic(markers, comp, ncomp, mask, lz, ly, lx, args, cp, cndi):
    """Assign each foreground voxel to a seed by CONNECTIVITY (not straight-line dist).

    Single-seed components are filled wholesale with their seed (connected by
    construction); multi-seed components are grown geodesically from their seeds so each
    cell stays connected to its seed (touching objects split at the geodesic midline).
    No-seed components stay 0 (dropped). Returns local seed labels (0 = background).

    With ``--grow-min-thickness t`` it is a two-stage growth: (1) grow each seed only
    through the mask's thick core (distance-to-background >= t) so object *territories*
    are set by the blobby cores and split at thin necks; (2) grow the SAME seeds through
    the remaining full mask so thin material is attached to its nearest seed. So no
    seeded object is dropped (every seed keeps its own connected id) while splits stay
    centred on the thick cores. Only foreground unreachable from any seed stays 0.
    """
    if args.grow_min_thickness > 0:
        edt = cndi.distance_transform_edt(mask)
        thick = (mask & (edt >= args.grow_min_thickness)) | (markers > 0)
        cores = _grow_labels(markers, thick, cp, cndi)  # thick cores set territories
        return _grow_labels(cores, mask, cp, cndi)  # attach thin material to same seeds
    n = int(ncomp) + 1
    seed_comp = comp[lz, ly, lx]  # each seed's component
    cnts = cp.bincount(seed_comp, minlength=n)  # seeds per component
    local_labels = cp.arange(1, lz.shape[0] + 1, dtype=cp.int32)
    comp_to_seed = cp.zeros(n, dtype=cp.int32)
    comp_to_seed[seed_comp] = local_labels  # last-wins; kept only for single-seed comps
    comp_to_seed = cp.where(cnts == 1, comp_to_seed, cp.int32(0))
    comp_to_seed[0] = 0
    assigned = comp_to_seed[comp]  # single-seed components filled directly
    is_multi = cnts >= 2
    if bool(is_multi.any()):
        multi_vox = is_multi[comp]
        grown = _grow_labels(
            cp.where(multi_vox, markers, cp.int32(0)), mask & multi_vox, cp, cndi
        )
        assigned = cp.where(assigned > 0, assigned, grown)
    return assigned


def _segment_block(norm, coords, ids, exp, args, cp, cndi):
    """GPU per-block: carve -> morphology -> geodesic seed assignment -> size filter.

    ``norm`` is the (smoothed) normalized intensity already on the GPU. Returns
    ``(labels_uint32, k)`` for the full expanded block; the caller crops to the core.
    cuCIM has no watershed, so voxels are assigned to seeds by marker-controlled
    geodesic growth through the carved mask (keeps each id one connected region).
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
    # EDT feature transform -> per-voxel nearest-seed brightness for the carve only.
    inds = cndi.distance_transform_edt(
        markers == 0, return_distances=False, return_indices=True
    )
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
    # Assign voxels to seeds by CONNECTIVITY (geodesic growth), so every id is one
    # connected region tied to its seed -- no straight-line wrap-around into a blob.
    comp, ncomp = cndi.label(mask)
    assigned = _assign_geodesic(markers, comp, ncomp, mask, lz, ly, lx, args, cp, cndi)
    # Map local seed labels -> compact object "groups": identity (one object per seed),
    # or a merge mode (flood/core) that collapses seeds sharing one blob.
    local_to_group, group_to_global = _object_groups(
        assigned, mask, norm, comp, lz, ly, lx, local_to_global, args, cp, cndi
    )
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
        "--merge-valley-frac",
        type=float,
        default=0.0,
        help="Merge seeds on one blob unless the intensity valley between them drops "
        "below this fraction of their OWN peak (relative to each seed, so it adapts to "
        "blobs dim in some regions). e.g. 0.3 tolerates a 30%%-of-peak dip; larger = "
        "merges across deeper valleys. Recommended merge mode; takes precedence over "
        "--merge-core-threshold. 0 = off.",
    )
    ap.add_argument(
        "--merge-valley-depth",
        type=float,
        default=0.0,
        help="Absolute-depth variant of --merge-valley-frac (flood floor = seed - "
        "this, normalized). Used only if --merge-valley-frac is 0. 0 = off.",
    )
    ap.add_argument(
        "--merge-levels",
        type=int,
        default=8,
        help="Number of flood levels for the valley merge (more = finer saddle-depth "
        "resolution, slower: adds this many label() passes on blocks that have "
        "co-located seeds).",
    )
    ap.add_argument(
        "--merge-max-distance",
        type=float,
        default=0.0,
        help="With the valley merge, only merge seeds whose centroids are within this "
        "many FULL-RESOLUTION (level-0) voxels; auto-scaled to the seg level by the "
        "pyramid factor. Seeds farther apart stay separate even if joined by a bright "
        "ridge. Single-linkage (chains of close seeds can still link). 0 = no limit.",
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
        "--grow-min-thickness",
        type=float,
        default=0.0,
        help="Blobbier splits without dropping points: territories are set by growing "
        "seeds through mask regions at least this thick (distance to background, seg "
        "voxels), splitting at thin necks; then thin material is attached to its "
        "nearest seed (no seeded object dropped). 0 = off (grow the full shape).",
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
    # --merge-max-distance is given in full-resolution (level-0) voxels; this per-axis
    # factor converts seg-level voxels -> level-0 voxels (~2**seg_level).
    s0 = _scale_zyx(by_path["0"]) if "0" in by_path else None
    if s0 is not None:
        args.merge_dist_scale = tuple(s_seg[a] / s0[a] for a in range(3))
    else:
        args.merge_dist_scale = (float(2 ** int(level)),) * 3
    logger.info(
        "Points @ level %s -> seg level %s, factor(z,y,x)=%s, seed-sphere r=%d, "
        "merge-dist scale(z,y,x)=%s",
        args.seeds_level,
        level,
        tuple(round(f, 3) for f in factor),
        args.seed_sphere_radius,
        tuple(round(f, 3) for f in args.merge_dist_scale),
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
