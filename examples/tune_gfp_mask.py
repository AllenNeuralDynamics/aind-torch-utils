"""Parameter-sweep tuner for the gfp-mask segmentation (per-combo tri-view on a level).

Runs the *exact same* GPU mask chain as the pipeline (``create_gfp_mask_gpu``: smooth +
threshold + opening, optionally preceded by flat-field correction + global
normalization) over a small coarse pyramid level, sweeping ``threshold`` x
``smooth_sigma`` x ``open_iterations``. Each parameter combination gets its own output
folder containing a single high-res PNG with the XY / ZX / ZY orthogonal MIP overlays
(the mask depth-coded by the depth of its brightest voxel along each view axis -- turbo,
near->far, with a per-panel colorbar; use ``--flat-mask`` for plain red) and a
``params.txt`` listing that combo's params plus the proposed params for a destination
resolution. A ``metrics.csv`` at the root ranks all combos.

Because ``smooth_sigma`` / ``open_iterations`` are in *voxels*, values tuned on a coarse
level do not transfer 1:1 to a finer processing level. Pass ``--target-level`` to record
the scaled equivalents (in each combo's ``params.txt``) for the level your run uses.

Reads the tune level from ``--in-spec`` (same convention as ``run_gfp_mask_example.py``:
the level is the trailing digit of the kvstore path). Normalization and flat-field
settings come from ``--params-json`` (same file the pipeline uses).

Example::

    python examples/tune_gfp_mask.py \\
        --in-spec /scratch/seg_tile_spec.json --params-json /scratch/params.json \\
        --thresholds 0.02 0.05 0.1 --smooth-sigmas 0.5,1,1 1,2,2 1,3,3 \\
        --open-iterations 0 1 --target-level 0 --dpi 200 --out tune

Output layout (one folder per combo)::

    tune/
      metrics.csv
      thr_0.05_sigma_0.5-1-1_open_1/{views.png, params.txt}
      ...

Requires the ``postprocess`` (cupy) + ``extras`` (matplotlib) optional dependencies.
"""
import argparse
import csv
import logging
import os
import re
import sys
from typing import List, Optional, Tuple

import cupy as cp
import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from aind_torch_utils.config import InferenceConfig  # noqa: E402
from aind_torch_utils.correction import (  # noqa: E402
    apply_flatfield,
    normalize_global,
    sample_background,
    scale_params,
)
from aind_torch_utils.postprocessing import create_gfp_mask_gpu  # noqa: E402
from aind_torch_utils.utils import open_ts_spec  # noqa: E402

from run_gfp_mask_example import (  # noqa: E402
    DEFAULT_IN_SPEC,
    _estimate_background,
    _kvstore_bucket_path,
    _load_spec_dict,
    _normalization_kwargs,
    _read_source_multiscales,
    _scale_zyx,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_sigma(text: str) -> Tuple[float, float, float]:
    """Parse a 'z,y,x' string into a float triple."""
    parts = [p for p in text.split(",") if p != ""]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"smooth-sigma must be 'z,y,x' (3 values), got '{text}'"
        )
    return tuple(float(p) for p in parts)


def _spec_level_and_group(spec) -> Tuple[str, str, str]:
    """Return (bucket, group_path, level) parsed from the spec's kvstore path."""
    bucket, kv_path, _ = _kvstore_bucket_path(_load_spec_dict(spec))
    kv_path = kv_path.rstrip("/")
    m = re.search(r"/(\d+)$", kv_path)
    if m is None:
        raise ValueError(
            f"kvstore path must end with a multiscale level, got '{kv_path}'"
        )
    return bucket, kv_path[: m.start()], m.group(1)


def _apply_normalization(vol: np.ndarray, cfg: InferenceConfig) -> np.ndarray:
    """Normalize a volume exactly as PrepWorker would (global/percentile/off)."""
    if cfg.normalize == "global":
        vol = normalize_global(vol, cfg.norm_lower, cfg.norm_upper, cfg.eps)
    elif cfg.normalize == "percentile":
        lo, hi = np.percentile(vol, [cfg.norm_lower, cfg.norm_upper])
        vol = (vol - lo) / max(hi - lo, cfg.eps)
    if cfg.clip_norm:
        if cfg.clip_norm is True:
            vol = np.clip(vol, 0.0, 1.0)
        else:
            lo, hi = cfg.clip_norm
            vol = np.clip(vol, lo, hi)
    return vol.astype(np.float32, copy=False)


def _preprocess(
    spec, cfg: InferenceConfig, raw: np.ndarray, bbox, full_shape, use_flatfield: bool
) -> np.ndarray:
    """Flat-field (optional) + normalize a raw crop, mirroring PrepWorker."""
    vol = raw.astype(np.float32, copy=False)
    if use_flatfield and cfg.flatfield:
        bg_field = _estimate_background(spec, cfg, full_shape)
        if bg_field is not None:
            z0, z1, y0, y1, x0, x1 = bbox
            bg = sample_background(
                bg_field.field, bg_field.scale, z0, z1, y0, y1, x0, x1
            )
            vol = apply_flatfield(
                vol, bg, mode=cfg.flatfield_mode, eps=cfg.eps, bg_mean=bg_field.mean
            )
    return _apply_normalization(vol, cfg)


def _mip_and_depth(mask, vol, axis: int):
    """Project a 3D mask: return (binary MIP, normalized depth) as numpy.

    ``depth`` is the index along ``axis`` of the **brightest masked voxel** in each
    output column (a true depth-coded MIP, consistent with the grayscale signal MIP
    shown underneath), normalized to ``[0, 1]`` (near -> far). Unambiguous for columns
    that contain mask at multiple depths, unlike a centroid. Only meaningful where the
    MIP is positive (unmasked columns argmax to 0 but are drawn with zero alpha).
    """
    length = mask.shape[axis]
    mask_mip = mask.max(axis=axis)
    # Depth of the brightest masked voxel: argmax of the signal restricted to the mask.
    masked = cp.where(mask > 0, vol, cp.float32(-cp.inf))
    depth_idx = masked.argmax(axis=axis).astype(cp.float32)
    depth_norm = depth_idx / max(length - 1, 1)
    return cp.asnumpy(mask_mip), cp.asnumpy(depth_norm)


def _overlay_rgb(
    gray: np.ndarray,
    mask_mip: np.ndarray,
    depth_norm: Optional[np.ndarray] = None,
    alpha: float = 0.6,
    cmap_name: str = "turbo",
) -> np.ndarray:
    """Build an (H, W, 3) uint8 image: grayscale signal with the mask overlaid.

    Min-max normalizes ``gray`` for display, then blends the mask on top where
    ``mask_mip`` is positive. If ``depth_norm`` is given, the mask is depth-coded via
    ``cmap_name`` (near -> far); otherwise it is flat red.
    """
    g = gray.astype(np.float32)
    lo, hi = float(g.min()), float(g.max())
    g = (g - lo) / (hi - lo) if hi > lo else np.zeros_like(g)
    rgb = np.repeat(g[..., None], 3, axis=2)  # grayscale base
    if depth_norm is not None:
        color = plt.get_cmap(cmap_name)(np.clip(depth_norm, 0.0, 1.0))[..., :3]
    else:
        color = np.zeros(rgb.shape, dtype=np.float32)
        color[..., 0] = 1.0  # flat red
    a = (mask_mip > 0).astype(np.float32)[..., None] * alpha
    rgb = rgb * (1.0 - a) + color * a
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def _overlay(
    ax,
    gray: np.ndarray,
    mask_mip: np.ndarray,
    title: str,
    depth_norm: Optional[np.ndarray] = None,
) -> None:
    """Draw a grayscale MIP with the (optionally depth-coded) mask overlaid."""
    ax.imshow(_overlay_rgb(gray, mask_mip, depth_norm))
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _combo_metrics(vol_cp, thr, sigma, op) -> dict:
    """Unsupervised quality proxies for one (threshold, sigma, open) mask.

    Returns foreground fraction, connected-component count (fragmentation), the share
    of foreground in the largest component (contiguity), and the intensity contrast
    between predicted foreground and background. Computed on the 3D mask, so it is
    independent of the projection axis.
    """
    import cupyx.scipy.ndimage as cndi  # GPU only; lazy import

    mask = create_gfp_mask_gpu(
        vol_cp, threshold=thr, smooth_sigma=sigma, open_iterations=op
    )
    fg_frac = float(mask.mean())
    labeled, n = cndi.label(mask)
    if n > 0:
        counts = cp.bincount(labeled.ravel())[1:]  # drop background label 0
        fg_vox = float(counts.sum())
        largest_frac = float(counts.max()) / fg_vox if fg_vox > 0 else 0.0
    else:
        largest_frac = 0.0
    mb = mask.astype(bool)
    n_fg = int(mb.sum())
    fg_mean = float(vol_cp[mb].mean()) if n_fg > 0 else 0.0
    bg_mean = float(vol_cp[~mb].mean()) if n_fg < mb.size else 0.0
    return {
        "fg_frac": fg_frac,
        "n_components": int(n),
        "largest_frac": largest_frac,
        "contrast": fg_mean - bg_mean,
    }


def _write_metrics_csv(path: str, rows: List[dict]) -> None:
    """Write the per-combo metrics table for offline ranking/sorting."""
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s (%d rows)", path, len(rows))


def _combo_dirname(
    thr: float, sigma: Tuple[float, float, float], op: int, ff_tag: str, multi_ff: bool
) -> str:
    """Folder name for one parameter combo, e.g. 'thr_0.05_sigma_0.5-1-1_open_1'."""
    sig = "-".join(f"{s:g}" for s in sigma)
    name = f"thr_{thr:g}_sigma_{sig}_open_{op}"
    return f"{name}_{ff_tag}" if multi_ff else name


# View axis -> (label, depth-axis name). XY projects along z, ZX along y, ZY along x.
_VIEWS = [(0, "XY", "z"), (1, "ZX", "y"), (2, "ZY", "x")]


def _render_triview(
    vol_cp, thr, sigma, op, metric: dict, depth_color: bool, dpi: int, out_path: str
) -> None:
    """Render one combo's XY/ZX/ZY depth-coded overlays as a single high-res PNG.

    The mask is computed once; each panel is a max-projection along one axis, with the
    mask depth-coded (turbo) by the depth of its brightest voxel along that axis and its
    own colorbar (the depth axis differs per view). ``dpi`` sets zoomable resolution.
    """
    mask = create_gfp_mask_gpu(
        vol_cp, threshold=thr, smooth_sigma=sigma, open_iterations=op
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(7 * 3, 7), squeeze=False, constrained_layout=True
    )
    for col, (axis, view, depth_axis) in enumerate(_VIEWS):
        ax = axes[0][col]
        gray = cp.asnumpy(vol_cp.max(axis=axis))
        if depth_color:
            mask_mip, depth_norm = _mip_and_depth(mask, vol_cp, axis)
        else:
            mask_mip, depth_norm = cp.asnumpy(mask.max(axis=axis)), None
        _overlay(ax, gray, mask_mip, view, depth_norm)
        if depth_color:
            length = int(vol_cp.shape[axis])
            sm = ScalarMappable(cmap="turbo", norm=Normalize(0, max(length - 1, 1)))
            fig.colorbar(
                sm,
                ax=ax,
                fraction=0.046,
                pad=0.02,
                label=f"{depth_axis} depth: near -> far (vox)",
            )
    fig.suptitle(
        f"thr={thr:g}  σ={sigma}  open={op}   "
        f"fg={metric['fg_frac']:.3f} n={metric['n_components']} "
        f"top={metric['largest_frac']:.2f} contrast={metric['contrast']:.3f}",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def _render_for_ff(
    vol, ff_tag, tune_level, out_dir, args, cfg, factor, multi_ff
) -> list:
    """One folder per combo (views.png + params.txt); return the metric rows for CSV."""
    vol_cp = cp.asarray(vol)
    depth_color = not args.flat_mask
    rows = []
    for thr in args.thresholds:
        for sigma in args.smooth_sigmas:
            for op in args.open_iterations:
                m = _combo_metrics(vol_cp, thr, sigma, op)
                sig_tag = "x".join(f"{s:g}" for s in sigma)
                rows.append(
                    {
                        "flatfield": ff_tag,
                        "threshold": thr,
                        "sigma": sig_tag,
                        "open": op,
                        **m,
                    }
                )
                combo_dir = os.path.join(
                    out_dir, _combo_dirname(thr, sigma, op, ff_tag, multi_ff)
                )
                os.makedirs(combo_dir, exist_ok=True)
                _render_triview(
                    vol_cp,
                    thr,
                    sigma,
                    op,
                    m,
                    depth_color,
                    args.dpi,
                    os.path.join(combo_dir, "views.png"),
                )
                _write_combo_params(
                    os.path.join(combo_dir, "params.txt"),
                    thr,
                    sigma,
                    op,
                    cfg,
                    tune_level,
                    args.target_level,
                    factor,
                )
    return rows


def _compute_scale_factor(bucket, group_path, tune_level, target_level):
    """Return the (Z,Y,X) tune->target voxel-size factor, or None if unavailable.

    Logs the factor + transfer guidance once. ``None`` when ``--target-level`` is unset
    or the source multiscales metadata doesn't contain both levels.
    """
    if target_level is None:
        return None
    ms_info = _read_source_multiscales(bucket, group_path)
    if ms_info is None:
        logger.warning("No multiscales metadata; cannot propose target-level params.")
        return None
    _, datasets, _ = ms_info
    by_path = {d.get("path"): d for d in datasets}
    if tune_level not in by_path or target_level not in by_path:
        logger.warning(
            "Levels %s/%s not both in datasets %s; no target-level proposals.",
            tune_level,
            target_level,
            list(by_path),
        )
        return None
    s_tune = _scale_zyx(by_path[tune_level])
    s_target = _scale_zyx(by_path[target_level])
    if s_tune is None or s_target is None:
        logger.warning("Missing scale transforms; no target-level proposals.")
        return None
    factor = tuple(t / s for t, s in zip(s_tune, s_target))
    logger.info(
        "Tune level %s -> target level %s, factor (Z,Y,X)=%s (per-combo params.txt "
        "carries the scaled values).",
        tune_level,
        target_level,
        tuple(round(f, 3) for f in factor),
    )
    return factor


_TRANSFER_GUIDANCE = (
    "Transfer notes: threshold is in normalized units and transfers as-is. The scaled "
    "sigma preserves the physical smoothing scale, but the finer level is noisier "
    "(~factor^1.5x per-voxel) with more voxels, so nudge threshold UP slightly and add "
    "~1 open_iteration, then validate on a small --crop at the target level."
)


def _write_combo_params(
    path, thr, sigma, op, cfg, tune_level, target_level, factor
) -> None:
    """Write a per-combo params.txt: the tuner params + proposed target-level params."""
    lines = [
        "# GFP-mask tuning parameters",
        f"tune_level: {tune_level}",
        "",
        "[mask params at this (tune) level]",
        f"threshold: {thr:g}",
        f"smooth_sigma: {list(sigma)}",
        f"open_iterations: {op}",
        "",
        "[normalization config]",
        f"normalize: {cfg.normalize}",
        f"norm_lower: {cfg.norm_lower}",
        f"norm_upper: {cfg.norm_upper}",
        f"flatfield: {cfg.flatfield}",
    ]
    if cfg.flatfield:
        lines += [
            f"flatfield_mode: {cfg.flatfield_mode}",
            f"flatfield_level: {cfg.flatfield_level}",
            f"flatfield_opening_radius: {cfg.flatfield_opening_radius}",
            f"flatfield_sigma: {cfg.flatfield_sigma}",
        ]
    lines.append("")
    if factor is not None:
        ssig, sop = scale_params(sigma, op, factor)
        lines += [
            f"[proposed params for target level {target_level}]",
            f"threshold: {thr:g}",
            f"smooth_sigma: {[round(s, 3) for s in ssig]}",
            f"open_iterations: {sop}",
            f"scale_factor_zyx: {tuple(round(f, 3) for f in factor)}",
            "",
            _TRANSFER_GUIDANCE,
        ]
    else:
        lines.append(
            "[proposed params] pass --target-level (with source multiscales) to get "
            "target-resolution proposals."
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Sweep gfp-mask parameters on a coarse level; one folder per combo."
    )
    ap.add_argument(
        "--in-spec",
        default=None,
        help="TensorStore JSON spec (file/JSON/dict). Trailing digit = tune level.",
    )
    ap.add_argument(
        "--target-level",
        default=None,
        help="Pyramid level the real run uses; enables the param-scaling report.",
    )
    ap.add_argument("--params-json", default=None, help="normalize/flatfield params.")
    ap.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.1],
        help="Threshold values to sweep.",
    )
    ap.add_argument(
        "--smooth-sigmas",
        type=_parse_sigma,
        nargs="+",
        default=[(0.5, 1, 1), (1, 2, 2), (1, 3, 3)],
        help="smooth_sigma 'z,y,x' triples to sweep.",
    )
    ap.add_argument(
        "--open-iterations",
        type=int,
        nargs="+",
        default=[1],
        help="open_iterations values to sweep.",
    )
    ap.add_argument(
        "--crop",
        type=int,
        nargs=6,
        default=None,
        metavar=("Z0", "Z1", "Y0", "Y1", "X0", "X1"),
        help="Optional sub-region (absolute tune-level voxel coords).",
    )
    ap.add_argument(
        "--flat-mask",
        action="store_true",
        help="Overlay the mask in flat red instead of depth-coding it (turbo).",
    )
    ap.add_argument(
        "--sweep-flatfield",
        action="store_true",
        help="Also render a flat-field OFF variant for comparison.",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output PNG resolution; raise (e.g. 300) for more zoomable detail.",
    )
    ap.add_argument(
        "--out",
        default="gfp_tune",
        help="Output root folder (trailing '.png' stripped). One subfolder per param "
        "combo (views.png + params.txt) plus metrics.csv.",
    )
    ap.add_argument("--aws-region", default="us-west-2")
    ap.add_argument(
        "--max-gb",
        type=float,
        default=3.0,
        help=(
            "Abort if the crop exceeds this many GB (float32). The GPU is the limiter: "
            "peak VRAM is ~4x the crop, so 3 GB ~= 12 GB on a 16 GB GPU."
        ),
    )
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Sweep gfp-mask params; one folder per combo (tri-view PNG + params.txt)."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    spec = args.in_spec if args.in_spec is not None else DEFAULT_IN_SPEC

    bucket, group_path, tune_level = _spec_level_and_group(spec)
    logger.info("Tuning on s3://%s/%s level %s", bucket, group_path, tune_level)

    store = open_ts_spec(spec)
    full_shape = tuple(store.domain.shape[-3:])
    cfg = InferenceConfig(**_normalization_kwargs(args.params_json))

    if args.crop is not None:
        z0, z1, y0, y1, x0, x1 = args.crop
    else:
        z0, y0, x0 = 0, 0, 0
        z1, y1, x1 = full_shape
    bbox = (z0, z1, y0, y1, x0, x1)

    # Guard against OOM: the crop is held in RAM as float32 and copied to the GPU,
    # with several more float32 buffers allocated per sweep cell. Abort early with
    # guidance instead of getting silently Killed.
    n_vox = (z1 - z0) * (y1 - y0) * (x1 - x0)
    gb = n_vox * 4 / 1e9
    logger.info(
        "Crop %s = %.2f GB float32 (needs ~3-4x that across RAM+GPU).",
        (z1 - z0, y1 - y0, x1 - x0),
        gb,
    )
    if gb > args.max_gb:
        raise SystemExit(
            f"Crop is {gb:.1f} GB (> --max-gb {args.max_gb}). Point --in-spec at a "
            f"coarser pyramid level (the tuner is meant for a downsampled level) or "
            f"restrict it with --crop Z0 Z1 Y0 Y1 X0 X1."
        )

    raw = (
        store[cfg.t_idx, cfg.c_idx, z0:z1, y0:y1, x0:x1]
        .read()
        .result()
        .astype(np.float32, copy=False)
    )
    logger.info("Loaded crop %s from full shape %s", raw.shape, full_shape)

    # Flat-field settings to render: configured one, plus OFF if --sweep-flatfield.
    ff_settings = [True]
    if args.sweep_flatfield and cfg.flatfield:
        ff_settings.append(False)

    # All outputs go into one root folder derived from --out (".png" stripped).
    out_dir = args.out[:-4] if args.out.lower().endswith(".png") else args.out
    os.makedirs(out_dir, exist_ok=True)
    logger.info("Writing outputs to %s/", out_dir)

    # Tune->target voxel-size factor (once) for the per-combo params.txt proposals.
    factor = _compute_scale_factor(bucket, group_path, tune_level, args.target_level)
    multi_ff = len(ff_settings) > 1

    all_rows = []
    for use_ff in ff_settings:
        vol = _preprocess(spec, cfg, raw, bbox, full_shape, use_ff)
        ff_tag = "ff" if (use_ff and cfg.flatfield) else "noff"
        all_rows += _render_for_ff(
            vol, ff_tag, tune_level, out_dir, args, cfg, factor, multi_ff
        )
    _write_metrics_csv(os.path.join(out_dir, "metrics.csv"), all_rows)


if __name__ == "__main__":
    main()
