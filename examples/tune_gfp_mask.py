"""Parameter-sweep tuner for the gfp-mask segmentation (MIP montage on a coarse level).

Runs the *exact same* GPU mask chain as the pipeline (``create_gfp_mask_gpu``: smooth +
threshold + opening, optionally preceded by flat-field correction + global
normalization) over a small coarse pyramid level, and renders a montage of
maximum-intensity-projection (MIP) overlays — the mask depth-coded (turbo, near->far)
over the grayscale signal (use ``--flat-mask`` for plain red) — for a grid of
``threshold`` x ``smooth_sigma`` values (one montage per ``open_iterations``).
This lets you pick mask parameters by eye in seconds instead of running the full
pipeline.

Because ``smooth_sigma`` / ``open_iterations`` are in *voxels*, values tuned on a coarse
level do not transfer 1:1 to a finer processing level. Pass ``--target-level`` to print
the scaled equivalents for the level your real run uses.

Reads the tune level from ``--in-spec`` (same convention as ``run_gfp_mask_example.py``:
the level is the trailing digit of the kvstore path). Normalization and flat-field
settings come from ``--params-json`` (same file the pipeline uses).

Example::

    python examples/tune_gfp_mask.py \\
        --in-spec /scratch/seg_tile_spec.json --params-json /scratch/params.json \\
        --thresholds 0.02 0.05 0.1 --smooth-sigmas 0.5,1,1 1,2,2 1,3,3 \\
        --open-iterations 0 1 --target-level 0 --ortho --rotate 24 --out tune

All outputs go into the ``--out`` folder. ``--ortho`` adds XY/XZ/YZ projection montages
(depth via three views); ``--rotate N`` writes a rotating-MIP overlay GIF (N frames,
parallax depth) per param combo into the folder.

Requires the ``postprocess`` (cupy) + ``extras`` (matplotlib) optional dependencies.
"""
import argparse
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


def _mip_and_depth(mask, axis: int):
    """Project a 3D mask: return (binary MIP, normalized centroid-depth) as numpy.

    ``depth`` is the intensity-weighted mean index of the foreground along ``axis``
    (the mask's centroid in depth for each output pixel), normalized to ``[0, 1]``
    (near -> far). Only meaningful where the MIP is positive.
    """
    length = mask.shape[axis]
    mask_mip = mask.max(axis=axis)
    shape = [1, 1, 1]
    shape[axis] = length
    idx = cp.arange(length, dtype=cp.float32).reshape(shape)
    mf = mask.astype(cp.float32)
    msum = mf.sum(axis=axis)
    depth = (mf * idx).sum(axis=axis) / cp.maximum(msum, 1.0)
    depth_norm = depth / max(length - 1, 1)
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
    ``cmap_name`` (near -> far); otherwise it is flat red. Used by montage cells and
    rotation frames.
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


def _render_montage(
    vol_cp,
    gray_mip: np.ndarray,
    thresholds: List[float],
    sigmas: List[Tuple[float, float, float]],
    open_iters: int,
    mip_axis: int,
    out_path: str,
    suptitle: str,
    depth_color: bool = True,
) -> None:
    """Sweep threshold x smooth_sigma at a fixed open_iters; save one montage PNG.

    When ``depth_color`` is set the mask is depth-coded (turbo, near->far) along the
    projection axis; otherwise it is flat red.
    """
    nrows, ncols = len(thresholds), len(sigmas)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows), squeeze=False
    )
    for r, thr in enumerate(thresholds):
        for c, sigma in enumerate(sigmas):
            mask = create_gfp_mask_gpu(
                vol_cp, threshold=thr, smooth_sigma=sigma, open_iterations=open_iters
            )
            fg = float(mask.mean())
            if depth_color:
                mask_mip, depth_norm = _mip_and_depth(mask, mip_axis)
            else:
                mask_mip, depth_norm = cp.asnumpy(mask.max(axis=mip_axis)), None
            title = f"thr={thr:g} σ={sigma} fg={fg:.3f}"
            _overlay(axes[r][c], gray_mip, mask_mip, title, depth_norm)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def _render_rotation_gif(
    vol_cp,
    threshold: float,
    sigma: Tuple[float, float, float],
    open_it: int,
    n_frames: int,
    out_path: str,
    depth_color: bool = True,
) -> None:
    """Save a rotating-MIP overlay GIF for one param set to ``out_path``.

    Spins the volume about the vertical (y) axis; for each angle it rotates, recomputes
    the mask, MIPs both along z (mask depth-coded along z when ``depth_color`` is set),
    and writes an animated GIF. Frames are rendered sequentially so peak GPU memory
    stays near one extra volume.
    """
    import cupyx.scipy.ndimage as cndi  # GPU only; lazy import

    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not installed; skipping rotation GIF %s", out_path)
        return

    frames = []
    for i in range(n_frames):
        angle = i * 360.0 / n_frames
        rot = cndi.rotate(vol_cp, angle, axes=(0, 2), reshape=False, order=1)
        mask = create_gfp_mask_gpu(
            rot, threshold=threshold, smooth_sigma=sigma, open_iterations=open_it
        )
        gray = cp.asnumpy(rot.max(axis=0))
        if depth_color:
            mask_mip, depth_norm = _mip_and_depth(mask, 0)
        else:
            mask_mip, depth_norm = cp.asnumpy(mask.max(axis=0)), None
        frames.append(Image.fromarray(_overlay_rgb(gray, mask_mip, depth_norm)))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
    )
    logger.info("Wrote %s (%d frames)", out_path, n_frames)


def _render_for_ff(vol, ff_tag, tune_level, out_dir, axes_list, axis_tag, args) -> None:
    """Render all montages (+ optional rotation frames) for one flat-field setting."""
    vol_cp = cp.asarray(vol)
    depth_color = not args.flat_mask
    depth_note = "  mask depth-coded (turbo near->far)" if depth_color else ""
    for op in args.open_iterations:
        for ax in axes_list:
            gray_mip = vol.max(axis=ax)
            suffix = f"{ff_tag}_open{op}_{axis_tag[ax]}"
            suptitle = (
                f"level {tune_level}  flatfield={'on' if ff_tag == 'ff' else 'off'}"
                f"  open={op}  proj={axis_tag[ax]}  (rows=threshold, cols=sigma)"
                f"{depth_note}"
            )
            _render_montage(
                vol_cp,
                gray_mip,
                args.thresholds,
                args.smooth_sigmas,
                op,
                ax,
                os.path.join(out_dir, f"montage_{suffix}.png"),
                suptitle,
                depth_color,
            )
    # Rotation: one animated GIF per parameter combo, in the output folder.
    if args.rotate > 0:
        for thr in args.thresholds:
            for sigma in args.smooth_sigmas:
                for op in args.open_iterations:
                    sig_tag = "x".join(f"{s:g}" for s in sigma)
                    gif_path = os.path.join(
                        out_dir, f"rotate_{ff_tag}_thr{thr:g}_sig{sig_tag}_open{op}.gif"
                    )
                    _render_rotation_gif(
                        vol_cp, thr, sigma, op, args.rotate, gif_path, depth_color
                    )


def _scaling_report(
    bucket: str,
    group_path: str,
    tune_level: str,
    target_level: str,
    sigmas: List[Tuple[float, float, float]],
    open_iters: List[int],
) -> None:
    """Print smooth_sigma/open_iterations scaled from the tune level to the target."""
    ms_info = _read_source_multiscales(bucket, group_path)
    if ms_info is None:
        logger.warning("No multiscales metadata; cannot compute scaling report.")
        return
    _, datasets, _ = ms_info
    by_path = {d.get("path"): d for d in datasets}
    if tune_level not in by_path or target_level not in by_path:
        logger.warning(
            "Levels %s/%s not both in datasets %s; skipping scaling report.",
            tune_level,
            target_level,
            list(by_path),
        )
        return
    s_tune = _scale_zyx(by_path[tune_level])
    s_target = _scale_zyx(by_path[target_level])
    if s_tune is None or s_target is None:
        logger.warning("Missing scale transforms; skipping scaling report.")
        return
    factor = tuple(t / s for t, s in zip(s_tune, s_target))
    logger.info(
        "Scaling report: tune level %s -> target level %s, factor (Z,Y,X)=%s "
        "(approximate; round as needed):",
        tune_level,
        target_level,
        tuple(round(f, 3) for f in factor),
    )
    for sigma in sigmas:
        for op in open_iters:
            ssig, sop = scale_params(sigma, op, factor)
            logger.info(
                "  tune (σ=%s, open=%d)  ->  target (σ=%s, open=%d)",
                sigma,
                op,
                tuple(round(s, 2) for s in ssig),
                sop,
            )
    yx = (factor[1] + factor[2]) / 2.0
    logger.info(
        "Transfer guidance: the finer target level has ~%.0fx more per-voxel noise and "
        "~%.0fx more voxels; the scaled sigma recovers most SNR but the noise tail "
        "needs calibration:",
        yx**1.5,
        factor[0] * factor[1] * factor[2],
    )
    logger.info("  - keep the scaled σ above (preserves the physical smoothing scale);")
    logger.info(
        "  - nudge threshold UP slightly and add ~1 open_iteration to clear the extra "
        "noise-tail specks at full resolution;"
    )
    logger.info(
        "  - VALIDATE: rerun this tuner with --in-spec at the target level + a small "
        "--crop to calibrate threshold/opening in the real noise regime (the coarse "
        "sweep only fixes the physical scale)."
    )


def _parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Sweep gfp-mask parameters on a coarse level; render MIP montages."
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
        help="Threshold values to sweep (montage rows).",
    )
    ap.add_argument(
        "--smooth-sigmas",
        type=_parse_sigma,
        nargs="+",
        default=[(0.5, 1, 1), (1, 2, 2), (1, 3, 3)],
        help="smooth_sigma 'z,y,x' triples to sweep (montage columns).",
    )
    ap.add_argument(
        "--open-iterations",
        type=int,
        nargs="+",
        default=[1],
        help="open_iterations values; one montage PNG per value.",
    )
    ap.add_argument(
        "--crop",
        type=int,
        nargs=6,
        default=None,
        metavar=("Z0", "Z1", "Y0", "Y1", "X0", "X1"),
        help="Optional sub-region (absolute tune-level voxel coords).",
    )
    ap.add_argument("--mip-axis", type=int, default=0, help="MIP axis (0=Z,1=Y,2=X).")
    ap.add_argument(
        "--flat-mask",
        action="store_true",
        help="Overlay the mask in flat red instead of depth-coding it by z (turbo).",
    )
    ap.add_argument(
        "--ortho",
        action="store_true",
        help="Render all three projections (XY/XZ/YZ) for depth, not just --mip-axis.",
    )
    ap.add_argument(
        "--rotate",
        type=int,
        default=0,
        help="If >0, save a rotating-MIP overlay GIF (N frames over 360deg) per param "
        "combo into the output folder, for pseudo-3D depth perception.",
    )
    ap.add_argument(
        "--sweep-flatfield",
        action="store_true",
        help="Also render a flat-field OFF montage for comparison.",
    )
    ap.add_argument(
        "--out",
        default="gfp_tune",
        help="Output folder (a trailing '.png' is stripped). Montages + rotation "
        "frame subfolders are written inside it.",
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
    """Render parameter-sweep MIP montages for the gfp-mask on a coarse level."""
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

    # All outputs go into one folder derived from --out (".png" stripped).
    out_dir = args.out[:-4] if args.out.lower().endswith(".png") else args.out
    os.makedirs(out_dir, exist_ok=True)
    logger.info("Writing outputs to %s/", out_dir)
    # --ortho renders all three projections; otherwise just the requested axis.
    axis_tag = {0: "xy", 1: "xz", 2: "yz"}
    axes_list = [0, 1, 2] if args.ortho else [args.mip_axis]
    if args.rotate > 0:
        n_rot = (
            len(args.thresholds)
            * len(args.smooth_sigmas)
            * len(args.open_iterations)
            * len(ff_settings)
        )
        logger.info(
            "Rotation enabled: %d param combos x %d frames each (reduce sweep lists "
            "if this is too many).",
            n_rot,
            args.rotate,
        )
    for use_ff in ff_settings:
        vol = _preprocess(spec, cfg, raw, bbox, full_shape, use_ff)
        ff_tag = "ff" if (use_ff and cfg.flatfield) else "noff"
        _render_for_ff(vol, ff_tag, tune_level, out_dir, axes_list, axis_tag, args)

    if args.target_level is not None:
        _scaling_report(
            bucket,
            group_path,
            tune_level,
            args.target_level,
            args.smooth_sigmas,
            args.open_iterations,
        )


if __name__ == "__main__":
    main()
