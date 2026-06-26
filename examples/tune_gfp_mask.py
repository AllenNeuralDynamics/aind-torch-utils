"""Parameter-sweep tuner for the gfp-mask segmentation (MIP montage on a coarse level).

Runs the *exact same* GPU mask chain as the pipeline (``create_gfp_mask_gpu``: smooth +
threshold + opening, optionally preceded by flat-field correction + global
normalization) over a small coarse pyramid level, and renders a montage of
maximum-intensity-projection (MIP) overlays — red mask over grayscale signal — for a
grid of ``threshold`` x ``smooth_sigma`` values (one montage per ``open_iterations``).
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
        --open-iterations 0 1 --target-level 0 --out tune.png

Requires the ``postprocess`` (cupy) + ``extras`` (matplotlib) optional dependencies.
"""
import argparse
import logging
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


def _overlay(ax, gray: np.ndarray, mask_mip: np.ndarray, title: str) -> None:
    """Draw a grayscale MIP with the mask MIP overlaid in translucent red."""
    ax.imshow(gray, cmap="gray")
    rgba = np.zeros(mask_mip.shape + (4,), dtype=np.float32)
    rgba[..., 0] = 1.0  # red channel
    rgba[..., 3] = (mask_mip > 0).astype(np.float32) * 0.5  # alpha where masked
    ax.imshow(rgba)
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
) -> None:
    """Sweep threshold x smooth_sigma at a fixed open_iters; save one montage PNG."""
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
            mask_mip = cp.asnumpy(mask.max(axis=mip_axis))
            title = f"thr={thr:g} σ={sigma} fg={fg:.3f}"
            _overlay(axes[r][c], gray_mip, mask_mip, title)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


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
        "--sweep-flatfield",
        action="store_true",
        help="Also render a flat-field OFF montage for comparison.",
    )
    ap.add_argument("--out", default="gfp_tune.png", help="Output PNG path (base).")
    ap.add_argument("--aws-region", default="us-west-2")
    ap.add_argument(
        "--max-gb", type=float, default=3.0,
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

    base = args.out[:-4] if args.out.lower().endswith(".png") else args.out
    for use_ff in ff_settings:
        vol = _preprocess(spec, cfg, raw, bbox, full_shape, use_ff)
        vol_cp = cp.asarray(vol)
        gray_mip = vol.max(axis=args.mip_axis)
        ff_tag = "ff" if (use_ff and cfg.flatfield) else "noff"
        for op in args.open_iterations:
            tag = f"{ff_tag}_open{op}"
            out_path = f"{base}_{tag}.png"
            suptitle = (
                f"level {tune_level}  flatfield={'on' if ff_tag == 'ff' else 'off'}  "
                f"open_iterations={op}  (rows=threshold, cols=smooth_sigma)"
            )
            _render_montage(
                vol_cp,
                gray_mip,
                args.thresholds,
                args.smooth_sigmas,
                op,
                args.mip_axis,
                out_path,
                suptitle,
            )

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
