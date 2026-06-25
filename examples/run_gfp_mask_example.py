"""GPU GFP-masking example on a public S3 multiscale zarr.

Reads a SmartSPIM/HCR fused volume from S3 at a chosen multiscale level, runs the
``gfp-mask`` model (a cupy/cuCIM masking routine driven by the inference pipeline),
and writes a ``uint8`` mask zarr back to S3.

Requirements
------------
- A CUDA GPU with the post-processing stack installed:
  ``pip install -e ".[postprocess]"`` (``cupy-cuda12x`` + ``cucim-cu12``). The
  ``gfp-mask`` model is GPU-only.
- AWS credentials with write access to the output bucket. The input bucket
  ``aind-open-data`` is public (region ``us-west-2``); reading uses the standard AWS
  credential chain. If your environment has no credentials and signed S3 reads fail,
  TensorStore's s3 driver may need anonymous access configured.

Usage
-----
python examples/run_gfp_mask_example.py \
    --out-bucket my-output-bucket \
    --out-prefix predictions/HCR_831990 \
    --devices cuda:0

By default it processes::

    s3://aind-open-data/HCR_831990-s2-ls1_2026-05-15_00-00-00_processed_2026-05-19_01-32-11/fusion/ch_488/fused.zarr

at multiscale level 5. Override the input with ``--in-bucket/--in-path/--level``.
Mask parameters can be customized via ``--params-json`` (see
``GfpMaskModel.from_json``); omit it to use defaults.
"""

import argparse
import logging
import sys
from typing import List, Optional

import tensorstore as ts

from aind_torch_utils.config import InferenceConfig
from aind_torch_utils.run import load_model, run
from aind_torch_utils.utils import open_ts_spec

DEFAULT_IN_BUCKET = "aind-open-data"
DEFAULT_IN_PATH = (
    "HCR_831990-s2-ls1_2026-05-15_00-00-00_processed_2026-05-19_01-32-11"
    "/fusion/ch_488/fused.zarr"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _open_or_create_s3_mask_zarr(
    bucket: str, path: str, shape, chunks, region: str
) -> ts.TensorStore:
    """Open (or create) a uint8 zarr2 array for the output mask on S3."""
    spec = {
        "driver": "zarr",
        "kvstore": {
            "driver": "s3",
            "bucket": bucket,
            "path": path,
            "aws_region": region,
        },
        "metadata": {
            "shape": list(shape),
            "chunks": list(chunks),
            "dtype": "|u1",  # uint8 mask
            "dimension_separator": "/",
        },
        "create": True,
        "delete_existing": True,
    }
    return ts.open(spec).result()


def _parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Run the gfp-mask model on an S3 multiscale zarr."
    )
    ap.add_argument("--in-bucket", default=DEFAULT_IN_BUCKET)
    ap.add_argument("--in-path", default=DEFAULT_IN_PATH)
    ap.add_argument("--level", type=int, default=5, help="Multiscale level to read")
    ap.add_argument("--out-bucket", required=True)
    ap.add_argument(
        "--out-prefix",
        required=True,
        help="Output written to s3://<out-bucket>/<out-prefix>/gfp_mask.zarr/0/",
    )
    ap.add_argument("--aws-region", default="us-west-2")
    ap.add_argument("--devices", nargs="+", default=["cuda:0"])
    ap.add_argument("--patch", type=int, nargs=3, default=(64, 64, 64))
    ap.add_argument("--block", type=int, nargs=3, default=(256, 256, 256))
    ap.add_argument("--overlap", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument(
        "--params-json",
        default=None,
        help="Optional JSON file of GfpMaskModel parameters.",
    )
    ap.add_argument("--prep-workers", type=int, default=2)
    ap.add_argument("--writer-workers", type=int, default=2)
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Read level-N input from S3, run gfp-mask, write the mask back to S3."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    # --- Input store (existing zarr; metadata read from the store) ---
    in_path = f"{args.in_path.rstrip('/')}/{args.level}/"
    in_spec = {
        "driver": "zarr",
        "kvstore": {
            "driver": "s3",
            "bucket": args.in_bucket,
            "path": in_path,
            "aws_region": args.aws_region,
        },
    }
    logger.info("Opening input: s3://%s/%s", args.in_bucket, in_path)
    input_store = open_ts_spec(in_spec)
    shape = tuple(input_store.domain.shape)
    logger.info("Input shape=%s dtype=%s", shape, input_store.dtype)

    # The pipeline indexes the store as [t_idx, c_idx, z, y, x], so it must be 5D.
    if len(shape) != 5:
        raise ValueError(
            f"Expected a 5D (T, C, Z, Y, X) input store, got shape {shape}. "
            "Point --in-path at an OME-Zarr multiscale level."
        )
    z, y, x = shape[2:]

    # Clamp patch/block to the (possibly small) downsampled volume.
    patch = tuple(min(p, d) for p, d in zip(args.patch, (z, y, x)))
    block = tuple(min(b, d) for b, d in zip(args.block, (z, y, x)))
    if patch != tuple(args.patch) or block != tuple(args.block):
        logger.info(
            "Clamped to volume (Z,Y,X)=(%d,%d,%d): patch=%s block=%s",
            z,
            y,
            x,
            patch,
            block,
        )

    # --- Output store (fresh uint8 mask zarr on S3) ---
    out_path = f"{args.out_prefix.rstrip('/')}/gfp_mask.zarr/0/"
    chunks = (1, 1, min(128, z), min(128, y), min(128, x))
    logger.info("Creating output: s3://%s/%s", args.out_bucket, out_path)
    output_store = _open_or_create_s3_mask_zarr(
        args.out_bucket, out_path, (1, 1, z, y, x), chunks, args.aws_region
    )

    # --- Model + config ---
    model = load_model("gfp-mask", args.params_json)
    cfg = InferenceConfig(
        patch=patch,
        block=block,
        overlap=args.overlap,
        batch_size=args.batch,
        devices=args.devices,
        amp=False,  # no torch math in the model; avoid a pointless float16 copy
        use_compile=False,  # torch.compile cannot trace the cupy/cuCIM region
        normalize="percentile",  # feed ~[0,1]; matches intensity_percentiles=(0,1)
        output_denormalize=False,  # mask is not in intensity space
        seam_mode="trim",
    )
    logger.info("Inference config:\n%s", cfg)

    run(
        model,
        input_store,
        output_store,
        cfg,
        num_prep_workers=max(1, args.prep_workers),
        num_writer_workers=max(1, args.writer_workers),
    )
    logger.info("Done. Mask written to s3://%s/%s", args.out_bucket, out_path)


if __name__ == "__main__":
    main()
