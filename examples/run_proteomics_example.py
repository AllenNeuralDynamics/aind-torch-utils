"""
Large-scale proteomics inference with a shared MAE encoder and N decoder heads.

Architecture
------------
SharedEncoderModel holds one MAE3DEncoderRope (frozen) and N DecoderPath heads.
The encoder runs once per forward pass; all heads receive the same latent and
skip features. Outputs are N probability maps [0,1], one per protein marker.

Multi-GPU note
--------------
GpuWorker deepcopies the model to each device, so the encoder is replicated
once per GPU. The gain is computational (FLOPs), not memory: each GPU runs
one encoder pass per batch instead of N, reducing encoder cost by (N-1)/N.

Usage
-----
python scripts/run_proteomics_example.py \
    --in-spec /path/to/input_spec.json \
    --out-bucket my-output-bucket \
    --out-prefix predictions/sample_001 \
    --encoder-weights /path/to/mae_encoder.pth \
    --decoder-weights /path/to/protein_a.pth /path/to/protein_b.pth \
    --output-names protein_a protein_b \
    --percentiles-dir /path/to/percentiles_dir

The input spec is a TensorStore JSON spec pointing to a 5D volume (T,C,Z,Y,X).
Decoder checkpoints may be raw DecoderPath state dicts, ProteinPredictionModel
state dicts (keys prefixed "decoder."), or Lightning checkpoints (keys under
"state_dict" with "model.decoder." prefix).
"""

import argparse
import json
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import tensorstore as ts
import torch

import boto3
from aind_proteomics_image_translator.models.protein_head import (
    DecoderPath,
    ProteinPredictionModel,
)
from aind_torch_utils.config import InferenceConfig
from aind_torch_utils.models import SharedEncoderModel
from aind_torch_utils.run import run
from aind_torch_utils.utils import open_ts_spec
from example_utils.omezarr_metadata import _get_pyramid_metadata, write_ome_ngff_metadata
from example_utils.utils import load_json
import time


def load_sample_percentiles(percentiles_dir: str) -> Dict[str, List[List[float]]]:
    """Load pre-computed percentiles for all samples in a directory.

    Reads every ``*_percentiles.npz`` file.  The sample_id is the filename
    with ``_percentiles.npz`` stripped.  Both the raw acquisition name and any
    ``_processed_YYYY-MM-DD`` variant are registered so S3 tile paths resolve.
    """
    result = {}
    for fname in os.listdir(percentiles_dir):
        if not fname.endswith("_percentiles.npz"):
            continue
        sample_id = fname[: -len("_percentiles.npz")]
        data = np.load(os.path.join(percentiles_dir, fname), allow_pickle=True)
        percentiles = data["combined_percentiles"].tolist()  # [[p_low, p_high], ...]
        result[sample_id] = percentiles
        base_id = re.sub(r"_processed_\d{4}-\d{2}-\d{2}.*$", "", sample_id)
        if base_id != sample_id:
            result[base_id] = percentiles
    return result


def extract_sample_id(s3_path: str) -> str:
    """Return the first path component after the bucket in an S3 URI."""
    return urlparse(s3_path).path.lstrip("/").split("/")[0]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _strip_decoder_prefix(state_dict: dict) -> dict:
    """Normalize a decoder checkpoint to bare DecoderPath keys.

    Handles:
    - Raw DecoderPath state dict (keys as-is)
    - ProteinPredictionModel state dict ("decoder." prefix)
    - Lightning checkpoint nested under "state_dict" ("model.decoder." prefix)
    """
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any(k.startswith("model.decoder.") for k in state_dict):
        return {
            k[len("model.decoder."):]: v
            for k, v in state_dict.items()
            if k.startswith("model.decoder.")
        }
    if any(k.startswith("decoder.") for k in state_dict):
        return {
            k[len("decoder."):]: v
            for k, v in state_dict.items()
            if k.startswith("decoder.")
        }
    return state_dict


def load_proteomics_model(
    encoder_checkpoint: str,
    decoder_checkpoints: List[str],
    img_size: Tuple[int, int, int] = (128, 128, 128),
    encoder_num_heads: int = 4,
    feature_size: int = 10,
    recover_layers: Tuple[int, int] = (2, 5),
    apply_sigmoid: bool = True,
) -> SharedEncoderModel:
    """Load a SharedEncoderModel from checkpoint files.

    Uses ProteinPredictionModel to load and configure the encoder (handles
    both .pth raw state dicts and .ckpt Lightning checkpoints, and infers
    all encoder hyperparameters from the checkpoint automatically).
    The decoder architecture is mirrored from the base model's decoder so
    all N heads share the same structural hyperparameters.

    Parameters
    ----------
    encoder_checkpoint : str
        Path to the pretrained MAE encoder checkpoint (.pth or .ckpt).
    decoder_checkpoints : list of str
        Paths to decoder state-dict files (.pth), one per protein head.
    img_size : tuple of int
        Input patch size (D, H, W). Must match MAE pretraining resolution.
    encoder_num_heads : int
        Attention heads in the MAE encoder. Must match pretraining; cannot
        be inferred from the checkpoint.
    feature_size : int
        Base channel count for each DecoderPath.
    recover_layers : tuple of int
        Two encoder layer indices for skip connections.
    apply_sigmoid : bool
        Apply sigmoid activation to all decoder outputs.

    Returns
    -------
    SharedEncoderModel
        In eval mode, encoder frozen, ready for run().
    """
    logger.info(f"Loading MAE encoder from: {encoder_checkpoint}")
    base = ProteinPredictionModel(
        img_size=img_size,
        mae_checkpoint_path=encoder_checkpoint,
        freeze_encoder=True,
        feature_size=feature_size,
        encoder_num_heads=encoder_num_heads,
    )
    encoder = base.encoder
    ref = base.decoder  # architectural template for all decoder heads

    logger.info(
        f"Encoder config — embed_dim={encoder.embed_dim}, "
        f"feat_size={ref.feat_size}, patch_size={ref.patch_size}, "
        f"n_register_tokens={ref.n_register_tokens}"
    )

    decoders = []
    for i, dec_ckpt in enumerate(decoder_checkpoints):
        logger.info(f"Loading decoder {i} from: {dec_ckpt}")
        dec = DecoderPath(
            feat_size=ref.feat_size,
            patch_size=ref.patch_size,
            n_register_tokens=ref.n_register_tokens,
            hidden_size=ref.hidden_size,
            feature_size=ref.feature_size,
        )
        raw_sd = torch.load(dec_ckpt, map_location="cpu", weights_only=False)
        sd = _strip_decoder_prefix(raw_sd)
        missing, unexpected = dec.load_state_dict(sd, strict=False)
        if missing:
            logger.warning(f"Decoder {i}: {len(missing)} missing keys: {missing[:5]}")
        if unexpected:
            logger.warning(f"Decoder {i}: {len(unexpected)} unexpected keys ignored.")
        decoders.append(dec)

    model = SharedEncoderModel(
        encoder=encoder,
        decoders=decoders,
        recover_layers=recover_layers,
        apply_sigmoid=apply_sigmoid,
    )
    model.eval()
    return model


def _open_or_create_s3_zarr(
    bucket: str, path: str, shape: Tuple, chunks: Tuple
) -> ts.TensorStore:
    """Open (or create) float32 zarr2 level-0 array inside an OME-zarr group on S3."""
    level0_path = path.rstrip("/") + "/0/"
    spec = {
        "driver": "zarr",
        "kvstore": {"driver": "s3", "bucket": bucket, "path": level0_path},
        "metadata": {
            "shape": list(shape),
            "chunks": list(chunks),
            "dtype": "<f4",
            "dimension_separator": "/",
        },
        "open": True,
        "create": True,
    }
    return ts.open(spec).result()


def _read_pyramid_spec_from_zarr(
    in_spec_path: str,
) -> Optional[Tuple[Tuple[float, float, float], List[List[int]], int]]:
    """Read pyramid spec from the input volume's zarr.json.

    Parameters
    ----------
    in_spec_path : str
        Path to the input TensorStore JSON spec file. The S3 URI in this spec is
        used to locate the zarr group and read its multiscales metadata.
    
    Returns
    -------
    Optional[Tuple[Tuple[float, float, float], List[List[int]], int]]
        Returns (voxel_size_zyx, scale_factors_per_level, n_extra_levels) or None
        if the input has no multiscales metadata.
    """
    with open(in_spec_path) as fh:
        spec_dict = json.load(fh)
    kv = spec_dict.get("kvstore", spec_dict)
    bucket = kv.get("bucket", "")
    path = kv.get("path", "").rstrip("/")

    # Strip trailing level digit (e.g. ".../0" → "...") to reach the group root
    group_path = re.sub(r"/\d+$", "", path)
    s3_group_uri = f"s3://{bucket}/{group_path}"

    # Try zarr v3 (zarr.json) first, fall back to zarr v2 (.zattrs)
    zarr_json = None
    for fname in ("zarr.json", ".zattrs"):
        try:
            zarr_json = load_json(s3_group_uri, fname)
            logger.info(f"Read multiscales metadata from {s3_group_uri}/{fname}")
            break
        except Exception:
            pass
    if zarr_json is None:
        logger.warning(f"Could not read zarr.json or .zattrs from {s3_group_uri}")
        return None

    # zarr v3: {"attributes": {"ome": {"multiscales": [...]}}}
    # zarr v2: {"multiscales": [...]}  or  {"ome": {"multiscales": [...]}}
    attrs = zarr_json.get("attributes", zarr_json)
    multiscales = attrs.get("ome", attrs).get("multiscales", [])
    if not multiscales:
        return None

    datasets = multiscales[0].get("datasets", [])
    if len(datasets) < 2:
        return None

    def _get_scale(ds):
        for ct in ds.get("coordinateTransformations", []):
            if ct["type"] == "scale":
                return ct["scale"]
        return None

    scales = [_get_scale(ds) for ds in datasets]
    if any(s is None for s in scales):
        return None

    # Voxel size at level 0 — last 3 values are Z, Y, X
    voxel_size_zyx = tuple(scales[0][-3:])

    # Scale factors between consecutive levels (ZYX only)
    scale_factors_per_level = [
        [max(1, round(scales[i + 1][j] / scales[i][j])) for j in range(-3, 0)]
        for i in range(len(scales) - 1)
    ]

    return voxel_size_zyx, scale_factors_per_level, len(datasets) - 1


def _write_pyramid_ts(
    bucket: str,
    zarr_s3_path: str,
    vol_shape: Tuple,
    voxel_size_zyx: Tuple[float, float, float],
    scale_factors_per_level: List[List[int]],
    n_lvls: int,
    image_name: str = "prediction",
    chunk_size: Tuple[int, int, int] = (128, 128, 128),
) -> None:
    """Generate pyramid levels 1..n_lvls using TensorStore's downsample driver.

    All levels are zarr v2, consistent with the inference output at level 0.
    OME-NGFF .zattrs is uploaded via boto3 at the group root.

    Parameters
    ----------
    bucket : str
        S3 bucket name for output.
    zarr_s3_path : str
        S3 key prefix for the zarr group, e.g. "predictions/sample_001/protein_a.zarr/". The function will write to "<prefix>/0/", "<prefix>/1/", etc.
    vol_shape : tuple
        Shape of the level-0 volume (T, C, Z, Y, X).
    voxel_size_zyx : tuple
        Voxel size in Z, Y, X order.
    scale_factors_per_level : list of list
        List of [sf_z, sf_y, sf_x] scale factors between consecutive levels.
    n_lvls : int
        Number of pyramid levels to generate (excluding level 0).
    image_name : str, optional
        Name of the image. Default is "prediction".
    chunk_size : tuple, optional
        Chunk size for the volume. Default is (128, 128, 128).
    """
    import asyncio

    base_path = zarr_s3_path.rstrip("/") + "/"
    chunk_5d = (1, 1) + chunk_size

    zattrs = write_ome_ngff_metadata(
        arr_shape=list(vol_shape),
        chunk_size=list(chunk_5d),
        image_name=image_name,
        n_lvls=n_lvls + 1,
        scale_factors=scale_factors_per_level,
        voxel_size=(1.0, 1.0) + tuple(voxel_size_zyx),
        origin=[0, 0, 0],
        metadata=_get_pyramid_metadata(),
    )
    s3_client = boto3.client("s3")
    zattrs_key = f"{base_path}.zattrs"
    s3_client.put_object(Bucket=bucket, Key=zattrs_key, Body=json.dumps(zattrs).encode())
    logger.info(f"Wrote OME-NGFF metadata to s3://{bucket}/{zattrs_key}")

    async def _build_levels():
        import asyncio as _aio

        # write() collects all lazy chunk results before writing anything,
        # so memory grows to the full level size on TB-scale data regardless of
        # cache_pool or s3_request_concurrency (github.com/google/tensorstore/issues/213).
        # Fix: acquire semaphore before each read() so TensorStore never sees more
        # than max_inflight chunks at once.
        # Peak RAM = max_inflight x input_chunk_bytes (e.g. 8 x 64 MB = 512 MB for sf=2).
        max_inflight = 8
        sem = _aio.Semaphore(max_inflight)
        ctx = ts.Context({"data_copy_concurrency": {"limit": 4}})

        for lvl in range(1, n_lvls + 1):
            sf = scale_factors_per_level[min(lvl - 1, len(scale_factors_per_level) - 1)]
            sf_padded = [1, 1] + list(sf)

            downsampled = await ts.open({
                "driver": "downsample",
                "downsample_factors": sf_padded,
                "downsample_method": "mean",
                "base": {
                    "driver": "zarr",
                    "kvstore": {
                        "driver": "s3",
                        "bucket": bucket,
                        "path": f"{base_path}{lvl - 1}/",
                    },
                },
            }, context=ctx)
            new_shape = list(downsampled.shape)
            nz, ny, nx = new_shape[2], new_shape[3], new_shape[4]
            cz, cy, cx = chunk_5d[2], chunk_5d[3], chunk_5d[4]

            output_spec = {
                "driver": "zarr",
                "kvstore": {
                    "driver": "s3",
                    "bucket": bucket,
                    "path": f"{base_path}{lvl}/",
                },
                "metadata": {
                    "shape": new_shape,
                    "chunks": list(chunk_5d),
                    "dtype": "<f4",
                    "dimension_separator": "/",
                },
                "create": True,
                "delete_existing": True,
            }
            output = await ts.open(output_spec, context=ctx)

            async def _write_chunk(z0, z1, y0, y1, x0, x1):
                async with sem:
                    data = await downsampled[:, :, z0:z1, y0:y1, x0:x1].read()
                    await output[:, :, z0:z1, y0:y1, x0:x1].write(data)
                    del data

            all_ranges = [
                (z0, min(z0 + cz, nz), y0, min(y0 + cy, ny), x0, min(x0 + cx, nx))
                for z0 in range(0, nz, cz)
                for y0 in range(0, ny, cy)
                for x0 in range(0, nx, cx)
            ]
            n_chunks = len(all_ranges)
            batch = 64
            for i in range(0, n_chunks, batch):
                await _aio.gather(*[_write_chunk(*r) for r in all_ranges[i : i + batch]])
                logger.info(f"[pyramid] level {lvl} chunk {min(i + batch, n_chunks)}/{n_chunks}")

            logger.info(f"[pyramid] level {lvl} done — shape {new_shape}")

    asyncio.run(_build_levels())


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Large-scale proteomics inference: shared MAE encoder + N decoder heads"
    )
    ap.add_argument("--in-spec", required=True,
                    help="TensorStore JSON spec file for the input volume.")
    ap.add_argument("--out-bucket", required=True,
                    help="S3 bucket name for output probability maps.")
    ap.add_argument("--out-prefix", required=True,
                    help="S3 key prefix; each output is written to '<prefix>/<name>.zarr/'.")
    ap.add_argument("--output-names", nargs="+", required=True,
                    help="Name per decoder output, e.g. '--output-names nucleus membrane'.")
    ap.add_argument("--encoder-weights", required=True,
                    help="MAE encoder checkpoint (.pth state dict or .ckpt Lightning).")
    ap.add_argument("--decoder-weights", nargs="+", required=True,
                    help="Decoder checkpoint files (.pth), one per output head.")
    ap.add_argument("--encoder-num-heads", type=int, default=4,
                    help="Attention heads in MAE encoder (default: 4). Must match pretraining.")
    ap.add_argument("--feature-size", type=int, default=10,
                    help="Base channel count for each DecoderPath (default: 10).")
    ap.add_argument("--recover-layers", nargs=2, type=int, default=[2, 5],
                    metavar=("SHALLOW", "DEEP"),
                    help="Encoder layer indices for skip connections (default: 2 5).")
    ap.add_argument("--no-sigmoid", action="store_true",
                    help="Output raw logits instead of sigmoid probabilities.")
    ap.add_argument("--patch", nargs=3, type=int, default=[128, 128, 128],
                    metavar=("Z", "Y", "X"),
                    help="Patch size in voxels (default: 128 128 128).")
    ap.add_argument("--overlap", type=int, default=16,
                    help="Patch overlap in voxels (default: 16).")
    ap.add_argument("--block", nargs=3, type=int, default=[512, 512, 512],
                    metavar=("Z", "Y", "X"),
                    help="Block size for volume tiling (default: 512 512 512).")
    ap.add_argument("--batch", type=int, default=4,
                    help="Inference batch size (default: 4).")
    ap.add_argument(
        "--devices",
        nargs="*",
        default=None,
        metavar="DEVICE",
        help=(
            "CUDA devices to use, e.g. --devices cuda:0 cuda:1. "
            "Defaults to all available GPUs (torch.cuda.device_count())."
        ),
    )
    ap.add_argument("--t", type=int, default=0, help="Time index (default: 0).")
    ap.add_argument("--c", type=int, default=0, help="Channel index (default: 0).")
    ap.add_argument("--prep-workers", type=int, default=4)
    ap.add_argument("--writer-workers", type=int, default=2)
    ap.add_argument("--max-inflight-batches", type=int, default=32)
    ap.add_argument("--metrics-json", default="metrics.json")
    ap.add_argument(
        "--percentiles-dir",
        default=None,
        help=(
            "Directory containing *_percentiles.npz files produced by "
            "compute_percentiles(). When provided, normalization is set to "
            "'global' using the sample's pre-computed percentiles, matching "
            "the training-time PercentileNormalizationd transform. "
            "Required for correct inference; omit only for debugging."
        ),
    )
    ap.add_argument(
        "--sample-id",
        default=None,
        help=(
            "Sample identifier used to look up percentiles "
            "(e.g. 'HCR_802702-s1-ls2_2026-03-13_18-00-00'). "
            "If omitted, auto-extracted from the first path component of the "
            "S3 URI in --in-spec."
        ),
    )
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if args.devices:
        devices = args.devices
        invalid = [d for d in devices if not str(d).startswith("cuda:")]
        if invalid:
            raise ValueError(
                "This proteomics pipeline requires CUDA devices. "
                f"Invalid device(s): {invalid}. "
                "Provide only CUDA devices, e.g. --devices cuda:0 cuda:1."
            )
    else:
        n = torch.cuda.device_count()
        if n <= 0 or not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA devices detected. This proteomics pipeline is GPU-only "
                "(GpuWorker uses CUDA streams/events) and cannot run on CPU."
            )
        devices = [f"cuda:{i}" for i in range(n)]
        logger.info(f"Auto-detected {len(devices)} CUDA device(s): {devices}")

    if len(args.decoder_weights) != len(args.output_names):
        raise ValueError(
            f"--decoder-weights ({len(args.decoder_weights)}) and "
            f"--output-names ({len(args.output_names)}) must have the same length."
        )

    norm_lower: Optional[float] = None
    norm_upper: Optional[float] = None

    if args.percentiles_dir:
        sample_percentiles: Dict = load_sample_percentiles(args.percentiles_dir)
        logger.info(
            f"Loaded percentiles for {len(sample_percentiles)} sample(s) "
            f"from {args.percentiles_dir}"
        )

        # Determine sample_id: explicit flag > auto-extract from in-spec path
        if args.sample_id:
            sample_id = args.sample_id
        else:
            with open(args.in_spec) as fh:
                spec_dict = json.load(fh)
            # Prefer kvstore path; fall back to top-level path
            kv = spec_dict.get("kvstore", spec_dict)
            s3_path = (
                f"s3://{kv.get('bucket', '')}/{kv.get('path', '').lstrip('/')}"
            )
            sample_id = extract_sample_id(s3_path)
            logger.info(f"Auto-extracted sample_id='{sample_id}' from in-spec")

        if sample_id not in sample_percentiles:
            raise KeyError(
                f"sample_id '{sample_id}' not found in percentiles. "
                f"Available: {list(sample_percentiles.keys())[:10]}"
            )
        p_low, p_high = sample_percentiles[sample_id][args.c]
        norm_lower, norm_upper = float(p_low), float(p_high)
        logger.info(
            f"Using percentile normalization: p_low={norm_lower}, p_high={norm_upper}"
        )
    else:
        logger.warning(
            "--percentiles-dir not provided; normalization disabled. "
            "Outputs may be incorrect if the model was trained with percentile normalization."
        )

    logger.info(f"Opening input volume: {args.in_spec}")
    in_store = open_ts_spec(args.in_spec)
    T, C, Z, Y, X = tuple(in_store.domain.shape)
    logger.info(f"Volume shape: T={T} C={C} Z={Z} Y={Y} X={X}")

    # This script writes compact outputs with a single (t, c) plane only.
    # Current pipeline uses the same cfg.t_idx/c_idx for both read and write,
    # so compact output requires writing at index (0, 0).
    if args.t != 0 or args.c != 0:
        raise ValueError(
            "This example writes compact output stores with shape (1,1,Z,Y,X), "
            "so --t and --c must both be 0. "
            f"Got --t {args.t}, --c {args.c}."
        )

    patch = tuple(args.patch)
    vol_shape = (1, 1, Z, Y, X)
    out_chunks = (1, 1) + patch  # one patch per zarr chunk
    out_stores = []
    for name in args.output_names:
        s3_path = f"{args.out_prefix.rstrip('/')}/{name}.zarr/"
        logger.info(f"Output store: s3://{args.out_bucket}/{s3_path}")
        out_stores.append(
            _open_or_create_s3_zarr(args.out_bucket, s3_path, vol_shape, out_chunks)
        )

    model = load_proteomics_model(
        encoder_checkpoint=args.encoder_weights,
        decoder_checkpoints=args.decoder_weights,
        img_size=patch,
        encoder_num_heads=args.encoder_num_heads,
        feature_size=args.feature_size,
        recover_layers=tuple(args.recover_layers),
        apply_sigmoid=not args.no_sigmoid,
    )

    if norm_lower is not None and norm_upper is not None:
        # Match preprocessing during training
        norm_kwargs = dict(
            normalize="global",
            norm_lower=norm_lower,
            norm_upper=norm_upper,
            output_denormalize=False,
        )
    else:
        norm_kwargs = dict(normalize=False, output_denormalize=False)

    cfg = InferenceConfig(
        patch=patch,
        overlap=args.overlap,
        block=tuple(args.block),
        batch_size=args.batch,
        t_idx=args.t,
        c_idx=args.c,
        devices=devices,
        amp=True,
        use_tf32=True,
        max_inflight_batches=args.max_inflight_batches,
        seam_mode="blend",
        trim_voxels=None,
        **norm_kwargs,
    )
    logger.info(f"Inference config:\n{cfg}")

    run(
        model=model,
        input_store=in_store,
        output_store=out_stores,
        cfg=cfg,
        metrics_json=args.metrics_json,
        num_prep_workers=args.prep_workers,
        num_writer_workers=args.writer_workers,
    )
    logger.info("Inference complete.")
    
    start_pyramid_time = time.time()
    pyramid_spec = _read_pyramid_spec_from_zarr(args.in_spec)
    if pyramid_spec is None:
        logger.info("Input has no multiscales metadata — skipping pyramid generation.")
    else:
        voxel_size_zyx, scale_factors_per_level, n_extra_levels = pyramid_spec
        logger.info(
            f"Pyramid spec from input: {n_extra_levels} extra level(s), "
            f"voxel_size_zyx={voxel_size_zyx}, scale_factors={scale_factors_per_level}"
        )
        for name in args.output_names:
            s3_path = f"{args.out_prefix.rstrip('/')}/{name}.zarr/"
            logger.info(f"Generating pyramid for {name}...")
            _write_pyramid_ts(
                bucket=args.out_bucket,
                zarr_s3_path=s3_path,
                vol_shape=vol_shape,
                voxel_size_zyx=voxel_size_zyx,
                scale_factors_per_level=scale_factors_per_level,
                n_lvls=n_extra_levels,
                image_name=name,
            )

    end_pyramid_time = time.time()
    logger.info(f"Pyramid generation time: {end_pyramid_time - start_pyramid_time:.2f} seconds")


if __name__ == "__main__":
    profile_name = os.environ["AWS_PROFILE"]
    print("Using profile:", profile_name)
    session = boto3.Session(profile_name=profile_name)
    creds = session.get_credentials().get_frozen_credentials()

    os.environ["AWS_ACCESS_KEY_ID"] = creds.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = creds.secret_key
    os.environ["AWS_SESSION_TOKEN"] = creds.token
    main()
