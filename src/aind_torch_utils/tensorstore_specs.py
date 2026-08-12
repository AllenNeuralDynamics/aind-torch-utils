"""Build TensorStore specifications for Beaker denoising runs."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Sequence
from urllib.parse import urlparse

import tensorstore as ts


ZARR_V2_FORMAT = 2
ZARR_V3_FORMAT = 3
OUTPUT_ZARR_FORMAT = ZARR_V2_FORMAT
OUTPUT_RESOLUTION = "0"
OUTPUT_FILL_VALUE = 0
OUTPUT_DIMENSION_SEPARATOR = "/"
OUTPUT_COMPRESSOR = {
    "id": "blosc",
    "cname": "zstd",
    "clevel": 6,
    "shuffle": 1,
}


@dataclass(frozen=True)
class ZarrLocation:
    """Normalized root and array locations for one multiscale Zarr."""

    root_uri: str
    array_uri: str
    resolution: str


@dataclass(frozen=True)
class InputArrayMetadata:
    """Metadata needed to construct matching input and output specs."""

    location: ZarrLocation
    driver: str
    zarr_format: int
    shape: tuple[int, ...]
    origin: tuple[int, ...]
    dtype: str
    chunks: tuple[int, ...]


def _is_s3(uri: str) -> bool:
    return uri.startswith("s3://")


def _parse_s3(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Not a complete S3 URI: {uri}")
    if parsed.query or parsed.fragment:
        raise ValueError("S3 Zarr URIs must not contain a query or fragment.")
    return parsed.netloc, parsed.path.strip("/")


def normalize_zarr_location(
    input_uri: str,
    default_resolution: str = "0",
) -> ZarrLocation:
    """Normalize an S3 Zarr root or explicit resolution-array URI."""
    bucket, key = _parse_s3(input_uri.rstrip("/"))
    parts = key.split("/")
    resolution = str(default_resolution).strip("/")
    if not resolution:
        raise ValueError("Input resolution must not be empty.")

    if parts[-1].isdigit() and len(parts) > 1 and parts[-2].endswith(".zarr"):
        resolution = parts[-1]
        root_key = "/".join(parts[:-1])
    elif parts[-1].endswith(".zarr"):
        root_key = key
    else:
        raise ValueError(
            "Input must point to a Zarr root ending in '.zarr' or to its "
            "numeric resolution array."
        )

    root_uri = f"s3://{bucket}/{root_key}"
    return ZarrLocation(
        root_uri=root_uri,
        array_uri=f"{root_uri}/{resolution}",
        resolution=resolution,
    )


def array_uri_at_resolution(root_uri: str, resolution: str) -> str:
    """Return an array URI below a normalized Zarr root."""
    normalized = normalize_zarr_location(root_uri, resolution)
    return f"{normalized.root_uri}/{str(resolution).strip('/')}"


def derive_output_root(input_root_uri: str) -> str:
    """Derive a sibling ``-denoised`` Zarr root from an input root."""
    bucket, key = _parse_s3(input_root_uri)
    parent, _, name = key.rpartition("/")
    if name.endswith(".ome.zarr"):
        output_name = f"{name[:-len('.ome.zarr')]}-denoised.ome.zarr"
    elif name.endswith(".zarr"):
        output_name = f"{name[:-len('.zarr')]}-denoised.zarr"
    else:  # normalize_zarr_location normally prevents this branch.
        raise ValueError(f"Input Zarr root does not end in '.zarr': {input_root_uri}")
    output_key = f"{parent}/{output_name}" if parent else output_name
    return f"s3://{bucket}/{output_key}"


def _input_driver_for_format(zarr_format: int) -> str:
    if zarr_format == ZARR_V2_FORMAT:
        return "zarr"
    if zarr_format == ZARR_V3_FORMAT:
        return "zarr3"
    raise ValueError(f"Unsupported input Zarr format: {zarr_format}")


def _output_driver_for_format(zarr_format: int) -> str:
    if zarr_format == ZARR_V2_FORMAT:
        return "zarr2"
    if zarr_format == ZARR_V3_FORMAT:
        return "zarr3"
    raise ValueError(f"Unsupported output Zarr format: {zarr_format}")


def inspect_input_array(location: ZarrLocation) -> InputArrayMetadata:
    """Open only array metadata and auto-detect Zarr v2 versus v3."""
    failures: list[str] = []
    store: Any | None = None
    zarr_format: int | None = None
    driver = ""
    for candidate_format in (ZARR_V2_FORMAT, ZARR_V3_FORMAT):
        candidate_driver = _input_driver_for_format(candidate_format)
        try:
            store = ts.open(
                {"driver": candidate_driver, "kvstore": location.array_uri},
                open=True,
            ).result()
            zarr_format = candidate_format
            driver = candidate_driver
            break
        except Exception as exc:  # TensorStore exposes backend-specific errors.
            failures.append(f"{candidate_driver}: {exc}")

    if store is None or zarr_format is None:
        details = "; ".join(failures)
        raise ValueError(
            f"Could not open input Zarr array {location.array_uri}: {details}"
        )

    shape = tuple(int(value) for value in store.domain.shape)
    origin = tuple(int(value) for value in store.domain.inclusive_min)
    try:
        chunks = tuple(int(value) for value in store.chunk_layout.read_chunk.shape)
    except Exception as exc:
        raise ValueError("Input Zarr has no determinate read chunk shape.") from exc
    dtype = store.dtype.numpy_dtype.str
    return InputArrayMetadata(
        location=location,
        driver=driver,
        zarr_format=zarr_format,
        shape=shape,
        origin=origin,
        dtype=dtype,
        chunks=chunks,
    )


def validate_input_metadata(metadata: InputArrayMetadata) -> None:
    """Validate the layout required by the denoising workflow."""
    if len(metadata.shape) != 5:
        raise ValueError(
            f"Expected a 5D T,C,Z,Y,X input; got shape {metadata.shape}."
        )
    if metadata.origin != (0, 0, 0, 0, 0):
        raise ValueError(
            f"Expected a zero-origin input; got origin {metadata.origin}."
        )
    if len(metadata.chunks) != 5 or any(value <= 0 for value in metadata.chunks):
        raise ValueError(f"Invalid input chunk shape: {metadata.chunks}.")


def validate_output_chunks(
    chunks: Sequence[int],
    block: Sequence[int],
) -> tuple[int, ...]:
    """Validate that spatial chunks are safe for independent Ray shards."""
    normalized = tuple(int(value) for value in chunks)
    spatial_block = tuple(int(value) for value in block)
    if len(normalized) != 5:
        raise ValueError("Output chunks must contain five T,C,Z,Y,X dimensions.")
    if len(spatial_block) != 3:
        raise ValueError("Inference block must contain three Z,Y,X dimensions.")
    if any(value <= 0 for value in normalized):
        raise ValueError(f"Output chunks must be positive; got {normalized}.")
    incompatible = [
        axis
        for axis, (block_size, chunk_size) in enumerate(
            zip(spatial_block, normalized[-3:])
        )
        if block_size % chunk_size != 0
    ]
    if incompatible:
        axes = ", ".join("ZYX"[axis] for axis in incompatible)
        raise ValueError(
            f"Output chunks {normalized[-3:]} do not divide block "
            f"{spatial_block} on axis/axes {axes}; pass --output-chunks Z Y X."
        )
    return normalized


def _build_output_kv(
    output_base: str,
    resolution: str,
    aws_region: str,
) -> dict[str, object]:
    if _is_s3(output_base):
        bucket, key = _parse_s3(output_base)
        output_kv: dict[str, object] = {
            "driver": "s3",
            "bucket": bucket,
            "path": f"{key.rstrip('/')}/{resolution}",
        }
        if aws_region:
            output_kv["aws_region"] = aws_region
        return output_kv

    output_path = os.path.abspath(os.path.join(output_base, resolution))
    return {"driver": "file", "path": output_path}


def _build_output_metadata(
    zarr_format: int,
    shape: Sequence[int],
    dtype_str: str,
    chunks: Sequence[int],
) -> dict[str, object]:
    if zarr_format != ZARR_V2_FORMAT:
        raise ValueError("Automated Beaker output currently supports only Zarr v2.")
    return {
        "shape": list(shape),
        "zarr_format": ZARR_V2_FORMAT,
        "fill_value": OUTPUT_FILL_VALUE,
        "chunks": list(chunks),
        "compressor": OUTPUT_COMPRESSOR,
        "dimension_separator": OUTPUT_DIMENSION_SEPARATOR,
        "dtype": dtype_str,
    }


def build_specs(
    metadata: InputArrayMetadata,
    output_root: str,
    chunks: Sequence[int],
    aws_region: str,
    output_resolution: str = OUTPUT_RESOLUTION,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build input and Zarr v2 output TensorStore specs."""
    input_spec: dict[str, Any] = {
        "driver": metadata.driver,
        "kvstore": metadata.location.array_uri,
    }
    output_spec: dict[str, Any] = {
        "driver": _output_driver_for_format(OUTPUT_ZARR_FORMAT),
        "kvstore": _build_output_kv(output_root, output_resolution, aws_region),
        "metadata": _build_output_metadata(
            OUTPUT_ZARR_FORMAT,
            metadata.shape,
            metadata.dtype,
            chunks,
        ),
        "recheck_cached_metadata": False,
        "recheck_cached_data": False,
        "open": True,
        "create": True,
        "delete_existing": False,
    }
    return input_spec, output_spec


def sanitized_run_slug(input_root_uri: str) -> str:
    """Return a compact filesystem/Beaker-safe name for a Zarr root."""
    _bucket, key = _parse_s3(input_root_uri)
    name = key.rsplit("/", 1)[-1]
    name = re.sub(r"(?:\.ome)?\.zarr$", "", name)
    slug = re.sub(r"[^a-zA-Z0-9-]+", "-", name).strip("-").lower()
    return slug[:48].rstrip("-") or "zarr"


def s3_prefix_name(input_root_uri: str) -> str:
    """Return the input's top-level S3 key prefix for Beaker naming."""
    _bucket, key = _parse_s3(input_root_uri)
    prefix = key.split("/", 1)[0]
    if not prefix:
        raise ValueError(f"Input S3 URI has no key prefix: {input_root_uri}")
    return prefix
