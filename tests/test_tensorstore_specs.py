"""Tests for automated TensorStore specification construction."""

import pytest

from aind_torch_utils.tensorstore_specs import (
    InputArrayMetadata,
    ZarrLocation,
    build_specs,
    derive_output_root,
    normalize_zarr_location,
    s3_prefix_name,
    validate_input_metadata,
    validate_output_chunks,
)


def _metadata(**overrides):
    values = {
        "location": ZarrLocation(
            "s3://bucket/sample.zarr",
            "s3://bucket/sample.zarr/0",
            "0",
        ),
        "driver": "zarr3",
        "zarr_format": 3,
        "shape": (1, 1, 512, 1024, 2048),
        "origin": (0, 0, 0, 0, 0),
        "dtype": "<u2",
        "chunks": (1, 1, 128, 256, 256),
    }
    values.update(overrides)
    return InputArrayMetadata(**values)


@pytest.mark.parametrize(
    ("uri", "default_resolution", "root", "array", "resolution"),
    [
        (
            "s3://bucket/path/fused.zarr",
            "0",
            "s3://bucket/path/fused.zarr",
            "s3://bucket/path/fused.zarr/0",
            "0",
        ),
        (
            "s3://bucket/path/fused.zarr/2/",
            "0",
            "s3://bucket/path/fused.zarr",
            "s3://bucket/path/fused.zarr/2",
            "2",
        ),
    ],
)
def test_normalize_zarr_location(
    uri, default_resolution, root, array, resolution
):
    location = normalize_zarr_location(uri, default_resolution)
    assert location.root_uri == root
    assert location.array_uri == array
    assert location.resolution == resolution


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("s3://bucket/path/fused.zarr", "s3://bucket/path/fused-denoised.zarr"),
        (
            "s3://bucket/path/sample.ome.zarr",
            "s3://bucket/path/sample-denoised.ome.zarr",
        ),
    ],
)
def test_derive_output_root(source, expected):
    assert derive_output_root(source) == expected


def test_s3_prefix_name_uses_top_level_dataset_prefix():
    assert (
        s3_prefix_name(
            "s3://bucket/exaSPIM_703070_2024-07-09/fused/channel.zarr"
        )
        == "exaSPIM_703070_2024-07-09"
    )


def test_build_specs_preserves_input_format_and_emits_v2_output():
    metadata = _metadata()
    input_spec, output_spec = build_specs(
        metadata,
        "s3://output-bucket/results/sample-denoised.zarr",
        metadata.chunks,
        "us-west-2",
    )

    assert input_spec == {
        "driver": "zarr3",
        "kvstore": "s3://bucket/sample.zarr/0",
    }
    assert output_spec["driver"] == "zarr2"
    assert output_spec["kvstore"] == {
        "driver": "s3",
        "bucket": "output-bucket",
        "path": "results/sample-denoised.zarr/0",
        "aws_region": "us-west-2",
    }
    assert output_spec["metadata"]["shape"] == list(metadata.shape)
    assert output_spec["metadata"]["chunks"] == list(metadata.chunks)
    assert output_spec["metadata"]["dtype"] == "<u2"
    assert output_spec["delete_existing"] is False


def test_validate_output_chunks_rejects_incompatible_input_chunks():
    with pytest.raises(ValueError, match="--output-chunks"):
        validate_output_chunks((1, 1, 96, 256, 256), (512, 512, 512))

    assert validate_output_chunks(
        (1, 1, 128, 128, 256), (512, 512, 512)
    ) == (1, 1, 128, 128, 256)


def test_validate_input_metadata_requires_5d_zero_origin():
    validate_input_metadata(_metadata())
    with pytest.raises(ValueError, match="zero-origin"):
        validate_input_metadata(_metadata(origin=(0, 0, 1, 0, 0)))
    with pytest.raises(ValueError, match="5D"):
        validate_input_metadata(
            _metadata(shape=(1, 512, 1024, 2048), origin=(0, 0, 0, 0))
        )
