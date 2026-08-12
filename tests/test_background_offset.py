"""Tests for the local background-offset prepass helpers."""

import numpy as np
import pytest

from aind_torch_utils import background_offset


def test_spec_with_stats_resolution_accepts_v2_and_v3():
    for driver in ("zarr", "zarr3"):
        original = {
            "driver": driver,
            "kvstore": "s3://bucket/sample.zarr/0",
        }
        rewritten = background_offset.spec_with_stats_resolution(original, "5")
        assert rewritten["kvstore"] == "s3://bucket/sample.zarr/5"
        assert original["kvstore"].endswith("/0")


def test_replace_nested_kvstore_resolution():
    kvstore = {
        "base": {"driver": "s3", "bucket": "bucket", "path": "sample.zarr/0"},
        "path": "unused",
    }
    rewritten = background_offset.replace_kvstore_resolution(kvstore, "5")
    assert rewritten["path"] == "5"


def test_compute_background_offset_ignores_zeros(monkeypatch):
    array = np.array([0, 0, 2, 4, 8, 16], dtype=np.uint16).reshape(1, 1, 1, 2, 3)
    monkeypatch.setattr(
        background_offset,
        "_percentile_tdigest",
        lambda sample, percentile: float(np.percentile(sample, percentile)),
    )
    assert background_offset.compute_background_offset(array, 0, 0, 0) == 2.0
    assert (
        background_offset.compute_background_offset(
            array, 0, 0, 0, ignore_zeros=False
        )
        == 0.0
    )


def test_compute_background_offset_validates_indices_and_percentile(monkeypatch):
    array = np.ones((1, 1, 2, 2, 2))
    monkeypatch.setattr(background_offset, "_percentile_tdigest", lambda *_: 1.0)
    with pytest.raises(IndexError, match="t_idx"):
        background_offset.compute_background_offset(array, 1, 0, 1)
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        background_offset.compute_background_offset(array, 0, 0, 101)


def test_build_workflow_params_does_not_require_local_checkpoint():
    assert background_offset.build_workflow_params("/config/model.pth", 2) == {
        "checkpoint_path": "/config/model.pth",
        "offset": 2.0,
    }
