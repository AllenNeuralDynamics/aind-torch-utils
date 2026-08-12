"""Regression tests for assets copied into the Beaker image."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_gpu_metrics_wrapper_has_unix_shebang_and_line_endings():
    wrapper = REPOSITORY_ROOT / "beaker/run-with-gpu-metrics.sh"
    contents = wrapper.read_bytes()

    assert contents.startswith(b"#!/usr/bin/env bash\n")
    assert b"\r" not in contents
