"""Tests for ExecutionPolicy and its synthesis from config."""
import torch

from aind_torch_utils.execution import ExecutionPolicy


def test_defaults():
    p = ExecutionPolicy()
    assert p.input_dtype == torch.float32
    assert p.autocast is False
    assert p.inference_mode is True
    assert p.compile is False
    assert p.channels_last is False


def test_from_config_amp_on():
    p = ExecutionPolicy.from_config(
        amp=True, use_compile=True, compile_mode="reduce-overhead", compile_dynamic=None
    )
    assert p.input_dtype == torch.float16
    assert p.autocast is True
    assert p.compile is True
    assert p.compile_mode == "reduce-overhead"


def test_from_config_amp_off():
    p = ExecutionPolicy.from_config(
        amp=False, use_compile=False, compile_mode="default", compile_dynamic=False
    )
    assert p.input_dtype == torch.float32
    assert p.autocast is False
    assert p.compile is False
    assert p.compile_dynamic is False
