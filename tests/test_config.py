import pytest

from aind_torch_utils.config import InferenceConfig


def test_precision_performance_defaults_to_false():
    cfg = InferenceConfig(
        patch=(16, 16, 16),
        overlap=4,
        trim_voxels=2,
        block=(32, 32, 32),
    )

    assert cfg.use_tf32 is False
    assert cfg.cudnn_benchmark is False
    assert cfg.compile_mode == "default"
    assert cfg.compile_dynamic is None


@pytest.mark.parametrize("field", ["use_tf32", "cudnn_benchmark"])
def test_precision_performance_config_overrides(field):
    cfg = InferenceConfig(
        patch=(16, 16, 16),
        overlap=4,
        trim_voxels=2,
        block=(32, 32, 32),
        **{field: True},
    )

    assert getattr(cfg, field) is True


@pytest.mark.parametrize(
    ("requested_mode", "expected_mode"),
    [
        ("reduce-overhead", "default"),
        ("max-autotune", "max-autotune-no-cudagraphs"),
    ],
)
def test_compile_cudagraph_modes_are_downgraded_for_multi_cuda(
    requested_mode, expected_mode
):
    with pytest.warns(RuntimeWarning, match="threaded multi-GPU"):
        cfg = InferenceConfig(
            patch=(16, 16, 16),
            overlap=4,
            trim_voxels=2,
            block=(32, 32, 32),
            devices=["cuda:0", "cuda:1"],
            use_compile=True,
            compile_mode=requested_mode,
        )

    assert cfg.compile_mode == expected_mode


def test_compile_cudagraph_mode_allowed_for_single_cuda():
    cfg = InferenceConfig(
        patch=(16, 16, 16),
        overlap=4,
        trim_voxels=2,
        block=(32, 32, 32),
        devices=["cuda:0"],
        use_compile=True,
        compile_mode="reduce-overhead",
    )

    assert cfg.compile_mode == "reduce-overhead"
