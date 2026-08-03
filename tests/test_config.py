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
    assert cfg.max_pending_writes == 2
    assert cfg.tensorstore_data_copy_concurrency == 8
    assert cfg.resume is False
    assert cfg.work_store == "none"


def test_max_pending_writes_must_be_positive():
    with pytest.raises(ValueError, match="max_pending_writes must be > 0"):
        InferenceConfig(max_pending_writes=0)


def test_tensorstore_data_copy_concurrency_must_be_positive():
    with pytest.raises(
        ValueError, match="tensorstore_data_copy_concurrency must be > 0"
    ):
        InferenceConfig(tensorstore_data_copy_concurrency=0)


def test_resume_selects_s3_markers_and_validates_namespaces():
    assert InferenceConfig(resume=True).work_store == "s3-markers"

    with pytest.raises(ValueError, match="resume_marker_prefix"):
        InferenceConfig(resume_marker_prefix="")
    with pytest.raises(ValueError, match="resume_run_id"):
        InferenceConfig(resume_run_id="")


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
@pytest.mark.parametrize(
    "devices",
    [["cuda:0"], ["cuda:0", "cuda:1"]],
    ids=["single_cuda", "multi_cuda"],
)
def test_compile_cudagraph_modes_are_downgraded_for_cuda(
    devices, requested_mode, expected_mode
):
    # Threaded capture fails on a single GPU too (capture runs on a worker
    # thread, separate from warmup), so the downgrade applies whenever any
    # CUDA device is present, not just multi-GPU.
    with pytest.warns(RuntimeWarning, match="threaded"):
        cfg = InferenceConfig(
            patch=(16, 16, 16),
            overlap=4,
            trim_voxels=2,
            block=(32, 32, 32),
            devices=devices,
            use_compile=True,
            compile_mode=requested_mode,
        )

    assert cfg.compile_mode == expected_mode


def test_non_cudagraph_compile_mode_is_left_unchanged_for_cuda():
    cfg = InferenceConfig(
        patch=(16, 16, 16),
        overlap=4,
        trim_voxels=2,
        block=(32, 32, 32),
        devices=["cuda:0"],
        use_compile=True,
        compile_mode="max-autotune-no-cudagraphs",
    )

    assert cfg.compile_mode == "max-autotune-no-cudagraphs"
