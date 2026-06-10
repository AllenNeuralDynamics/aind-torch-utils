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
