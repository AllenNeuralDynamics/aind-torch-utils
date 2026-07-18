"""Tests for the checkpoint-aware denoise-net workflow recipe."""

from types import SimpleNamespace

import numpy as np
import pytest

import aind_torch_utils.recipes.denoise_net as recipe
from aind_torch_utils.transforms import IntensityTransformAdapter
from aind_torch_utils.workflow import WorkflowRegistry


class _BaseTransform:
    def __init__(self, scale=32.0):
        self.scale = scale

    def forward(self, array):
        return np.asarray(array, dtype=np.float32) / self.scale

    def inverse(self, array):
        return np.asarray(array, dtype=np.float32) * self.scale


class _OffsetTransform:
    def __init__(self, base, offset):
        self.base_transform = base
        self.offset = offset
        self.scale = base.scale

    def forward(self, array):
        return self.base_transform.forward(np.asarray(array) - self.offset)

    def inverse(self, array):
        return self.base_transform.inverse(array) + self.offset


def _install_fake_source(monkeypatch, loaded=None):
    calls = {"load": [], "offset": []}
    model = SimpleNamespace(model_config={"width_multiplier": 2})
    base = _BaseTransform(scale=16.0)

    def load_model(path, device="cuda"):
        calls["load"].append((path, device))
        return loaded if loaded is not None else (model, base)

    def build_volume_transform(transform, *, offset):
        calls["offset"].append((transform, offset))
        return _OffsetTransform(transform, offset)

    source = SimpleNamespace(
        load_model=load_model,
        build_volume_transform=build_volume_transform,
    )
    monkeypatch.setattr(recipe, "import_module", lambda name: source)
    return calls, model, base


def test_denoise_net_recipe_is_registered():
    assert "denoise-net" in WorkflowRegistry.list_workflows()


def test_workflow_loads_structured_checkpoint_on_cpu(monkeypatch):
    calls, model, base = _install_fake_source(monkeypatch)

    workflow = WorkflowRegistry.build(
        "denoise-net", {"checkpoint_path": "configured-checkpoint.pth"}
    )

    assert calls["load"] == [("configured-checkpoint.pth", "cpu")]
    assert calls["offset"] == []
    assert workflow.processor is model
    assert isinstance(workflow.preprocess, IntensityTransformAdapter)
    assert workflow.preprocess.transform is base
    assert workflow.outputs is None
    assert workflow.output_spec_factory is None
    assert workflow.execution is None


def test_omitting_offset_preserves_source_default_transform(monkeypatch):
    """A source loader's bare-checkpoint asinh default is used unchanged."""
    calls, _, source_default = _install_fake_source(monkeypatch)

    workflow = WorkflowRegistry.build(
        "denoise-net", {"checkpoint_path": "bare-state-dict.pth"}
    )

    assert calls["offset"] == []
    assert workflow.preprocess.transform is source_default


def test_explicit_offset_uses_source_composition_without_changing_scale(monkeypatch):
    calls, _, base = _install_fake_source(monkeypatch)

    workflow = WorkflowRegistry.build(
        "denoise-net",
        {"checkpoint_path": "checkpoint.pth", "offset": 73.5},
    )

    assert calls["offset"] == [(base, 73.5)]
    shifted = workflow.preprocess.transform
    assert shifted.base_transform is base
    assert shifted.scale == base.scale == 16.0
    values = np.array([73.5, 89.5, 16073.5])
    np.testing.assert_array_equal(
        shifted.forward(values), base.forward(values - 73.5)
    )


@pytest.mark.parametrize(
    "params,error,match",
    [
        ({}, ValueError, "checkpoint_path"),
        ({"checkpoint_path": ""}, ValueError, "checkpoint_path"),
        ({"checkpoint_path": 123}, ValueError, "checkpoint_path"),
        (
            {"checkpoint_path": "x.pth", "offset": "73.5"},
            TypeError,
            "offset",
        ),
        (
            {"checkpoint_path": "x.pth", "offset": float("nan")},
            ValueError,
            "finite",
        ),
        (
            {"checkpoint_path": "x.pth", "surprise": True},
            ValueError,
            "Unknown",
        ),
    ],
)
def test_workflow_rejects_invalid_parameters(monkeypatch, params, error, match):
    _install_fake_source(monkeypatch)
    with pytest.raises(error, match=match):
        WorkflowRegistry.build("denoise-net", params)


def test_workflow_reports_missing_optional_dependency(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(recipe, "import_module", missing)
    with pytest.raises(ImportError, match=r"aind-torch-utils\[denoise-net\]"):
        WorkflowRegistry.build("denoise-net", {"checkpoint_path": "x.pth"})


def test_workflow_reports_outdated_source_api(monkeypatch):
    monkeypatch.setattr(
        recipe,
        "import_module",
        lambda name: SimpleNamespace(load_model=lambda *a, **k: object()),
    )
    with pytest.raises(ImportError, match="checkpoint-aware transform API"):
        WorkflowRegistry.build("denoise-net", {"checkpoint_path": "x.pth"})


def test_workflow_reports_legacy_loader_return(monkeypatch):
    _install_fake_source(monkeypatch, loaded=object())
    with pytest.raises(ImportError, match=r"does not return \(model, transform\)"):
        WorkflowRegistry.build("denoise-net", {"checkpoint_path": "x.pth"})


def test_workflow_reports_transform_without_required_api(monkeypatch):
    _install_fake_source(monkeypatch, loaded=(object(), object()))
    with pytest.raises(ImportError, match="required forward/inverse API"):
        WorkflowRegistry.build("denoise-net", {"checkpoint_path": "x.pth"})
