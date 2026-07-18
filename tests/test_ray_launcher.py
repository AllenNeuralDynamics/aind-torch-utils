"""Tests for model and workflow dispatch in the Ray launcher."""

import json

import pytest

import aind_torch_utils.distributed.ray_launcher as launcher
from aind_torch_utils.workflow import Workflow


class _ImmediateRemote:
    """Small Ray remote-function stand-in that executes synchronously."""

    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _ImmediateRay:
    """Only the Ray surface used by ``ray_launcher.main``."""

    def __init__(self):
        self.init_kwargs = None
        self.shutdown_called = False

    def init(self, **kwargs):
        self.init_kwargs = kwargs

    def remote(self, **options):
        del options
        return _ImmediateRemote

    def get(self, futures):
        return futures

    def shutdown(self):
        self.shutdown_called = True


def _workflow_args(tmp_path, *extra):
    in_spec = tmp_path / "in.json"
    out_spec = tmp_path / "out.json"
    params = tmp_path / "params.json"
    in_spec.write_text(json.dumps({"driver": "input"}))
    out_spec.write_text(json.dumps({"driver": "output", "create": True}))
    params.write_text(json.dumps({"checkpoint_path": "checkpoint.pth"}))
    return [
        "--in-spec",
        str(in_spec),
        "--out-spec",
        str(out_spec),
        "--workflow",
        "test-workflow",
        "--workflow-params",
        str(params),
        "--num-shards",
        "1",
        *extra,
    ]


@pytest.mark.parametrize("local_fallback", [True, False])
def test_workflow_runs_in_local_and_ray_paths(
    tmp_path, monkeypatch, local_fallback
):
    captured = {}
    workflow = Workflow(processor=object())

    def build(name, params):
        captured["build"] = (name, params)
        return workflow

    def run_workflow(selected, input_store, output_stores, cfg, **kwargs):
        captured["run"] = (
            selected,
            input_store,
            output_stores,
            cfg.shard_index,
            kwargs,
        )

    monkeypatch.setattr(launcher.WorkflowRegistry, "build", build)
    monkeypatch.setattr(launcher, "run_workflow", run_workflow)
    monkeypatch.setattr(launcher, "open_ts_spec", lambda spec, **kwargs: spec)
    monkeypatch.setattr(
        launcher,
        "load_model",
        lambda *args, **kwargs: pytest.fail("legacy model path was used"),
    )

    extra = ["--local-fallback"] if local_fallback else []
    fake_ray = None
    if not local_fallback:
        fake_ray = _ImmediateRay()
        monkeypatch.setattr(launcher, "_import_ray", lambda: fake_ray)

    launcher.main(_workflow_args(tmp_path, *extra))

    assert captured["build"] == (
        "test-workflow",
        {"checkpoint_path": "checkpoint.pth"},
    )
    assert captured["run"][0] is workflow
    assert captured["run"][1] == {"driver": "input"}
    assert captured["run"][2] == [
        {
            "driver": "output",
            "create": False,
            "delete_existing": False,
            "open": True,
        }
    ]
    assert captured["run"][3] == 0
    if fake_ray is not None:
        assert fake_ray.shutdown_called is True


def test_model_path_still_loads_registered_model(monkeypatch):
    args = launcher.parse_inference_args(
        [
            "--in-spec",
            "in.json",
            "--out-spec",
            "out.json",
            "--model-type",
            "legacy",
            "--weights",
            "weights.pth",
        ]
    )
    cfg = launcher.InferenceConfig()
    model = object()
    captured = {}

    def run_model(selected, input_store, output_stores, selected_cfg, **kwargs):
        del kwargs
        captured.update(
            selected=selected,
            input_store=input_store,
            output_stores=output_stores,
            cfg=selected_cfg,
        )

    monkeypatch.setattr(launcher, "open_ts_spec", lambda spec, **kwargs: spec)
    monkeypatch.setattr(launcher, "load_model", lambda *args: model)
    monkeypatch.setattr(launcher, "run", run_model)

    launcher._run_shard(
        args,
        {},
        cfg,
        {"driver": "input"},
        [{"driver": "output"}],
        "metrics.json",
    )

    assert captured == {
        "selected": model,
        "input_store": {"driver": "input"},
        "output_stores": [{"driver": "output"}],
        "cfg": cfg,
    }


def test_run_shard_uses_one_tensorstore_context_for_all_stores(monkeypatch):
    args = launcher.parse_inference_args(
        [
            "--in-spec",
            "in.json",
            "--out-spec",
            "out.json",
            "--model-type",
            "legacy",
        ]
    )
    cfg = launcher.InferenceConfig(
        tensorstore_data_copy_concurrency=10,
    )
    shared_context = object()
    opened = []

    def open_store(spec, *, context=None):
        opened.append((spec, context))
        return spec

    monkeypatch.setattr(
        launcher,
        "_make_shard_tensorstore_context",
        lambda selected_cfg: shared_context,
    )
    monkeypatch.setattr(launcher, "open_ts_spec", open_store)
    monkeypatch.setattr(launcher, "load_model", lambda *args: object())
    monkeypatch.setattr(launcher, "run", lambda *args, **kwargs: None)

    launcher._run_shard(
        args,
        {},
        cfg,
        {"driver": "input"},
        [{"driver": "output-0"}, {"driver": "output-1"}],
        None,
    )

    assert [spec for spec, _ in opened] == [
        {"driver": "input"},
        {"driver": "output-0"},
        {"driver": "output-1"},
    ]
    assert all(context is shared_context for _, context in opened)


def test_shard_tensorstore_context_uses_configured_copy_limit():
    cfg = launcher.InferenceConfig(tensorstore_data_copy_concurrency=12)

    context = launcher._make_shard_tensorstore_context(cfg)

    assert context.spec.to_json() == {
        "data_copy_concurrency": {"limit": 12}
    }
