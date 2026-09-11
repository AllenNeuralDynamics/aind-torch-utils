"""Tests for model and workflow dispatch in the Ray launcher."""

import json
from types import SimpleNamespace

import pytest

import aind_torch_utils.distributed.ray_launcher as launcher
from aind_torch_utils.workflow import Workflow


@pytest.fixture(autouse=True)
def inline_supervision(monkeypatch):
    """Dispatch tests run inline; subprocess behavior is tested separately."""
    monkeypatch.setattr(
        launcher,
        "supervise_shard",
        lambda target, args, cfg, metrics_json: target(*args),
    )


class _ImmediateRemote:
    """Small Ray remote-function stand-in that executes synchronously."""

    def __init__(self, fn, ray):
        self._fn = fn
        self._ray = ray

    def remote(self, *args, **kwargs):
        self._ray.remote_calls.append((args, kwargs))
        return self._fn(*args, **kwargs)


class _ImmediateRay:
    """Only the Ray surface used by ``ray_launcher.main``."""

    def __init__(self):
        self.init_kwargs = None
        self.shutdown_called = False
        self.get_called = False
        self.remote_options = []
        self.remote_calls = []

    def init(self, **kwargs):
        self.init_kwargs = kwargs

    def remote(self, **options):
        self.remote_options.append(options)

        def decorate(fn):
            return _ImmediateRemote(fn, self)

        return decorate

    def get(self, futures):
        self.get_called = True
        return futures

    def shutdown(self):
        self.shutdown_called = True


class _GetFailureRay(_ImmediateRay):
    """Ray stand-in that reports a remote failure from ``ray.get``."""

    def remote(self, **options):
        self.remote_options.append(options)
        ray = self

        class _DeferredRemote:
            @staticmethod
            def remote(*args, **kwargs):
                ray.remote_calls.append((args, kwargs))
                return object()

        def decorate(fn):
            del fn
            return _DeferredRemote

        return decorate

    def get(self, futures):
        self.get_called = True
        raise RuntimeError("shard failed")


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


def test_workflow_params_accept_inline_json(tmp_path):
    params = {"checkpoint_path": "/config/model.pth", "offset": 2.5}
    assert launcher._load_workflow_params(json.dumps(params)) == params

    path = tmp_path / "params.json"
    path.write_text(json.dumps(params))
    assert launcher._load_workflow_params(str(path)) == params


@pytest.mark.parametrize("local_fallback", [True, False])
def test_workflow_runs_in_local_and_ray_paths(tmp_path, monkeypatch, local_fallback):
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
            "--thread-dump-interval",
            "42",
        ]
    )
    cfg = launcher.InferenceConfig()
    model = object()
    captured = {}

    def run_model(selected, input_store, output_stores, selected_cfg, **kwargs):
        captured.update(
            selected=selected,
            input_store=input_store,
            output_stores=output_stores,
            cfg=selected_cfg,
            thread_dump_interval=kwargs["thread_dump_interval"],
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
        "thread_dump_interval": 42.0,
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

    assert context.spec.to_json() == {"data_copy_concurrency": {"limit": 12}}


@pytest.mark.parametrize(
    "chunk",
    [
        (1, 1, 64, 64, 64),
        (1, 1, 32, 16, 8),
    ],
)
def test_multi_shard_zarr_layout_accepts_equal_and_divisor_chunks(chunk):
    launcher._validate_zarr_chunk_layout(
        (64, 64, 64),
        chunk,
        (0, 0, 0, 0, 0),
        output_label="output 0",
    )


@pytest.mark.parametrize(
    ("chunk", "origin"),
    [
        ((1, 1, 128, 64, 64), (0, 0, 0, 0, 0)),
        ((1, 1, 48, 64, 64), (0, 0, 0, 0, 0)),
        ((1, 1, 32, 64, 64), (0, 0, 16, 0, 0)),
        ((1, 1, None, 64, 64), (0, 0, 0, 0, 0)),
    ],
)
def test_multi_shard_zarr_layout_rejects_unsafe_chunks(chunk, origin):
    with pytest.raises(
        ValueError,
        match=r"block=.*chunk=.*grid_origin=",
    ):
        launcher._validate_zarr_chunk_layout(
            (64, 64, 64),
            chunk,
            origin,
            output_label="output 0",
        )


def test_declared_zarr2_alias_is_validated():
    spec = {
        "driver": "zarr2",
        "kvstore": {"driver": "memory"},
        "metadata": {
            "shape": [1, 1, 128, 256, 256],
            "chunks": [1, 1, 64, 96, 64],
            "dtype": "<u2",
        },
        "create": True,
    }
    with pytest.raises(ValueError, match="Unsafe multi-shard Zarr"):
        launcher._validate_declared_zarr_output(
            spec, (64, 64, 64), output_label="output 0"
        )


def test_declared_unsafe_layout_fails_before_destructive_open_or_ray(
    tmp_path, monkeypatch
):
    args = _workflow_args(tmp_path, "--num-shards", "2")
    output_spec_path = tmp_path / "out.json"
    output_spec_path.write_text(
        json.dumps(
            {
                "driver": "zarr",
                "kvstore": {"driver": "memory"},
                "metadata": {
                    "shape": [1, 1, 128, 256, 256],
                    "chunks": [1, 1, 64, 96, 64],
                    "dtype": "<u2",
                },
                "create": True,
                "delete_existing": True,
            }
        )
    )
    monkeypatch.setattr(
        launcher,
        "open_ts_spec",
        lambda *args, **kwargs: pytest.fail("destructive output open occurred"),
    )
    monkeypatch.setattr(
        launcher,
        "_import_ray",
        lambda: pytest.fail("Ray was initialized"),
    )

    with pytest.raises(ValueError, match="Unsafe multi-shard Zarr"):
        launcher.main(args)


def test_effective_unsafe_layout_fails_before_ray_init(tmp_path, monkeypatch):
    args = _workflow_args(tmp_path, "--num-shards", "2")
    output_spec_path = tmp_path / "out.json"
    output_spec_path.write_text(
        json.dumps(
            {
                "driver": "zarr",
                "kvstore": {"driver": "memory"},
                "open": True,
            }
        )
    )
    layout = SimpleNamespace(
        write_chunk=SimpleNamespace(shape=(1, 1, 64, 64, 64)),
        grid_origin=(0, 0, 32, 0, 0),
    )
    monkeypatch.setattr(
        launcher,
        "open_ts_spec",
        lambda *args, **kwargs: SimpleNamespace(chunk_layout=layout),
    )
    monkeypatch.setattr(
        launcher,
        "_import_ray",
        lambda: pytest.fail("Ray was initialized"),
    )

    with pytest.raises(ValueError, match="grid_origin=\\(32, 0, 0\\)"):
        launcher.main(args)


def test_resume_rejects_delete_existing_before_output_open(tmp_path, monkeypatch):
    config_path = tmp_path / "resume.json"
    config_path.write_text(json.dumps({"resume": True}))
    args = _workflow_args(tmp_path, "--config", str(config_path))
    output_spec_path = tmp_path / "out.json"
    output_spec_path.write_text(
        json.dumps(
            {
                "driver": "zarr",
                "kvstore": "s3://bucket/out.zarr",
                "delete_existing": True,
            }
        )
    )
    monkeypatch.setattr(
        launcher,
        "open_ts_spec",
        lambda *args, **kwargs: pytest.fail("destructive output open occurred"),
    )

    with pytest.raises(ValueError, match="delete_existing"):
        launcher.main(args)


def test_single_node_ray_launches_eight_one_gpu_shards(tmp_path, monkeypatch):
    fake_ray = _ImmediateRay()
    shard_runs = []

    def capture_shard(
        run_args,
        workflow_params,
        cfg,
        input_spec,
        output_specs,
        metrics_json,
    ):
        shard_runs.append((cfg, metrics_json))

    monkeypatch.setattr(launcher, "_import_ray", lambda: fake_ray)
    monkeypatch.setattr(launcher, "open_ts_spec", lambda spec, **kwargs: spec)
    monkeypatch.setattr(launcher, "_run_shard", capture_shard)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")

    launcher.main(
        _workflow_args(
            tmp_path,
            "--num-shards",
            "8",
            "--cpus-per-shard",
            "8",
            "--gpus-per-shard",
            "1",
            "--metrics-json-template",
            "/results/metrics_shard{shard}.json",
        )
    )

    assert fake_ray.init_kwargs == {}
    assert fake_ray.remote_options == [
        {"num_cpus": 8.0, "num_gpus": 1.0, "max_retries": 0}
    ]
    assert len(fake_ray.remote_calls) == 8
    assert [cfg.shard_index for cfg, _ in shard_runs] == list(range(8))
    assert all(cfg.devices == ["cuda:0"] for cfg, _ in shard_runs)
    assert [path for _, path in shard_runs] == [
        f"/results/metrics_shard{shard}.json" for shard in range(8)
    ]
    assert fake_ray.get_called is True
    assert fake_ray.shutdown_called is True


def test_ray_shutdown_after_shard_failure(tmp_path, monkeypatch):
    fake_ray = _GetFailureRay()
    monkeypatch.setattr(launcher, "_import_ray", lambda: fake_ray)
    monkeypatch.setattr(launcher, "open_ts_spec", lambda spec, **kwargs: spec)

    with pytest.raises(RuntimeError, match="shard failed"):
        launcher.main(_workflow_args(tmp_path))

    assert fake_ray.get_called is True
    assert fake_ray.shutdown_called is True
