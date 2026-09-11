import threading
import time
import unittest.mock

import numpy as np
import pytest
import tensorstore as ts
import torch
from torch import nn

from aind_torch_utils.config import InferenceConfig
from aind_torch_utils.execution import ExecutionPolicy
from aind_torch_utils.models import SharedEncoderModel
from aind_torch_utils.outputs import OutputSpec
from aind_torch_utils.recovery import PipelineTimeoutError
from aind_torch_utils.run import (
    _resolve_output_specs,
    _validate_inversion,
    run,
    run_workflow,
)
from aind_torch_utils.workflow import Workflow, WorkflowRegistry


def test_watchdog_and_shared_join_deadline_bound_uncooperative_workers(
    dummy_data,
    monkeypatch,
    tmp_path,
):
    input_store, output_store = dummy_data
    release = threading.Event()
    threads = [
        threading.Thread(target=release.wait, name=f"stuck-{i}") for i in range(3)
    ]
    reports = []
    monkeypatch.setattr(
        "aind_torch_utils.run._setup_worker_threads",
        lambda *args, **kwargs: ([threads[0]], [threads[1]], [threads[2]]),
    )
    monkeypatch.setattr(
        "aind_torch_utils.run.capture_diagnostics", lambda *args: reports.append(args)
    )
    cfg = InferenceConfig(
        devices=["cpu"],
        progress_timeout_s=0.05,
        shutdown_timeout_s=0.1,
        diagnostics_dir=str(tmp_path),
    )
    started = time.monotonic()
    try:
        with pytest.raises(PipelineTimeoutError, match="Shutdown deadline"):
            run(DummyModel(), input_store, output_store, cfg)
        assert time.monotonic() - started < 1
        assert reports and "No pipeline progress" in reports[0][3]
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=1)


def test_worker_failure_cannot_deadlock_sentinel_delivery_into_full_queue(
    dummy_data,
    monkeypatch,
):
    input_store, output_store = dummy_data

    def setup(*args, **kwargs):
        stop, prep_q, errors = args[5], args[7], args[-1]
        prep_q.put(object())

        def fail():
            errors.append(("gpu-0", RuntimeError("GPU failed")))
            stop.set()

        return [], [threading.Thread(target=fail)], []

    monkeypatch.setattr("aind_torch_utils.run._setup_worker_threads", setup)
    cfg = InferenceConfig(
        devices=["cpu"], max_inflight_batches=1, shutdown_timeout_s=0.1
    )
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="Worker thread.*gpu-0"):
        run(DummyModel(), input_store, output_store, cfg)
    assert time.monotonic() - started < 1


class DummyModel(nn.Module):
    """A dummy model that returns its input."""

    def forward(self, x):
        return x


class DummyMultiOutputModel(nn.Module):
    """Returns N copies of its input stacked along dim 1: (B, N, Z, Y, X)."""

    def __init__(self, n_outputs: int = 2):
        super().__init__()
        self.n_outputs = n_outputs

    def forward(self, x):
        # x: (B, 1, Z, Y, X) → repeat N times along dim 1 → (B, N, Z, Y, X)
        return x.squeeze(1).unsqueeze(1).expand(-1, self.n_outputs, -1, -1, -1)


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return tmp_path


@pytest.fixture
def dummy_data(temp_dir):
    """Create dummy data for testing."""
    shape = (1, 1, 32, 32, 32)
    dtype = "<u2"
    input_path = temp_dir / "input.zarr"
    output_path = temp_dir / "output.zarr"

    # Create dummy input data
    spec = {
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": str(input_path)},
        "metadata": {
            "shape": shape,
            "chunks": (1, 1, 16, 16, 16),
            "dtype": dtype,
        },
    }
    data = ts.open(spec, create=True).result()
    # Min value 1 to identify gaps in output due to incorrect stitching
    arr = np.random.randint(1, 65535, size=shape, dtype=np.uint16)
    data.write(arr).result()

    # Prepare output spec
    output_spec = {
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": str(output_path)},
        "metadata": {
            "shape": shape,
            "chunks": (1, 1, 16, 16, 16),
            "dtype": dtype,
        },
    }
    ts.open(output_spec, create=True).result()

    return ts.open(spec).result(), ts.open(output_spec).result()


def test_run_pipeline(temp_dir, dummy_data):
    """
    Test the run function with a dummy model and data.
    This is an integration test for the pipeline.
    """
    input_store, output_store = dummy_data
    metrics_json = temp_dir / "metrics.json"

    # Use CPU for testing if no CUDA devices are available
    devices = ["cuda:0"] if torch.cuda.is_available() else ["cpu"]

    # If no CUDA devices, we need to patch GpuWorker to not use cuda streams
    if not torch.cuda.is_available():
        with unittest.mock.patch("aind_torch_utils.run.GpuWorker") as mock_gpu_worker:
            # The mock needs to have a run method that can be called in a thread
            mock_gpu_worker.return_value.run.side_effect = lambda stop_event: None
            _run_test_logic(
                input_store, output_store, metrics_json, devices, DummyModel()
            )
    else:
        _run_test_logic(input_store, output_store, metrics_json, devices, DummyModel())

    # Assertions
    assert metrics_json.exists()

    # Verify output data is same as input for DummyModel
    input_data = input_store.read().result()
    output_data = output_store.read().result()

    # The mock GpuWorker does not process data, so we only check for equality if cuda is available
    if torch.cuda.is_available():
        np.testing.assert_array_equal(input_data, output_data)
        # assert all values are > 0 to check for gaps during stitching
        assert np.all(output_data > 0)
    else:
        # If no cuda, output will be empty
        assert np.all(output_data == 0)


def _run_test_logic(input_store, output_store, metrics_json, devices, model):
    """Helper function to run the test logic."""
    cfg = InferenceConfig(
        patch=(16, 16, 16),
        overlap=4,
        trim_voxels=2,
        seam_mode="trim",
        block=(32, 32, 32),
        batch_size=4,
        t_idx=0,
        c_idx=0,
        devices=devices,
        amp=False,
        max_inflight_batches=10,
        normalize=False,  # so we get the same data out as we put in
    )

    run(
        model=model,
        input_store=input_store,
        output_store=output_store,
        cfg=cfg,
        metrics_json=str(metrics_json),
        metrics_interval=0.1,
        num_prep_workers=1,
        num_writer_workers=1,
    )


def test_run_requires_work_store_when_resume_is_enabled(dummy_data):
    input_store, output_store = dummy_data
    cfg = InferenceConfig(
        patch=(16, 16, 16),
        overlap=4,
        trim_voxels=2,
        block=(32, 32, 32),
        devices=["cpu"],
        amp=False,
        normalize=False,
        output_denormalize=False,
        resume=True,
    )

    with pytest.raises(ValueError, match="requires a BlockWorkStore"):
        run(DummyModel(), input_store, output_store, cfg)


def test_run_prepares_and_forwards_resume_store(dummy_data, monkeypatch):
    input_store, output_store = dummy_data
    captured = {}

    class _WorkStore:
        def prepare(self, shard_spec):
            captured["prepared"] = shard_spec

    class _Monitor:
        def join(self, timeout=None):
            return None

        def get_data(self):
            return []

    def setup_threads(*args, **kwargs):
        captured["forwarded"] = args[-2]
        return [], [], []

    monkeypatch.setattr(
        "aind_torch_utils.run._setup_monitors",
        lambda *args: (_Monitor(), _Monitor()),
    )
    monkeypatch.setattr(
        "aind_torch_utils.run._setup_worker_threads",
        setup_threads,
    )
    work_store = _WorkStore()
    cfg = InferenceConfig(
        patch=(16, 16, 16),
        overlap=4,
        trim_voxels=2,
        block=(32, 32, 32),
        devices=["cpu"],
        amp=False,
        normalize=False,
        output_denormalize=False,
        resume=True,
    )

    run(DummyModel(), input_store, output_store, cfg, work_store=work_store)

    assert captured["prepared"].index == 0
    assert captured["forwarded"] is work_store


@pytest.fixture
def multi_output_data(tmp_path):
    """Two output stores (float32) matching the 32³ input volume."""
    shape = (1, 1, 32, 32, 32)
    input_path = tmp_path / "input_mo.zarr"
    input_spec = {
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": str(input_path)},
        "metadata": {"shape": shape, "chunks": (1, 1, 16, 16, 16), "dtype": "<u2"},
    }
    data = ts.open(input_spec, create=True).result()
    arr = np.random.randint(1, 65535, size=shape, dtype=np.uint16)
    data.write(arr).result()
    in_store = ts.open(input_spec).result()

    out_stores = []
    for i in range(2):
        out_path = tmp_path / f"output_mo_{i}.zarr"
        out_spec = {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": str(out_path)},
            "metadata": {"shape": shape, "chunks": (1, 1, 16, 16, 16), "dtype": "<f4"},
        }
        ts.open(out_spec, create=True).result()
        out_stores.append(ts.open(out_spec).result())

    return in_store, out_stores


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for GPU pipeline"
)
def test_run_multi_output_pipeline(multi_output_data, tmp_path):
    """Pipeline writes two independent output stores from a multi-output model."""
    in_store, out_stores = multi_output_data
    metrics_json = tmp_path / "metrics_mo.json"

    cfg = InferenceConfig(
        patch=(16, 16, 16),
        overlap=4,
        trim_voxels=2,
        seam_mode="trim",
        block=(32, 32, 32),
        batch_size=4,
        t_idx=0,
        c_idx=0,
        devices=["cuda:0"],
        amp=False,
        max_inflight_batches=10,
        normalize=False,
    )

    run(
        model=DummyMultiOutputModel(n_outputs=2),
        input_store=in_store,
        output_store=out_stores,  # list of 2 stores
        cfg=cfg,
        metrics_json=str(metrics_json),
        metrics_interval=0.1,
        num_prep_workers=1,
        num_writer_workers=1,
    )

    assert metrics_json.exists()
    # Both outputs should be non-zero (DummyMultiOutputModel copies input to both channels)
    for store in out_stores:
        out = store.read().result()
        assert np.all(out > 0), "Found zero voxels — stitching gap or missing writes"
    # Both outputs should be identical (same input repeated)
    out0 = out_stores[0].read().result().astype(np.float32)
    out1 = out_stores[1].read().result().astype(np.float32)
    np.testing.assert_allclose(out0, out1, rtol=1e-4)


def test_resolve_output_specs_synthesizes_from_store():
    cfg = InferenceConfig(devices=["cpu"], output_denormalize=True)
    specs = _resolve_output_specs(object(), None, cfg)
    assert len(specs) == 1
    assert specs[0].invert is True  # driven by output_denormalize
    assert specs[0].postprocess is None
    assert specs[0].accumulator_factory is not None


def test_resolve_output_specs_list_store():
    cfg = InferenceConfig(devices=["cpu"], output_denormalize=False)
    specs = _resolve_output_specs([object(), object()], None, cfg)
    assert len(specs) == 2
    assert all(s.invert is False for s in specs)


def test_resolve_output_specs_passthrough_explicit():
    cfg = InferenceConfig(devices=["cpu"])
    factory = object()
    explicit = [OutputSpec(store=object(), accumulator_factory=factory)]
    assert _resolve_output_specs(None, explicit, cfg) is explicit


def test_resolve_output_specs_rejects_both_and_neither():
    cfg = InferenceConfig(devices=["cpu"])
    with pytest.raises(ValueError, match="not both"):
        _resolve_output_specs(object(), [OutputSpec(object(), object())], cfg)
    with pytest.raises(ValueError, match="Provide output_store"):
        _resolve_output_specs(None, None, cfg)
    with pytest.raises(ValueError, match="non-empty"):
        _resolve_output_specs(None, [], cfg)


def test_resolve_output_specs_rejects_none_store():
    """A None store must fail here, not as an AttributeError in a writer thread."""
    cfg = InferenceConfig(devices=["cpu"])
    specs = [OutputSpec(store=None, accumulator_factory=object())]
    with pytest.raises(ValueError, match="store=None"):
        _resolve_output_specs(None, specs, cfg)


def test_validate_inversion_rejects_invert_without_inverse():
    """invert=True + forward-only preprocess must error, not silently skip."""

    class _ForwardOnly:
        def forward(self, block, ctx):
            return block, None

    specs = [OutputSpec(store=object(), accumulator_factory=object(), invert=True)]
    with pytest.raises(ValueError, match="defines no inverse"):
        _validate_inversion(_ForwardOnly(), specs)
    # invert=False everywhere -> a forward-only preprocess is fine.
    _validate_inversion(
        _ForwardOnly(),
        [OutputSpec(store=object(), accumulator_factory=object(), invert=False)],
    )


def test_validate_inversion_rejects_unknown_stage():
    """A typo'd inverse_stage must error, not silently disable inversion."""

    class _TypoStage:
        inverse_stage = "after-finalize"  # hyphen typo

        def forward(self, block, ctx):
            return block, None

        def inverse(self, block, state, ctx):
            return block

    specs = [OutputSpec(store=object(), accumulator_factory=object(), invert=True)]
    with pytest.raises(ValueError, match="Unknown inverse_stage"):
        _validate_inversion(_TypoStage(), specs)


def test_validate_inversion_defaults_missing_stage():
    """A duck-typed invertible transform without inverse_stage is accepted
    (the writer defaults it to after_finalize)."""

    class _NoStage:
        def forward(self, block, ctx):
            return block, None

        def inverse(self, block, state, ctx):
            return block

    specs = [OutputSpec(store=object(), accumulator_factory=object(), invert=True)]
    _validate_inversion(_NoStage(), specs)  # must not raise


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for GPU pipeline"
)
def test_run_workflow_end_to_end(temp_dir, dummy_data):
    """A registered Workflow (identity processor, no normalization) runs through
    run_workflow and reproduces the input."""
    input_store, output_store = dummy_data

    @WorkflowRegistry.register("test-identity-workflow")
    def _build(params):
        return Workflow(
            processor=DummyModel(),
            preprocess=None,  # run() synthesizes from cfg (normalize=False)
            execution=ExecutionPolicy.from_config(
                amp=False,
                use_compile=False,
                compile_mode="default",
                compile_dynamic=None,
            ),
        )

    cfg = InferenceConfig(
        patch=(16, 16, 16),
        overlap=4,
        trim_voxels=2,
        seam_mode="trim",
        block=(32, 32, 32),
        batch_size=4,
        devices=["cuda:0"],
        amp=False,
        max_inflight_batches=10,
        normalize=False,
    )

    workflow = WorkflowRegistry.build("test-identity-workflow")
    run_workflow(
        workflow,
        input_store,
        output_store,
        cfg,
        metrics_json=str(temp_dir / "wf_metrics.json"),
        metrics_interval=0.1,
    )

    np.testing.assert_array_equal(
        input_store.read().result(), output_store.read().result()
    )


def test_shared_encoder_model_forward():
    """SharedEncoderModel runs encoder once and fans out to N decoders."""
    encoder = nn.Identity()
    decoder_a = nn.Identity()
    decoder_b = nn.Identity()
    model = SharedEncoderModel(encoder, [decoder_a, decoder_b])

    x = torch.randn(2, 1, 8, 8, 8)
    out = model(x)
    # Expected: (B=2, N=2, Z=8, Y=8, X=8)
    assert out.shape == (2, 2, 8, 8, 8), f"Unexpected output shape: {out.shape}"
    # Both decoder outputs should equal the input (identity chain)
    np.testing.assert_allclose(out[:, 0].numpy(), x.squeeze(1).numpy(), rtol=1e-5)
    np.testing.assert_allclose(out[:, 1].numpy(), x.squeeze(1).numpy(), rtol=1e-5)


@pytest.mark.parametrize("error", [RuntimeError("main failed"), KeyboardInterrupt()])
def test_main_loop_errors_escape_after_cleanup(
    error, dummy_data, monkeypatch, tmp_path
):
    input_store, output_store = dummy_data

    class _Thread:
        def __init__(self, fail_on_first_check=False):
            self.daemon = None
            self.started = False
            self.joined = False
            self.checks = 0
            self.fail_on_first_check = fail_on_first_check

        def start(self):
            self.started = True

        def is_alive(self):
            self.checks += 1
            if self.fail_on_first_check and self.checks == 1:
                raise error
            return self.started and not self.joined

        def join(self, timeout=None):
            self.joined = True

    class _Monitor:
        def __init__(self):
            self.joined = False

        def join(self, timeout=None):
            self.joined = True

        def get_data(self):
            return []

    prep = _Thread(fail_on_first_check=True)
    gpu = _Thread()
    writer = _Thread()
    queue_monitor = _Monitor()
    system_monitor = _Monitor()
    captured = {}
    scheduled_dumps = []
    cancelled_dumps = []

    def setup_monitors(prep_q, write_queues, interval, stop_event):
        captured["stop_event"] = stop_event
        return queue_monitor, system_monitor

    monkeypatch.setattr(
        "aind_torch_utils.run._setup_monitors",
        setup_monitors,
    )
    monkeypatch.setattr(
        "aind_torch_utils.run._setup_worker_threads",
        lambda *args, **kwargs: ([prep], [gpu], [writer]),
    )
    monkeypatch.setattr(
        "aind_torch_utils.run.faulthandler.dump_traceback_later",
        lambda interval, repeat: scheduled_dumps.append((interval, repeat)),
    )
    monkeypatch.setattr(
        "aind_torch_utils.run.faulthandler.cancel_dump_traceback_later",
        lambda: cancelled_dumps.append(True),
    )

    cfg = InferenceConfig(
        patch=(16, 16, 16),
        overlap=4,
        trim_voxels=2,
        block=(32, 32, 32),
        devices=["cpu"],
        amp=False,
        normalize=False,
    )
    metrics_path = tmp_path / "failure-metrics.json"

    with pytest.raises(type(error), match=str(error) or None):
        run(
            DummyModel(),
            input_store,
            output_store,
            cfg,
            metrics_json=str(metrics_path),
            thread_dump_interval=17.5,
        )

    assert captured["stop_event"].is_set()
    assert all(thread.joined for thread in (prep, gpu, writer))
    assert queue_monitor.joined is True
    assert system_monitor.joined is True
    assert metrics_path.exists()
    assert scheduled_dumps == [(17.5, True)]
    assert cancelled_dumps == [True]
