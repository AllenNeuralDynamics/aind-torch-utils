"""Fault injection tests for bounded waits and process-isolated recovery."""

import ctypes
import json
import os
import signal
import socket
import threading
import time
from pathlib import Path

import pytest
import tensorstore as ts

from aind_torch_utils.config import InferenceConfig
from aind_torch_utils.distributed.supervisor import _poll_messages, supervise_shard
from aind_torch_utils.recovery import (
    PipelineProgress,
    PipelineStopped,
    PipelineTimeoutError,
    capture_diagnostics,
    stall_reason,
    wait_for_future,
)


def test_future_wait_times_out_without_resubmitting():
    promise, future = ts.Promise.new()
    started = time.monotonic()
    with pytest.raises(PipelineTimeoutError, match="input block"):
        wait_for_future(
            future, threading.Event(), started + 0.15, "input block (1,2,3)"
        )
    assert time.monotonic() - started < 1
    assert not future.done()
    promise.set_result("eventual result")
    assert future.result() == "eventual result"


def test_future_wait_stops_promptly_and_can_complete_normally():
    promise, future = ts.Promise.new()
    stop = threading.Event()
    timer = threading.Timer(0.05, stop.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(PipelineStopped):
            wait_for_future(future, stop, started + 10, "input")
    finally:
        timer.join()
    assert time.monotonic() - started < 1
    promise.set_result(42)
    assert (
        wait_for_future(future, threading.Event(), time.monotonic() + 1, "input") == 42
    )


def test_completed_future_error_is_not_mistaken_for_poll_timeout():
    promise, future = ts.Promise.new()
    promise.set_exception(TimeoutError("storage error"))
    with pytest.raises(TimeoutError, match="storage error"):
        wait_for_future(future, threading.Event(), time.monotonic() + 1, "input")


def test_future_completion_racing_a_poll_timeout_returns_actual_result():
    class RacingFuture:
        calls = 0

        def result(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("poll expired just before completion")
            return 42

        def done(self):
            return True

    assert (
        wait_for_future(
            RacingFuture(), threading.Event(), time.monotonic() + 1, "input"
        )
        == 42
    )


def test_other_workers_progress_does_not_hide_an_expired_read():
    cfg = InferenceConfig(read_timeout_s=0.01)
    progress = PipelineProgress()
    progress.running()
    token = progress.begin("read", {"block": [0, 0, 4, 5, 6]}, cfg.read_timeout_s)
    time.sleep(0.02)
    progress.advance()
    snapshot = progress.snapshot()
    assert "read timed out" in stall_reason(snapshot, cfg, time.monotonic())
    assert snapshot["outstanding"][token]["native_id"] == threading.get_native_id()
    progress.end(token)
    assert stall_reason(progress.snapshot(), cfg, time.monotonic()) is None


def _faulty_attempt(root, mode, *, progress):
    """Picklable child target; record process identity and emulate durable work."""
    root = Path(root)
    with (root / "attempts").open("a") as stream:
        stream.write(f"{os.getpid()}\n")
    if mode == "startup":
        threading.Event().wait()
    progress.running()
    if mode == "error":
        raise ValueError("invalid model")
    if mode == "crash":
        os._exit(7)
    if mode == "ignore_term":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if mode == "native":
        # PyDLL deliberately retains the GIL while blocked inside libc.
        ctypes.PyDLL(None).sleep(20)
    if mode == "recover" and (root / "done").exists():
        (root / "resumed").write_text("skipped completed block")
        return
    (root / "done").write_text("committed output and marker")
    if mode == "read":
        progress.begin("read", {"block": [0, 0, 1, 2, 3], "bbox": [[1, 2]]}, 0.15)
        while True:
            progress.advance()  # Simulate other healthy workers.
            time.sleep(0.02)
    if mode == "timeout_error":
        raise PipelineTimeoutError("input read deadline")
    threading.Event().wait()  # Simulate an uninterruptible native call.


def _supervisor_cfg(tmp_path, **overrides):
    return InferenceConfig(
        **{
            "devices": ["cpu"],
            "resume": True,
            "progress_timeout_s": 0.3,
            "startup_timeout_s": 10,
            "read_timeout_s": 0.15,
            "shutdown_timeout_s": 0.2,
            "diagnostic_timeout_s": 0.2,
            "retry_backoff_s": 0,
            "max_shard_retries": 1,
            "diagnostics_dir": str(tmp_path / "diagnostics"),
            **overrides,
        }
    )


def test_supervisor_restarts_in_fresh_process_and_preserves_completed_work(tmp_path):
    supervise_shard(
        _faulty_attempt, (str(tmp_path), "recover"), _supervisor_cfg(tmp_path)
    )
    pids = [int(pid) for pid in (tmp_path / "attempts").read_text().splitlines()]
    assert len(pids) == 2 and pids[0] != pids[1]
    assert (tmp_path / "resumed").read_text() == "skipped completed block"
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    reports = list((tmp_path / "diagnostics").glob("*.json"))
    assert len(reports) == 1
    assert "No pipeline progress" in json.loads(reports[0].read_text())["reason"]
    assert "_faulty_attempt" in reports[0].with_suffix(".python.txt").read_text()
    assert reports[0].with_suffix(".native.txt").exists()


@pytest.mark.parametrize(
    "mode", ["hang", "read", "crash", "timeout_error", "ignore_term", "native"]
)
def test_supervisor_exhausts_finite_retry_budget(tmp_path, mode):
    with pytest.raises(PipelineTimeoutError, match=r"after 2 attempt\(s\)"):
        supervise_shard(
            _faulty_attempt, (str(tmp_path), mode), _supervisor_cfg(tmp_path)
        )
    assert len((tmp_path / "attempts").read_text().splitlines()) == 2
    reports = list((tmp_path / "diagnostics").glob("*.json"))
    assert len(reports) == 2
    if mode == "read":
        report = json.loads(reports[0].read_text())
        operation = next(iter(report["outstanding"].values()))
        assert operation["block"] == [0, 0, 1, 2, 3]
        assert operation["elapsed_s"] >= 0.15


@pytest.mark.parametrize(
    "resume,mode,error",
    [
        (False, "hang", PipelineTimeoutError),
        (True, "error", RuntimeError),
    ],
)
def test_supervisor_does_not_retry_without_resume_or_on_model_error(
    tmp_path, resume, mode, error
):
    with pytest.raises(error):
        supervise_shard(
            _faulty_attempt,
            (str(tmp_path), mode),
            _supervisor_cfg(tmp_path, resume=resume),
        )
    assert len((tmp_path / "attempts").read_text().splitlines()) == 1


def test_supervisor_bounds_startup_before_any_progress(tmp_path):
    cfg = _supervisor_cfg(tmp_path, startup_timeout_s=0.2, max_shard_retries=0)
    with pytest.raises(PipelineTimeoutError, match="startup timed out"):
        supervise_shard(_faulty_attempt, (str(tmp_path), "startup"), cfg)


class _DiskS3Client:
    """Exercise the production marker backend across separate test processes."""

    def __init__(self, root):
        self.root = Path(root)

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self

    def paginate(self, Bucket, Prefix):
        yield {
            "Contents": [
                {"Key": str(p.relative_to(self.root))}
                for p in self.root.rglob("*.done")
            ]
        }

    def put_object(self, Bucket, Key, Body, **kwargs):
        path = self.root / Key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Body)


def _resume_marker_attempt(root, *, progress):
    from aind_torch_utils.work_state import BlockKey, S3MarkerBlockWorkStore

    root = Path(root)
    work_store = S3MarkerBlockWorkStore(
        bucket="test",
        prefix="resume",
        t_idx=0,
        c_idx=0,
        run_id="unchanged",
        s3_client=_DiskS3Client(root / "markers"),
    )
    work_store.prepare(None)
    progress.running()
    for k in range(2):
        lease = work_store.claim_block(BlockKey(0, 0, 0, 0, k, k))
        if lease is None:
            continue
        # Exclusive creation fails if a retry incorrectly reprocesses marked work.
        with (root / f"output{k}").open("x") as stream:
            stream.write("committed output")
        work_store.complete_block(lease)
        progress.advance()
        if k == 0:
            threading.Event().wait()


def test_fresh_attempt_loads_production_resume_markers(tmp_path):
    supervise_shard(_resume_marker_attempt, (str(tmp_path),), _supervisor_cfg(tmp_path))
    assert (tmp_path / "output0").read_text() == "committed output"
    assert (tmp_path / "output1").read_text() == "committed output"
    assert len(list((tmp_path / "markers").rglob("*.done"))) == 2


def test_native_diagnostic_failure_is_recorded_without_blocking(tmp_path, monkeypatch):
    import subprocess

    import aind_torch_utils.recovery as recovery

    monkeypatch.setattr(recovery.shutil, "which", lambda name: "/usr/bin/gdb")

    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 0.1
        raise subprocess.TimeoutExpired(args[0], 0.1)

    monkeypatch.setattr(recovery.subprocess, "run", timeout)
    prefix = tmp_path / "diagnostic"
    capture_diagnostics(os.getpid(), prefix, {"outstanding": {}}, "test stall", 0.1)
    assert "GDB timed out" in prefix.with_suffix(".native.txt").read_text()
    assert json.loads(prefix.with_suffix(".json").read_text())["reason"] == "test stall"


def test_partial_progress_message_cannot_block_the_watchdog():
    receiver, sender = socket.socketpair()
    receiver.setblocking(False)
    try:
        sender.sendall(b'["pro')
        started = time.monotonic()
        messages, buffered, closed = _poll_messages(receiver, b"")
        assert time.monotonic() - started < 0.5
        assert messages == [] and not closed
        sender.sendall(b'gress", {"phase": "running"}]\n')
        messages, buffered, closed = _poll_messages(receiver, buffered)
        assert messages == [["progress", {"phase": "running"}]]
        assert buffered == b"" and not closed
    finally:
        receiver.close()
        sender.close()


@pytest.mark.parametrize(
    "field",
    [
        "read_timeout_s",
        "write_timeout_s",
        "progress_timeout_s",
        "startup_timeout_s",
        "shutdown_timeout_s",
        "diagnostic_timeout_s",
    ],
)
@pytest.mark.parametrize("invalid", [0, -1, float("inf"), float("nan")])
def test_deadlines_must_be_positive_and_finite(field, invalid):
    with pytest.raises(ValueError):
        InferenceConfig(**{field: invalid})
