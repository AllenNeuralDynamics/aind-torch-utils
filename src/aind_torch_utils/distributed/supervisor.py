"""Run each shard attempt in a fresh process with an external progress deadline."""

from __future__ import annotations

import ctypes
import faulthandler
import json
import logging
import multiprocessing
import os
import select
import signal
import socket
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from aind_torch_utils.recovery import (
    PipelineProgress,
    PipelineTimeoutError,
    capture_diagnostics,
    diagnostic_directory,
    stall_reason,
)

logger = logging.getLogger(__name__)


class _ProgressSender:
    """Serialize concurrent worker messages onto a newline-framed socket."""

    def __init__(self, connection: socket.socket):
        self.connection = connection
        self.lock = threading.Lock()

    def send(self, message: tuple) -> None:
        """Send a complete JSON record without interleaving worker threads."""
        data = json.dumps(message).encode() + b"\n"
        with self.lock:
            self.connection.sendall(data)


def _poll_messages(connection: socket.socket, buffered: bytes) -> tuple:
    """Never block waiting for the remainder of a partially written message.

    A native call could hold the child's GIL halfway through a send. A blocking
    multiprocessing Connection.recv() would then disable the external watchdog.
    """
    if not select.select([connection], [], [], 0.1)[0]:
        return [], buffered, False
    chunk = connection.recv(65536)
    if not chunk:
        return [], buffered, True
    lines = (buffered + chunk).split(b"\n")
    return [json.loads(line) for line in lines[:-1]], lines[-1], False


def _child_entry(
    target: Callable,
    args: tuple,
    connection: Any,
    prefix: str,
    publish_interval_s: float,
) -> None:
    """Arm diagnostics before constructing stores or touching CUDA."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    if os.name == "posix":
        os.setsid()
    if os.path.exists("/proc/self"):
        parent = os.getppid()
        libc = ctypes.CDLL(None, use_errno=True)
        # Permit the supervisor's GDB subprocess under Yama's parent-only policy.
        # Container seccomp/capability restrictions can still deny attachment.
        libc.prctl(0x59616D61, parent, 0, 0, 0)  # PR_SET_PTRACER
        libc.prctl(1, signal.SIGKILL, 0, 0, 0)  # PR_SET_PDEATHSIG
        if os.getppid() != parent:
            os._exit(1)
    path = Path(prefix).with_suffix(".python.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        faulthandler.enable(file=stream, all_threads=True)
        signal_ready = hasattr(signal, "SIGUSR1")
        if signal_ready:
            faulthandler.register(signal.SIGUSR1, file=stream, all_threads=True)
        sender = _ProgressSender(connection)
        progress = PipelineProgress(
            lambda state: sender.send(("progress", state)),
            publish_interval_s,
        )
        sender.send(("ready", {"signal_ready": signal_ready}))
        try:
            target(*args, progress=progress)
        except BaseException as exc:
            sender.send(
                (
                    "failure",
                    {
                        "retryable": isinstance(exc, PipelineTimeoutError),
                        "error": traceback.format_exc(),
                        "snapshot": progress.snapshot(),
                    },
                )
            )
            # Keep the failing process and outstanding native work available for
            # diagnostics. Only the supervisor may retire it and start a retry.
            threading.Event().wait()
        else:
            sender.send(("success", None))
        finally:
            if signal_ready:
                faulthandler.unregister(signal.SIGUSR1)
            faulthandler.disable()
            connection.close()


def _stop_process(process: Any, timeout_s: float, grouped: bool) -> None:
    """Terminate, escalate to kill, and reap before allowing another attempt."""
    if process.is_alive():
        if grouped and os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            process.terminate()
        process.join(timeout_s)
    if grouped and os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.is_alive():
        process.kill()
    process.join(2)
    if process.is_alive():
        raise RuntimeError(
            f"Cannot reap shard process {process.pid}; refusing an overlapping retry"
        )


def supervise_shard(
    target: Callable,
    args: tuple,
    cfg: Any,
    metrics_json: Optional[str] = None,
) -> None:
    """Retry timeouts/crashes only with durable resume, never arbitrary errors.

    The spawn method deliberately avoids inheriting initialized CUDA or
    TensorStore state. Deadlines and native diagnostics live outside that state.
    """
    context = multiprocessing.get_context("spawn")
    retries = (
        cfg.max_shard_retries if cfg.resume and cfg.work_store == "s3-markers" else 0
    )
    directory = diagnostic_directory(cfg, metrics_json)
    directory.mkdir(parents=True, exist_ok=True)
    run_tag = f"shard{cfg.shard_index}-{time.time_ns()}"
    for attempt in range(retries + 1):
        prefix = directory / f"{run_tag}-attempt{attempt + 1}"
        receive, send = socket.socketpair()
        receive.setblocking(False)
        process = context.Process(
            target=_child_entry,
            args=(
                target,
                args,
                send,
                str(prefix),
                min(0.5, cfg.progress_timeout_s / 4),
            ),
        )
        started = time.monotonic()
        snapshot: dict = {"phase": "startup", "outstanding": {}}
        ready = False
        signal_ready = False
        reason = None
        buffered = b""
        retryable = True
        try:
            process.start()
        except BaseException:
            receive.close()
            send.close()
            process.close()
            raise
        send.close()
        logger.info(
            "Shard %d attempt %d/%d started pid=%d",
            cfg.shard_index,
            attempt + 1,
            retries + 1,
            process.pid,
        )
        try:
            while reason is None:
                messages, buffered, closed = _poll_messages(receive, buffered)
                for kind, payload in messages:
                    if kind == "ready":
                        ready = True
                        signal_ready = payload["signal_ready"]
                    elif kind == "progress":
                        snapshot = payload
                    elif kind == "failure":
                        snapshot = payload["snapshot"]
                        reason = payload["error"]
                        retryable = payload["retryable"]
                    elif kind == "success":
                        process.join(cfg.shutdown_timeout_s)
                        if not process.is_alive() and process.exitcode == 0:
                            return
                        reason = "Shard reported success but its process did not exit cleanly"
                    if reason is not None:
                        break
                if reason is None and closed:
                    reason = (
                        "Shard process closed its progress channel without a result"
                    )
                # Check even with continuous updates: other workers must not mask
                # an individual outstanding read/write that exceeded its deadline.
                if reason is None:
                    reason = stall_reason(snapshot, cfg, started)

            logger.error(
                "Shard %d attempt %d failed: %s", cfg.shard_index, attempt + 1, reason
            )
            capture_diagnostics(
                process.pid,
                prefix,
                snapshot,
                reason,
                cfg.diagnostic_timeout_s,
                signal_ready,
            )
        finally:
            try:
                _stop_process(process, cfg.shutdown_timeout_s, ready)
            finally:
                receive.close()
                if not process.is_alive():
                    process.close()
        if not retryable:
            raise RuntimeError(
                f"Shard {cfg.shard_index} failed (not retryable): {reason}"
            )
        if attempt == retries:
            raise PipelineTimeoutError(
                f"Shard {cfg.shard_index} failed after {attempt + 1} attempt(s): {reason}"
            )
        logger.warning(
            "Restarting shard %d from completion markers in %.1fs",
            cfg.shard_index,
            cfg.retry_backoff_s,
        )
        time.sleep(cfg.retry_backoff_s)
