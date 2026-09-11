"""Bounded waits, progress reporting, and best-effort native stall diagnostics."""

from __future__ import annotations

import faulthandler
import json
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class PipelineTimeoutError(RuntimeError):
    """A storage operation or pipeline stopped making progress."""


class PipelineStopped(RuntimeError):
    """A cooperative wait was interrupted by pipeline shutdown."""


class PipelineProgress:
    """Track useful work and outstanding operations, not idle queue polling.

    An optional callback streams snapshots to a supervisor in another process.
    Transitions of storage operations are always reported; other progress is
    throttled to avoid sending one message per skipped resume marker.
    """

    def __init__(
        self,
        publish: Optional[Callable[[dict], None]] = None,
        publish_interval_s: float = 0.5,
    ):
        self._lock = threading.Lock()
        self._publish = publish
        self._last_sent = 0.0
        self._publish_interval_s = publish_interval_s
        self.last_progress = time.monotonic()
        self.phase = "startup"
        self.outstanding: dict[str, dict] = {}

    def _snapshot(self) -> dict:
        return {
            "phase": self.phase,
            "last_progress": self.last_progress,
            "outstanding": {k: dict(v) for k, v in self.outstanding.items()},
        }

    def snapshot(self) -> dict:
        """Return a consistent, serializable snapshot."""
        with self._lock:
            return self._snapshot()

    def _send(self, force: bool = False) -> None:
        now = time.monotonic()
        if self._publish is not None and (
            force or now - self._last_sent >= self._publish_interval_s
        ):
            self._publish(self._snapshot())
            self._last_sent = now

    def advance(self) -> None:
        """Record a completed read, batch, merge, write, or resume skip."""
        with self._lock:
            self.last_progress = time.monotonic()
            self._send()

    def running(self) -> None:
        """Start the progress deadline after setup and model warmup."""
        with self._lock:
            self.phase = "running"
            self.last_progress = time.monotonic()
            self._send(force=True)

    def begin(self, kind: str, details: dict, timeout_s: float) -> str:
        """Record an operation before entering any potentially blocking API."""
        with self._lock:
            key = f"{threading.current_thread().name}:{kind}:{time.monotonic_ns()}"
            self.outstanding[key] = {
                "kind": kind,
                "thread": threading.current_thread().name,
                "thread_ident": threading.get_ident(),
                "native_id": threading.get_native_id(),
                "started": time.monotonic(),
                "timeout_s": timeout_s,
                **details,
            }
            self._send(force=True)
            return key

    def end(self, key: str) -> None:
        """Remove an operation only after successful completion."""
        with self._lock:
            self.outstanding.pop(key, None)
            self.last_progress = time.monotonic()
            self._send(force=True)


def stall_reason(snapshot: dict, cfg: Any, started: float) -> Optional[str]:
    """Evaluate deadlines independently of the potentially stalled process."""
    now = time.monotonic()
    for operation in snapshot.get("outstanding", {}).values():
        elapsed = now - operation["started"]
        if elapsed >= operation["timeout_s"]:
            return f"{operation['kind']} timed out after {elapsed:.1f}s: {operation}"
    if snapshot.get("phase", "startup") == "startup":
        if now - started >= cfg.startup_timeout_s:
            return f"Shard startup timed out after {now - started:.1f}s"
    else:
        idle = now - snapshot["last_progress"]
        if idle >= cfg.progress_timeout_s:
            return f"No pipeline progress for {idle:.1f}s"
    return None


def wait_for_future(
    future: Any,
    stop_event: threading.Event,
    deadline: float,
    description: str,
) -> Any:
    """Poll a TensorStore future without turning poll timeouts into retries.

    The absolute deadline includes time since submission, including queueing.
    Cancellation of a future is not proof that native work has stopped; the
    process supervisor is responsible for isolating and retiring failed work.
    """
    while True:
        if stop_event.is_set():
            raise PipelineStopped(f"Stopped while waiting for {description}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PipelineTimeoutError(f"Deadline exceeded waiting for {description}")
        try:
            return future.result(timeout=min(0.1, remaining))
        except TimeoutError:
            if future.done():
                # It may have completed between the timed wait and done(). Read
                # the actual outcome instead of raising a stale polling timeout.
                return future.result(timeout=0)


def diagnostic_directory(cfg: Any, metrics_json: Optional[str]) -> Path:
    """Resolve the shared directory used by supervisor and child."""
    if cfg.diagnostics_dir:
        return Path(cfg.diagnostics_dir)
    return (Path(metrics_json).parent if metrics_json else Path.cwd()) / "diagnostics"


def capture_diagnostics(
    pid: int,
    prefix: Path,
    snapshot: dict,
    reason: str,
    timeout_s: float,
    python_signal_ready: bool = False,
) -> None:
    """Capture outstanding I/O, Python stacks, and native stacks before teardown.

    GDB is optional and subject to container ptrace permissions. Record failure
    explicitly, together with Linux kernel wait locations as a fallback. Never
    request locals or environment variables, which may contain credentials.
    """
    try:
        prefix.parent.mkdir(parents=True, exist_ok=True)
        now = time.monotonic()
        snapshot = dict(snapshot)
        snapshot["outstanding"] = {
            k: {**v, "elapsed_s": now - v["started"]}
            for k, v in snapshot.get("outstanding", {}).items()
        }
        prefix.with_suffix(".json").write_text(
            json.dumps({"pid": pid, "reason": reason, **snapshot}, indent=2)
        )
        if pid == os.getpid():
            with prefix.with_suffix(".python.txt").open("a") as stream:
                faulthandler.dump_traceback(file=stream, all_threads=True)
        elif python_signal_ready:
            try:
                os.kill(pid, signal.SIGUSR1)
            except ProcessLookupError:
                logger.warning(
                    "Process %d exited before Python stacks could be captured", pid
                )

        with prefix.with_suffix(".native.txt").open("w") as stream:
            debugger = shutil.which("gdb")
            if debugger:
                try:
                    result = subprocess.run(
                        [
                            debugger,
                            "--batch",
                            "--nx",
                            "-p",
                            str(pid),
                            "-ex",
                            "set pagination off",
                            "-ex",
                            "thread apply all bt",
                            "-ex",
                            "detach",
                        ],
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        timeout=timeout_s,
                        check=False,
                    )
                    stream.write(f"\nGDB exit status: {result.returncode}\n")
                except subprocess.TimeoutExpired:
                    stream.write(f"\nGDB timed out after {timeout_s}s\n")
            else:
                stream.write(
                    "GDB unavailable; install gdb for native user-space stacks.\n"
                )
            stream.write(
                "\nLinux thread wait locations (kernel stacks may require permission):\n"
            )
            for task in sorted(Path(f"/proc/{pid}/task").glob("*")):
                for name in ("comm", "wchan", "stack"):
                    try:
                        stream.write(
                            f"{task.name}/{name}: {task.joinpath(name).read_text()}\n"
                        )
                    except OSError as exc:
                        stream.write(f"{task.name}/{name}: {exc}\n")
        logger.error("Stall diagnostics saved to %s.*", prefix)
    except Exception:
        logger.exception("Could not capture stall diagnostics for pid=%s", pid)
