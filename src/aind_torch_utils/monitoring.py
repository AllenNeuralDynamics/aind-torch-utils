import threading
import time
from dataclasses import dataclass, field
from queue import Queue
from typing import Dict, List, Optional, Set, Tuple

import psutil

BYTES_PER_MB = 1024 * 1024
BlockIndex = Tuple[int, int, int]
TIMING_SUMMARY_KEYS = (
    "preparation_s",
    "inference_s",
    "write_s",
    "prep_queue_put_wait_s",
    "prep_queue_residence_s",
    "gpu_transfer_overhead_s",
    "write_queue_put_wait_s",
    "write_queue_residence_s",
    "output_ready_wait_s",
    "total_block_processing_s",
)


@dataclass
class _BlockTiming:
    start_s: Optional[float] = None
    timings: Dict[str, float] = field(default_factory=dict)


class BlockProgressMonitor:
    """
    Tracks block-level progress and summary timing statistics.

    The monitor is thread-safe and stores only active block state plus aggregate
    duration samples for completed blocks. Exported data is JSON-serializable
    and intentionally summary-only.
    """

    def __init__(
        self,
        total_blocks: int,
        t0: Optional[float] = None,
    ) -> None:
        """
        Initializes the BlockProgressMonitor.

        Parameters
        ----------
        total_blocks : int
            The total number of blocks expected for the run.
        t0 : Optional[float], optional
            The reference start time (from time.perf_counter()). If None, the
            current time is used, by default None.
        """
        self.total_blocks = max(0, int(total_blocks))
        self.t0 = time.perf_counter() if t0 is None else t0
        self._lock = threading.Lock()
        self._active_blocks: Dict[BlockIndex, _BlockTiming] = {}
        self._completed_blocks: Set[BlockIndex] = set()
        self._timing_values: Dict[str, List[float]] = {
            key: [] for key in TIMING_SUMMARY_KEYS
        }

    @staticmethod
    def _key(block_idx: BlockIndex) -> BlockIndex:
        return tuple(block_idx)

    @staticmethod
    def _now(now_s: Optional[float] = None) -> float:
        return time.perf_counter() if now_s is None else now_s

    @staticmethod
    def _percentile(sorted_values: List[float], q: float) -> Optional[float]:
        if not sorted_values:
            return None
        if len(sorted_values) == 1:
            return sorted_values[0]

        pos = (len(sorted_values) - 1) * (q / 100.0)
        lo = int(pos)
        hi = min(lo + 1, len(sorted_values) - 1)
        frac = pos - lo
        return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac

    @classmethod
    def _summary(cls, values: List[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {
                "count": 0,
                "min": None,
                "mean": None,
                "max": None,
                "p50": None,
                "p95": None,
            }

        ordered = sorted(values)
        return {
            "count": len(values),
            "min": ordered[0],
            "mean": sum(values) / len(values),
            "max": ordered[-1],
            "p50": cls._percentile(ordered, 50.0),
            "p95": cls._percentile(ordered, 95.0),
        }

    def _add_timing(
        self, block_idx: BlockIndex, timing_key: str, duration_s: float
    ) -> None:
        if duration_s < 0:
            return
        key = self._key(block_idx)
        with self._lock:
            if key in self._completed_blocks:
                return
            rec = self._active_blocks.setdefault(key, _BlockTiming())
            rec.timings[timing_key] = rec.timings.get(timing_key, 0.0) + float(
                duration_s
            )

    def start_block(
        self, block_idx: BlockIndex, now_s: Optional[float] = None
    ) -> None:
        """
        Records the wall-clock start for a block if it has not started.

        Parameters
        ----------
        block_idx : BlockIndex
            The (z, y, x) index of the block.
        now_s : Optional[float], optional
            Explicit timestamp for testing. Defaults to time.perf_counter().
        """
        key = self._key(block_idx)
        now = self._now(now_s)
        with self._lock:
            if key in self._completed_blocks:
                return
            rec = self._active_blocks.get(key)
            if rec is None:
                self._active_blocks[key] = _BlockTiming(start_s=now)
            elif rec.start_s is None:
                rec.start_s = now

    def record_preparation_time(
        self, block_idx: BlockIndex, duration_s: float
    ) -> None:
        """
        Adds active preparation time for a block.

        Parameters
        ----------
        block_idx : BlockIndex
            The (z, y, x) index of the block.
        duration_s : float
            Active preparation duration in seconds.
        """
        self._add_timing(block_idx, "preparation_s", duration_s)

    def add_inference_time(self, block_idx: BlockIndex, duration_s: float) -> None:
        """
        Adds GPU inference time for one batch in a block.

        Parameters
        ----------
        block_idx : BlockIndex
            The (z, y, x) index of the block.
        duration_s : float
            Inference duration in seconds.
        """
        self._add_timing(block_idx, "inference_s", duration_s)

    def add_write_time(self, block_idx: BlockIndex, duration_s: float) -> None:
        """
        Adds active writer-stage time for a block.

        Parameters
        ----------
        block_idx : BlockIndex
            The (z, y, x) index of the block.
        duration_s : float
            Active writer-stage duration in seconds.
        """
        self._add_timing(block_idx, "write_s", duration_s)

    def add_prep_queue_put_wait_time(
        self, block_idx: BlockIndex, duration_s: float
    ) -> None:
        """Adds time spent blocked on a full preparation queue."""
        self._add_timing(block_idx, "prep_queue_put_wait_s", duration_s)

    def add_prep_queue_residence_time(
        self, block_idx: BlockIndex, duration_s: float
    ) -> None:
        """Adds time a batch spent resident in the preparation queue."""
        self._add_timing(block_idx, "prep_queue_residence_s", duration_s)

    def add_gpu_transfer_overhead_time(
        self, block_idx: BlockIndex, duration_s: float
    ) -> None:
        """Adds GPU stage time outside model inference."""
        self._add_timing(block_idx, "gpu_transfer_overhead_s", duration_s)

    def add_write_queue_put_wait_time(
        self, block_idx: BlockIndex, duration_s: float
    ) -> None:
        """Adds time spent blocked on a full writer queue."""
        self._add_timing(block_idx, "write_queue_put_wait_s", duration_s)

    def add_write_queue_residence_time(
        self, block_idx: BlockIndex, duration_s: float
    ) -> None:
        """Adds time a prediction batch spent resident in a writer queue."""
        self._add_timing(block_idx, "write_queue_residence_s", duration_s)

    def add_output_ready_wait_time(
        self, block_idx: BlockIndex, duration_s: float
    ) -> None:
        """Adds writer time spent waiting for GPU output readiness."""
        self._add_timing(block_idx, "output_ready_wait_s", duration_s)

    def complete_block(
        self, block_idx: BlockIndex, now_s: Optional[float] = None
    ) -> None:
        """
        Marks a block complete and commits its stage timings to summaries.

        Parameters
        ----------
        block_idx : BlockIndex
            The (z, y, x) index of the block.
        now_s : Optional[float], optional
            Explicit timestamp for testing. Defaults to time.perf_counter().
        """
        key = self._key(block_idx)
        now = self._now(now_s)
        with self._lock:
            if key in self._completed_blocks:
                return

            rec = self._active_blocks.pop(key, _BlockTiming(start_s=now))
            self._completed_blocks.add(key)

            for timing_key, duration_s in rec.timings.items():
                if timing_key in self._timing_values:
                    self._timing_values[timing_key].append(duration_s)
            if rec.start_s is not None:
                self._timing_values["total_block_processing_s"].append(
                    max(0.0, now - rec.start_s)
                )

    def get_data(self, now_s: Optional[float] = None) -> Dict:
        """
        Return a JSON-serializable snapshot of block progress and timing stats.

        Parameters
        ----------
        now_s : Optional[float], optional
            Explicit timestamp for testing. Defaults to time.perf_counter().

        Returns
        -------
        Dict
            A dictionary containing progress counts, rates, ETA, and summary
            timing statistics.
        """
        now = self._now(now_s)
        with self._lock:
            completed_blocks = len(self._completed_blocks)
            active_blocks = len(self._active_blocks)
            timing_values = {
                key: list(values) for key, values in self._timing_values.items()
            }

        incomplete_blocks = max(self.total_blocks - completed_blocks, 0)
        percent_complete = (
            100.0
            if self.total_blocks == 0
            else min(100.0, completed_blocks / self.total_blocks * 100.0)
        )
        elapsed_s = max(0.0, now - self.t0)
        blocks_per_sec = completed_blocks / elapsed_s if elapsed_s > 0 else 0.0
        if incomplete_blocks == 0:
            eta_s = 0.0
        elif blocks_per_sec > 0:
            eta_s = incomplete_blocks / blocks_per_sec
        else:
            eta_s = None

        return {
            "total_blocks": self.total_blocks,
            "completed_blocks": completed_blocks,
            "incomplete_blocks": incomplete_blocks,
            "active_blocks": active_blocks,
            "percent_complete": percent_complete,
            "elapsed_s": elapsed_s,
            "blocks_per_sec": blocks_per_sec,
            "eta_s": eta_s,
            "timing_summary": {
                key: self._summary(timing_values.get(key, []))
                for key in TIMING_SUMMARY_KEYS
            },
        }

    def format_progress(self, now_s: Optional[float] = None) -> str:
        """
        Return a concise human-readable progress line.

        Parameters
        ----------
        now_s : Optional[float], optional
            Explicit timestamp for testing. Defaults to time.perf_counter().

        Returns
        -------
        str
            A formatted progress message.
        """
        data = self.get_data(now_s)
        eta_s = data["eta_s"]
        eta_text = "unknown" if eta_s is None else f"{eta_s:.1f}s"
        return (
            "Block progress: "
            f"{data['completed_blocks']}/{data['total_blocks']} "
            f"({data['percent_complete']:.1f}%), "
            f"{data['blocks_per_sec']:.3f} blocks/s, ETA {eta_text}"
        )


@dataclass
class QueueSample:
    """
    A single sample of queue sizes at a specific time.

    Attributes
    ----------
    t_rel_s : float
        The time of the sample in seconds, relative to the monitor's start time.
    sizes : Dict[str, Optional[int]]
        A dictionary mapping queue names to their sizes. A size of None
        indicates that the queue size could not be determined.
    """

    t_rel_s: float
    sizes: Dict[str, Optional[int]]  # None when qsize() is not implemented


class QueueMonitor:
    """
    Periodically sample Queue.qsize() for a set of queues and expose results.

    Notes:
    - Queue.qsize() is approximate and may raise NotImplementedError on some
      platforms/queue types. When that happens, size is recorded as None.
    - Uses the shared stop_event and an interruptible sleep to exit promptly.
    - Samples are stored in-memory with an optional max_samples ring buffer.
    - Thread-safe snapshotting for get_data().
    """

    def __init__(
        self,
        queues: Dict[str, Queue],
        stop_event: threading.Event,
        interval_s: float = 0.25,
        t0: Optional[float] = None,
        max_samples: Optional[int] = 10000,
        daemon: bool = True,
    ) -> None:
        """
        Initializes the QueueMonitor.

        Parameters
        ----------
        queues : Dict[str, Queue]
            A dictionary mapping names to the queue objects to monitor.
        stop_event : threading.Event
            An event to signal the monitoring thread to stop.
        interval_s : float, optional
            The interval in seconds at which to sample the queues, by default 0.25.
        t0 : Optional[float], optional
            The reference start time (from time.perf_counter()). If None, the
            current time is used, by default None.
        max_samples : Optional[int], optional
            The maximum number of samples to store. If None, all samples are
            kept. If exceeded, older samples are discarded, by default 10000.
        daemon : bool, optional
            Whether the monitoring thread should be a daemon thread, by default True.
        """
        self.queues = queues
        self.stop_event = stop_event
        self.interval_s = interval_s
        self.samples: List[QueueSample] = []
        self._lock = threading.Lock()
        self._max_samples = max_samples
        self.t0 = time.perf_counter() if t0 is None else t0
        self._th = threading.Thread(
            target=self._run, name="queue-monitor", daemon=daemon
        )

    def start(self) -> None:
        """Starts the monitoring thread."""
        self._th.start()

    def join(self, timeout: Optional[float] = None) -> None:
        """
        Waits for the monitoring thread to finish.

        Parameters
        ----------
        timeout : Optional[float], optional
            The maximum time in seconds to wait for the thread to join,
            by default None.
        """
        if self._th.is_alive():
            self._th.join(timeout)

    def stop(self, timeout: Optional[float] = None) -> None:
        """
        Stops the monitoring thread.

        Parameters
        ----------
        timeout : Optional[float], optional
            The maximum time in seconds to wait for the thread to join after
            setting the stop event, by default None.
        """
        self.stop_event.set()
        self.join(timeout)

    def _run(self) -> None:
        """The main run loop for the monitoring thread."""
        try:
            while not self.stop_event.is_set():
                t_rel = time.perf_counter() - self.t0
                snap: Dict[str, Optional[int]] = {}
                for name, q in self.queues.items():
                    try:
                        snap[name] = q.qsize()
                    except NotImplementedError:
                        snap[name] = None  # not supported on this platform/queue type
                with self._lock:
                    self._append_sample(QueueSample(t_rel, snap))
                # interruptible sleep
                if self.stop_event.wait(self.interval_s):
                    break
        except Exception:
            # Ensure other threads can react; avoid silent thread death
            self.stop_event.set()
            raise

    def _append_sample(self, sample: QueueSample) -> None:
        """
        Appends a new sample, enforcing the max_samples limit.

        Parameters
        ----------
        sample : QueueSample
            The sample to append.
        """
        self.samples.append(sample)
        if self._max_samples is not None and self._max_samples >= 0:
            excess = len(self.samples) - self._max_samples
            if excess > 0:
                del self.samples[:excess]

    def get_data(self) -> Dict:
        """
        Return a JSON-serializable snapshot of recorded queue sizes.

        Returns
        -------
        Dict
            A dictionary containing the recorded queue size data, structured
            for easy conversion to JSON.
        """
        with self._lock:
            samples = tuple(self.samples)
        if not samples:
            return {}

        data: Dict[str, Dict] = {"queues": {}}
        for name, q in self.queues.items():
            data["queues"][name] = {
                "max_size": getattr(q, "maxsize", 0) or 0,
                "samples": [],
            }
        for sample in samples:
            for q_name, q_size in sample.sizes.items():
                if q_name in data["queues"]:
                    data["queues"][q_name]["samples"].append(
                        {"t_rel_s": sample.t_rel_s, "size": q_size}
                    )
        return data


@dataclass
class SystemSample:
    """
    A single sample of system metrics at a specific time.

    Attributes
    ----------
    t_rel_s : float
        The time of the sample in seconds, relative to the monitor's start time.
    cpu_percent : float
        The CPU usage percentage for the current process. May exceed 100% on
        multi-core systems unless normalized.
    ram_rss_mb : float
        The Resident Set Size (RSS) memory usage of the current process in MiB.
    net_io_mbytes_sec : Dict[str, float]
        A dictionary with system-wide network I/O rates in MiB/s for "sent"
        and "recv".
    disk_io_mbytes_sec : Dict[str, float]
        A dictionary with system-wide disk I/O rates in MiB/s for "read"
        and "write".
    """

    t_rel_s: float
    cpu_percent: float  # May exceed 100% on multi-core unless normalized
    ram_rss_mb: float
    net_io_mbytes_sec: Dict[str, float]  # system-wide bytes sent/recv → MB/s
    disk_io_mbytes_sec: Dict[str, float]  # system-wide bytes read/write → MB/s


class SystemMonitor:
    """
    Periodically sample **process** CPU% and RAM, plus **system-wide** network and
    disk I/O rates.

    - CPU% uses psutil.Process.cpu_percent(interval=None). On multi-core systems,
      values can exceed 100%. Set normalize_cpu_percent=True to scale to 0–100%
      by logical CPUs.
    - RAM is Resident Set Size (RSS) in MiB (computed with 1024*1024).
    - Network/Disk rates are **system-wide totals**, not per-process (psutil lacks a
      cross-platform per-process network API). Units are MB/s (MiB/s by calculation).
    - Uses interruptible sleep via Event.wait and supports an optional ring buffer.
    - Thread-safe snapshotting for get_data().
    """

    def __init__(
        self,
        stop_event: threading.Event,
        interval_s: float = 1.0,
        t0: Optional[float] = None,
        max_samples: Optional[int] = 10000,
        daemon: bool = True,
        normalize_cpu_percent: bool = False,
    ) -> None:
        """
        Initializes the SystemMonitor.

        Parameters
        ----------
        stop_event : threading.Event
            An event to signal the monitoring thread to stop.
        interval_s : float, optional
            The interval in seconds at which to sample metrics, by default 1.0.
        t0 : Optional[float], optional
            The reference start time (from time.perf_counter()). If None, the
            current time is used, by default None.
        max_samples : Optional[int], optional
            The maximum number of samples to store. If None, all samples are
            kept. If exceeded, older samples are discarded, by default 10000.
        daemon : bool, optional
            Whether the monitoring thread should be a daemon thread, by default True.
        normalize_cpu_percent : bool, optional
            If True, normalizes CPU percentage to a 0-100% range based on the
            number of logical CPUs, by default False.
        """
        self.proc = psutil.Process()
        self.stop_event = stop_event
        self.interval_s = interval_s
        self.samples: List[SystemSample] = []
        self._lock = threading.Lock()
        self._max_samples = max_samples
        self.t0 = time.perf_counter() if t0 is None else t0
        self._th = threading.Thread(
            target=self._run, name="system-monitor", daemon=daemon
        )
        self.normalize_cpu_percent = normalize_cpu_percent

        # Initialize cpu_percent baseline (non-blocking)
        self.proc.cpu_percent(interval=None)

        # Initialize I/O counters and timestamp as close together as possible
        self.last_net_io = psutil.net_io_counters()
        self.last_disk_io = psutil.disk_io_counters()
        self.last_sample_time = time.perf_counter()

    def start(self) -> None:
        """Starts the monitoring thread."""
        self._th.start()

    def join(self, timeout: Optional[float] = None) -> None:
        """
        Waits for the monitoring thread to finish.

        Parameters
        ----------
        timeout : Optional[float], optional
            The maximum time in seconds to wait for the thread to join,
            by default None.
        """
        if self._th.is_alive():
            self._th.join(timeout)

    def stop(self, timeout: Optional[float] = None) -> None:
        """
        Stops the monitoring thread.

        Parameters
        ----------
        timeout : Optional[float], optional
            The maximum time in seconds to wait for the thread to join after
            setting the stop event, by default None.
        """
        self.stop_event.set()
        self.join(timeout)

    @staticmethod
    def _rate(delta_bytes: int, dt: float) -> float:
        """
        Calculate a rate in MiB/s.

        Parameters
        ----------
        delta_bytes : int
            The change in bytes over the interval.
        dt : float
            The time interval in seconds.

        Returns
        -------
        float
            The calculated rate in MiB/s. Returns 0.0 if dt is non-positive
            or if delta_bytes is negative (counter rollover).
        """
        if dt <= 0:
            return 0.0
        return max(0.0, float(delta_bytes) / BYTES_PER_MB / dt)

    def _run(self) -> None:
        """The main run loop for the monitoring thread."""
        try:
            while not self.stop_event.is_set():
                # interruptible sleep first to respect desired cadence
                if self.stop_event.wait(self.interval_s):
                    break

                now = time.perf_counter()
                t_rel = now - self.t0

                # CPU percent (may exceed 100 on multi-core)
                cpu = self.proc.cpu_percent(interval=None)
                if self.normalize_cpu_percent:
                    n = psutil.cpu_count(logical=True) or 1
                    cpu = cpu / n * 100.0

                # RSS in MiB
                ram_rss_mb = self.proc.memory_info().rss / BYTES_PER_MB

                # Compute deltas against last snapshot
                dt = now - self.last_sample_time
                if dt <= 0:
                    dt = self.interval_s

                # Network I/O (system-wide)
                current_net_io = psutil.net_io_counters()
                if current_net_io is None or self.last_net_io is None:
                    net_io_mbytes_sec = {"sent": 0.0, "recv": 0.0}
                else:
                    net_io_mbytes_sec = {
                        "sent": self._rate(
                            current_net_io.bytes_sent - self.last_net_io.bytes_sent,
                            dt,
                        ),
                        "recv": self._rate(
                            current_net_io.bytes_recv - self.last_net_io.bytes_recv,
                            dt,
                        ),
                    }
                if current_net_io is not None:
                    self.last_net_io = current_net_io

                # Disk I/O (system-wide)
                current_disk_io = psutil.disk_io_counters()
                if current_disk_io is None or self.last_disk_io is None:
                    disk_io_mbytes_sec = {"read": 0.0, "write": 0.0}
                else:
                    disk_io_mbytes_sec = {
                        "read": self._rate(
                            current_disk_io.read_bytes - self.last_disk_io.read_bytes,
                            dt,
                        ),
                        "write": self._rate(
                            current_disk_io.write_bytes - self.last_disk_io.write_bytes,
                            dt,
                        ),
                    }
                if current_disk_io is not None:
                    self.last_disk_io = current_disk_io

                self.last_sample_time = now

                with self._lock:
                    self._append_sample(
                        SystemSample(
                            t_rel,
                            cpu,
                            ram_rss_mb,
                            net_io_mbytes_sec,
                            disk_io_mbytes_sec,
                        )
                    )
        except Exception:
            self.stop_event.set()
            raise

    def _append_sample(self, sample: SystemSample) -> None:
        """
        Appends a new sample, enforcing the max_samples limit.

        Parameters
        ----------
        sample : SystemSample
            The sample to append.
        """
        self.samples.append(sample)
        if self._max_samples is not None and self._max_samples >= 0:
            excess = len(self.samples) - self._max_samples
            if excess > 0:
                del self.samples[:excess]

    def get_data(self) -> Dict:
        """
        Return a JSON-serializable snapshot of recorded system stats.

        Returns
        -------
        Dict
            A dictionary containing the recorded system metrics, structured
            for easy conversion to JSON.
        """
        with self._lock:
            samples = tuple(self.samples)
        if not samples:
            return {}

        data = {
            "t_rel_s": [s.t_rel_s for s in samples],
            "cpu_percent": [s.cpu_percent for s in samples],
            "ram_rss_mb": [s.ram_rss_mb for s in samples],
            "net_io_mbytes_sec": {
                "sent": [s.net_io_mbytes_sec["sent"] for s in samples],
                "recv": [s.net_io_mbytes_sec["recv"] for s in samples],
            },
            "disk_io_mbytes_sec": {
                "read": [s.disk_io_mbytes_sec["read"] for s in samples],
                "write": [s.disk_io_mbytes_sec["write"] for s in samples],
            },
        }
        return data
