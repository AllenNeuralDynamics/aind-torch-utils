import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Dict, List, Optional

import psutil

BYTES_PER_MB = 1024 * 1024


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
