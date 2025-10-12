from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from copy import deepcopy
from typing import Any, List, Optional, Tuple

from torch import nn

import aind_torch_utils.models  # This registers all models when imported
from aind_torch_utils.config import InferenceConfig
from aind_torch_utils.model_registry import ModelRegistry
from aind_torch_utils.monitoring import QueueMonitor, SystemMonitor
from aind_torch_utils.distributed.sharding import ShardSpec, make_shard_spec
from aind_torch_utils.utils import open_ts_spec
from aind_torch_utils.workers import GpuWorker, PrepWorker, WriterWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _put_until_stop(
    q: queue.Queue,
    item: Any,
    stop_event: threading.Event,
    timeout: float = 0.1,
) -> None:
    """Puts an item on a queue, waiting until space is available or a stop
    event is set.

    Parameters
    ----------
    q : queue.Queue
        The queue to put the item on.
    item : Any
        The item to put on the queue.
    stop_event : threading.Event
        An event that signals to stop waiting.
    timeout : float, optional
        The timeout for the queue put operation, by default 0.1.
    """
    while True:
        try:
            q.put(item, timeout=timeout)
            break
        except queue.Full:
            if stop_event.is_set():
                break
            continue


def _write_metrics_json(
    metrics_json: str, monitor: QueueMonitor, sys_monitor: SystemMonitor
) -> None:
    """Write queue and system metrics to a JSON file.

    Parameters
    ----------
    metrics_json : str
        Path to the output JSON file.
    monitor : QueueMonitor
        The queue monitor instance.
    sys_monitor : SystemMonitor
        The system monitor instance.
    """
    try:
        metrics = {
            "queue_monitor": monitor.get_data(),
            "system_monitor": sys_monitor.get_data(),
        }
        # Ensure parent directory exists before writing
        parent_dir = os.path.dirname(metrics_json)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(metrics_json, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Wrote metrics to {metrics_json}")
    except Exception:
        logger.exception("Failed to write metrics JSON")


def _setup_queues(
    num_writer_queues: int, maxsize: int
) -> Tuple[queue.Queue, List[queue.Queue]]:
    """Sets up the queues for the pipeline.

    Parameters
    ----------
    num_writer_queues : int
        The number of writer queues to create.
    maxsize : int
        The maximum size of the queues.

    Returns
    -------
    Tuple[queue.Queue, List[queue.Queue]]
        A tuple containing the prep queue and a list of writer queues.
    """
    prep_q = queue.Queue(maxsize=maxsize)
    write_queues = [queue.Queue(maxsize=maxsize) for _ in range(num_writer_queues)]
    return prep_q, write_queues


def _setup_monitors(
    prep_q: queue.Queue,
    write_queues: List[queue.Queue],
    metrics_interval: float,
    stop_event: threading.Event,
) -> Tuple[QueueMonitor, SystemMonitor]:
    """Sets up the queue and system monitors.

    Parameters
    ----------
    prep_q : queue.Queue
        The prep queue.
    write_queues : List[queue.Queue]
        A list of writer queues.
    metrics_interval : float
        The interval at which to sample the queues and system.
    stop_event : threading.Event
        An event to signal the monitors to stop.

    Returns
    -------
    Tuple[QueueMonitor, SystemMonitor]
        A tuple containing the queue monitor and system monitor.
    """
    # start queue monitor
    q_monitor = QueueMonitor(
        queues={
            "prep": prep_q,
            **{f"write-{i}": q for i, q in enumerate(write_queues)},
        },
        stop_event=stop_event,
        interval_s=metrics_interval,
    )
    q_monitor.start()

    # start system monitor
    sys_monitor = SystemMonitor(
        stop_event=stop_event,
        interval_s=metrics_interval,
    )
    sys_monitor.start()

    return q_monitor, sys_monitor


def _setup_workers(
    model: nn.Module,
    input_store: Any,
    output_store: Any,
    cfg: InferenceConfig,
    shard_spec: ShardSpec,
    num_prep_workers: int,
    prep_q: queue.Queue,
    write_queues: List[queue.Queue],
) -> Tuple[List[PrepWorker], List[GpuWorker], List[WriterWorker]]:
    """Sets up the workers for the pipeline.

    Parameters
    ----------
    model : nn.Module
        The model to use for denoising.
    input_store : Any
        The input data store.
    output_store : Any
        The output data store.
    cfg : InferenceConfig
        The denoising configuration.
    num_prep_workers : int
        The number of prep workers to create.
    prep_q : queue.Queue
        The prep queue.
    write_queues : List[queue.Queue]
        A list of writer queues.

    Returns
    -------
    Tuple[List[PrepWorker], List[GpuWorker], List[WriterWorker]]
        A tuple containing the prep workers, GPU workers, and writer workers.
    """
    local_prep = max(1, num_prep_workers)
    global_worker_count = max(1, local_prep * cfg.shard_count)
    global_worker_offset = shard_spec.index * local_prep
    prep_workers = [
        PrepWorker(
            cfg,
            input_store,
            prep_q,
            cfg.patch,
            shard_spec,
            worker_id=i,
            num_workers=local_prep,
            global_worker_offset=global_worker_offset,
            global_worker_count=global_worker_count,
        )
        for i in range(local_prep)
    ]
    gpu_workers = [
        GpuWorker(cfg, deepcopy(model), device, prep_q, write_queues)
        for device in cfg.devices
    ]
    writer_workers = [
        WriterWorker(cfg, output_store, write_queues[i])
        for i in range(len(write_queues))
    ]
    return prep_workers, gpu_workers, writer_workers


def _setup_worker_threads(
    model: nn.Module,
    input_store: Any,
    output_store: Any,
    cfg: InferenceConfig,
    shard_spec: ShardSpec,
    stop_event: threading.Event,
    num_prep_workers: int,
    prep_q: queue.Queue,
    write_queues: List[queue.Queue],
) -> Tuple[List[threading.Thread], List[threading.Thread], List[threading.Thread]]:
    """Sets up the worker threads for the pipeline.

    Parameters
    ----------
    model : nn.Module
        The model to use for denoising.
    input_store : Any
        The input data store.
    output_store : Any
        The output data store.
    cfg : InferenceConfig
        The denoising configuration.
    shard_spec : ShardSpec
        Description of the spatial shard handled by this process.
    stop_event : threading.Event
        An event to signal the threads to stop.
    num_prep_workers : int
        The number of prep workers.
    prep_q : queue.Queue
        The prep queue.
    write_queues : List[queue.Queue]
        A list of writer queues.

    Returns
    -------
    Tuple[List[threading.Thread], List[threading.Thread], List[threading.Thread]]
        A tuple containing the prep threads, GPU threads, and writer threads.
    """

    # Workers
    prep_workers, gpu_workers, writer_workers = _setup_workers(
        model,
        input_store,
        output_store,
        cfg,
        shard_spec,
        num_prep_workers,
        prep_q,
        write_queues,
    )

    # Threads
    prep_threads = [
        threading.Thread(target=w.run, args=(stop_event,), name=f"prep-{i}")
        for i, w in enumerate(prep_workers)
    ]
    gpu_threads = [
        threading.Thread(target=worker.run, args=(stop_event,), name=f"gpu-{i}")
        for i, worker in enumerate(gpu_workers)
    ]
    writer_threads = [
        threading.Thread(target=w.run, args=(stop_event,), name=f"writer-{i}")
        for i, w in enumerate(writer_workers)
    ]

    return prep_threads, gpu_threads, writer_threads


def run(
    model: nn.Module,
    input_store: Any,
    output_store: Any,
    cfg: InferenceConfig,
    metrics_json: Optional[str] = None,
    metrics_interval: float = 0.5,
    num_prep_workers: int = 1,
    num_writer_workers: int = 1,
) -> None:
    """Runs the denoising pipeline.

    Parameters
    ----------
    model : nn.Module
        The model to use for denoising.
    input_store : Any
        The input data store.
    output_store : Any
        The output data store.
    cfg : InferenceConfig
        The denoising configuration.
    metrics_json : Optional[str], optional
        Path to the output JSON file for metrics, by default None.
    metrics_interval : float, optional
        The interval at which to sample the queues and system, by default 0.5.
    num_prep_workers : int, optional
        The number of prep workers to use, by default 1.
    num_writer_workers : int, optional
        The number of writer workers to use, by default 1.
    """
    # Validate shapes
    T, C, Z, Y, X = tuple(input_store.domain.shape)
    assert 0 <= cfg.t_idx < T and 0 <= cfg.c_idx < C, "Invalid t/c indices"

    shard_spec = make_shard_spec(
        (Z, Y, X),
        cfg.block,
        cfg.shard_count,
        cfg.shard_index,
        cfg.shard_strategy,
    )
    logger.info(
        "Shard %d/%d strategy=%s blocks=%s->%s tiles=%s idx=%s",
        shard_spec.index,
        shard_spec.count,
        shard_spec.strategy,
        shard_spec.block_start,
        shard_spec.block_stop,
        shard_spec.tiles_per_axis,
        shard_spec.tile_index,
    )

    # Queues
    prep_q, write_queues = _setup_queues(
        num_writer_workers, maxsize=cfg.max_inflight_batches
    )

    stop_event = threading.Event()

    # Monitors
    q_monitor, sys_monitor = _setup_monitors(
        prep_q, write_queues, metrics_interval, stop_event
    )

    # Threads
    prep_threads, gpu_threads, writer_threads = _setup_worker_threads(
        model,
        input_store,
        output_store,
        cfg,
        shard_spec,
        stop_event,
        num_prep_workers,
        prep_q,
        write_queues,
    )
    all_threads = prep_threads + gpu_threads + writer_threads

    prep_sentinels_sent = False
    writer_sentinels_sent = False

    t0 = time.perf_counter()
    try:
        for th in all_threads:
            th.daemon = False
            th.start()

        while any(th.is_alive() for th in all_threads):
            # when all prep threads are done, send GPU sentinels
            if (not prep_sentinels_sent) and all(
                not th.is_alive() for th in prep_threads
            ):
                for _ in range(len(cfg.devices)):
                    _put_until_stop(prep_q, None, stop_event, timeout=0.1)
                prep_sentinels_sent = True

            # when ALL GPU threads finish, close ALL writers (one sentinel per writer)
            if (not writer_sentinels_sent) and all(
                not th.is_alive() for th in gpu_threads
            ):
                for wq in write_queues:
                    _put_until_stop(wq, None, stop_event, timeout=0.1)
                writer_sentinels_sent = True

            # cooperative wait
            stop_event.wait(0.05)

    except (KeyboardInterrupt, Exception) as e:
        logger.exception(f"Caught {type(e).__name__}, initiating shutdown.")
    finally:
        logger.info("Setting stop event for all threads.")
        # GUARANTEE sentinel delivery on shutdown
        if not prep_sentinels_sent:
            for _ in range(len(cfg.devices)):
                _put_until_stop(prep_q, None, stop_event, timeout=0.1)
        if not writer_sentinels_sent:
            for wq in write_queues:
                _put_until_stop(wq, None, stop_event, timeout=0.1)

        stop_event.set()

        # Final join to ensure all threads have exited
        for th in all_threads:
            if th.is_alive():
                th.join()
        # Stop monitors
        q_monitor.join()
        sys_monitor.join()

        if metrics_json:
            _write_metrics_json(metrics_json, q_monitor, sys_monitor)

    t1 = time.perf_counter()
    throughput = (Z * Y * X * input_store.dtype.numpy_dtype.itemsize) / 1e6 / (t1 - t0)
    logger.info(f"Total time: {t1-t0:.2f}s")
    logger.info(f"Throughput: {throughput:.2f}MB/s")


def load_model(model_type: str, weights_path: Optional[str] = None) -> nn.Module:
    """Loads a model from the registry.

    Parameters
    ----------
    model_type : str
        Type of model to load (must be registered).
    weights_path : Optional[str], optional
        Path to the model weights file, by default None.

    Returns
    -------
    nn.Module
        The loaded model.

    Raises
    ------
    FileNotFoundError
        If the weights file is specified but not found.
    KeyError
        If the model type is not registered.
    """
    if weights_path and not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")

    return ModelRegistry.load_model(model_type, weights_path)


def _parse_args(argv: List[str]) -> argparse.Namespace:
    """Parses command line arguments.

    Parameters
    ----------
    argv : List[str]
        The command line arguments.

    Returns
    -------
    argparse.Namespace
        The parsed arguments.
    """
    ap = argparse.ArgumentParser(description="Scalable pytorch inference pipeline")
    ap.add_argument("--in-spec", type=str, required=True)
    ap.add_argument("--out-spec", type=str, required=True)
    ap.add_argument(
        "--model-type",
        type=str,
        required=True,
        help="Type of model to use (must be registered)",
    )
    ap.add_argument("--weights", type=str, help="Model weights path")
    ap.add_argument("--t", type=int, default=0)
    ap.add_argument("--c", type=int, default=0)
    ap.add_argument("--patch", type=int, nargs=3, default=(64, 64, 64))
    ap.add_argument("--overlap", type=int, default=10)
    ap.add_argument("--block", type=int, nargs=3, default=(256, 256, 256))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument(
        "--devices",
        nargs="+",
        default=["cuda:0"],
    )
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument(
        "--no-tf32", action="store_true", help="Disable TF32 (enabled by default)"
    )
    ap.add_argument("--compile", action="store_true", help="Enable torch.compile")
    ap.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=["reduce-overhead", "max-autotune"],
        help="torch.compile mode",
    )
    ap.add_argument(
        "--no-compile-dynamic",
        action="store_true",
        help="Disable dynamic shape support for torch.compile",
    )
    ap.add_argument(
        "--max-inflight-batches",
        type=int,
        default=64,
        help="Maximum prepared batches allowed in queues",
    )
    ap.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Total number of spatial shards (processes or Ray actors).",
    )
    ap.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Index of this shard in [0, shard-count).",
    )
    ap.add_argument(
        "--shard-strategy",
        type=str,
        default="contiguous-z",
        choices=["contiguous-z", "stride"],
        help="Spatial sharding strategy across processes.",
    )
    ap.add_argument(
        "--seam-mode",
        type=str,
        default="trim",
        choices=["trim", "blend"],
        help="Seam handling strategy",
    )
    ap.add_argument(
        "--trim-voxels",
        type=int,
        default=5,
        help="Voxels to trim from each non-boundary patch edge in trim mode",
    )
    ap.add_argument(
        "--halo",
        type=int,
        default=None,
        help=(
            "Explicit halo size. If omitted: in trim mode defaults to trim_voxels; "
            "in blend mode defaults to a small positive heuristic."
        ),
    )
    ap.add_argument(
        "--min-blend-weight",
        type=float,
        default=0.05,
        help="Minimum blend weight floor (blend mode)",
    )
    ap.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Numerical epsilon for divisions",
    )
    ap.add_argument(
        "--norm-lower",
        type=float,
        default=0.5,
        help="Lower percentile for per-patch normalization or global min.",
    )
    ap.add_argument(
        "--norm-upper",
        type=float,
        default=99.9,
        help="Upper percentile for per-patch normalization or global max.",
    )
    ap.add_argument(
        "--normalize",
        type=str,
        default="percentile",
        choices=["percentile", "global", "false"],
        help="Normalization strategy.",
    )
    ap.add_argument(
        "--clip-norm",
        nargs="*",
        type=float,
        default=None,
        metavar=("LO", "HI"),
        help=(
            "Optional clipping after percentile normalization. Usage: "
            "'--clip-norm' (no values) => clip to [0,1]; "
            "'--clip-norm LO HI' => clip to [LO,HI]; omit flag => no clipping."
        ),
    )
    ap.add_argument(
        "--metrics-json",
        type=str,
        default="metrics.json",
        help="Write metrics over time to this JSON file",
    )
    ap.add_argument(
        "--metrics-interval",
        type=float,
        default=0.5,
        help="Queue sampling interval in seconds (default: 0.5)",
    )
    ap.add_argument(
        "--prep-workers",
        type=int,
        default=4,
        help="Number of CPU prep workers",
    )
    ap.add_argument(
        "--writer-workers",
        type=int,
        default=4,
        help="Number of writer workers",
    )
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """The main entry point for the script.

    Parameters
    ----------
    argv : Optional[List[str]], optional
        The command line arguments, by default None.
    """
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    model = load_model(args.model_type, args.weights)

    in_arr = open_ts_spec(args.in_spec)
    out_arr = open_ts_spec(args.out_spec)

    cfg = InferenceConfig.from_cli_args(args)
    logger.info(f"Inference config:\n{cfg}")

    run(
        model,
        in_arr,
        out_arr,
        cfg,
        args.metrics_json,
        args.metrics_interval,
        num_prep_workers=max(1, args.prep_workers),
        num_writer_workers=max(1, args.writer_workers),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
