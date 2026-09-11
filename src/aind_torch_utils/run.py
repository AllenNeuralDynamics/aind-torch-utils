"""
Module for running the inference pipeline with multiple threads and monitoring.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import os
import queue
import sys
import threading
import time
from copy import deepcopy
from typing import Any, List, Optional, Tuple, Union

from torch import nn

import aind_torch_utils.models  # This registers all models when imported
import aind_torch_utils.recipes  # This registers all workflow recipes when imported
from aind_torch_utils import transforms
from aind_torch_utils.accumulators import weighted_average_factory
from aind_torch_utils.config import InferenceConfig
from aind_torch_utils.distributed.sharding import ShardSpec, make_shard_spec
from aind_torch_utils.execution import ExecutionPolicy
from aind_torch_utils.model_registry import ModelRegistry
from aind_torch_utils.monitoring import QueueMonitor, SystemMonitor
from aind_torch_utils.outputs import OutputSpec
from aind_torch_utils.recovery import (
    PipelineProgress,
    PipelineStopped,
    PipelineTimeoutError,
    capture_diagnostics,
    diagnostic_directory,
    stall_reason,
)
from aind_torch_utils.transforms import BlockPreprocessor
from aind_torch_utils.utils import load_ts_spec, open_ts_spec
from aind_torch_utils.work_state import (
    BlockWorkStore,
    NoopBlockWorkStore,
    build_block_work_store,
    validate_resume_output_specs,
)
from aind_torch_utils.workers import GpuWorker, PrepWorker, WriterWorker
from aind_torch_utils.workflow import Workflow, WorkflowRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _start_periodic_thread_dumps(interval_s: Optional[float], shard_index: int) -> bool:
    """Schedule repeating all-thread tracebacks for diagnosing a stalled run.

    ``faulthandler.dump_traceback_later`` writes directly to stderr from a
    watchdog thread, so it remains useful when Python worker threads are
    deadlocked or blocked in native extension calls. The timer is process-global;
    inference runs are isolated in separate Ray worker processes.
    """
    if interval_s is None or interval_s <= 0:
        return False
    logger.info(
        "Shard %d scheduling periodic all-thread dumps every %.1fs.",
        shard_index,
        interval_s,
    )
    faulthandler.dump_traceback_later(interval_s, repeat=True)
    return True


def _cancel_periodic_thread_dumps(scheduled: bool, shard_index: int) -> None:
    """Cancel a thread-dump watchdog previously scheduled by this run."""
    if not scheduled:
        return
    faulthandler.cancel_dump_traceback_later()
    logger.info("Shard %d cancelled periodic all-thread dumps.", shard_index)


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
    output_specs: List[OutputSpec],
    cfg: InferenceConfig,
    shard_spec: ShardSpec,
    num_prep_workers: int,
    prep_q: queue.Queue,
    write_queues: List[queue.Queue],
    preprocess: BlockPreprocessor,
    execution: ExecutionPolicy,
    work_store: BlockWorkStore,
) -> Tuple[List[PrepWorker], List[GpuWorker], List[WriterWorker]]:
    """Sets up the workers for the pipeline.

    Parameters
    ----------
    model : nn.Module
        The model to use for inference.
    input_store : Any
        The input data store.
    output_specs : List[OutputSpec]
        One spec per model output channel (store + merge + post + invert).
    cfg : InferenceConfig
        The inference configuration.
    shard_spec : ShardSpec
        Description of the spatial shard handled by this process.
    num_prep_workers : int
        The number of prep workers to create.
    prep_q : queue.Queue
        The prep queue.
    write_queues : List[queue.Queue]
        A list of writer queues.
    preprocess : BlockPreprocessor
        The injected block transform shared by prep (forward) and writer (inverse).
    execution : ExecutionPolicy
        How the GPU processor runs (dtype/autocast/compile/channels_last).

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
            preprocess,
            execution,
            shard_spec=shard_spec,
            worker_id=i,
            num_workers=local_prep,
            global_worker_offset=global_worker_offset,
            global_worker_count=global_worker_count,
            work_store=work_store,
        )
        for i in range(local_prep)
    ]
    # Per-device copies only make sense for nn.Modules (independent weights on
    # each GPU). A plain-callable BlockProcessor is shared as-is: it manages
    # its own state and may hold handles deepcopy cannot pickle.
    gpu_workers = [
        GpuWorker(
            cfg,
            deepcopy(model) if isinstance(model, nn.Module) else model,
            device,
            prep_q,
            write_queues,
            execution,
        )
        for device in cfg.devices
    ]
    writer_workers = [
        WriterWorker(
            cfg,
            output_specs,
            write_queues[i],
            preprocess,
            work_store,
        )
        for i in range(len(write_queues))
    ]
    return prep_workers, gpu_workers, writer_workers


def _guarded_worker(
    run_fn: Any,
    stop_event: threading.Event,
    errors: List[Tuple[str, BaseException]],
    name: str,
) -> Any:
    """Wrap a worker's run() so an uncaught exception stops the pipeline.

    Without this, a dead worker leaves its peers spinning on full queues
    forever (hang) or lets run() return "successfully" with nothing written.
    """

    def target() -> None:
        try:
            run_fn(stop_event)
        except Exception as exc:  # noqa: BLE001 - any worker death is fatal
            if isinstance(exc, PipelineStopped) and stop_event.is_set():
                return
            logger.exception("Worker thread %s died; stopping pipeline.", name)
            errors.append((name, exc))
            stop_event.set()

    return target


def _setup_worker_threads(
    model: nn.Module,
    input_store: Any,
    output_specs: List[OutputSpec],
    cfg: InferenceConfig,
    shard_spec: ShardSpec,
    stop_event: threading.Event,
    num_prep_workers: int,
    prep_q: queue.Queue,
    write_queues: List[queue.Queue],
    preprocess: BlockPreprocessor,
    execution: ExecutionPolicy,
    work_store: BlockWorkStore,
    worker_errors: List[Tuple[str, BaseException]],
    *,
    progress: Optional[PipelineProgress] = None,
) -> Tuple[List[threading.Thread], List[threading.Thread], List[threading.Thread]]:
    """Sets up the worker threads for the pipeline.

    Parameters
    ----------
    model : nn.Module
        The model to use for inference.
    input_store : Any
        The input data store.
    output_specs : List[OutputSpec]
        One spec per model output channel.
    cfg : InferenceConfig
        The inference configuration.
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
    preprocess : BlockPreprocessor
        The injected block transform shared by prep and writer.
    execution : ExecutionPolicy
        How the GPU processor runs.
    worker_errors : list
        Receives ``(thread_name, exception)`` for any worker that dies; the
        caller re-raises after joins so failures are never silent.

    Returns
    -------
    Tuple[List[threading.Thread], List[threading.Thread], List[threading.Thread]]
        A tuple containing the prep threads, GPU threads, and writer threads.
    """

    # Workers
    prep_workers, gpu_workers, writer_workers = _setup_workers(
        model,
        input_store,
        output_specs,
        cfg,
        shard_spec,
        num_prep_workers,
        prep_q,
        write_queues,
        preprocess,
        execution,
        work_store,
    )
    for worker in prep_workers + gpu_workers + writer_workers:
        worker.progress = progress

    # Threads
    prep_threads = [
        threading.Thread(
            target=_guarded_worker(w.run, stop_event, worker_errors, f"prep-{i}"),
            name=f"prep-{i}",
        )
        for i, w in enumerate(prep_workers)
    ]
    gpu_threads = [
        threading.Thread(
            target=_guarded_worker(w.run, stop_event, worker_errors, f"gpu-{i}"),
            name=f"gpu-{i}",
        )
        for i, w in enumerate(gpu_workers)
    ]
    writer_threads = [
        threading.Thread(
            target=_guarded_worker(w.run, stop_event, worker_errors, f"writer-{i}"),
            name=f"writer-{i}",
        )
        for i, w in enumerate(writer_workers)
    ]

    return prep_threads, gpu_threads, writer_threads


def _resolve_output_specs(
    output_store: Union[Any, List[Any], None],
    outputs: Optional[List[OutputSpec]],
    cfg: InferenceConfig,
) -> List[OutputSpec]:
    """Return the per-output specs, synthesizing them from ``output_store`` if needed.

    Exactly one of ``output_store`` / ``outputs`` must be provided. When synthesizing,
    each store gets the default trim/blend merge and ``invert=cfg.output_denormalize``,
    reproducing the pre-refactor single-policy writer.
    """
    if outputs is not None:
        if output_store is not None:
            raise ValueError("Pass either output_store or outputs, not both.")
        if not outputs:
            raise ValueError("outputs must be a non-empty list of OutputSpec.")
        # Catch a None store here, in the caller's thread, instead of as an
        # opaque AttributeError when the first block completes in a writer.
        for i, spec in enumerate(outputs):
            if spec.store is None:
                raise ValueError(
                    f"OutputSpec {i} has store=None; every output needs a "
                    "destination store."
                )
        return outputs

    if output_store is None:
        raise ValueError("Provide output_store (or outputs).")

    output_stores = output_store if isinstance(output_store, list) else [output_store]
    factory = weighted_average_factory(
        cfg.eps,
        cfg.overlap,
        cfg.seam_mode,
        cfg.trim_voxels,
        cfg.min_blend_weight,
    )
    return [
        OutputSpec(
            store=store,
            accumulator_factory=factory,
            postprocess=None,
            invert=cfg.output_denormalize,
        )
        for store in output_stores
    ]


_KNOWN_INVERSE_STAGES = ("before_accumulate", "after_finalize")


def _validate_inversion(
    preprocess: BlockPreprocessor, output_specs: List[OutputSpec]
) -> None:
    """Fail fast on inversion misconfiguration, before any thread starts.

    The writer silently skips inversion when its stage gates do not match, so a
    non-invertible preprocess paired with ``invert=True`` (or an unrecognized
    ``inverse_stage`` string) would otherwise write normalized-space data with
    no error.
    """
    if not transforms.is_invertible(preprocess):
        if any(spec.invert for spec in output_specs):
            raise ValueError(
                "An OutputSpec requests invert=True but the preprocess "
                f"({type(preprocess).__name__}) defines no inverse. Set "
                "invert=False (or cfg.output_denormalize=False) to write "
                "outputs untransformed, or use an invertible preprocess."
            )
        return
    # May raise for Sequential members that disagree — better here than in a
    # writer thread.
    stage = getattr(preprocess, "inverse_stage", "after_finalize")
    if stage not in _KNOWN_INVERSE_STAGES:
        raise ValueError(
            f"Unknown inverse_stage {stage!r} on {type(preprocess).__name__}; "
            f"expected one of {_KNOWN_INVERSE_STAGES}."
        )


def run(
    model: nn.Module,
    input_store: Any,
    output_store: Union[Any, List[Any], None],
    cfg: InferenceConfig,
    metrics_json: Optional[str] = None,
    metrics_interval: float = 0.5,
    num_prep_workers: int = 1,
    num_writer_workers: int = 1,
    preprocess: Optional[BlockPreprocessor] = None,
    outputs: Optional[List[OutputSpec]] = None,
    execution: Optional[ExecutionPolicy] = None,
    thread_dump_interval: Optional[float] = None,
    work_store: Optional[BlockWorkStore] = None,
    progress: Optional[PipelineProgress] = None,
) -> None:
    """Runs the inference pipeline.

    Parameters
    ----------
    model : nn.Module
        The model to use for inference. For multi-output models (e.g.,
        SharedEncoderModel) the forward pass must return a tensor of shape
        ``(B, N, Z, Y, X)`` and there must be N outputs.
    input_store : Any
        The input data store.
    output_store : Any or list of Any or None
        Output TensorStore(s). Pass a single store for single-output models, or
        a list of N stores when the model returns N output channels. May be
        ``None`` only when ``outputs`` is given.
    cfg : InferenceConfig
        The inference configuration.
    metrics_json : Optional[str], optional
        Path to the output JSON file for metrics, by default None.
    metrics_interval : float, optional
        The interval at which to sample the queues and system, by default 0.5.
    num_prep_workers : int, optional
        The number of prep workers to use, by default 1.
    num_writer_workers : int, optional
        The number of writer workers to use, by default 1.
    preprocess : Optional[BlockPreprocessor], optional
        Injected input-domain transform applied per block (its inverse runs in
        the writer). When ``None`` (default), one is synthesized from the legacy
        config fields (``normalize``/``norm_lower``/``norm_upper``/``clip_norm``),
        preserving existing behavior.
    outputs : Optional[List[OutputSpec]], optional
        Per-output specs (store + merge factory + post-processor + invert flag).
        When ``None`` (default), specs are synthesized from ``output_store`` with
        the default trim/blend merge and ``invert=cfg.output_denormalize`` for
        every output, preserving existing behavior. Provide this for per-output
        merge/post/dtype control; ``output_store`` is then optional.
    execution : Optional[ExecutionPolicy], optional
        How the GPU processor runs (input dtype / autocast / inference_mode /
        compile / channels_last). When ``None`` (default), synthesized from cfg
        (``amp``/``use_compile``/``compile_mode``/``compile_dynamic``), so AMP-on
        stays the legacy default.
    thread_dump_interval : Optional[float], optional
        Emit all Python thread stacks to stderr at this interval in seconds while
        the worker pipeline is running. ``None`` or a non-positive value disables
        the watchdog. Dumps repeat until the run completes.
    work_store : BlockWorkStore, optional
        Block completion backend. Required when ``cfg.resume`` is enabled.
    progress : PipelineProgress, optional
        Progress channel supplied by the process supervisor. Direct callers
        can omit it to use in-process deadline checks and diagnostic capture.
    """
    # Validate shapes
    T, C, Z, Y, X = tuple(input_store.domain.shape)
    assert 0 <= cfg.t_idx < T and 0 <= cfg.c_idx < C, "Invalid t/c indices"

    # Synthesize the default block transform from config when none is injected,
    # so existing callers keep their current normalization behavior.
    if preprocess is None:
        preprocess = transforms.from_config(
            cfg.normalize,
            cfg.norm_lower,
            cfg.norm_upper,
            cfg.eps,
            cfg.clip_norm,
        )

    # Resolve the per-output specs. Explicit `outputs` win; otherwise synthesize
    # one spec per store with the default trim/blend merge and a per-output invert
    # flag driven by cfg.output_denormalize -- byte-identical to the old writer.
    output_specs = _resolve_output_specs(output_store, outputs, cfg)

    # Fail fast (in this thread) on invert/inverse_stage misconfiguration.
    _validate_inversion(preprocess, output_specs)

    # Default execution policy from cfg keeps AMP/compile behavior identical.
    if execution is None:
        execution = ExecutionPolicy.from_config(
            cfg.amp, cfg.use_compile, cfg.compile_mode, cfg.compile_dynamic
        )

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

    if cfg.resume:
        if work_store is None:
            raise ValueError("cfg.resume=True requires a BlockWorkStore")
        active_work_store = work_store
    else:
        active_work_store = NoopBlockWorkStore()
    active_work_store.prepare(shard_spec)

    # Queues
    prep_q, write_queues = _setup_queues(
        num_writer_workers, maxsize=cfg.max_inflight_batches
    )

    stop_event = threading.Event()

    externally_supervised = progress is not None
    progress = progress if progress is not None else PipelineProgress()

    # Collects (thread_name, exception) from any worker that dies; re-raised
    # after joins so a failed run never looks like a successful one.
    worker_errors: List[Tuple[str, BaseException]] = []

    # Threads
    prep_threads, gpu_threads, writer_threads = _setup_worker_threads(
        model,
        input_store,
        output_specs,
        cfg,
        shard_spec,
        stop_event,
        num_prep_workers,
        prep_q,
        write_queues,
        preprocess,
        execution,
        active_work_store,
        worker_errors,
        progress=progress,
    )
    all_threads = prep_threads + gpu_threads + writer_threads

    q_monitor, sys_monitor = _setup_monitors(
        prep_q, write_queues, metrics_interval, stop_event
    )

    prep_sentinels_sent = False
    writer_sentinels_sent = False
    thread_dumps_scheduled = _start_periodic_thread_dumps(
        thread_dump_interval, shard_spec.index
    )

    t0 = time.perf_counter()
    started_threads = []
    gpu_sentinels_remaining = len(cfg.devices)
    writer_queues_to_close = list(write_queues)
    progress.running()
    try:
        for th in all_threads:
            # A failed native call cannot be killed as a Python thread. Direct
            # callers get a bounded failure; supervised attempts retire the process.
            th.daemon = True
            th.start()
            started_threads.append(th)

        while any(th.is_alive() for th in all_threads):
            if stop_event.is_set():
                break
            reason = stall_reason(progress.snapshot(), cfg, t0)
            if reason:
                raise PipelineTimeoutError(reason)
            # when all prep threads are done, send GPU sentinels
            if (not prep_sentinels_sent) and all(
                not th.is_alive() for th in prep_threads
            ):
                while gpu_sentinels_remaining:
                    try:
                        prep_q.put_nowait(None)
                    except queue.Full:
                        break
                    gpu_sentinels_remaining -= 1
                prep_sentinels_sent = gpu_sentinels_remaining == 0

            # when ALL GPU threads finish, close ALL writers (one sentinel per writer)
            if (not writer_sentinels_sent) and all(
                not th.is_alive() for th in gpu_threads
            ):
                for wq in list(writer_queues_to_close):
                    try:
                        wq.put_nowait(None)
                    except queue.Full:
                        continue
                    writer_queues_to_close.remove(wq)
                writer_sentinels_sent = not writer_queues_to_close

            # cooperative wait
            stop_event.wait(0.05)

        if worker_errors:
            names = ", ".join(name for name, _ in worker_errors)
            error_type = (
                PipelineTimeoutError
                if any(
                    isinstance(exc, PipelineTimeoutError) for _, exc in worker_errors
                )
                else RuntimeError
            )
            raise error_type(
                f"Worker thread(s) failed: {names}; see logged tracebacks."
            ) from worker_errors[0][1]

    except (KeyboardInterrupt, Exception) as e:
        logger.exception(f"Caught {type(e).__name__}, initiating shutdown.")
        if isinstance(e, PipelineTimeoutError) and not externally_supervised:
            capture_diagnostics(
                os.getpid(),
                diagnostic_directory(cfg, metrics_json)
                / f"shard{cfg.shard_index}-{time.time_ns()}",
                progress.snapshot(),
                str(e),
                cfg.diagnostic_timeout_s,
            )
        raise
    finally:
        try:
            logger.info("Setting stop event for all threads.")
            stop_event.set()
            deadline = time.monotonic() + cfg.shutdown_timeout_s
            # One shared shutdown budget, not a full timeout for every thread.
            for th in started_threads:
                if th.is_alive():
                    th.join(timeout=max(0, deadline - time.monotonic()))
            # Stop monitors
            q_monitor.join(timeout=max(0, deadline - time.monotonic()))
            sys_monitor.join(timeout=max(0, deadline - time.monotonic()))

            alive = [th.name for th in started_threads if th.is_alive()]
            if alive:
                reason = f"Shutdown deadline exceeded; discard this process: {alive}"
                if not externally_supervised:
                    capture_diagnostics(
                        os.getpid(),
                        diagnostic_directory(cfg, metrics_json)
                        / f"shard{cfg.shard_index}-{time.time_ns()}-shutdown",
                        progress.snapshot(),
                        reason,
                        cfg.diagnostic_timeout_s,
                    )
                raise PipelineTimeoutError(reason)

            if metrics_json:
                _write_metrics_json(metrics_json, q_monitor, sys_monitor)
        finally:
            _cancel_periodic_thread_dumps(thread_dumps_scheduled, shard_spec.index)

    t1 = time.perf_counter()
    throughput = (Z * Y * X * input_store.dtype.numpy_dtype.itemsize) / 1e6 / (t1 - t0)
    logger.info(f"Total time: {t1-t0:.2f}s")
    logger.info(f"Throughput: {throughput:.2f}MB/s")


def run_workflow(
    workflow: Workflow,
    input_store: Any,
    output_store: Union[Any, List[Any], None],
    cfg: InferenceConfig,
    **run_kwargs: Any,
) -> None:
    """Run a :class:`Workflow` by unpacking its injected objects into :func:`run`.

    The workflow supplies the processor, preprocessing, per-output specs, and
    execution policy; everything else (metrics, worker counts) is forwarded via
    ``run_kwargs``.

    Parameters
    ----------
    workflow : Workflow
        The recipe to run. When the workflow leaves ``preprocess`` or
        ``execution`` unset (``None``), :func:`run` synthesizes them from
        ``cfg``, so config/CLI normalization and AMP/compile flags apply; a
        recipe sets them to pin its own behavior.
    input_store : Any
        The input data store.
    output_store : Any or list of Any or None
        Output store(s). Ignored when the workflow supplies its own ``outputs``;
        required when it supplies an ``output_spec_factory``.
    cfg : InferenceConfig
        The runtime configuration.
    **run_kwargs : Any
        Forwarded to :func:`run` (e.g. ``metrics_json``, ``num_prep_workers``).
    """
    if workflow.output_spec_factory is not None and output_store is None:
        raise ValueError(
            "This workflow builds its output specs from the opened output "
            "stores; provide output_store."
        )
    output_stores = output_store if isinstance(output_store, list) else [output_store]
    outputs = workflow.resolve_outputs(output_stores)
    run(
        workflow.processor,
        input_store,
        None if outputs is not None else output_store,
        cfg,
        preprocess=workflow.preprocess,
        outputs=outputs,
        execution=workflow.execution,
        **run_kwargs,
    )


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
    ap.add_argument(
        "--out-spec",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Path(s) to output TensorStore JSON spec file(s). Provide one per "
            "model output channel. Required unless the workflow supplies its "
            "own fixed outputs."
        ),
    )
    ap.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Type of model to use (must be registered). Alternative to --workflow.",
    )
    ap.add_argument("--weights", type=str, help="Model weights path")
    ap.add_argument(
        "--workflow",
        type=str,
        default=None,
        help=(
            "Named workflow recipe to run (registered in aind_torch_utils.recipes). "
            "Alternative to --model-type; brings its own preprocessing/outputs."
        ),
    )
    ap.add_argument(
        "--workflow-params",
        type=str,
        default=None,
        help=(
            "Path to a JSON file or inline JSON object of parameters passed "
            "to the workflow builder."
        ),
    )
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to JSON file or inline JSON string with InferenceConfig fields "
            "(defaults to built-in config)"
        ),
    )
    ap.add_argument(
        "--no-output-denormalize",
        action="store_true",
        help=(
            "Disable inverse normalization before writing outputs. "
            "Use for probability/segmentation models whose outputs are not in input intensity space."
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
    ap.add_argument(
        "--thread-dump-interval",
        type=float,
        default=0.0,
        help=(
            "Emit repeating all-thread stack dumps at this interval in seconds; "
            "non-positive values disable dumps (default: disabled)"
        ),
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

    if bool(args.model_type) == bool(args.workflow):
        raise SystemExit("Provide exactly one of --model-type or --workflow.")
    if args.workflow and args.weights:
        raise SystemExit(
            "--weights only applies with --model-type; pass weights to the "
            "workflow via --workflow-params."
        )

    if args.config:
        cfg = InferenceConfig.from_json(args.config)
    else:
        cfg = InferenceConfig()
    if args.no_output_denormalize:
        cfg.output_denormalize = False
    logger.info(f"Inference config:\n{cfg}")

    input_spec = load_ts_spec(args.in_spec)
    output_spec_dicts = [load_ts_spec(spec) for spec in (args.out_spec or [])]
    if output_spec_dicts or not args.workflow:
        validate_resume_output_specs(cfg, output_spec_dicts)
    in_arr = open_ts_spec(deepcopy(input_spec))

    run_kwargs = dict(
        metrics_json=args.metrics_json,
        metrics_interval=args.metrics_interval,
        num_prep_workers=max(1, args.prep_workers),
        num_writer_workers=max(1, args.writer_workers),
        thread_dump_interval=args.thread_dump_interval,
    )

    if args.workflow:
        params: dict = {}
        if args.workflow_params:
            stripped = args.workflow_params.lstrip()
            if stripped.startswith("{") or stripped.startswith("["):
                params = json.loads(stripped)
            else:
                with open(args.workflow_params) as f:
                    params = json.load(f)
        workflow = WorkflowRegistry.build(args.workflow, params)
        # Decide the stores' fate BEFORE opening them: an open with a
        # create/delete_existing spec mutates the target, and a fixed-outputs
        # workflow would then silently ignore it.
        if workflow.outputs is not None:
            if args.out_spec:
                raise SystemExit(
                    f"Workflow '{args.workflow}' supplies its own outputs; "
                    "remove --out-spec (it would be ignored)."
                )
            if cfg.resume:
                raise SystemExit(
                    "CLI resume requires --out-spec; fixed-output workflows can "
                    "resume only through the programmatic work_store API."
                )
            out_arr = None
        else:
            if not args.out_spec:
                raise SystemExit(
                    f"--out-spec is required: workflow '{args.workflow}' does "
                    "not supply fixed outputs."
                )
            out_arr = [open_ts_spec(deepcopy(s)) for s in output_spec_dicts]
        work_store = build_block_work_store(
            cfg=cfg,
            input_spec=input_spec,
            output_specs=output_spec_dicts,
            input_store=in_arr,
            output_stores=out_arr or [],
            workload={
                "kind": "workflow",
                "name": args.workflow,
                "params": params,
            },
        )
        run_workflow(
            workflow,
            in_arr,
            out_arr,
            cfg,
            work_store=work_store,
            **run_kwargs,
        )
    else:
        if not args.out_spec:
            raise SystemExit("--out-spec is required with --model-type.")
        out_arr = [open_ts_spec(deepcopy(s)) for s in output_spec_dicts]
        model = load_model(args.model_type, args.weights)
        work_store = build_block_work_store(
            cfg=cfg,
            input_spec=input_spec,
            output_specs=output_spec_dicts,
            input_store=in_arr,
            output_stores=out_arr,
            workload={
                "kind": "model",
                "model_type": args.model_type,
                "weights_path": args.weights,
            },
        )
        run(model, in_arr, out_arr, cfg, work_store=work_store, **run_kwargs)


if __name__ == "__main__":  # pragma: no cover
    main()
