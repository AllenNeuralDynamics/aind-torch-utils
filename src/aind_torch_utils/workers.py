import logging
import queue
import threading
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import tensorstore as ts
import torch
from torch import nn

from aind_torch_utils.accumulators import BlockAccumulator
from aind_torch_utils.config import InferenceConfig
from aind_torch_utils.context import BlockContext
from aind_torch_utils.distributed.sharding import ShardSpec, make_shard_spec
from aind_torch_utils.execution import ExecutionPolicy, cuda_safe_compile_mode
from aind_torch_utils.outputs import OutputSpec
from aind_torch_utils.recovery import (
    PipelineProgress,
    PipelineStopped,
    PipelineTimeoutError,
    wait_for_future,
)
from aind_torch_utils.transforms import BlockPreprocessor, is_invertible
from aind_torch_utils.utils import iter_blocks_zyx, iter_patch_starts
from aind_torch_utils.work_state import (
    BlockKey,
    BlockLease,
    BlockWorkStore,
    NoopBlockWorkStore,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Batch:
    """
    A batch of image patches to be processed by a GpuWorker.

    Attributes
    ----------
    block_idx : Tuple[int, int, int]
        The (z, y, x) index of the block.
    block_bbox : Tuple[slice, slice, slice]
        The core bounding box of the block in the full volume.
    writer_key : int
        Dense, shard-local block index used to select a writer worker.
    starts_in_block : List[Tuple[int, int, int]]
        List of (z, y, x) start coordinates for each patch in the batch,
        relative to the expanded block.
    host_in : torch.Tensor
        The input tensor of patches, pinned to host memory. When compiling
        (execution.compile), tail batches are zero-padded up to batch_size so
        the model sees a constant input shape, and rows beyond
        len(starts_in_block) are padding. In eager mode it has exactly
        len(starts_in_block) rows.
    valid_sizes : List[Tuple[int, int, int]]
        List of (dz, dy, dx) valid dimensions for each patch, handling
        boundary conditions.
    transform_state : Any
        Opaque, **per-block** state produced by the preprocessing/normalization step
        and consumed (only) by its inverse in the writer. The runtime never inspects
        it. Today it holds the affine ``(mn, mx)`` used for inverse normalization; a
        future preprocessor may store anything its paired inverse needs (or ``None``
        when no inversion is required).
    total_patches_in_block : int
        The total number of patches in the entire block.
    acc_shape : Tuple[int, int, int]
        The shape of the expanded (core + halo) accumulator for this block.
    halo_left : Tuple[int, int, int]
        The size of the halo on the (-z, -y, -x) sides of the block.
    ctx : BlockContext
        Absolute placement of the block, built once in the prep stage and
        carried through to the writer so both stages share one derivation.
    lease : BlockLease, optional
        Work-store lease proving this block should be processed.
    """

    block_idx: Tuple[int, int, int]
    block_bbox: Tuple[slice, slice, slice]
    writer_key: int
    starts_in_block: List[Tuple[int, int, int]]
    host_in: torch.Tensor
    valid_sizes: List[Tuple[int, int, int]]
    transform_state: Any  # opaque, per-block (today: affine (mn, mx))
    total_patches_in_block: int
    acc_shape: Tuple[int, int, int]  # shape of expanded (core+halo) accumulator
    halo_left: Tuple[int, int, int]  # halo size on the -Z/-Y/-X sides
    ctx: BlockContext
    lease: Optional[BlockLease] = None


@dataclass(slots=True)
class Preds:
    """
    A batch of model predictions from a GpuWorker, ready for a WriterWorker.

    Attributes
    ----------
    block_idx : Tuple[int, int, int]
        The (z, y, x) index of the block.
    block_bbox : Tuple[slice, slice, slice]
        The core bounding box of the block in the full volume.
    writer_key : int
        Dense, shard-local block index used to select a writer worker.
    starts_in_block : List[Tuple[int, int, int]]
        List of (z, y, x) start coordinates for each patch in the batch,
        relative to the expanded block.
    host_out : torch.Tensor
        The output tensor of predictions, pinned to host memory.
    valid_sizes : List[Tuple[int, int, int]]
        List of (dz, dy, dx) valid dimensions for each patch.
    transform_state : Any
        Opaque, **per-block** state passed through from the matching :class:`Batch`
        (the GPU stage does not touch it) and consumed by the preprocessing inverse
        in the writer. Today it holds the affine ``(mn, mx)`` for denormalization.
    total_patches_in_block : int
        The total number of patches in the entire block.
    acc_shape : Tuple[int, int, int]
        The shape of the expanded (core + halo) accumulator for this block.
    halo_left : Tuple[int, int, int]
        The size of the halo on the (-z, -y, -x) sides of the block.
    ctx : BlockContext
        Absolute placement of the block, passed through unchanged from the
        matching :class:`Batch` (the GPU stage does not touch it).
    ready_event : Optional[torch.cuda.Event]
        A CUDA event that signals when the D2H copy of `host_out` is complete.
    lease : BlockLease, optional
        Work-store lease passed through unchanged from the matching batch.
    """

    block_idx: Tuple[int, int, int]
    block_bbox: Tuple[slice, slice, slice]
    writer_key: int
    starts_in_block: List[Tuple[int, int, int]]
    host_out: torch.Tensor
    valid_sizes: List[Tuple[int, int, int]]
    transform_state: Any
    total_patches_in_block: int
    acc_shape: Tuple[int, int, int]
    halo_left: Tuple[int, int, int]
    ctx: BlockContext
    lease: Optional[BlockLease] = None
    # CUDA event to signal the D2H copy completed
    ready_event: Optional["torch.cuda.Event"] = field(
        default=None, repr=False, compare=False
    )


def writer_for_key(writer_key: int, num_writers: int) -> int:
    """
    Determine which writer should handle a shard-local block key.

    Parameters
    ----------
    writer_key : int
        Dense block index within the current shard.
    num_writers : int
        The total number of writer workers.

    Returns
    -------
    int
        The writer index to use for this block.
    """
    return writer_key % num_writers


class PrepWorker:
    """
    Worker that reads data blocks, prepares patches, and puts them in a queue.
    """

    def __init__(
        self,
        cfg: InferenceConfig,
        reader: "ts.TensorStore",
        prep_q: "queue.Queue[Batch]",
        model_patch: Tuple[int, int, int],
        preprocess: BlockPreprocessor,
        execution: ExecutionPolicy,
        shard_spec: Optional[ShardSpec] = None,
        worker_id: int = 0,
        num_workers: int = 1,
        global_worker_offset: int = 0,
        global_worker_count: int = 1,
        work_store: Optional[BlockWorkStore] = None,
    ):
        """
        Initializes the PrepWorker.

        Parameters
        ----------
        cfg : InferenceConfig
            The denoising configuration.
        reader : ts.TensorStore
            The TensorStore reader for the input data.
        prep_q : queue.Queue[Batch]
            The queue to which prepared batches will be added.
        model_patch : Tuple[int, int, int]
            The (z, y, x) size of the model's input patches.
        preprocess : BlockPreprocessor
            Injected input-domain transform applied to each block. Its
            ``forward(block, ctx)`` returns the processed block and an opaque
            per-block ``transform_state`` carried to the writer. The prep stage
            no longer branches on normalization mode; that logic lives in the
            transform object (see :mod:`aind_torch_utils.transforms`).
        execution : ExecutionPolicy
            Supplies the host ``input_dtype`` for allocated patches and whether
            the processor is compiled (tail batches are padded to a constant
            shape only when compiling).
        shard_spec : Optional[ShardSpec], optional
            Description of the spatial shard assigned to this process. If
            omitted, it is derived from the input shape and configuration.
        worker_id : int, optional
            The ID of this worker, by default 0.
        num_workers : int, optional
            The total number of preparation workers, by default 1.
        global_worker_offset : int, optional
            Offset applied to convert the local worker id into a global id when
            using strided sharding, by default 0.
        global_worker_count : int, optional
            Total number of global prep workers across all shards, by default 1.
        work_store : BlockWorkStore, optional
            Completion backend used to skip blocks before reading input.
        """
        self.cfg = cfg
        self.reader = reader
        self.prep_q = prep_q
        self.patch = model_patch
        self.preprocess = preprocess
        self.execution = execution
        self.full_zyx = self.reader.shape[-3:]
        self.worker_id = worker_id
        self.num_workers = max(1, num_workers)
        self.shard_spec = shard_spec or make_shard_spec(
            self.full_zyx,
            self.cfg.block,
            self.cfg.shard_count,
            self.cfg.shard_index,
            self.cfg.shard_strategy,
        )
        self.shard_strategy = self.shard_spec.strategy
        self.block_start = self.shard_spec.block_start
        self.block_stop = self.shard_spec.block_stop
        self.global_worker_count = max(1, global_worker_count)
        self.global_worker_id = global_worker_offset + worker_id
        self.global_worker_offset = global_worker_offset
        self.work_store = work_store or NoopBlockWorkStore()
        self.progress: Optional[PipelineProgress] = None

    def _block_in_shard(self, block_idx: Tuple[int, int, int]) -> bool:
        """
        Check whether a block index lies within the shard's block bounds.
        """
        sz0, sy0, sx0 = self.block_start
        sz1, sy1, sx1 = self.block_stop
        bz, by, bx = block_idx
        return sz0 <= bz < sz1 and sy0 <= by < sy1 and sx0 <= bx < sx1

    def _writer_key_for_block(
        self, global_block_index: int, block_idx: Tuple[int, int, int]
    ) -> int:
        """Return the dense index of an owned block within this shard.

        Strided shards own consecutive groups of prep-worker slots in each
        global scheduling round. Contiguous shards own rectangular regions of
        the block grid, for which a local row-major index is dense.
        """
        if self.shard_strategy == "stride":
            global_slot = global_block_index % self.global_worker_count
            local_slot = global_slot - self.global_worker_offset
            if not 0 <= local_slot < self.num_workers:
                raise ValueError(
                    f"Block {global_block_index} is not owned by shard "
                    f"{self.shard_spec.index}."
                )
            scheduling_round = global_block_index // self.global_worker_count
            return scheduling_round * self.num_workers + local_slot

        sz0, sy0, sx0 = self.block_start
        _, sy1, sx1 = self.block_stop
        bz, by, bx = block_idx
        if not self._block_in_shard(block_idx):
            raise ValueError(
                f"Block {block_idx} is not owned by shard " f"{self.shard_spec.index}."
            )
        local_y = sy1 - sy0
        local_x = sx1 - sx0
        return ((bz - sz0) * local_y + (by - sy0)) * local_x + (bx - sx0)

    def run(self, stop_event: threading.Event) -> None:
        """
        The main run loop for the worker.

        Iterates over blocks assigned to this worker, reads them, creates
        batches of patches, and puts them into the preparation queue.

        Parameters
        ----------
        stop_event : threading.Event
            An event that signals the worker to stop.
        """
        t, c = self.cfg.t_idx, self.cfg.c_idx
        Z, Y, X = self.full_zyx
        pz, py, px = self.patch
        halo = int(
            self.cfg.halo if self.cfg.halo is not None else (self.cfg.trim_voxels or 0)
        )
        starts_cache: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]] = {}

        # Iterate block grid in Z, Y, X order
        for k, (block_idx, core_bbox) in enumerate(
            iter_blocks_zyx(self.full_zyx, self.cfg.block)
        ):
            if self.shard_strategy == "stride":
                if (k % self.global_worker_count) != self.global_worker_id:
                    continue
            else:
                if not self._block_in_shard(block_idx):
                    continue
                if (k % self.num_workers) != self.worker_id:
                    continue

            if stop_event.is_set():
                break

            block_key = BlockKey(
                t=t,
                c=c,
                z=block_idx[0],
                y=block_idx[1],
                x=block_idx[2],
                linear_k=k,
            )
            lease = self.work_store.claim_block(block_key)
            if lease is None:
                logger.debug("Skipping completed block %s", block_key)
                if self.progress is not None:
                    self.progress.advance()
                continue

            zsl, ysl, xsl = core_bbox
            z0, z1 = zsl.start, zsl.stop
            y0, y1 = ysl.start, ysl.stop
            x0, x1 = xsl.start, xsl.stop

            # expanded ("core + halo") bbox, clipped to volume
            z0e, z1e = max(z0 - halo, 0), min(z1 + halo, Z)
            y0e, y1e = max(y0 - halo, 0), min(y1 + halo, Y)
            x0e, x1e = max(x0 - halo, 0), min(x1 + halo, X)

            acc_shape = (z1e - z0e, y1e - y0e, x1e - x0e)

            # absolute expanded bbox: used to read the block and to place it in the
            # volume for the injected transform (seam-free coordinate sampling).
            expanded_bbox = (slice(z0e, z1e), slice(y0e, y1e), slice(x0e, x1e))
            ctx = BlockContext.from_block(
                block_idx=block_idx,
                core_bbox=core_bbox,
                expanded_bbox=expanded_bbox,
                full_shape=(Z, Y, X),
                t_idx=t,
                c_idx=c,
            )
            # how much halo was actually added on the - sides (derived once, in ctx)
            halo_left = ctx.halo_left

            # read expanded block and cast to float32 for the injected transform
            ez_sl, ey_sl, ex_sl = expanded_bbox
            view = self.reader[t, c, ez_sl, ey_sl, ex_sl]
            started = time.monotonic()
            details = {
                "block": block_key.coords,
                "linear_k": k,
                "bbox": [(s.start, s.stop) for s in expanded_bbox],
            }
            token = (
                self.progress.begin("read", details, self.cfg.read_timeout_s)
                if self.progress is not None
                else None
            )
            logger.debug("Reading block %s bbox=%s", block_key, details["bbox"])
            try:
                read_array = wait_for_future(
                    view.read(),
                    stop_event,
                    started + self.cfg.read_timeout_s,
                    f"input block {block_key} bbox={details['bbox']}",
                )
            except Exception:
                logger.error(
                    "Input read failed/stopped: block=%s bbox=%s elapsed=%.3fs",
                    block_key,
                    details["bbox"],
                    time.monotonic() - started,
                )
                raise
            if token is not None:
                self.progress.end(token)
            logger.debug(
                "Read block %s in %.3fs", block_key, time.monotonic() - started
            )
            norm_block = read_array.astype(np.float32, copy=False)
            bz, by, bx = acc_shape

            # The injected transform owns all normalization/correction math and
            # returns opaque per-block state consumed by its inverse in the writer
            # (None when nothing needs to travel there). The prep stage no longer
            # branches on normalization mode.
            norm_block, transform_state = self.preprocess.forward(norm_block, ctx)

            # patch starts over the expanded region (same stride/overlap)
            if (bz, by, bx) not in starts_cache:
                starts_cache[(bz, by, bx)] = list(
                    iter_patch_starts((bz, by, bx), self.patch, self.cfg.overlap)
                )
            starts = starts_cache[(bz, by, bx)]
            total_patches = len(starts)
            writer_key = self._writer_key_for_block(k, block_idx)

            # batch over those starts
            for i in range(0, total_patches, self.cfg.batch_size):
                if stop_event.is_set():
                    return
                batch_starts = starts[i : i + self.cfg.batch_size]
                n_real = len(batch_starts)
                pin_memory = any("cuda" in d for d in self.cfg.devices)
                # When compiling, pad the tail batch up to batch_size so the
                # model always sees a constant input shape; this prevents
                # torch.compile from recompiling at runtime (not thread-safe
                # across GPU workers). Writers ignore padded rows since they
                # only index rows in starts_in_block. In eager mode there is
                # no shape constraint, so allocate exactly n_real rows and
                # avoid wasting compute and copy bandwidth on padding.
                n_rows = self.cfg.batch_size if self.execution.compile else n_real
                host_in = torch.zeros(
                    (n_rows, 1, pz, py, px),
                    dtype=self.execution.input_dtype,
                    pin_memory=pin_memory,
                )
                valid_sizes = []

                for bi, (sz, sy, sx) in enumerate(batch_starts):
                    ez, ey, ex = (
                        min(sz + pz, bz),
                        min(sy + py, by),
                        min(sx + px, bx),
                    )
                    dz, dy, dx = ez - sz, ey - sy, ex - sx

                    # Slice the already-normalized block
                    norm = norm_block[sz:ez, sy:ey, sx:ex]

                    host_in[bi, 0, :dz, :dy, :dx].copy_(torch.from_numpy(norm))

                    valid_sizes.append((dz, dy, dx))

                batch = Batch(
                    block_idx=block_idx,
                    block_bbox=core_bbox,  # keep the *core* bbox for writing
                    writer_key=writer_key,
                    starts_in_block=batch_starts,  # coords are in expanded space
                    host_in=host_in,
                    valid_sizes=valid_sizes,
                    transform_state=transform_state,
                    total_patches_in_block=total_patches,
                    acc_shape=acc_shape,
                    halo_left=halo_left,
                    ctx=ctx,
                    lease=lease,
                )
                while not stop_event.is_set():
                    try:
                        self.prep_q.put(batch, timeout=0.1)
                        if self.progress is not None:
                            self.progress.advance()
                        break
                    except queue.Full:
                        continue


class GpuWorker:
    """
    Worker that runs model inference on a GPU.
    """

    progress: Optional[PipelineProgress] = None

    def __init__(
        self,
        cfg: InferenceConfig,
        model: nn.Module,
        device: str,
        prep_q: "queue.Queue[Optional[Batch]]",
        write_queues: "List[queue.Queue[Optional[Preds]]]",
        execution: ExecutionPolicy,
    ):
        """
        Initializes the GpuWorker.

        Parameters
        ----------
        cfg : InferenceConfig
            The denoising configuration.
        model : nn.Module or callable
            The processor to run. An ``nn.Module`` is moved to the device and
            put in eval mode; any other tensor-in/tensor-out callable (see
            :class:`~aind_torch_utils.workflow.BlockProcessor`) is used as-is
            and manages its own device placement.
        device : str
            The CUDA device to use (e.g., "cuda:0").
        prep_q : queue.Queue[Optional[Batch]]
            The queue from which to get prepared batches.
        write_queues : List[queue.Queue[Optional[Preds]]]
            A list of queues to send predictions to, one for each writer worker.
        execution : ExecutionPolicy
            How to execute the processor: autocast, inference_mode, compile,
            channels_last. Detaches these from global config so a processor's
            constraints travel with it.
        """
        self.cfg = cfg
        self.model = model
        self.device = torch.device(device)
        self.prep_q = prep_q
        self.write_queues = write_queues
        self.num_writers = len(write_queues)
        self.execution = execution

        torch.backends.cuda.matmul.allow_tf32 = self.cfg.use_tf32
        torch.backends.cudnn.benchmark = self.cfg.cudnn_benchmark

        # BlockProcessor promises "any callable" works: only nn.Modules get
        # device placement / eval; plain callables manage their own state.
        if isinstance(self.model, nn.Module):
            self.model.to(self.device)
            if self.execution.channels_last:
                self.model = self.model.to(memory_format=torch.channels_last_3d)
            self.model.eval()

        self.copy_stream = torch.cuda.Stream(device=self.device)

        if getattr(torch, "compile", None) and self.execution.compile:
            self._compile_model()

    def _autocast_context(self):
        return (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.execution.autocast
            else nullcontext()
        )

    def _inference_context(self):
        if self.execution.inference_mode:
            return torch.inference_mode()
        return nullcontext()

    def _compile_model(self) -> None:
        # Keep a handle to the original module so we can fall back to eager
        # execution if compilation fails. torch.compile returns a new wrapper
        # and does not mutate the original, so this reference stays valid.
        eager_model = self.model
        # InferenceConfig._validate applies the same downgrade for the default
        # (config-synthesized) policy; a directly-injected ExecutionPolicy
        # bypasses that validator, so guard here too: CUDA-graph capture runs
        # lazily on a worker thread and aborts with
        # cudaErrorStreamCaptureInvalidated.
        compile_mode = self.execution.compile_mode
        if self.device.type == "cuda":
            safe_mode = cuda_safe_compile_mode(compile_mode)
            if safe_mode != compile_mode:
                logger.warning(
                    "torch.compile mode %r enables CUDA graphs, which fail in "
                    "this pipeline's threaded GPU workers; using %r instead.",
                    compile_mode,
                    safe_mode,
                )
                compile_mode = safe_mode
        try:
            try:
                # PrepWorker pads tail batches to batch_size, so input shapes
                # are constant and no runtime recompiles are expected
                # regardless of the `dynamic` setting.
                self.model = torch.compile(
                    self.model,
                    mode=compile_mode,
                    dynamic=self.execution.compile_dynamic,
                )
                logger.info("Compiled model on %s.", self.device)
            except TypeError:
                # older PyTorch without `dynamic` kwarg
                self.model = torch.compile(self.model, mode=compile_mode)
                logger.info("Compiled model on %s (older pytorch).", self.device)

            # Compilation is lazy: the graph is traced on the first forward,
            # so tracing/guard errors surface here in warmup, not above.
            self._warmup_compiled_model()
        except Exception as exc:
            # Some models do host-side numpy/Python work in forward that
            # dynamo cannot trace. Fall back to eager so the run proceeds
            # instead of aborting. Warmup runs on the main thread, so this
            # also keeps the failure off the worker threads.
            logger.warning(
                "torch.compile failed on %s (%s); falling back to eager " "execution.",
                self.device,
                type(exc).__name__,
                exc_info=True,
            )
            self.model = eager_model

    def _warmup_compiled_model(self) -> None:
        torch.cuda.set_device(self.device)
        dtype = self.execution.input_dtype
        shape = (self.cfg.batch_size, 1, *self.cfg.patch)
        warmup_in = torch.zeros(shape, dtype=dtype, device=self.device)
        if self.execution.channels_last:
            warmup_in = warmup_in.contiguous(memory_format=torch.channels_last_3d)

        logger.info("Warming compiled model on %s with shape %s.", self.device, shape)
        with self._inference_context():
            with self._autocast_context():
                warmup_out = self.model(warmup_in)
        torch.cuda.synchronize(self.device)
        del warmup_in, warmup_out
        logger.info("Finished compiled model warmup on %s.", self.device)

    def run(self, stop_event: threading.Event) -> None:
        """
        The main run loop for the worker.

        Gets batches from the prep queue, runs model inference, and puts
        the predictions into the appropriate writer queue.

        Parameters
        ----------
        stop_event : threading.Event
            An event that signals the worker to stop.
        """
        autocast_ctx = self._autocast_context()

        # Ensure the current device matches self.device for streams/events
        torch.cuda.set_device(self.device)

        while not stop_event.is_set():
            try:
                batch = self.prep_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if batch is None:
                break

            # Allocate device input per batch (simple path)
            dev_in = torch.empty_like(
                batch.host_in,
                device=self.device,
                memory_format=torch.contiguous_format,
            )

            # H2D
            dev_in.copy_(batch.host_in, non_blocking=True)
            if self.execution.channels_last:
                dev_in = dev_in.contiguous(memory_format=torch.channels_last_3d)

            # Inference
            with self._inference_context():
                with autocast_ctx:
                    out = self.model(dev_in)

            # D2H into pinned buffer sized to actual model output (async on a dedicated stream)
            pin_memory = "cuda" in str(self.device)
            host_out = torch.empty(
                out.shape,
                dtype=out.dtype,
                pin_memory=pin_memory,
            )

            # Ensure the copy stream waits for the default stream's compute to finish
            cur = torch.cuda.current_stream(self.device)
            self.copy_stream.wait_stream(cur)

            # Create the event on the correct device
            with torch.cuda.device(self.device):
                evt = torch.cuda.Event(blocking=False, enable_timing=False)

            # Enqueue the async D2H copy and record an event on the copy stream
            with torch.cuda.stream(self.copy_stream):
                host_out.copy_(out, non_blocking=True)
                evt.record()  # marks completion of the D2H on copy_stream

            # out is produced on the compute stream but consumed by copy_stream.
            # The caching allocator only tracks the producing stream, so without
            # this it could hand out's memory to a later compute-stream
            # allocation while this async D2H is still reading it (a
            # write-after-read hazard). record_stream makes the allocator also
            # wait for copy_stream before recycling the block.
            out.record_stream(self.copy_stream)

            preds = Preds(
                block_idx=batch.block_idx,
                block_bbox=batch.block_bbox,
                writer_key=batch.writer_key,
                starts_in_block=batch.starts_in_block,
                host_out=host_out,
                valid_sizes=batch.valid_sizes,
                transform_state=batch.transform_state,  # opaque pass-through
                total_patches_in_block=batch.total_patches_in_block,
                acc_shape=batch.acc_shape,
                halo_left=batch.halo_left,
                ctx=batch.ctx,
                lease=batch.lease,
                ready_event=evt,  # <-- writer will synchronize this
            )

            # Route every batch for a block to its shard-local writer.
            wid = writer_for_key(preds.writer_key, self.num_writers)
            target_q = self.write_queues[wid]

            while not stop_event.is_set():
                try:
                    target_q.put(preds, timeout=0.1)
                    if self.progress is not None:
                        self.progress.advance()
                    break
                except queue.Full:
                    continue


@dataclass
class _BlockState:
    """Writer-side bookkeeping for one in-flight block.

    The writer counts merged patches itself (``seen``), so accumulators only
    merge; an accumulator cannot silently stall block completion by forgetting
    counter plumbing.
    """

    accs: List[BlockAccumulator]
    seen: int = 0


@dataclass(slots=True)
class _PendingWrite:
    """A write future and the source array it may still reference."""

    future: Any
    source: np.ndarray
    group: "_BlockWriteGroup"
    deadline: float
    token: Optional[str]


# A cancelled/timed-out write may still reference its source in native code.
# Keep these buffers until the failed process is retired by the supervisor.
_ABANDONED_WRITES: List[_PendingWrite] = []


@dataclass
class _BlockWriteGroup:
    """Track all asynchronous output writes associated with one block lease."""

    lease: Optional[BlockLease]
    work_store: BlockWorkStore
    remaining: int = 0
    sealed: bool = False
    failed: bool = False
    failure_reported: bool = False
    completed: bool = False

    def add_write(self) -> None:
        """Register a future before it enters the pending-write queue."""
        self.remaining += 1

    def finish_write(self, exc: Optional[BaseException]) -> None:
        """Record one committed or failed output write."""
        self.remaining -= 1
        if exc is not None:
            self._record_failure(exc)
        self._maybe_complete()

    def seal(self, exc: Optional[BaseException] = None) -> None:
        """Declare that no more output writes will be added to this block."""
        self.sealed = True
        if exc is not None:
            self._record_failure(exc)
        self._maybe_complete()

    def _record_failure(self, exc: BaseException) -> None:
        self.failed = True
        if self.lease is not None and not self.failure_reported:
            self.failure_reported = True
            self.work_store.fail_block(self.lease, exc)

    def _maybe_complete(self) -> None:
        if (
            self.sealed
            and self.remaining == 0
            and not self.failed
            and not self.completed
            and self.lease is not None
        ):
            self.work_store.complete_block(self.lease)
            self.completed = True


class _BoundedPendingWrites:
    """Keep a bounded set of asynchronous writes alive and observable."""

    def __init__(
        self,
        limit: int,
        timeout_s: float = 300,
        stop_event: Optional[threading.Event] = None,
        progress: Optional[PipelineProgress] = None,
        shutdown_timeout_s: float = 30,
    ):
        self.limit = limit
        self.timeout_s = timeout_s
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.progress = progress
        self.shutdown_timeout_s = shutdown_timeout_s
        self._writes: Deque[_PendingWrite] = deque()

    def __enter__(self) -> "_BoundedPendingWrites":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if exc_type is None:
            self.flush()
        else:
            try:
                deadline = time.monotonic() + self.shutdown_timeout_s
                for pending in self._writes:
                    pending.deadline = min(pending.deadline, deadline)
                self.flush()
            except Exception:  # noqa: BLE001 - preserve the original worker error
                logger.exception(
                    "A pending TensorStore write also failed during shutdown."
                )
        return False

    def _wait_for_oldest(self) -> None:
        pending = self._writes.popleft()
        # Keep `pending.source` strongly referenced until TensorStore reports
        # that the write is committed. Calling result() also surfaces failures.
        try:
            wait_for_future(
                pending.future,
                self.stop_event,
                pending.deadline,
                f"output block {pending.group.lease}",
            )
        except Exception as exc:
            if isinstance(exc, (PipelineStopped, PipelineTimeoutError)):
                _ABANDONED_WRITES.append(pending)
            pending.group.finish_write(exc)
            raise
        else:
            pending.group.finish_write(None)
            if self.progress is not None and pending.token is not None:
                self.progress.end(pending.token)

    def submit(self, target: Any, source: np.ndarray, group: _BlockWriteGroup) -> None:
        """Start a write, waiting first when the pending-write limit is full."""
        if len(self._writes) >= self.limit:
            self._wait_for_oldest()
        if self.stop_event.is_set():
            raise PipelineStopped("Stopped before output submission")
        started = time.monotonic()
        token = (
            self.progress.begin(
                "write",
                {
                    "block": (
                        group.lease.block.coords if group.lease is not None else None
                    )
                },
                self.timeout_s,
            )
            if self.progress is not None
            else None
        )
        future = target.write(source)
        group.add_write()
        self._writes.append(
            _PendingWrite(
                future=future,
                source=source,
                group=group,
                deadline=started + self.timeout_s,
                token=token,
            )
        )

    def poll(self) -> None:
        """Observe committed writes even when no next block arrives."""
        while self._writes and self._writes[0].future.done():
            self._wait_for_oldest()

    def flush(self) -> None:
        """Wait for every pending write and raise the first asynchronous error."""
        first_error: Optional[Exception] = None
        while self._writes:
            try:
                self._wait_for_oldest()
            except Exception as exc:  # noqa: BLE001 - drain all writes before raising
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


class WriterWorker:
    """
    Worker that accumulates predictions for a block and writes the result.

    Each model output channel is described by an :class:`OutputSpec` carrying its
    own merge accumulator, optional post-processor, inversion flag, and destination
    store. The model output tensor is expected to have shape ``(B, N, Z, Y, X)``
    where N equals ``len(outputs)``.

    Per block, once complete, each output is finalized through the ordered pipeline
    (issue #25 §4.4): ``finalize -> invert (if requested) -> postprocess -> crop
    halo -> cast to store dtype -> write``.
    """

    progress: Optional[PipelineProgress] = None

    def __init__(
        self,
        cfg: InferenceConfig,
        outputs: List[OutputSpec],
        write_q: "queue.Queue[Optional[Preds]]",
        preprocess: BlockPreprocessor,
        work_store: Optional[BlockWorkStore] = None,
    ):
        """
        Initializes the WriterWorker.

        Parameters
        ----------
        cfg : InferenceConfig
            The inference configuration.
        outputs : list of OutputSpec
            One spec per model output channel: store + merge factory + optional
            post-processor + per-output ``invert`` flag.
        write_q : queue.Queue[Optional[Preds]]
            The queue from which to get model predictions.
        preprocess : BlockPreprocessor
            The same transform the prep stage applied. Its ``inverse`` is applied
            to the outputs whose ``OutputSpec.invert`` is set (only when it is
            invertible), using the per-block ``transform_state`` carried on
            ``Preds``. The transform's ``inverse_stage`` decides whether inversion
            happens per patch (``before_accumulate``) or once per finalized block
            (``after_finalize``).
        work_store : BlockWorkStore, optional
            Completion backend notified after every output for a block commits.
        """
        self.cfg = cfg
        self.outputs = outputs
        self.write_q = write_q
        self.preprocess = preprocess
        self.work_store = work_store or NoopBlockWorkStore()
        # maps block_idx → in-flight accumulation state for that block
        self.blocks: Dict[Tuple[int, int, int], _BlockState] = {}

    def _make_accumulators(
        self, acc_shape: Tuple[int, int, int], ctx: BlockContext
    ) -> List[BlockAccumulator]:
        return [spec.accumulator_factory(acc_shape, ctx) for spec in self.outputs]

    def _cast_to_store(self, core: np.ndarray, store: Any) -> np.ndarray:
        target_dtype = store.dtype.numpy_dtype
        if np.issubdtype(target_dtype, np.integer):
            info = np.iinfo(target_dtype)
            return np.clip(core, info.min, info.max).astype(target_dtype, copy=False)
        return core.astype(target_dtype, copy=False)

    def run(self, stop_event: threading.Event) -> None:
        """
        The main run loop for the worker.

        Gets predictions from the write queue, accumulates them until a block
        is complete, finalizes the block, and writes it to each output store.

        Parameters
        ----------
        stop_event : threading.Event
            An event that signals the worker to stop.
        """
        invertible = is_invertible(self.preprocess)
        # Duck-typed invertible transforms may omit inverse_stage; default to
        # the common linear case (matching Sequential's fallback) instead of
        # dying with AttributeError. run() validates the value up front.
        inverse_stage = (
            getattr(self.preprocess, "inverse_stage", "after_finalize")
            if invertible
            else None
        )

        with _BoundedPendingWrites(
            self.cfg.max_pending_writes,
            self.cfg.write_timeout_s,
            stop_event,
            self.progress,
            self.cfg.shutdown_timeout_s,
        ) as pending_writes:
            while not stop_event.is_set():
                pending_writes.poll()
                try:
                    preds = self.write_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                if preds is None:
                    break  # single sentinel closes the writer

                if getattr(preds, "ready_event", None) is not None:
                    preds.ready_event.synchronize()

                zsl, ysl, xsl = preds.block_bbox
                core_bz, core_by, core_bx = (
                    zsl.stop - zsl.start,
                    ysl.stop - ysl.start,
                    xsl.stop - xsl.start,
                )

                # Block placement is identical for every Preds of this block; it was
                # built once in the prep stage and rides on the carrier.
                ctx = preds.ctx

                state = self.blocks.get(preds.block_idx)
                if state is None:
                    state = _BlockState(
                        accs=self._make_accumulators(preds.acc_shape, ctx)
                    )
                    self.blocks[preds.block_idx] = state
                accs = state.accs

                out_np = preds.host_out.numpy()
                # Ensure the tensor has a channel dimension that matches the outputs
                if out_np.ndim == 4:
                    # legacy single-output (B, pz, py, px) — add channel dim
                    out_np = out_np[:, np.newaxis]

                if out_np.ndim != 5:
                    raise ValueError(
                        "Expected model output with shape (B, N, Z, Y, X) "
                        f"(or legacy (B, Z, Y, X)); got shape {out_np.shape}"
                    )

                if out_np.shape[1] != len(self.outputs):
                    raise ValueError(
                        "Mismatch between model output channels and output specs: "
                        f"got N={out_np.shape[1]} channels but {len(self.outputs)} "
                        f"output(s) for block {preds.block_idx}."
                    )

                # Merge patches in the transform's *output* space. For an output that
                # inverts with a nonlinear transform declaring 'before_accumulate', the
                # inverse must run per patch (it does not commute with averaging); the
                # common linear case inverts once per block after finalize (below).
                for bi, (sz, sy, sx) in enumerate(preds.starts_in_block):
                    dz, dy, dx = preds.valid_sizes[bi]
                    for n, (spec, acc) in enumerate(zip(self.outputs, accs)):
                        pp = out_np[bi, n].astype(np.float32, copy=False)
                        if spec.invert and inverse_stage == "before_accumulate":
                            pp = self.preprocess.inverse(pp, preds.transform_state, ctx)
                        acc.add(pp, (sz, sy, sx), (dz, dy, dx))
                state.seen += len(preds.starts_in_block)
                if self.progress is not None:
                    self.progress.advance()

                if state.seen >= preds.total_patches_in_block:
                    lz, ly, lx = preds.halo_left
                    write_group = _BlockWriteGroup(
                        lease=preds.lease,
                        work_store=self.work_store,
                    )
                    try:
                        for spec, acc in zip(self.outputs, accs):
                            ext = acc.finalize()  # expanded (core + halo)
                            if spec.invert and inverse_stage == "after_finalize":
                                ext = self.preprocess.inverse(
                                    ext, preds.transform_state, ctx
                                )
                            if spec.postprocess is not None:
                                ext = spec.postprocess(ext, ctx)
                            core = ext[
                                lz : lz + core_bz,
                                ly : ly + core_by,
                                lx : lx + core_bx,
                            ]
                            out_arr = self._cast_to_store(core, spec.store)
                            target = spec.store[
                                self.cfg.t_idx, self.cfg.c_idx, zsl, ysl, xsl
                            ]
                            pending_writes.submit(target, out_arr, write_group)
                    except Exception as exc:
                        write_group.seal(exc)
                        raise
                    else:
                        write_group.seal()
                    del self.blocks[preds.block_idx]
