import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorstore as ts
import torch
from torch import nn

from aind_torch_utils.accumulators import BlockAccumulator
from aind_torch_utils.config import InferenceConfig
from aind_torch_utils.utils import iter_blocks_zyx, iter_patch_starts

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
    linear_k : int
        The linear index of the block.
    starts_in_block : List[Tuple[int, int, int]]
        List of (z, y, x) start coordinates for each patch in the batch,
        relative to the expanded block.
    host_in : torch.Tensor
        The input tensor of patches, pinned to host memory.
    valid_sizes : List[Tuple[int, int, int]]
        List of (dz, dy, dx) valid dimensions for each patch, handling
        boundary conditions.
    per_patch_mnmx : List[Tuple[float, float]]
        List of (min, max) percentile values for each patch used for normalization.
    total_patches_in_block : int
        The total number of patches in the entire block.
    acc_shape : Tuple[int, int, int]
        The shape of the expanded (core + halo) accumulator for this block.
    halo_left : Tuple[int, int, int]
        The size of the halo on the (-z, -y, -x) sides of the block.
    """

    block_idx: Tuple[int, int, int]
    block_bbox: Tuple[slice, slice, slice]
    linear_k: int
    starts_in_block: List[Tuple[int, int, int]]
    host_in: torch.Tensor
    valid_sizes: List[Tuple[int, int, int]]
    per_patch_mnmx: List[Tuple[float, float]]  # per-patch (mn, mx)
    total_patches_in_block: int
    acc_shape: Tuple[int, int, int]  # shape of expanded (core+halo) accumulator
    halo_left: Tuple[int, int, int]  # halo size on the -Z/-Y/-X sides


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
    linear_k : int
        The linear index of the block.
    starts_in_block : List[Tuple[int, int, int]]
        List of (z, y, x) start coordinates for each patch in the batch,
        relative to the expanded block.
    host_out : torch.Tensor
        The output tensor of predictions, pinned to host memory.
    valid_sizes : List[Tuple[int, int, int]]
        List of (dz, dy, dx) valid dimensions for each patch.
    per_patch_mnmx : List[Tuple[float, float]]
        List of (min, max) percentile values for each patch for denormalization.
    total_patches_in_block : int
        The total number of patches in the entire block.
    acc_shape : Tuple[int, int, int]
        The shape of the expanded (core + halo) accumulator for this block.
    halo_left : Tuple[int, int, int]
        The size of the halo on the (-z, -y, -x) sides of the block.
    ready_event : Optional[torch.cuda.Event]
        A CUDA event that signals when the D2H copy of `host_out` is complete.
    """

    block_idx: Tuple[int, int, int]
    block_bbox: Tuple[slice, slice, slice]
    linear_k: int
    starts_in_block: List[Tuple[int, int, int]]
    host_out: torch.Tensor
    valid_sizes: List[Tuple[int, int, int]]
    per_patch_mnmx: List[Tuple[float, float]]
    total_patches_in_block: int
    acc_shape: Tuple[int, int, int]
    halo_left: Tuple[int, int, int]
    # CUDA event to signal the D2H copy completed
    ready_event: Optional["torch.cuda.Event"] = field(
        default=None, repr=False, compare=False
    )


def shard_for_block_linear(linear_k: int, num_writers: int) -> int:
    """
    Determines which writer shard should handle a given block.

    Parameters
    ----------
    linear_k : int
        The linear index of the block.
    num_writers : int
        The total number of writer workers.

    Returns
    -------
    int
        The index of the writer shard to use for this block.
    """
    return linear_k % num_writers


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
        worker_id: int = 0,
        num_workers: int = 1,
    ):
        """
        Initializes the PrepWorker.

        Parameters
        ----------
        cfg : DenoiseConfig
            The denoising configuration.
        reader : ts.TensorStore
            The TensorStore reader for the input data.
        prep_q : queue.Queue[Batch]
            The queue to which prepared batches will be added.
        model_patch : Tuple[int, int, int]
            The (z, y, x) size of the model's input patches.
        worker_id : int, optional
            The ID of this worker, by default 0.
        num_workers : int, optional
            The total number of preparation workers, by default 1.
        """
        self.cfg = cfg
        self.reader = reader
        self.prep_q = prep_q
        self.patch = model_patch
        self.full_zyx = self.reader.shape[-3:]
        self.worker_id = worker_id
        self.num_workers = max(1, num_workers)

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

        # Strided partition over blocks: worker i handles k where k % num_workers == i
        for k, (block_idx, core_bbox) in enumerate(
            iter_blocks_zyx(self.full_zyx, self.cfg.block)
        ):
            if (k % self.num_workers) != self.worker_id:
                continue

            if stop_event.is_set():
                break

            zsl, ysl, xsl = core_bbox
            z0, z1 = zsl.start, zsl.stop
            y0, y1 = ysl.start, ysl.stop
            x0, x1 = xsl.start, xsl.stop

            # expanded ("core + halo") bbox, clipped to volume
            z0e, z1e = max(z0 - halo, 0), min(z1 + halo, Z)
            y0e, y1e = max(y0 - halo, 0), min(y1 + halo, Y)
            x0e, x1e = max(x0 - halo, 0), min(x1 + halo, X)

            # how much halo was actually added on the - sides
            halo_left = (z0 - z0e, y0 - y0e, x0 - x0e)
            acc_shape = (z1e - z0e, y1e - y0e, x1e - x0e)

            # read expanded block and cast to float32 for normalization
            view = self.reader[t, c, slice(z0e, z1e), slice(y0e, y1e), slice(x0e, x1e)]
            norm_block = view.read().result().astype(np.float32, copy=False)
            bz, by, bx = acc_shape

            if self.cfg.norm_percentile_lower == self.cfg.norm_percentile_upper:
                # Bypass normalization entirely (identity). We pretend (mn,mx)=(0,1)
                # so the writer performs a no-op inverse transform.
                block_mn, block_mx = 0.0, 1.0
            else:
                # calculate percentiles
                block_mn, block_mx = np.percentile(
                    norm_block,
                    [
                        self.cfg.norm_percentile_lower,
                        self.cfg.norm_percentile_upper,
                    ],
                )
                block_scale = max(block_mx - block_mn, self.cfg.eps)
                # normalize the block in-place
                norm_block -= block_mn
                norm_block /= block_scale

            # patch starts over the expanded region (same stride/overlap)
            if (bz, by, bx) not in starts_cache:
                starts_cache[(bz, by, bx)] = list(
                    iter_patch_starts((bz, by, bx), self.patch, self.cfg.overlap)
                )
            starts = starts_cache[(bz, by, bx)]
            total_patches = len(starts)

            # batch over those starts
            for i in range(0, total_patches, self.cfg.batch_size):
                batch_starts = starts[i : i + self.cfg.batch_size]
                B = len(batch_starts)
                pin_memory = any("cuda" in d for d in self.cfg.devices)
                host_in = torch.zeros(
                    (B, 1, pz, py, px),
                    dtype=torch.float16 if self.cfg.amp else torch.float32,
                    pin_memory=pin_memory,
                )
                valid_sizes, per_patch_mnmx = [], []

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
                    # Every patch in the block uses the same (mn, mx)
                    per_patch_mnmx.append((float(block_mn), float(block_mx)))

                batch = Batch(
                    block_idx=block_idx,
                    block_bbox=core_bbox,  # keep the *core* bbox for writing
                    linear_k=k,
                    starts_in_block=batch_starts,  # coords are in expanded space
                    host_in=host_in,
                    valid_sizes=valid_sizes,
                    per_patch_mnmx=per_patch_mnmx,
                    total_patches_in_block=total_patches,
                    acc_shape=acc_shape,
                    halo_left=halo_left,
                )
                while not stop_event.is_set():
                    try:
                        self.prep_q.put(batch, timeout=0.1)
                        break
                    except queue.Full:
                        continue


class nullcontext:
    """A context manager that does nothing."""

    def __enter__(self) -> "nullcontext":
        """Enter the context."""
        return self

    def __exit__(self, *args: Any) -> bool:
        """Exit the context."""
        return False


class GpuWorker:
    """
    Worker that runs model inference on a GPU.
    """

    def __init__(
        self,
        cfg: InferenceConfig,
        model: nn.Module,
        device: str,
        prep_q: "queue.Queue[Optional[Batch]]",
        write_queues: "List[queue.Queue[Optional[Preds]]]",
    ):
        """
        Initializes the GpuWorker.

        Parameters
        ----------
        cfg : DenoiseConfig
            The denoising configuration.
        model : nn.Module
            The PyTorch model to run.
        device : str
            The CUDA device to use (e.g., "cuda:0").
        prep_q : queue.Queue[Optional[Batch]]
            The queue from which to get prepared batches.
        write_queues : List[queue.Queue[Optional[Preds]]]
            A list of queues to send predictions to, one for each writer worker.
        """
        self.cfg = cfg
        self.model = model
        self.device = torch.device(device)
        self.prep_q = prep_q
        self.write_queues = write_queues
        self.num_writers = len(write_queues)

        if self.cfg.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        self.model.to(self.device)
        self.model.eval()

        self.copy_stream = torch.cuda.Stream(device=self.device)

        if getattr(torch, "compile", None) and self.cfg.use_compile:
            try:
                # dynamic=True avoids recompiles when the final batch is smaller
                self.model = torch.compile(
                    self.model,
                    mode=self.cfg.compile_mode,
                    dynamic=self.cfg.compile_dynamic,
                )
                logger.info("Successfully compiled model.")
            except TypeError:
                # older PyTorch without `dynamic` kwarg
                self.model = torch.compile(self.model, mode=self.cfg.compile_mode)
                logger.info("Successfully compiled model (older pytorch).")

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
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.cfg.amp
            else nullcontext()
        )

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

            # Inference
            with torch.inference_mode():
                with autocast_ctx:
                    out = self.model(dev_in)
                if out.dtype != batch.host_in.dtype:
                    out = out.to(batch.host_in.dtype)

            # D2H into pinned buffer (async on a dedicated stream)
            host_out = torch.empty_like(batch.host_in, pin_memory=True)

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

            preds = Preds(
                block_idx=batch.block_idx,
                block_bbox=batch.block_bbox,
                linear_k=batch.linear_k,
                starts_in_block=batch.starts_in_block,
                host_out=host_out,
                valid_sizes=batch.valid_sizes,
                per_patch_mnmx=batch.per_patch_mnmx,  # pass-through
                total_patches_in_block=batch.total_patches_in_block,
                acc_shape=batch.acc_shape,
                halo_left=batch.halo_left,
                ready_event=evt,  # <-- writer will synchronize this
            )

            # route to shard
            wid = shard_for_block_linear(preds.linear_k, self.num_writers)
            target_q = self.write_queues[wid]

            while not stop_event.is_set():
                try:
                    target_q.put(preds, timeout=0.1)
                    break
                except queue.Full:
                    continue


class WriterWorker:
    """
    Worker that accumulates predictions for a block and writes the result.
    """

    def __init__(
        self,
        cfg: InferenceConfig,
        writer: "ts.TensorStore",
        write_q: "queue.Queue[Optional[Preds]]",
    ):
        """
        Initializes the WriterWorker.

        Parameters
        ----------
        cfg : DenoiseConfig
            The denoising configuration.
        writer : ts.TensorStore
            The TensorStore writer for the output data.
        write_q : queue.Queue[Optional[Preds]]
            The queue from which to get model predictions.
        """
        self.cfg = cfg
        self.writer = writer
        self.write_q = write_q
        self.blocks: Dict[Tuple[int, int, int], BlockAccumulator] = {}

    def run(self, stop_event: threading.Event) -> None:
        """
        The main run loop for the worker.

        Gets predictions from the write queue, accumulates them until a block
        is complete, finalizes the block, and writes it to the output.

        Parameters
        ----------
        stop_event : threading.Event
            An event that signals the worker to stop.
        """
        while not stop_event.is_set():
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

            acc = self.blocks.get(preds.block_idx)
            if acc is None:
                acc = BlockAccumulator(
                    preds.acc_shape,
                    self.cfg.patch,
                    self.cfg.eps,
                    overlap=self.cfg.overlap,
                    seam_mode=self.cfg.seam_mode,
                    trim_voxels=self.cfg.trim_voxels,
                    min_blend_weight=self.cfg.min_blend_weight,
                )
                acc.total = preds.total_patches_in_block
                self.blocks[preds.block_idx] = acc

            out_np = preds.host_out.numpy()
            for bi, (sz, sy, sx) in enumerate(preds.starts_in_block):
                dz, dy, dx = preds.valid_sizes[bi]
                patch_pred = out_np[bi, 0]
                mn, mx = preds.per_patch_mnmx[bi]
                scale = max(mx - mn, self.cfg.eps)
                pp = patch_pred.astype(np.float32, copy=False)
                denorm = (pp * np.float32(scale) + np.float32(mn)).astype(
                    np.float32, copy=False
                )
                acc.add(denorm, (sz, sy, sx), (dz, dy, dx))

            if acc.count >= acc.total:
                ext = acc.finalize()
                del self.blocks[preds.block_idx]

                lz, ly, lx = preds.halo_left
                core = ext[lz : lz + core_bz, ly : ly + core_by, lx : lx + core_bx]
                out_u16 = np.clip(core, 0.0, 65535.0).astype(np.uint16, copy=False)
                self.writer[self.cfg.t_idx, self.cfg.c_idx, zsl, ysl, xsl].write(
                    out_u16
                ).result()
