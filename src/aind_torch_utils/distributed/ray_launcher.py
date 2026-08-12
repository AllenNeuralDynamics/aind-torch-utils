from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import tensorstore as ts

from aind_torch_utils.config import InferenceConfig
from aind_torch_utils.run import _parse_args as parse_inference_args
from aind_torch_utils.run import (
    load_model,
    run,
    run_workflow,
)
from aind_torch_utils.utils import open_ts_spec
from aind_torch_utils.work_state import (
    build_block_work_store,
    validate_resume_output_specs,
)
from aind_torch_utils.workflow import WorkflowRegistry

logger = logging.getLogger(__name__)


def _import_ray():
    try:
        import ray  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "Ray is not installed. Install ray to use the Ray launcher."
        ) from exc
    return ray


def _build_inference_config(
    args: argparse.Namespace,
    shard_count: Optional[int] = None,
) -> InferenceConfig:
    """
    Construct the base InferenceConfig shared by all shards.
    """
    if args.config:
        cfg = InferenceConfig.from_json(
            args.config,
            shard_count=shard_count,
            shard_index=0,
        )
    else:
        data: Dict[str, Any] = {}
        if shard_count is not None:
            data["shard_count"] = shard_count
        data["shard_index"] = 0
        cfg = InferenceConfig.model_validate(data)
    if args.no_output_denormalize:
        cfg.output_denormalize = False
    return cfg


def _default_metrics_template(base: Optional[str]) -> Optional[str]:
    if not base:
        return None
    stem, ext = os.path.splitext(base)
    if not ext:
        ext = ".json"
    return f"{stem}_shard{{shard}}{ext}"


def _resolve_metrics_path(
    base: Optional[str], template: Optional[str], shard: int
) -> Optional[str]:
    if template:
        return template.format(shard=shard)
    if base:
        return _default_metrics_template(base).format(shard=shard)
    return None


def _detect_visible_gpus() -> int:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not vis:
        return 0
    return len([d for d in vis.split(",") if d.strip()])


def _canonical_devices(requested: Sequence[str]) -> Sequence[str]:
    if not requested:
        return requested
    visible = _detect_visible_gpus()
    if visible == 0:
        return requested
    return [f"cuda:{idx}" for idx in range(min(len(requested), visible))]


def _make_shard_payload(
    base_cfg: InferenceConfig,
    metrics_template: Optional[str],
    metrics_base: Optional[str],
    shard_idx: int,
) -> Dict[str, Any]:
    cfg_dict = base_cfg.model_dump()
    cfg_dict["shard_index"] = shard_idx
    metrics_path = _resolve_metrics_path(metrics_base, metrics_template, shard_idx)
    return {"config": cfg_dict, "metrics_json": metrics_path}


def _load_workflow_params(path: Optional[str]) -> Dict[str, Any]:
    """Load workflow builder parameters from a JSON file or inline object."""
    if not path:
        return {}
    stripped = path.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        params = json.loads(stripped)
    else:
        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)
    if not isinstance(params, dict):
        raise TypeError("Workflow parameters must be a JSON object.")
    return params


def _make_shard_tensorstore_context(cfg: InferenceConfig) -> ts.Context:
    """Create the TensorStore resource context shared by one shard's stores."""
    return ts.Context(
        {
            "data_copy_concurrency": {
                "limit": cfg.tensorstore_data_copy_concurrency,
            }
        }
    )


def _run_shard(
    run_args: argparse.Namespace,
    workflow_params: Dict[str, Any],
    cfg: InferenceConfig,
    input_spec: Dict[str, Any],
    output_specs: Sequence[Dict[str, Any]],
    metrics_json: Optional[str],
) -> None:
    """Open a shard's stores and execute its selected model or workflow."""
    store_context = _make_shard_tensorstore_context(cfg)
    logger.info(
        "Shard %d TensorStore data_copy_concurrency.limit=%d",
        cfg.shard_index,
        cfg.tensorstore_data_copy_concurrency,
    )
    input_store = open_ts_spec(
        copy.deepcopy(input_spec),
        context=store_context,
    )
    output_stores = [
        open_ts_spec(copy.deepcopy(spec), context=store_context)
        for spec in output_specs
    ]
    if run_args.workflow:
        workload = {
            "kind": "workflow",
            "name": run_args.workflow,
            "params": workflow_params,
        }
    else:
        workload = {
            "kind": "model",
            "model_type": run_args.model_type,
            "weights_path": run_args.weights,
        }
    work_store = build_block_work_store(
        cfg=cfg,
        input_spec=input_spec,
        output_specs=output_specs,
        input_store=input_store,
        output_stores=output_stores,
        workload=workload,
    )
    run_kwargs = dict(
        metrics_json=metrics_json,
        metrics_interval=run_args.metrics_interval,
        num_prep_workers=max(1, run_args.prep_workers),
        num_writer_workers=max(1, run_args.writer_workers),
        thread_dump_interval=run_args.thread_dump_interval,
        work_store=work_store,
    )

    if run_args.workflow:
        workflow = WorkflowRegistry.build(
            run_args.workflow, copy.deepcopy(workflow_params)
        )
        run_workflow(workflow, input_store, output_stores, cfg, **run_kwargs)
    else:
        model = load_model(run_args.model_type, run_args.weights)
        run(model, input_store, output_stores, cfg, **run_kwargs)


def _launch_locally(
    base_cfg: InferenceConfig,
    run_args: argparse.Namespace,
    workflow_params: Dict[str, Any],
    shards: int,
    metrics_template: Optional[str],
    input_spec: Dict[str, Any],
    output_specs: Sequence[Dict[str, Any]],
) -> None:
    logger.info("Running locally across %d shard(s).", shards)
    for shard in range(shards):
        logger.info("Starting local shard %d/%d", shard, shards)
        payload = _make_shard_payload(
            base_cfg,
            metrics_template,
            run_args.metrics_json,
            shard,
        )
        cfg = InferenceConfig(**payload["config"])
        cfg.devices = list(_canonical_devices(cfg.devices))
        _run_shard(
            run_args,
            workflow_params,
            cfg,
            input_spec,
            output_specs,
            payload["metrics_json"],
        )


def _parse_ray_args(
    argv: Optional[Sequence[str]] = None,
) -> Tuple[argparse.Namespace, argparse.Namespace]:
    ray_parser = argparse.ArgumentParser(add_help=False)
    ray_parser.add_argument("--ray-address", type=str, default=None)
    ray_parser.add_argument("--num-shards", type=int, default=None)
    ray_parser.add_argument("--cpus-per-shard", type=float, default=None)
    ray_parser.add_argument("--gpus-per-shard", type=float, default=None)
    ray_parser.add_argument(
        "--metrics-json-template",
        type=str,
        default=None,
        help="Template for shard metrics file paths (use '{shard}' placeholder).",
    )
    ray_parser.add_argument(
        "--local-fallback",
        action="store_true",
        help="Run shards sequentially in-process if Ray is unavailable.",
    )
    ray_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan Ray tasks without executing them.",
    )
    ray_args, remaining = ray_parser.parse_known_args(argv)
    run_args = parse_inference_args(remaining)
    return ray_args, run_args


def _load_spec_arg(arg: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(arg, dict):
        return copy.deepcopy(arg)
    stripped = arg.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return json.loads(stripped)
    with open(arg, "r", encoding="utf-8") as f:
        return json.load(f)


def _prepare_output_spec_for_shards(spec: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = copy.deepcopy(spec)
    sanitized["delete_existing"] = False
    sanitized["create"] = False
    sanitized["open"] = True
    return sanitized


def _validate_zarr_chunk_layout(
    block: Sequence[int],
    chunk_shape: Optional[Sequence[Optional[int]]],
    grid_origin: Optional[Sequence[Optional[int]]],
    *,
    output_label: str,
) -> None:
    """Ensure spatial blocks do not share Zarr chunks across Ray shards."""
    spatial_block = tuple(block)
    spatial_chunk = (
        tuple(chunk_shape[-3:])
        if chunk_shape is not None and len(chunk_shape) == 5
        else None
    )
    spatial_origin = (
        tuple(grid_origin[-3:])
        if grid_origin is not None and len(grid_origin) == 5
        else None
    )
    layout = (
        f"block={spatial_block}, chunk={spatial_chunk}, "
        f"grid_origin={spatial_origin}"
    )

    layout_is_indeterminate = spatial_chunk is None or spatial_origin is None
    if not layout_is_indeterminate:
        layout_is_indeterminate = any(
            value is None for value in (*spatial_chunk, *spatial_origin)
        )
    if len(spatial_block) != 3 or layout_is_indeterminate:
        raise ValueError(
            f"Unsafe multi-shard Zarr output layout for {output_label}: "
            f"{layout}. The 5D write chunk shape and grid origin must be "
            "determinate before Ray tasks start."
        )

    for axis, (block_dim, chunk_dim, origin) in enumerate(
        zip(spatial_block, spatial_chunk, spatial_origin)
    ):
        invalid_dimension = chunk_dim is None or origin is None
        if not invalid_dimension:
            invalid_dimension = any(
                (
                    chunk_dim <= 0,
                    block_dim % chunk_dim != 0,
                    origin % chunk_dim != 0,
                )
            )
        if invalid_dimension:
            axis_name = "ZYX"[axis]
            raise ValueError(
                f"Unsafe multi-shard Zarr output layout for {output_label} "
                f"on spatial axis {axis_name}: {layout}. Each spatial chunk "
                "dimension must divide its block dimension and the chunk grid "
                "must align with the zero-based block grid."
            )


def _validate_declared_zarr_output(
    spec: Dict[str, Any],
    block: Sequence[int],
    *,
    output_label: str,
) -> None:
    """Validate declared Zarr metadata without opening or mutating its target."""
    if spec.get("driver") not in {"zarr", "zarr2"} or "metadata" not in spec:
        return
    try:
        layout = ts.Spec(copy.deepcopy(spec)).chunk_layout
        chunk_shape = layout.write_chunk.shape
        grid_origin = layout.grid_origin
    except Exception as exc:
        raise ValueError(
            f"Cannot determine multi-shard Zarr output layout for "
            f"{output_label} from declared metadata."
        ) from exc
    _validate_zarr_chunk_layout(
        block,
        chunk_shape,
        grid_origin,
        output_label=output_label,
    )


def _validate_opened_zarr_output(
    spec: Dict[str, Any],
    store: Any,
    block: Sequence[int],
    *,
    output_label: str,
) -> None:
    """Validate the effective write layout reported by an opened Zarr store."""
    if spec.get("driver") not in {"zarr", "zarr2"}:
        return
    try:
        layout = store.chunk_layout
        chunk_shape = layout.write_chunk.shape
        grid_origin = layout.grid_origin
    except Exception as exc:
        raise ValueError(
            f"Cannot determine effective multi-shard Zarr output layout for "
            f"{output_label} after opening it."
        ) from exc
    _validate_zarr_chunk_layout(
        block,
        chunk_shape,
        grid_origin,
        output_label=output_label,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    ray_args, run_args = _parse_ray_args(argv)

    if bool(run_args.model_type) == bool(run_args.workflow):
        raise SystemExit("Provide exactly one of --model-type or --workflow.")
    if run_args.workflow and run_args.weights:
        raise SystemExit(
            "--weights only applies with --model-type; pass weights to the "
            "workflow via --workflow-params."
        )
    workflow_params = (
        _load_workflow_params(run_args.workflow_params) if run_args.workflow else {}
    )

    base_cfg = _build_inference_config(run_args)
    shards = ray_args.num_shards or max(1, base_cfg.shard_count)
    if shards != base_cfg.shard_count:
        base_cfg = _build_inference_config(run_args, shards)
    metrics_template = ray_args.metrics_json_template or _default_metrics_template(
        run_args.metrics_json
    )

    input_spec_dict = _load_spec_arg(run_args.in_spec)
    output_spec_dicts = [_load_spec_arg(arg) for arg in run_args.out_spec]
    validate_resume_output_specs(base_cfg, output_spec_dicts)

    if shards > 1:
        for index, output_spec in enumerate(output_spec_dicts):
            _validate_declared_zarr_output(
                output_spec,
                base_cfg.block,
                output_label=f"output {index}",
            )

    if ray_args.dry_run:
        logger.info(
            "Dry run: would launch %d shard(s) with strategy=%s and metrics template=%s",
            shards,
            base_cfg.shard_strategy,
            metrics_template,
        )
        return

    logger.info(
        "Preparing output store (create/delete as specified) before sharded run."
    )
    for index, output_spec in enumerate(output_spec_dicts):
        output_store = open_ts_spec(copy.deepcopy(output_spec))
        if shards > 1:
            _validate_opened_zarr_output(
                output_spec,
                output_store,
                base_cfg.block,
                output_label=f"output {index}",
            )
    output_specs_for_shards = [
        _prepare_output_spec_for_shards(spec) for spec in output_spec_dicts
    ]

    if ray_args.local_fallback:
        _launch_locally(
            base_cfg,
            run_args,
            workflow_params,
            shards,
            metrics_template,
            input_spec_dict,
            output_specs_for_shards,
        )
        return

    try:
        ray = _import_ray()
    except RuntimeError:
        raise

    init_kwargs: Dict[str, Any] = {}
    if ray_args.ray_address:
        init_kwargs["address"] = ray_args.ray_address
    ray.init(**init_kwargs)

    try:
        cpus = ray_args.cpus_per_shard or max(
            1.0, run_args.prep_workers + run_args.writer_workers
        )
        requested_gpus = ray_args.gpus_per_shard
        if requested_gpus is None:
            requested_gpus = max(0.0, len(base_cfg.devices))

        @ray.remote(num_cpus=cpus, num_gpus=requested_gpus)
        def shard_task(shard_idx: int, payload: Dict[str, Any]) -> None:
            cfg_dict = payload["config"]
            cfg_dict["shard_index"] = shard_idx
            cfg = InferenceConfig(**cfg_dict)
            cfg.devices = list(_canonical_devices(cfg.devices))
            metrics_json = payload["metrics_json"]
            _run_shard(
                run_args,
                workflow_params,
                cfg,
                input_spec_dict,
                output_specs_for_shards,
                metrics_json,
            )

        futures = []
        for shard_idx in range(shards):
            payload = _make_shard_payload(
                base_cfg,
                metrics_template,
                run_args.metrics_json,
                shard_idx,
            )
            futures.append(shard_task.remote(shard_idx, payload))

        ray.get(futures)
    finally:
        ray.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
