from __future__ import annotations

import argparse
import logging
import os
import sys
import copy
import json
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from aind_torch_utils.config import InferenceConfig
from aind_torch_utils.run import (
    _parse_args as parse_inference_args,
    load_model,
    run,
)
from aind_torch_utils.utils import open_ts_spec

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
        return InferenceConfig.from_json_file(
            args.config,
            shard_count=shard_count,
            shard_index=0,
        )
    data: Dict[str, Any] = {}
    if shard_count is not None:
        data["shard_count"] = shard_count
    data["shard_index"] = 0
    return InferenceConfig.model_validate(data)


def _default_metrics_template(base: Optional[str]) -> Optional[str]:
    if not base:
        return None
    stem, ext = os.path.splitext(base)
    if not ext:
        ext = ".json"
    return f"{stem}_shard{{shard}}{ext}"


def _resolve_metrics_path(base: Optional[str], template: Optional[str], shard: int) -> Optional[str]:
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


def _launch_locally(
    base_cfg: InferenceConfig,
    run_args: argparse.Namespace,
    shards: int,
    metrics_template: Optional[str],
    input_spec: Dict[str, Any],
    output_spec: Dict[str, Any],
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
        model = load_model(run_args.model_type, run_args.weights)
        input_store = open_ts_spec(copy.deepcopy(input_spec))
        output_store = open_ts_spec(copy.deepcopy(output_spec))
        run(
            model,
            input_store,
            output_store,
            cfg,
            metrics_json=payload["metrics_json"],
            metrics_interval=run_args.metrics_interval,
            num_prep_workers=max(1, run_args.prep_workers),
            num_writer_workers=max(1, run_args.writer_workers),
        )


def _parse_ray_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, argparse.Namespace]:
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


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    ray_args, run_args = _parse_ray_args(argv)

    base_cfg = _build_inference_config(run_args)
    shards = ray_args.num_shards or max(1, base_cfg.shard_count)
    if shards != base_cfg.shard_count:
        base_cfg = _build_inference_config(run_args, shards)
    metrics_template = ray_args.metrics_json_template or _default_metrics_template(
        run_args.metrics_json
    )

    input_spec_dict = _load_spec_arg(run_args.in_spec)
    output_spec_dict = _load_spec_arg(run_args.out_spec)

    if ray_args.dry_run:
        logger.info(
            "Dry run: would launch %d shard(s) with strategy=%s and metrics template=%s",
            shards,
            base_cfg.shard_strategy,
            metrics_template,
        )
        return

    logger.info("Preparing output store (create/delete as specified) before sharded run.")
    open_ts_spec(copy.deepcopy(output_spec_dict))
    output_spec_for_shards = _prepare_output_spec_for_shards(output_spec_dict)

    if ray_args.local_fallback:
        _launch_locally(
            base_cfg,
            run_args,
            shards,
            metrics_template,
            input_spec_dict,
            output_spec_for_shards,
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
        cpus = ray_args.cpus_per_shard or max(1.0, run_args.prep_workers + run_args.writer_workers)
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
            model = load_model(run_args.model_type, run_args.weights)
            input_store = open_ts_spec(copy.deepcopy(input_spec_dict))
            output_store = open_ts_spec(copy.deepcopy(output_spec_for_shards))
            run(
                model,
                input_store,
                output_store,
                cfg,
                metrics_json=metrics_json,
                metrics_interval=run_args.metrics_interval,
                num_prep_workers=max(1, run_args.prep_workers),
                num_writer_workers=max(1, run_args.writer_workers),
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
