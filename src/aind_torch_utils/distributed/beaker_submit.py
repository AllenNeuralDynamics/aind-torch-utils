"""Render and submit reproducible Beaker denoising experiments."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import tensorstore as ts

from aind_torch_utils.background_offset import run_prepass, write_json
from aind_torch_utils.config import InferenceConfig
from aind_torch_utils.tensorstore_specs import (
    InputArrayMetadata,
    build_specs,
    derive_output_root,
    inspect_input_array,
    normalize_zarr_location,
    s3_prefix_name,
    sanitized_run_slug,
    validate_input_metadata,
    validate_output_chunks,
)


IMAGE_METRICS_WRAPPER = "/opt/aind-torch-utils/beaker/run-with-gpu-metrics.sh"
CHECKPOINT_MOUNT = "/config"


def _import_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required; install aind-torch-utils[beaker]."
        ) from exc
    return yaml


def _default_template() -> Path:
    repository_template = (
        Path(__file__).resolve().parents[3] / "beaker" / "single-node-ray.yaml"
    )
    if repository_template.exists():
        return repository_template
    return Path("beaker/single-node-ray.yaml")


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _read_template(path: Path) -> dict[str, Any]:
    yaml = _import_yaml()
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    tasks = value.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], dict):
        raise ValueError("The Beaker base YAML must contain exactly one task.")
    return value


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    yaml = _import_yaml()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)


def _compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def _task(spec: dict[str, Any]) -> dict[str, Any]:
    return spec["tasks"][0]


def _set_argument(arguments: list[str], option: str, value: str) -> None:
    try:
        index = arguments.index(option)
    except ValueError:
        arguments.extend([option, value])
        return
    if index + 1 >= len(arguments):
        raise ValueError(f"Base YAML argument {option} is missing its value.")
    arguments[index + 1] = value


def _set_env_value(task: dict[str, Any], name: str, value: str) -> None:
    env_vars = task.setdefault("envVars", [])
    for item in env_vars:
        if item.get("name") == name:
            item.pop("secret", None)
            item["value"] = value
            return
    env_vars.append({"name": name, "value": value})


def _set_env_secret(task: dict[str, Any], name: str, secret: str | None) -> None:
    env_vars = task.setdefault("envVars", [])
    env_vars[:] = [item for item in env_vars if item.get("name") != name]
    if secret:
        env_vars.append({"name": name, "secret": secret})


def _checkpoint_dataset(task: dict[str, Any]) -> tuple[dict[str, Any], str]:
    datasets = task.get("datasets", [])
    matches = [item for item in datasets if item.get("mountPath") == CHECKPOINT_MOUNT]
    if len(matches) != 1:
        raise ValueError(
            "Base YAML must mount exactly one checkpoint dataset at "
            f"{CHECKPOINT_MOUNT}."
        )
    source = matches[0].get("source")
    if not isinstance(source, dict) or not isinstance(source.get("beaker"), str):
        raise ValueError("Checkpoint dataset source must contain a Beaker ID.")
    return matches[0], source["beaker"]


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
    )


def _find_checkpoint(dataset: str, requested: str | None) -> str:
    if requested:
        if "/" in requested or requested in {".", ".."}:
            raise ValueError("--checkpoint-name must be a dataset-root filename.")
        return requested
    if shutil.which("beaker") is None:
        raise RuntimeError("The Beaker CLI is not installed or not on PATH.")
    result = _run_command(["beaker", "dataset", "ls", dataset, "--format", "json"])
    listing = json.loads(result.stdout)
    checkpoints = sorted(
        item["path"]
        for item in listing
        if isinstance(item, dict)
        and isinstance(item.get("path"), str)
        and item["path"].lower().endswith((".pth", ".pt", ".ckpt"))
        and "/" not in item["path"]
    )
    if len(checkpoints) != 1:
        raise ValueError(
            f"Checkpoint dataset {dataset} contains {len(checkpoints)} root-level "
            "checkpoint files; pass --checkpoint-name explicitly."
        )
    return checkpoints[0]


def _read_existing_output(output_spec: dict[str, Any]) -> Any | None:
    probe = {"driver": "zarr", "kvstore": copy.deepcopy(output_spec["kvstore"])}
    try:
        return ts.open(probe, open=True).result()
    except Exception as exc:
        if "NOT_FOUND" in str(exc):
            return None
        raise RuntimeError("Could not determine whether the output exists.") from exc


def _validate_existing_output(
    store: Any,
    metadata: InputArrayMetadata,
    chunks: Sequence[int],
) -> None:
    actual_shape = tuple(int(value) for value in store.domain.shape)
    actual_origin = tuple(int(value) for value in store.domain.inclusive_min)
    actual_chunks = tuple(int(value) for value in store.chunk_layout.write_chunk.shape)
    actual_dtype = store.dtype.numpy_dtype.str
    expected_chunks = tuple(int(value) for value in chunks)
    mismatches = []
    if actual_shape != metadata.shape:
        mismatches.append(f"shape {actual_shape} != {metadata.shape}")
    if actual_origin != metadata.origin:
        mismatches.append(f"origin {actual_origin} != {metadata.origin}")
    if actual_chunks != expected_chunks:
        mismatches.append(f"chunks {actual_chunks} != {expected_chunks}")
    if actual_dtype != metadata.dtype:
        mismatches.append(f"dtype {actual_dtype} != {metadata.dtype}")
    if mismatches:
        raise ValueError("Existing output is incompatible: " + "; ".join(mismatches))


def _apply_infrastructure_overrides(
    experiment: dict[str, Any], args: argparse.Namespace
) -> tuple[str, str]:
    task = _task(experiment)
    dataset_mount, dataset_id = _checkpoint_dataset(task)
    if args.dataset:
        dataset_id = args.dataset
        dataset_mount["source"]["beaker"] = dataset_id
    if args.image:
        task.setdefault("image", {})["beaker"] = args.image
    if args.cluster:
        task.setdefault("context", {})["cluster"] = args.cluster
    if args.budget:
        experiment["budget"] = args.budget
    if args.aws_region:
        _set_env_value(task, "AWS_REGION", args.aws_region)
        _set_env_value(task, "AWS_DEFAULT_REGION", args.aws_region)
    if args.aws_access_key_secret:
        _set_env_secret(
            task, "AWS_ACCESS_KEY_ID", args.aws_access_key_secret
        )
    if args.aws_secret_access_key_secret:
        _set_env_secret(
            task, "AWS_SECRET_ACCESS_KEY", args.aws_secret_access_key_secret
        )
    if args.aws_session_token_secret is not None:
        _set_env_secret(task, "AWS_SESSION_TOKEN", args.aws_session_token_secret)
    return dataset_id, args.aws_region or _env_value(task, "AWS_REGION") or ""


def _env_value(task: dict[str, Any], name: str) -> str | None:
    for item in task.get("envVars", []):
        if item.get("name") == name and isinstance(item.get("value"), str):
            return item["value"]
    return None


def _render_experiment(
    base: dict[str, Any],
    *,
    task_name: str,
    input_spec: dict[str, Any],
    output_spec: dict[str, Any],
    workflow_params: dict[str, Any],
    inference_config: dict[str, Any],
) -> dict[str, Any]:
    experiment = copy.deepcopy(base)
    task = _task(experiment)
    task["name"] = task_name
    task["command"] = [IMAGE_METRICS_WRAPPER]
    arguments = task.get("arguments")
    if not isinstance(arguments, list) or not all(
        isinstance(value, str) for value in arguments
    ):
        raise ValueError("Base YAML task arguments must be a list of strings.")
    _set_argument(arguments, "--in-spec", _compact_json(input_spec))
    _set_argument(arguments, "--out-spec", _compact_json(output_spec))
    _set_argument(arguments, "--workflow-params", _compact_json(workflow_params))
    _set_argument(arguments, "--config", _compact_json(inference_config))
    experiment["description"] = f"Denoise {input_spec['kvstore']}"
    return experiment


def _default_run_dir(template: Path, slug: str, timestamp: str) -> Path:
    return template.parent / "runs" / f"{timestamp}-{slug}"


def _parse_submission(stdout: str) -> Any:
    stripped = stdout.strip()
    if not stripped:
        return {"stdout": ""}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return {"stdout": stripped}


def _write_any_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect an S3 Zarr, estimate its background offset, render a "
            "Beaker experiment, and submit it."
        )
    )
    parser.add_argument("input_uri", help="S3 Zarr root or resolution-array URI.")
    parser.add_argument("--output-uri", default=None, help="Output Zarr root URI.")
    parser.add_argument("--input-resolution", default="0")
    parser.add_argument("--stats-resolution", default="5")
    parser.add_argument("--output-resolution", default="0")
    parser.add_argument("--offset-percentile", type=float, default=1.0)
    parser.add_argument("--include-zeros", action="store_true")
    parser.add_argument("--offset-workers", type=int, default=None)
    parser.add_argument("--offset-threads-per-worker", type=int, default=1)
    parser.add_argument("--offset-no-processes", action="store_true")
    parser.add_argument(
        "--output-chunks",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=None,
    )
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--template", type=Path, default=_default_template())
    parser.add_argument("--inference-config", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--no-submit", action="store_true")
    parser.add_argument("--image", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--budget", default=None)
    parser.add_argument("--cluster", default=None)
    parser.add_argument("--aws-region", default=None)
    parser.add_argument("--aws-access-key-secret", default=None)
    parser.add_argument("--aws-secret-access-key-secret", default=None)
    parser.add_argument(
        "--aws-session-token-secret",
        default=None,
        help="Set to an empty string to remove AWS_SESSION_TOKEN from the task.",
    )
    return parser.parse_args(argv)


def submit(args: argparse.Namespace) -> tuple[Path, Any | None]:
    """Prepare one run and optionally submit it, returning its directory/result."""
    template = args.template.resolve()
    base = _read_template(template)
    infrastructure = copy.deepcopy(base)
    dataset_id, aws_region = _apply_infrastructure_overrides(infrastructure, args)

    input_location = normalize_zarr_location(
        args.input_uri, args.input_resolution
    )
    metadata = inspect_input_array(input_location)
    validate_input_metadata(metadata)
    output_location = normalize_zarr_location(
        args.output_uri or derive_output_root(input_location.root_uri),
        args.output_resolution,
    )
    if output_location.resolution != str(args.output_resolution):
        raise ValueError(
            "An explicit output resolution must match --output-resolution."
        )
    if output_location.root_uri == input_location.root_uri:
        raise ValueError("Input and output Zarr roots must be different.")

    inference_path = (
        args.inference_config.resolve()
        if args.inference_config
        else template.parent / "inference.json"
    )
    inference_config = _read_json_object(inference_path)
    validated_config = InferenceConfig.model_validate(inference_config)
    output_chunks = tuple(metadata.chunks)
    if args.output_chunks:
        output_chunks = (*metadata.chunks[:2], *tuple(args.output_chunks))
    output_chunks = validate_output_chunks(output_chunks, validated_config.block)
    input_spec, output_spec = build_specs(
        metadata,
        output_location.root_uri,
        output_chunks,
        aws_region,
        output_resolution=str(args.output_resolution),
    )

    existing_output = _read_existing_output(output_spec)
    if existing_output is not None:
        if not args.resume_existing:
            raise FileExistsError(
                f"Output {output_location.array_uri} already exists; pass "
                "--resume-existing to validate and reuse it."
            )
        if not validated_config.resume:
            raise ValueError(
                "--resume-existing requires resume=true in the inference config."
            )
        _validate_existing_output(existing_output, metadata, output_chunks)

    checkpoint_name = _find_checkpoint(dataset_id, args.checkpoint_name)
    checkpoint_path = f"{CHECKPOINT_MOUNT}/{checkpoint_name}"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    slug = sanitized_run_slug(input_location.root_uri)
    task_name = args.name or f"denoise-{s3_prefix_name(input_location.root_uri)}"
    run_dir = (
        args.run_dir.resolve()
        if args.run_dir
        else _default_run_dir(template, slug, timestamp)
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "input.json", input_spec)
    write_json(run_dir / "output.json", output_spec)
    write_json(run_dir / "inference.json", inference_config)

    workflow_params, stats = run_prepass(
        in_spec=input_spec,
        checkpoint_path=checkpoint_path,
        stats_resolution=args.stats_resolution,
        offset_percentile=args.offset_percentile,
        config=inference_config,
        ignore_zeros=not args.include_zeros,
        workers=args.offset_workers,
        threads_per_worker=args.offset_threads_per_worker,
        processes=not args.offset_no_processes,
    )
    write_json(run_dir / "workflow.json", workflow_params)
    write_json(run_dir / "offset-stats.json", stats)

    experiment = _render_experiment(
        infrastructure,
        task_name=task_name,
        input_spec=input_spec,
        output_spec=output_spec,
        workflow_params=workflow_params,
        inference_config=inference_config,
    )
    experiment_path = run_dir / "experiment.yaml"
    _write_yaml(experiment_path, experiment)
    if args.no_submit:
        return run_dir, None
    if shutil.which("beaker") is None:
        raise RuntimeError("The Beaker CLI is not installed or not on PATH.")
    command = [
        "beaker",
        "experiment",
        "create",
        str(experiment_path),
        "--format",
        "json",
        "--name",
        task_name,
    ]
    if args.workspace:
        command.extend(["--workspace", args.workspace])
    try:
        result = _run_command(command)
    except subprocess.CalledProcessError as exc:
        failure = {
            "command": list(exc.cmd),
            "returncode": exc.returncode,
            "stdout": exc.stdout,
            "stderr": exc.stderr,
        }
        _write_any_json(run_dir / "submission-error.json", failure)
        raise
    submission = _parse_submission(result.stdout)
    _write_any_json(run_dir / "submission.json", submission)
    return run_dir, submission


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    try:
        run_dir, submission = submit(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Run artifacts: {run_dir}")
    if submission is None:
        print("Experiment was not submitted (--no-submit).")
    else:
        print(json.dumps(submission, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
