"""Tests for Beaker experiment rendering and submission preparation."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aind_torch_utils.distributed import beaker_submit
from aind_torch_utils.tensorstore_specs import InputArrayMetadata, ZarrLocation


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _metadata():
    return InputArrayMetadata(
        location=ZarrLocation(
            "s3://input-bucket/sample.zarr",
            "s3://input-bucket/sample.zarr/0",
            "0",
        ),
        driver="zarr3",
        zarr_format=3,
        shape=(1, 1, 512, 1024, 2048),
        origin=(0, 0, 0, 0, 0),
        dtype="<u2",
        chunks=(1, 1, 128, 256, 256),
    )


def test_find_checkpoint_requires_exactly_one_candidate(monkeypatch):
    monkeypatch.setattr(beaker_submit.shutil, "which", lambda _name: "/bin/beaker")
    monkeypatch.setattr(
        beaker_submit,
        "_run_command",
        lambda _command: type(
            "Result",
            (),
            {"stdout": json.dumps([{"path": "model.pth"}, {"path": "notes.txt"}])},
        )(),
    )
    assert beaker_submit._find_checkpoint("dataset-id", None) == "model.pth"


def test_submit_generates_reproducible_artifacts_without_submission(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        beaker_submit, "inspect_input_array", lambda _location: _metadata()
    )
    monkeypatch.setattr(beaker_submit, "_read_existing_output", lambda _spec: None)
    monkeypatch.setattr(
        beaker_submit,
        "run_prepass",
        lambda **kwargs: (
            {"checkpoint_path": kwargs["checkpoint_path"], "offset": 2.5},
            {"stats_resolution": kwargs["stats_resolution"], "offset": 2.5},
        ),
    )
    run_dir = tmp_path / "run"
    args = beaker_submit.parse_args(
        [
            "s3://input-bucket/sample.zarr",
            "--output-uri",
            "s3://output-bucket/sample-denoised.zarr",
            "--template",
            str(REPOSITORY_ROOT / "beaker/single-node-ray.yaml"),
            "--inference-config",
            str(REPOSITORY_ROOT / "beaker/inference.json"),
            "--checkpoint-name",
            "model.pth",
            "--run-dir",
            str(run_dir),
            "--no-submit",
        ]
    )

    actual_run_dir, submission = beaker_submit.submit(args)

    assert actual_run_dir == run_dir
    assert submission is None
    assert sorted(path.name for path in run_dir.iterdir()) == [
        "experiment.yaml",
        "inference.json",
        "input.json",
        "offset-stats.json",
        "output.json",
        "workflow.json",
    ]
    experiment = beaker_submit._read_template(run_dir / "experiment.yaml")
    task = experiment["tasks"][0]
    assert task["name"] == "denoised-sample.zarr"
    assert task["command"] == [beaker_submit.IMAGE_METRICS_WRAPPER]
    arguments = task["arguments"]
    input_arg = arguments[arguments.index("--in-spec") + 1]
    output_arg = arguments[arguments.index("--out-spec") + 1]
    workflow_arg = arguments[arguments.index("--workflow-params") + 1]
    assert json.loads(input_arg)["driver"] == "zarr3"
    assert json.loads(output_arg)["driver"] == "zarr2"
    assert json.loads(workflow_arg) == {
        "checkpoint_path": "/config/model.pth",
        "offset": 2.5,
    }


def test_existing_output_requires_explicit_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(
        beaker_submit, "inspect_input_array", lambda _location: _metadata()
    )
    monkeypatch.setattr(beaker_submit, "_read_existing_output", lambda _spec: object())
    args = beaker_submit.parse_args(
        [
            "s3://input-bucket/sample.zarr",
            "--template",
            str(REPOSITORY_ROOT / "beaker/single-node-ray.yaml"),
            "--inference-config",
            str(REPOSITORY_ROOT / "beaker/inference.json"),
            "--checkpoint-name",
            "model.pth",
            "--run-dir",
            str(tmp_path / "run"),
            "--no-submit",
        ]
    )
    with pytest.raises(FileExistsError, match="--resume-existing"):
        beaker_submit.submit(args)


def test_validate_existing_output_checks_shape_dtype_and_chunks():
    store = SimpleNamespace(
        domain=SimpleNamespace(
            shape=(1, 1, 512, 1024, 2048),
            inclusive_min=(0, 0, 0, 0, 0),
        ),
        chunk_layout=SimpleNamespace(
            write_chunk=SimpleNamespace(shape=(1, 1, 128, 256, 256))
        ),
        dtype=SimpleNamespace(numpy_dtype=SimpleNamespace(str="<u2")),
    )
    beaker_submit._validate_existing_output(store, _metadata(), _metadata().chunks)
    store.domain.shape = (1, 1, 1, 1, 1)
    with pytest.raises(ValueError, match="shape"):
        beaker_submit._validate_existing_output(
            store, _metadata(), _metadata().chunks
        )


def test_submit_invokes_beaker_and_records_response(tmp_path, monkeypatch):
    monkeypatch.setattr(
        beaker_submit, "inspect_input_array", lambda _location: _metadata()
    )
    monkeypatch.setattr(beaker_submit, "_read_existing_output", lambda _spec: None)
    monkeypatch.setattr(
        beaker_submit,
        "run_prepass",
        lambda **kwargs: (
            {"checkpoint_path": kwargs["checkpoint_path"], "offset": 2.5},
            {"stats_resolution": kwargs["stats_resolution"], "offset": 2.5},
        ),
    )
    monkeypatch.setattr(beaker_submit.shutil, "which", lambda _name: "/bin/beaker")
    commands = []

    def run_command(command):
        commands.append(command)
        return SimpleNamespace(stdout=json.dumps({"id": "experiment-id"}))

    monkeypatch.setattr(beaker_submit, "_run_command", run_command)
    run_dir = tmp_path / "submitted-run"
    args = beaker_submit.parse_args(
        [
            "s3://input-bucket/sample.zarr",
            "--output-uri",
            "s3://output-bucket/sample-denoised.zarr",
            "--template",
            str(REPOSITORY_ROOT / "beaker/single-node-ray.yaml"),
            "--inference-config",
            str(REPOSITORY_ROOT / "beaker/inference.json"),
            "--checkpoint-name",
            "model.pth",
            "--run-dir",
            str(run_dir),
        ]
    )

    _, submission = beaker_submit.submit(args)

    assert submission == {"id": "experiment-id"}
    assert commands[0][:3] == ["beaker", "experiment", "create"]
    assert json.loads((run_dir / "submission.json").read_text()) == submission
