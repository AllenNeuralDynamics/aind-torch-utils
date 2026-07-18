import json

import pytest

import aind_torch_utils.run as run_mod
from aind_torch_utils.run import _parse_args, main
from aind_torch_utils.workflow import Workflow


def _minimal_required_args(extra=None):
    base = [
        "--in-spec",
        "in.json",
        "--out-spec",
        "out.json",
        "--model-type",
        "dummy",
    ]
    if extra:
        base.extend(extra)
    return base


def test_cli_output_denormalize_default_enabled():
    args = _parse_args(_minimal_required_args())
    assert args.no_output_denormalize is False


def test_cli_output_denormalize_can_be_disabled():
    args = _parse_args(_minimal_required_args(["--no-output-denormalize"]))
    assert args.no_output_denormalize is True


def test_cli_workflow_flag_parsed():
    args = _parse_args(
        ["--in-spec", "in.json", "--out-spec", "out.json", "--workflow", "seg"]
    )
    assert args.workflow == "seg"
    assert args.model_type is None


def test_main_requires_exactly_one_of_model_or_workflow():
    # Neither --model-type nor --workflow -> fail fast before any I/O.
    with pytest.raises(SystemExit):
        main(["--in-spec", "in.json", "--out-spec", "out.json"])
    # Both -> also rejected.
    with pytest.raises(SystemExit):
        main(
            [
                "--in-spec", "in.json",
                "--out-spec", "out.json",
                "--model-type", "d",
                "--workflow", "w",
            ]
        )


def test_main_rejects_weights_with_workflow():
    """--weights would be silently ignored in workflow mode; refuse it instead."""
    with pytest.raises(SystemExit, match="--weights only applies"):
        main(
            [
                "--in-spec", "in.json",
                "--out-spec", "out.json",
                "--workflow", "w",
                "--weights", "w.pt",
            ]
        )


def test_main_forwards_json_parameters_to_workflow(tmp_path, monkeypatch):
    params = {"checkpoint_path": "/models/denoise.pth", "offset": 73.5}
    params_path = tmp_path / "denoise-params.json"
    params_path.write_text(json.dumps(params))
    captured = {}

    def build(name, received):
        captured["name"] = name
        captured["params"] = received
        return Workflow(processor=object())

    monkeypatch.setattr(run_mod.WorkflowRegistry, "build", build)
    monkeypatch.setattr(run_mod, "open_ts_spec", lambda spec: spec)
    monkeypatch.setattr(
        run_mod,
        "run_workflow",
        lambda workflow, in_arr, out_arr, cfg, **kwargs: captured.update(
            workflow=workflow, in_arr=in_arr, out_arr=out_arr
        ),
    )

    main(
        [
            "--in-spec",
            "in.json",
            "--out-spec",
            "out.json",
            "--workflow",
            "denoise-net",
            "--workflow-params",
            str(params_path),
        ]
    )

    assert captured["name"] == "denoise-net"
    assert captured["params"] == params
    assert captured["in_arr"] == "in.json"
    assert captured["out_arr"] == ["out.json"]
