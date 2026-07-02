import pytest

from aind_torch_utils.run import _parse_args, main


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
