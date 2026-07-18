from aind_torch_utils.run import _parse_args


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
