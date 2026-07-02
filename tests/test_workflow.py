"""Tests for Workflow, WorkflowRegistry, and run_workflow unpacking."""
import pytest

import aind_torch_utils.run as run_mod
from aind_torch_utils.execution import ExecutionPolicy
from aind_torch_utils.outputs import OutputSpec
from aind_torch_utils.workflow import Workflow, WorkflowRegistry


def _factory():
    return object()


def test_workflow_defaults():
    wf = Workflow(processor=object())
    assert wf.preprocess is None
    assert wf.outputs is None
    assert isinstance(wf.execution, ExecutionPolicy)
    assert wf.resolve_outputs([object()]) is None


def test_workflow_resolve_fixed_outputs():
    specs = [OutputSpec(store=object(), accumulator_factory=_factory())]
    wf = Workflow(processor=object(), outputs=specs)
    assert wf.resolve_outputs([object(), object()]) is specs


def test_workflow_resolve_via_factory():
    stores = [object(), object()]
    seen = {}

    def make_specs(s):
        seen["stores"] = s
        return [OutputSpec(store=st, accumulator_factory=_factory()) for st in s]

    wf = Workflow(processor=object(), output_spec_factory=make_specs)
    specs = wf.resolve_outputs(stores)
    assert seen["stores"] is stores
    assert len(specs) == 2


def test_workflow_rejects_outputs_and_factory():
    with pytest.raises(ValueError, match="not both"):
        Workflow(
            processor=object(),
            outputs=[OutputSpec(store=object(), accumulator_factory=_factory())],
            output_spec_factory=lambda s: [],
        )


def test_registry_register_build_list():
    @WorkflowRegistry.register("unit-test-workflow")
    def _build(params):
        return Workflow(
            processor=object(),
            execution=ExecutionPolicy(compile=params.get("c", False)),
        )

    assert "unit-test-workflow" in WorkflowRegistry.list_workflows()
    wf = WorkflowRegistry.build("unit-test-workflow", {"c": True})
    assert wf.execution.compile is True
    # missing params -> empty dict
    assert WorkflowRegistry.build("unit-test-workflow").execution.compile is False


def test_registry_unknown_raises():
    with pytest.raises(KeyError, match="not found in registry"):
        WorkflowRegistry.build("does-not-exist")


def test_run_workflow_forwards_processor_and_defaults(monkeypatch):
    captured = {}

    def fake_run(model, input_store, output_store, cfg, **kw):
        captured["model"] = model
        captured["output_store"] = output_store
        captured.update(kw)

    monkeypatch.setattr(run_mod, "run", fake_run)

    proc, pp = object(), object()
    exec_policy = ExecutionPolicy(autocast=True)
    wf = Workflow(processor=proc, preprocess=pp, execution=exec_policy)
    run_mod.run_workflow(wf, "IN", "OUT", "CFG", metrics_json="m.json")

    assert captured["model"] is proc
    assert captured["preprocess"] is pp
    assert captured["execution"] is exec_policy
    assert captured["outputs"] is None
    # No explicit outputs -> the store is forwarded for run() to synthesize specs.
    assert captured["output_store"] == "OUT"
    assert captured["metrics_json"] == "m.json"


def test_run_workflow_with_fixed_outputs_passes_none_store(monkeypatch):
    captured = {}

    def fake_run(model, input_store, output_store, cfg, **kw):
        captured["output_store"] = output_store
        captured["outputs"] = kw["outputs"]

    monkeypatch.setattr(run_mod, "run", fake_run)

    specs = [OutputSpec(store=object(), accumulator_factory=_factory())]
    wf = Workflow(processor=object(), outputs=specs)
    run_mod.run_workflow(wf, "IN", "OUT", "CFG")

    # Explicit specs -> output_store must be None (run rejects both).
    assert captured["output_store"] is None
    assert captured["outputs"] is specs
