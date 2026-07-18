"""Workflow: one object bundling a run's injected math, plus a name registry.

After PRs 1-5 every workflow-owned decision -- preprocessing, the processor, its
execution policy, and per-output merge/post/dtype -- is an injectable object. A
:class:`Workflow` groups them so a recipe can hand the runtime a single value and the
generic CLI can reach a named recipe via ``--workflow`` with no workflow-specific flags.

Scope note (issue #25 §3.7): the registry is intentionally the *last* piece and stays
deliberately small. It is provisional until a second real workflow has exercised these
contracts; concrete recipes live under :mod:`aind_torch_utils.recipes`.
"""
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

from aind_torch_utils.execution import ExecutionPolicy
from aind_torch_utils.outputs import OutputSpec
from aind_torch_utils.transforms import BlockPreprocessor

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


@runtime_checkable
class BlockProcessor(Protocol):
    """Tensor-in/tensor-out GPU processor: ``(B, Cin, Z, Y, X) -> (B, N, Z, Y, X)``.

    Any ``nn.Module`` satisfies this; so does any callable hiding CUDA/CuPy/compiled
    code. The runtime never assumes it is a neural network.
    """

    def __call__(self, batch: "torch.Tensor") -> "torch.Tensor":
        """Run the processor on a batch of patches."""
        ...


# A factory that turns the opened output stores into per-output specs. Lets a recipe
# defer spec construction until it sees the actual stores (e.g. to read their dtypes).
OutputSpecFactory = Callable[[List[Any]], List[OutputSpec]]


@dataclass
class Workflow:
    """A complete recipe: the processor plus the objects the runtime injects.

    Attributes
    ----------
    processor : BlockProcessor
        The model / GPU function run on each batch of patches.
    preprocess : BlockPreprocessor, optional
        Input-domain transform; ``None`` lets the runtime synthesize one from config.
    outputs : list of OutputSpec, optional
        Fixed per-output specs. Mutually exclusive with ``output_spec_factory``.
    output_spec_factory : OutputSpecFactory, optional
        Builds the specs from the opened output stores, when they are needed to
        construct the specs. Mutually exclusive with ``outputs``.
    execution : ExecutionPolicy, optional
        How the processor runs. ``None`` (default) lets the runtime synthesize
        the policy from config — the same rule as ``preprocess`` — so CLI/config
        AMP and compile flags keep working; a recipe sets it to declare its
        processor's constraints, which then override config.
    """

    processor: BlockProcessor
    preprocess: Optional[BlockPreprocessor] = None
    outputs: Optional[List[OutputSpec]] = None
    output_spec_factory: Optional[OutputSpecFactory] = None
    execution: Optional[ExecutionPolicy] = None

    def __post_init__(self):
        if self.outputs is not None and self.output_spec_factory is not None:
            raise ValueError(
                "Set either outputs or output_spec_factory on a Workflow, not both."
            )

    def resolve_outputs(
        self, output_stores: List[Any]
    ) -> Optional[List[OutputSpec]]:
        """Return per-output specs for these stores, or ``None`` to let run() default.

        Fixed ``outputs`` win; otherwise the factory (if any) builds them from the
        stores; otherwise ``None`` (the runtime synthesizes defaults from config).
        """
        if self.outputs is not None:
            return self.outputs
        if self.output_spec_factory is not None:
            return self.output_spec_factory(output_stores)
        return None


class WorkflowRegistry:
    """Registry mapping a workflow name to a builder ``build(params) -> Workflow``."""

    _registry: Dict[str, Callable[[Dict[str, Any]], Workflow]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator registering a workflow builder under ``name``."""

        def decorator(
            func: Callable[[Dict[str, Any]], Workflow]
        ) -> Callable[[Dict[str, Any]], Workflow]:
            cls._registry[name] = func
            return func

        return decorator

    @classmethod
    def build(cls, name: str, params: Optional[Dict[str, Any]] = None) -> Workflow:
        """Build the named workflow from ``params`` (a plain dict of recipe args)."""
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(
                f"Workflow '{name}' not found in registry. Available: {available}"
            )
        return cls._registry[name](params or {})

    @classmethod
    def list_workflows(cls) -> List[str]:
        """List all registered workflow names."""
        return list(cls._registry.keys())
