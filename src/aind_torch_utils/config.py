import warnings
from typing import Any, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, model_validator


class InferenceConfig(BaseModel):
    # Geometry
    patch: Tuple[int, int, int] = Field(
        default=(64, 64, 64),
        description="Patch size for the model (Z, Y, X)",
    )
    overlap: int = Field(default=10, description="Overlap between patches, in pixels")
    block: Tuple[int, int, int] = Field(
        default=(256, 256, 256),
        description="Block size for processing (Z, Y, X)",
    )
    batch_size: int = Field(default=32, description="Batch size for inference")

    # Indices (single t, c) to process
    t_idx: int = Field(default=0, description="Time index to process")
    c_idx: int = Field(default=0, description="Channel index to process")

    # Device & precision
    devices: List[str] = Field(
        default_factory=lambda: ["cuda:0"],
        description="List of torch devices to use",
    )
    amp: bool = Field(default=True, description="Use AMP")
    use_tf32: bool = Field(default=True, description="Use TF32")
    use_compile: bool = Field(default=False, description="Use torch.compile")
    compile_mode: str = Field(
        default="reduce-overhead",
        description="Torch.compile mode",  # or "max-autotune" if you want extra tuning time
    )
    compile_dynamic: bool = Field(
        default=True, description="Torch.compile with dynamic shapes"
    )  # tolerate last-batch size changes

    # Concurrency / queues
    max_inflight_batches: int = Field(default=64, description="Max in-flight batches")

    # Sharding
    shard_count: int = Field(
        default=1,
        description="Total number of shards (processes or distributed workers).",
    )
    shard_index: int = Field(
        default=0,
        description="Index of this shard in [0, shard_count).",
    )
    shard_strategy: Literal["contiguous-z", "stride"] = Field(
        default="contiguous-z",
        description="Spatial partitioning strategy across shards.",
    )

    # Seam handling
    # 'blend' = edge-aware Hann/Tukey blending inside blocks
    # 'trim'  = crop overlapped margins (keeps constant contribution)
    seam_mode: str = Field(default="trim", description="Seam handling mode")
    trim_voxels: Optional[int] = Field(
        default=5, description="Voxels to trim for seam handling"
    )
    halo: Optional[int] = Field(
        default=None,
        description=(
            "Halo size. If None: in trim mode defaults to trim_voxels; in blend "
            "mode defaults to a small positive heuristic (min(8, max(1, min(patch)//8)))."
        ),
    )
    min_blend_weight: float = Field(
        default=0.05, description="Minimum blend weight"
    )  # floor to avoid near-zero weights

    # Misc
    eps: float = Field(default=1e-6, description="Epsilon for division")
    norm_lower: float = Field(
        default=0.5, description="Lower percentile for normalization, or global min."
    )
    norm_upper: float = Field(
        default=99.9, description="Upper percentile for normalization, or global max."
    )
    normalize: Union[Literal["percentile", "global"], bool] = Field(
        default="percentile",
        description="Normalization strategy: 'percentile', 'global', or False to disable.",
    )
    clip_norm: Union[bool, Tuple[float, float]] = Field(
        default=False,
        description="If True, clip normalized values to [0,1]; if (lo,hi), clip to that range.",
    )

    @classmethod
    def from_cli_args(
        cls,
        args: Any,
        *,
        shard_count: Optional[int] = None,
        shard_index: Optional[int] = None,
    ) -> "InferenceConfig":
        """
        Build an InferenceConfig from an argparse.Namespace (or compatible object).

        Parameters
        ----------
        args : Any
            Parsed CLI arguments providing configuration values.
        shard_count : Optional[int]
            Override for shard_count, useful for launchers that manage sharding.
        shard_index : Optional[int]
            Override for shard_index, useful for launchers that manage sharding.
        """

        def _tuple_value(name: str) -> Tuple[int, int, int]:
            value = getattr(args, name, None)
            if value is None:
                default = cls.model_fields[name].default
                if default is None:
                    raise ValueError(f"Missing required argument '{name}'.")
                value = default
            return tuple(value)

        def _with_default(name: str) -> Any:
            value = getattr(args, name, None)
            if value is not None:
                return value
            field = cls.model_fields[name]
            default_factory = getattr(field, "default_factory", None)
            if default_factory is not None:
                return default_factory()
            return field.default

        clip_norm_arg = getattr(args, "clip_norm", None)
        if clip_norm_arg is None:
            clip_norm: Union[bool, Tuple[float, float]] = False
        elif len(clip_norm_arg) == 0:
            clip_norm = True
        else:
            clip_norm = tuple(clip_norm_arg)  # type: ignore[arg-type]

        normalize_arg = getattr(args, "normalize", "percentile")
        normalize = False if normalize_arg == "false" else normalize_arg

        return cls(
            patch=_tuple_value("patch"),
            overlap=getattr(args, "overlap", cls.model_fields["overlap"].default),
            block=_tuple_value("block"),
            batch_size=getattr(args, "batch", cls.model_fields["batch_size"].default),
            t_idx=getattr(args, "t", cls.model_fields["t_idx"].default),
            c_idx=getattr(args, "c", cls.model_fields["c_idx"].default),
            devices=_with_default("devices"),
            amp=not getattr(args, "no_amp", False),
            use_tf32=not getattr(args, "no_tf32", False),
            use_compile=getattr(args, "compile", cls.model_fields["use_compile"].default),
            compile_mode=getattr(args, "compile_mode", cls.model_fields["compile_mode"].default),
            compile_dynamic=not getattr(args, "no_compile_dynamic", False),
            max_inflight_batches=getattr(
                args,
                "max_inflight_batches",
                cls.model_fields["max_inflight_batches"].default,
            ),
            seam_mode=getattr(args, "seam_mode", cls.model_fields["seam_mode"].default),
            trim_voxels=getattr(args, "trim_voxels", cls.model_fields["trim_voxels"].default),
            halo=getattr(args, "halo", cls.model_fields["halo"].default),
            min_blend_weight=getattr(
                args,
                "min_blend_weight",
                cls.model_fields["min_blend_weight"].default,
            ),
            eps=getattr(args, "eps", cls.model_fields["eps"].default),
            norm_lower=getattr(args, "norm_lower", cls.model_fields["norm_lower"].default),
            norm_upper=getattr(args, "norm_upper", cls.model_fields["norm_upper"].default),
            normalize=normalize,
            clip_norm=clip_norm,
            shard_count=(
                shard_count
                if shard_count is not None
                else getattr(args, "shard_count", cls.model_fields["shard_count"].default)
            ),
            shard_index=(
                shard_index
                if shard_index is not None
                else getattr(args, "shard_index", cls.model_fields["shard_index"].default)
            ),
            shard_strategy=getattr(
                args,
                "shard_strategy",
                cls.model_fields["shard_strategy"].default,
            ),
        )

    @model_validator(mode="after")
    def _validate(self):
        """
        Cross-field validation & sanity checks.
        """
        # Basic geometry
        if len(self.patch) != 3:
            raise ValueError("patch must have exactly 3 dimensions (Z,Y,X)")
        if any(d <= 0 for d in self.patch):
            raise ValueError(f"patch dims must be >0, got {self.patch}")
        if len(self.block) != 3:
            raise ValueError("block must have exactly 3 dimensions (Z,Y,X)")
        if any(d <= 0 for d in self.block):
            raise ValueError(f"block dims must be >0, got {self.block}")

        if self.overlap < 0:
            raise ValueError("overlap must be >= 0")
        min_patch = min(self.patch)
        if self.overlap >= min_patch:
            raise ValueError(
                f"overlap ({self.overlap}) must be < smallest patch dim ({min_patch})"
            )

        # Block >= patch
        if any(b < p for b, p in zip(self.block, self.patch)):
            raise ValueError(
                f"All block dims {self.block} must be >= patch dims {self.patch}"
            )

        # Stride validity
        stride = tuple(p - self.overlap for p in self.patch)
        if any(s <= 0 for s in stride):
            raise ValueError(
                f"Stride (patch - overlap) must be >0 in each dim; patch={self.patch} overlap={self.overlap} -> stride={stride}"
            )

        # Seam handling
        if self.seam_mode not in {"trim", "blend"}:
            raise ValueError("seam_mode must be 'trim' or 'blend'")

        if self.seam_mode == "trim":
            if self.trim_voxels is None:
                raise ValueError("trim_voxels must be set when seam_mode='trim'")
            if self.trim_voxels < 0:
                raise ValueError("trim_voxels must be >= 0")
            # Required overlap already validated below, but compute now
            required = 2 * int(self.trim_voxels)
            if self.overlap < required:
                raise ValueError(
                    f"overlap ({self.overlap}) must be >= 2 * trim_voxels ({self.trim_voxels}) == {required} when seam_mode='trim'"
                )
            if required >= min_patch:
                raise ValueError(
                    f"2 * trim_voxels ({required}) must be < smallest patch dim ({min_patch})"
                )
        else:  # blend mode
            if not (0 < self.min_blend_weight <= 1):
                raise ValueError(
                    f"min_blend_weight ({self.min_blend_weight}) must be in (0,1] for blend mode"
                )
            if self.overlap == 0:
                raise ValueError("blend seam_mode requires overlap > 0")
            if self.trim_voxels is not None:
                warnings.warn(
                    "trim_voxels specified but seam_mode='blend'; value will be ignored.",
                    RuntimeWarning,
                )

        # Halo: infer a safe default and validate
        min_patch = min(self.patch)
        if self.halo is None:
            if self.seam_mode == "trim":
                # Ensure the core never touches untrimmed patch borders at block edges
                self.halo = int(self.trim_voxels or 0)
            else:  # blend mode
                # Provide a small positive default to reduce block-edge artifacts
                # Heuristic: cap at 8, scale with patch size, and ensure >=1
                default_blend_halo = max(1, min(8, max(1, min_patch // 8)))
                self.halo = int(default_blend_halo)
        # Validate and enforce relationships
        if self.halo < 0:
            raise ValueError(f"halo ({self.halo}) must be >= 0")
        if self.seam_mode == "trim":
            # If a user-set halo is smaller than trim_voxels, bump to be safe.
            required_halo = int(self.trim_voxels or 0)
            if self.halo < required_halo:
                warnings.warn(
                    (
                        "halo < trim_voxels; increasing halo to trim_voxels to avoid "
                        "untrimmed edges at block boundaries"
                    ),
                    RuntimeWarning,
                )
                self.halo = required_halo
        if self.halo == 0:
            warnings.warn(
                "halo=0 may cause block-edge artifacts for models with non-trivial receptive fields",
                RuntimeWarning,
            )

        # Normalization
        if self.normalize == "percentile":
            if not (
                0.0 <= self.norm_lower
                and self.norm_lower < self.norm_upper
                and self.norm_upper <= 100.0
            ):
                raise ValueError(
                    "For 'percentile' normalization, percentiles must be in [0, 100] with lower <= upper."
                )
        elif self.normalize == "global":
            if self.norm_lower >= self.norm_upper:
                raise ValueError(
                    "For 'global' normalization, norm_lower must be < norm_upper."
                )

        # clip_norm validation
        if isinstance(self.clip_norm, tuple):
            if len(self.clip_norm) != 2:
                raise ValueError("clip_norm tuple must be (low, high)")
            lo, hi = self.clip_norm
            if not (lo < hi):
                raise ValueError("clip_norm lower bound must be < upper bound")

        # Numerical params
        if self.eps <= 0:
            raise ValueError("eps must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.max_inflight_batches <= 0:
            raise ValueError("max_inflight_batches must be > 0")

        # Devices
        if not self.devices:
            raise ValueError("devices list must not be empty")

        # Sharding
        if self.shard_count <= 0:
            raise ValueError("shard_count must be > 0")
        if not (0 <= self.shard_index < self.shard_count):
            raise ValueError(
                f"shard_index ({self.shard_index}) must be in [0, shard_count ({self.shard_count}))"
            )

        return self
