import warnings
from typing import List, Literal, Optional, Tuple, Union

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

    # Seam handling
    # 'blend' = edge-aware Hann/Tukey blending inside blocks
    # 'trim'  = crop overlapped margins (keeps constant contribution)
    seam_mode: str = Field(default="trim", description="Seam handling mode")
    trim_voxels: Optional[int] = Field(
        default=5, description="Voxels to trim for seam handling"
    )
    halo: Optional[int] = Field(
        default=None, description="Halo size"
    )  # if None, we’ll use trim_voxels
    min_blend_weight: float = Field(
        default=0.05, description="Minimum blend weight"
    )  # floor to avoid near-zero weights

    # Misc
    eps: float = Field(default=1e-6, description="Epsilon for division")
    norm_percentile_lower: float = Field(
        default=0.5, description="Lower percentile for normalization, or global min."
    )
    norm_percentile_upper: float = Field(
        default=99.9, description="Upper percentile for normalization, or global max."
    )
    normalization_strategy: Union[Literal["percentile", "global"], bool] = Field(
        default="percentile",
        description="Normalization strategy: 'percentile', 'global', or False to disable.",
    )
    clip_norm: Union[bool, Tuple[float, float]] = Field(
        default=False,
        description="If True, clip normalized values to [0,1]; if (lo,hi), clip to that range.",
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

        # Halo
        if self.halo is not None:
            if self.halo < 0:
                raise ValueError(f"halo ({self.halo}) must be >= 0")

        # Normalization
        if self.normalization_strategy == "percentile":
            if not (
                0.0 <= self.norm_percentile_lower
                and self.norm_percentile_lower < self.norm_percentile_upper
                and self.norm_percentile_upper <= 100.0
            ):
                raise ValueError(
                    "For 'percentile' normalization, percentiles must be in [0, 100] with lower <= upper."
                )
        elif self.normalization_strategy == "global":
            if self.norm_percentile_lower >= self.norm_percentile_upper:
                raise ValueError(
                    "For 'global' normalization, norm_percentile_lower must be < norm_percentile_upper."
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

        return self
