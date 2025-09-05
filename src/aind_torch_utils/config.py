from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


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
        default=0.5, description="Lower percentile for normalization"
    )
    norm_percentile_upper: float = Field(
        default=99.9, description="Upper percentile for normalization"
    )
