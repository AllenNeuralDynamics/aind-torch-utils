"""Tests for the GPU GFP-masking post-processing model.

These require a CUDA GPU with cupy + cuCIM installed and are skipped otherwise
(e.g. on CPU-only CI or macOS dev machines).
"""
import json

import pytest

cp = pytest.importorskip("cupy")
pytest.importorskip("cucim")

import torch  # noqa: E402

from aind_torch_utils.postprocessing import (  # noqa: E402
    GfpMaskModel,
    create_gfp_mask_gpu,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)


def _synthetic_volume(shape=(32, 64, 64)):
    """Build a noisy background with a couple of bright blobs."""
    zz, yy, xx = cp.meshgrid(
        cp.arange(shape[0]),
        cp.arange(shape[1]),
        cp.arange(shape[2]),
        indexing="ij",
    )
    vol = cp.zeros(shape, dtype=cp.float32)
    for cz, cy, cx in [(8, 16, 16), (20, 45, 45)]:
        r2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        vol += cp.exp(-r2 / (2.0 * 5.0**2)).astype(cp.float32)
    vol = cp.clip(vol, 0, 1)
    return vol


def test_create_gfp_mask_gpu_basic():
    vol = _synthetic_volume()
    mask = create_gfp_mask_gpu(vol, intensity_percentiles=(0.0, 1.0))

    assert mask.dtype == cp.uint8
    assert mask.shape == vol.shape
    assert int(cp.unique(mask).max()) <= 1
    # Blob centers should be masked; far corner background should be empty.
    assert int(mask[8, 16, 16]) == 1
    assert float(mask.mean()) < 0.5  # mostly background


def test_gfp_mask_model_roundtrip():
    vol = _synthetic_volume()
    x = torch.as_tensor(vol, device="cuda")[None, None]  # (1, 1, Z, Y, X)

    model = GfpMaskModel(intensity_percentiles=(0.0, 1.0))
    out = model(x)

    assert out.shape == x.shape
    assert out.dtype == torch.uint8
    assert out.is_cuda
    assert out.device == x.device


def test_gfp_mask_model_batched():
    vol = _synthetic_volume()
    x = torch.as_tensor(vol, device="cuda")[None, None].repeat(3, 1, 1, 1, 1)

    out = GfpMaskModel(intensity_percentiles=(0.0, 1.0))(x)

    assert out.shape == x.shape
    # All batch items are identical inputs -> identical masks.
    assert torch.equal(out[0], out[1]) and torch.equal(out[1], out[2])


def test_forward_matches_per_volume_function():
    # Distinct volumes per batch item (shifted blobs) so batching can't hide a bug.
    vols = [_synthetic_volume((24, 48, 48)) for _ in range(3)]
    for i, v in enumerate(vols):
        vols[i] = cp.roll(v, shift=i * 3, axis=0)
    x = torch.stack([torch.as_tensor(v, device="cuda") for v in vols])[:, None]

    model = GfpMaskModel(intensity_percentiles=(0.0, 1.0))
    batched = model(x)

    # Batched forward must equal stacking the single-volume function per volume.
    for b in range(len(vols)):
        ref = create_gfp_mask_gpu(vols[b], intensity_percentiles=(0.0, 1.0))
        assert torch.equal(
            batched[b, 0].cpu(), torch.as_tensor(ref, device="cuda").cpu()
        )


def test_gfp_mask_model_from_json(tmp_path):
    params = {
        "intensity_percentiles": [0.0, 1.0],
        "background_sigma": [2, 8, 8],
        "low_thresh": 0.1,
        "high_thresh": 0.3,
        "min_object_size": 50,
    }
    path = tmp_path / "params.json"
    path.write_text(json.dumps(params))

    model = GfpMaskModel.from_json(str(path))

    # Sequence params parsed as tuples; scalars applied; missing keys defaulted.
    assert model.intensity_percentiles == (0.0, 1.0)
    assert model.background_sigma == (2, 8, 8)
    assert model.low_thresh == 0.1
    assert model.high_thresh == 0.3
    assert model.min_object_size == 50
    assert model.hole_size == 64  # default
    assert model.min_intensity is None  # default


def test_gfp_mask_model_from_json_rejects_unknown_key(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"not_a_param": 1}))

    with pytest.raises(TypeError):
        GfpMaskModel.from_json(str(path))
