from pathlib import Path

import numpy as np
import pytest
import torch

from inference_s2 import build_s2_model, load_checkpoint, prepare_s2_tensor


def test_prepare_s2_tensor_shape_and_band_validation() -> None:
    tensor = prepare_s2_tensor(np.ones((3, 5, 7), dtype=np.float32), 3, img_size=8)
    assert tensor.shape == (1, 1, 3, 8, 8)
    with pytest.raises(ValueError, match="Expected 4 S2 bands"):
        prepare_s2_tensor(np.ones((3, 5, 7), dtype=np.float32), 4)


def test_load_wrapped_checkpoint(tmp_path: Path) -> None:
    model = build_s2_model(3, (4, 8))
    checkpoint = tmp_path / "model.pth"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    loaded = load_checkpoint(build_s2_model(3, (4, 8)), checkpoint, "cpu")
    assert loaded(torch.randn(1, 1, 3, 8, 8)).shape == (1, 2, 8, 8)
