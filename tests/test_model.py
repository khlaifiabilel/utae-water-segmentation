import pytest
import torch

from models.utae_water_segmentation import UTAE_WaterSegmentation


def test_model_t1_output_and_backward() -> None:
    model = UTAE_WaterSegmentation(
        input_dim=3,
        encoder_widths=(4, 7, 11),
        out_conv=(5,),
        temporal_attention=False,
        spatial_attention=False,
    )
    inputs = torch.randn(2, 1, 3, 16, 16, requires_grad=True)
    outputs = model(inputs)
    assert outputs.shape == (2, 2, 16, 16)
    outputs.mean().backward()
    assert inputs.grad is not None


def test_model_temporal_attention_t2() -> None:
    model = UTAE_WaterSegmentation(
        input_dim=3,
        encoder_widths=(4, 8),
        out_conv=(),
        temporal_attention=True,
        spatial_attention=False,
        n_head=2,
    )
    assert model(torch.randn(1, 2, 3, 12, 12)).shape == (1, 2, 12, 12)


def test_model_validates_input_contract() -> None:
    model = UTAE_WaterSegmentation(3, encoder_widths=(4, 8))
    with pytest.raises(ValueError, match=r"\[B,T,C,H,W\]"):
        model(torch.randn(1, 3, 8, 8))
    with pytest.raises(ValueError, match="3 input channels"):
        model(torch.randn(1, 1, 2, 8, 8))
