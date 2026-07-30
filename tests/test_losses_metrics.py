import pytest
import torch
from torch.utils.data import DataLoader

from utils.losses import ComboLoss, DiceLoss, FocalLoss, get_loss_function
from utils.metrics import accuracy, evaluate_model, f1_score, iou


@pytest.mark.parametrize(
    "criterion", [DiceLoss(ignore_index=255), FocalLoss(ignore_index=255)]
)
def test_loss_ignores_configured_pixels(criterion: torch.nn.Module) -> None:
    targets = torch.tensor([[[0, 255], [1, 0]]])
    logits = torch.randn(1, 2, 2, 2)
    changed = logits.clone()
    changed[:, :, 0, 1] = torch.tensor([1000.0, -1000.0])
    assert torch.allclose(criterion(logits, targets), criterion(changed, targets))


def test_loss_factory_uses_type_key() -> None:
    assert isinstance(get_loss_function({"type": "dice"}), DiceLoss)


@pytest.mark.parametrize(
    "criterion",
    [
        get_loss_function({"type": "cross_entropy", "ignore_index": 255}),
        ComboLoss(ignore_index=255),
    ],
)
def test_losses_handle_all_ignored_targets(criterion: torch.nn.Module) -> None:
    logits = torch.randn(1, 2, 2, 2, requires_grad=True)
    loss = criterion(logits, torch.full((1, 2, 2), 255))
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None


def test_metrics_ignore_pixels() -> None:
    outputs = torch.tensor([[[[5.0, 0.0]], [[0.0, 5.0]]]])
    targets = torch.tensor([[[0, 255]]])
    assert accuracy(outputs, targets, ignore_index=255) == pytest.approx(1.0)
    class_iou, _ = iou(outputs, targets, ignore_index=255)
    class_f1, _ = f1_score(outputs, targets, ignore_index=255)
    assert class_iou[0] == pytest.approx(1.0)
    assert class_f1[0] == pytest.approx(1.0)


def test_evaluate_model_handles_empty_loader() -> None:
    model = torch.nn.Identity()
    results = evaluate_model(model, DataLoader([], batch_size=1), "cpu")
    assert results["accuracy"] == 0.0
