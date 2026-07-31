import math

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


def test_metrics_support_negative_ignore_index() -> None:
    outputs = torch.tensor([[[[5.0, 0.0]], [[0.0, 5.0]]]])
    targets = torch.tensor([[[0, -1]]])
    assert accuracy(outputs, targets, ignore_index=-1) == pytest.approx(1.0)
    class_iou, mean_iou = iou(outputs, targets, ignore_index=-1)
    assert class_iou[0] == pytest.approx(1.0)
    assert math.isnan(class_iou[1])
    assert mean_iou == pytest.approx(1.0)


def test_multiclass_metrics_match_manual_confusion_matrix() -> None:
    predicted = torch.tensor([[[0, 2, 2, 1]]])
    outputs = torch.nn.functional.one_hot(predicted, num_classes=3).permute(0, 3, 1, 2)
    targets = torch.tensor([[[0, 1, 2, 2]]])

    assert accuracy(outputs, targets) == pytest.approx(0.5)
    class_iou, mean_iou = iou(outputs, targets)
    class_f1, mean_f1 = f1_score(outputs, targets)
    assert class_iou == pytest.approx([1.0, 0.0, 1 / 3])
    assert mean_iou == pytest.approx(4 / 9)
    assert class_f1 == pytest.approx([1.0, 0.0, 0.5])
    assert mean_f1 == pytest.approx(0.5)


def test_evaluate_model_is_invariant_to_batching() -> None:
    samples = [
        {"image": torch.tensor([[[3.0]], [[0.0]]]), "mask": torch.tensor([[0]])},
        {"image": torch.tensor([[[0.0]], [[3.0]]]), "mask": torch.tensor([[1]])},
        {"image": torch.tensor([[[0.0]], [[3.0]]]), "mask": torch.tensor([[0]])},
    ]
    model = torch.nn.Identity()
    first = evaluate_model(model, DataLoader(samples, batch_size=1), "cpu")
    second = evaluate_model(model, DataLoader(samples, batch_size=2), "cpu")
    assert first == second


def test_metrics_handle_all_ignored_targets() -> None:
    outputs = torch.randn(1, 2, 2, 2)
    targets = torch.full((1, 2, 2), -1)
    assert accuracy(outputs, targets) == 0.0
    class_iou, mean_iou = iou(outputs, targets)
    class_f1, mean_f1 = f1_score(outputs, targets)
    assert all(math.isnan(value) for value in class_iou + class_f1)
    assert mean_iou == 0.0
    assert mean_f1 == 0.0


def test_evaluate_model_returns_only_requested_metrics() -> None:
    samples = [{"image": torch.tensor([[[3.0]], [[0.0]]]), "mask": torch.tensor([[0]])}]
    model = torch.nn.Identity()
    loader = DataLoader(samples, batch_size=1)
    assert set(evaluate_model(model, loader, "cpu", metrics=["iou"])) == {
        "iou",
        "mean_iou",
    }
    assert set(evaluate_model(model, loader, "cpu", metrics=["accuracy", "f1"])) == {
        "accuracy",
        "f1",
        "mean_f1",
    }
    assert evaluate_model(model, loader, "cpu", metrics=[]) == {}


def test_metrics_reject_invalid_inputs() -> None:
    outputs = torch.randn(1, 2, 2, 2)
    with pytest.raises(ValueError, match="spatial"):
        accuracy(outputs, torch.zeros(1, 3, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="outside"):
        accuracy(outputs, torch.full((1, 2, 2), 2))
    with pytest.raises(ValueError, match="Unsupported"):
        evaluate_model(
            torch.nn.Identity(), DataLoader([], batch_size=1), "cpu", metrics=["dice"]
        )


def test_evaluate_model_handles_empty_loader() -> None:
    model = torch.nn.Identity()
    results = evaluate_model(model, DataLoader([], batch_size=1), "cpu")
    assert results["accuracy"] == 0.0
