from collections.abc import Iterable, Mapping, Sequence

import torch
from torch import nn


def _confusion_matrix(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int | None = None,
    ignore_index: int | None = -1,
) -> torch.Tensor:
    if outputs.ndim < 3:
        raise ValueError("outputs must have shape [B, C, ...]")
    if targets.shape != (outputs.shape[0], *outputs.shape[2:]):
        raise ValueError("targets must match the batch and spatial output dimensions")

    output_classes = outputs.shape[1]
    if num_classes is None:
        num_classes = output_classes
    if num_classes <= 0 or output_classes != num_classes:
        raise ValueError("num_classes must match the output channel count")

    predicted = outputs.argmax(dim=1)
    valid = torch.ones_like(targets, dtype=torch.bool)
    if ignore_index is not None:
        valid &= targets != ignore_index

    valid_targets = targets[valid]
    if valid_targets.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64)
    if torch.any((valid_targets < 0) | (valid_targets >= num_classes)):
        raise ValueError("targets contain a class outside the configured range")

    indices = valid_targets.to(torch.int64) * num_classes + predicted[valid]
    return (
        torch.bincount(indices, minlength=num_classes**2)
        .reshape(num_classes, num_classes)
        .cpu()
    )


def _scores(confusion: torch.Tensor) -> dict[str, float | list[float]]:
    confusion = confusion.to(torch.float64)
    true_positive = confusion.diag()
    target_count = confusion.sum(dim=1)
    predicted_count = confusion.sum(dim=0)
    total = confusion.sum()

    accuracy_value = (true_positive.sum() / total).item() if total else 0.0
    iou_denominator = target_count + predicted_count - true_positive
    f1_denominator = target_count + predicted_count
    class_iou = torch.where(
        iou_denominator > 0,
        true_positive / iou_denominator,
        torch.nan,
    )
    class_f1 = torch.where(
        f1_denominator > 0,
        2 * true_positive / f1_denominator,
        torch.nan,
    )

    def macro(values: torch.Tensor) -> float:
        present = values[~values.isnan()]
        return present.mean().item() if present.numel() else 0.0

    return {
        "accuracy": accuracy_value,
        "iou": class_iou.tolist(),
        "mean_iou": macro(class_iou),
        "f1": class_f1.tolist(),
        "mean_f1": macro(class_f1),
    }


def accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int | None = -1,
) -> float:
    """Return pixel accuracy over non-ignored targets, or 0.0 if none are valid."""
    confusion = _confusion_matrix(outputs, targets, ignore_index=ignore_index)
    return _scores(confusion)["accuracy"]


def iou(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int | None = None,
    ignore_index: int | None = -1,
) -> tuple[list[float], float]:
    """Return per-class and macro IoU; absent classes have a NaN score."""
    scores = _scores(_confusion_matrix(outputs, targets, num_classes, ignore_index))
    return scores["iou"], scores["mean_iou"]


def f1_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int | None = None,
    ignore_index: int | None = -1,
) -> tuple[list[float], float]:
    """Return per-class and macro F1; absent classes have a NaN score."""
    scores = _scores(_confusion_matrix(outputs, targets, num_classes, ignore_index))
    return scores["f1"], scores["mean_f1"]


def evaluate_model(
    model: nn.Module,
    data_loader: Iterable[Mapping[str, torch.Tensor]],
    device: torch.device | str,
    metrics: Sequence[str] | None = None,
    ignore_index: int | None = -1,
) -> dict[str, float | list[float]]:
    """Evaluate requested metrics from dataset-level confusion counts."""
    requested = ("accuracy", "iou", "f1") if metrics is None else tuple(metrics)
    unknown = set(requested) - {"accuracy", "iou", "f1"}
    if unknown:
        raise ValueError(f"Unsupported metrics: {', '.join(sorted(unknown))}")

    model.eval()
    confusion = None
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch["image"].to(device))
            batch_confusion = _confusion_matrix(
                outputs,
                batch["mask"].to(device),
                ignore_index=ignore_index,
            )
            confusion = (
                batch_confusion if confusion is None else confusion + batch_confusion
            )

    if confusion is None:
        all_scores: dict[str, float | list[float]] = {
            "accuracy": 0.0,
            "iou": [],
            "mean_iou": 0.0,
            "f1": [],
            "mean_f1": 0.0,
        }
    else:
        all_scores = _scores(confusion)

    results: dict[str, float | list[float]] = {}
    if "accuracy" in requested:
        results["accuracy"] = all_scores["accuracy"]
    if "iou" in requested:
        results["iou"] = all_scores["iou"]
        results["mean_iou"] = all_scores["mean_iou"]
    if "f1" in requested:
        results["f1"] = all_scores["f1"]
        results["mean_f1"] = all_scores["mean_f1"]
    return results
