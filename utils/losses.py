"""Segmentation losses and configuration factory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, ignore_index: int = -1) -> None:
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        classes = logits.shape[1]
        valid = targets != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        safe_targets = targets.masked_fill(~valid, 0)
        one_hot = F.one_hot(safe_targets.long(), classes).permute(0, 3, 1, 2)
        one_hot = one_hot.to(dtype=logits.dtype)
        valid_channels = valid.unsqueeze(1)
        probabilities = F.softmax(logits, dim=1) * valid_channels
        one_hot = one_hot * valid_channels
        dimensions = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dimensions)
        cardinality = (probabilities + one_hot).sum(dimensions)
        dice = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = -1
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = targets != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        safe_targets = targets.masked_fill(~valid, 0)
        cross_entropy = F.cross_entropy(logits, safe_targets, reduction="none")
        true_probability = torch.exp(-cross_entropy)
        loss = self.alpha * (1 - true_probability) ** self.gamma * cross_entropy
        return loss[valid].mean()


class MaskedCrossEntropyLoss(nn.CrossEntropyLoss):
    """Cross entropy that remains finite when every target is ignored."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if not (targets != self.ignore_index).any():
            return logits.sum() * 0.0
        return super().forward(logits, targets)


class ComboLoss(nn.Module):
    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: Sequence[float] | None = None,
        ignore_index: int = -1,
    ) -> None:
        super().__init__()
        weights = torch.tensor(class_weights) if class_weights is not None else None
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = MaskedCrossEntropyLoss(weight=weights, ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(
            logits, targets
        )


def get_loss_function(loss_config: Mapping[str, Any]) -> nn.Module:
    """Create a loss from the canonical ``loss.type`` configuration key."""
    loss_type = loss_config.get("type", "cross_entropy")
    class_weights = loss_config.get("class_weights")
    ignore_index = int(loss_config.get("ignore_index", -1))
    if loss_type in {"cross_entropy", "weighted_ce"}:
        weights = torch.tensor(class_weights) if class_weights is not None else None
        return MaskedCrossEntropyLoss(weight=weights, ignore_index=ignore_index)
    if loss_type == "dice":
        return DiceLoss(
            smooth=float(loss_config.get("smooth", 1.0)), ignore_index=ignore_index
        )
    if loss_type == "focal":
        return FocalLoss(
            alpha=float(loss_config.get("alpha", 0.25)),
            gamma=float(loss_config.get("gamma", 2.0)),
            ignore_index=ignore_index,
        )
    if loss_type == "combo":
        return ComboLoss(
            ce_weight=float(loss_config.get("ce_weight", 0.5)),
            dice_weight=float(loss_config.get("dice_weight", 0.5)),
            class_weights=class_weights,
            ignore_index=ignore_index,
        )
    raise ValueError(f"Unknown loss type: {loss_type}")
