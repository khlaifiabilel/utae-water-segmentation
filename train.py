"""Training entry point for multimodal flood segmentation."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from data.dataset import create_data_loaders
from models.utae_water_segmentation import create_water_segmentation_model
from utils.losses import get_loss_function


def train_epoch(
    model: nn.Module,
    data_loader: Iterable[Mapping[str, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> float:
    """Run one optimization epoch and return sample-weighted mean loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in data_loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.shape[0]
        total_samples += images.shape[0]
    if total_samples == 0:
        raise ValueError("Training data loader is empty")
    return total_loss / total_samples


def validate_epoch(
    model: nn.Module,
    data_loader: Iterable[Mapping[str, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device | str,
) -> float:
    """Run one validation epoch and return sample-weighted mean loss."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            loss = criterion(model(images), masks)
            total_loss += loss.item() * images.shape[0]
            total_samples += images.shape[0]
    if total_samples == 0:
        raise ValueError("Validation data loader is empty")
    return total_loss / total_samples


def build_model(config: Mapping[str, Any]) -> nn.Module:
    model_config = config["model"]
    return create_water_segmentation_model(
        input_dim=int(model_config["s1_channels"]) + int(model_config["s2_channels"]),
        temporal_length=int(model_config.get("temporal_length", 1)),
        n_classes=int(model_config.get("n_classes", 2)),
        encoder_widths=model_config["encoder_widths"],
        out_conv=model_config.get("out_conv", (32, 32)),
        temporal_attention=model_config.get("temporal_attention"),
        spatial_attention=bool(model_config.get("spatial_attention", True)),
        attention_head_dims=model_config.get("attention_head_dims"),
        n_head=int(model_config.get("n_head", 8)),
    )


def train_model(config: Mapping[str, Any], device: str | None = None) -> Path:
    """Train from configuration and save the best validation checkpoint."""
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_loader, validation_loader, _ = create_data_loaders(
        config, config["paths"]["data_dir"]
    )
    model = build_model(config).to(selected_device)
    criterion = get_loss_function(config["loss"]).to(selected_device)
    training_config = config["training"]
    optimizer = Adam(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pth"
    best_validation_loss = float("inf")

    for epoch in tqdm(range(int(training_config["epochs"])), desc="Epochs"):
        training_loss = train_epoch(
            model, train_loader, criterion, optimizer, selected_device
        )
        validation_loss = validate_epoch(
            model, validation_loader, criterion, selected_device
        )
        print(
            f"Epoch {epoch + 1}: train_loss={training_loss:.6f} "
            f"validation_loss={validation_loss:.6f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": validation_loss,
                },
                checkpoint_path,
            )
    return checkpoint_path


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise TypeError("Training configuration must be a YAML mapping")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train U-TAE water segmentation")
    parser.add_argument(
        "--config", default="config/training_config.yaml", help="YAML config path"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()
    checkpoint = train_model(load_config(args.config), device=args.device)
    print(f"Best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
