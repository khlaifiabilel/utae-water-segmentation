"""Dataset and data-loader helpers for processed flood detection samples."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FloodDetectionDataset(Dataset[dict[str, Any]]):
    """Load processed S1/S2 samples and expose a temporal model input."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        transform: Callable[..., Mapping[str, Any]] | None = None,
        config_path: str | Path = "config/training_config.yaml",
        load_to_memory: bool = False,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.load_to_memory = load_to_memory
        if config is None:
            with Path(config_path).open(encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file)
        self.config = dict(config)
        self.s1_channels = int(self.config["model"]["s1_channels"])
        self.s2_channels = int(self.config["model"]["s2_channels"])

        split_file = self.data_dir / split / f"{split}_processed.pt"
        if not split_file.exists():
            raise FileNotFoundError(f"Processed data file not found: {split_file}")
        self.samples = torch.load(split_file, map_location="cpu", weights_only=False)
        if not isinstance(self.samples, (list, tuple)):
            raise TypeError(f"Expected a sequence of samples in {split_file}")

        self.memory_samples: list[dict[str, Any]] | None = None
        if load_to_memory:
            logger.info("Loading %d samples to memory", len(self.samples))
            self.memory_samples = [
                self._prepare_sample(sample) for sample in tqdm(self.samples)
            ]

    @staticmethod
    def _modality_tensor(value: Any, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError(
                f"{name} must have shape [C,H,W] or [T,C,H,W], got {tuple(tensor.shape)}"
            )
        return tensor.clone()

    def _prepare_sample(self, raw_sample: Mapping[str, Any]) -> dict[str, Any]:
        if "s1_data" not in raw_sample or "s2_data" not in raw_sample:
            raise KeyError("Each sample must contain s1_data and s2_data")
        if "mask" not in raw_sample:
            raise KeyError("Each sample must contain mask")

        s1_data = self._modality_tensor(raw_sample["s1_data"], "s1_data")
        s2_data = self._modality_tensor(raw_sample["s2_data"], "s2_data")
        if s1_data.shape[1] != self.s1_channels:
            raise ValueError(
                f"Expected {self.s1_channels} S1 channels, got {s1_data.shape[1]}"
            )
        if s2_data.shape[1] != self.s2_channels:
            raise ValueError(
                f"Expected {self.s2_channels} S2 channels, got {s2_data.shape[1]}"
            )
        if s1_data.shape[0] != s2_data.shape[0]:
            raise ValueError("S1 and S2 temporal dimensions must match")
        if s1_data.shape[-2:] != s2_data.shape[-2:]:
            raise ValueError("S1 and S2 spatial dimensions must match")

        mask = torch.as_tensor(raw_sample["mask"], dtype=torch.long)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        if mask.ndim != 2:
            raise ValueError(f"mask must have shape [H,W], got {tuple(mask.shape)}")
        if mask.shape != s1_data.shape[-2:]:
            raise ValueError("Mask and modality spatial dimensions must match")

        return {
            "s1_data": s1_data,
            "s2_data": s2_data,
            "image": torch.cat((s1_data, s2_data), dim=1),
            "mask": mask.clone(),
            "timestamp": torch.as_tensor(raw_sample.get("timestamp", 0)).long(),
            "location": raw_sample.get("location", "unknown"),
        }

    @staticmethod
    def _clone_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in sample.items()
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.memory_samples is None:
            sample = self._prepare_sample(self.samples[idx])
        else:
            sample = self._clone_sample(self.memory_samples[idx])
        if self.transform is not None:
            sample = self._apply_transforms(sample)
        return sample

    def _apply_transforms(self, sample: dict[str, Any]) -> dict[str, Any]:
        time_steps, _, height, width = sample["image"].shape
        combined = sample["image"].permute(2, 3, 0, 1).reshape(height, width, -1)
        transformed = self.transform(
            image=combined.numpy(), mask=sample["mask"].numpy()
        )
        image = torch.as_tensor(np.asarray(transformed["image"]))
        if image.ndim != 3:
            raise ValueError(
                "Transform must return image with shape [H,W,C] or [C,H,W]"
            )
        total_channels = time_steps * (self.s1_channels + self.s2_channels)
        if image.shape[-1] == total_channels:
            image = image.permute(2, 0, 1)
        elif image.shape[0] != total_channels:
            raise ValueError("Transform changed the image channel count")
        image = image.reshape(
            time_steps,
            self.s1_channels + self.s2_channels,
            image.shape[-2],
            image.shape[-1],
        ).float()
        sample["image"] = image
        sample["s1_data"] = image[:, : self.s1_channels].clone()
        sample["s2_data"] = image[:, self.s1_channels :].clone()
        sample["mask"] = torch.as_tensor(np.asarray(transformed["mask"])).long()
        if sample["mask"].ndim == 3 and sample["mask"].shape[0] == 1:
            sample["mask"] = sample["mask"].squeeze(0)
        if sample["mask"].shape != image.shape[-2:]:
            raise ValueError("Transform returned mismatched image and mask dimensions")
        return sample

    def get_class_distribution(self) -> dict[int, int]:
        class_counts: dict[int, int] = {}
        for index in tqdm(range(len(self)), desc="Class distribution"):
            unique, counts = torch.unique(self[index]["mask"], return_counts=True)
            for class_id, count in zip(unique.tolist(), counts.tolist(), strict=True):
                class_counts[class_id] = class_counts.get(class_id, 0) + count
        return class_counts


def get_data_transforms(config: Mapping[str, Any], split: str) -> Any:
    """Build Albumentations transforms, importing the optional module on demand."""
    try:
        import albumentations as A
    except ImportError as exc:
        raise ImportError(
            "Albumentations is required when dataset transforms are enabled"
        ) from exc

    if split == "train":
        augmentation = config.get("data", {}).get("augmentation", {})
        return A.Compose(
            [
                A.HorizontalFlip(p=augmentation.get("horizontal_flip", 0.5)),
                A.VerticalFlip(p=augmentation.get("vertical_flip", 0.5)),
                A.RandomRotate90(p=augmentation.get("rotate_90", 0.5)),
            ]
        )
    return A.Compose([])


def create_data_loaders(
    config: Mapping[str, Any],
    data_dir: str | Path,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test loaders from processed split files."""
    batch_size = config["data"]["batch_size"] if batch_size is None else batch_size
    num_workers = config["data"]["num_workers"] if num_workers is None else num_workers
    load_to_memory = bool(config["data"].get("load_to_memory", False))
    datasets = [
        FloodDetectionDataset(
            data_dir,
            split=split,
            transform=get_data_transforms(config, split),
            load_to_memory=load_to_memory,
            config=config,
        )
        for split in ("train", "validation", "test")
    ]
    loaders = [
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=index == 0,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for index, dataset in enumerate(datasets)
    ]
    return loaders[0], loaders[1], loaders[2]
