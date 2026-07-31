"""
Utility functions for data handling
Author: Bilel Khlaifi
Date: 2025-06-11
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def calculate_class_weights(
    masks: list[np.ndarray], num_classes: int = 2, ignore_index: int | None = -1
) -> torch.Tensor:
    """
    Calculate class weights for imbalanced dataset

    Args:
        masks: List of mask arrays
        num_classes: Number of classes

    Returns:
        Class weights tensor
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    class_counts = np.zeros(num_classes, dtype=np.int64)

    for mask in masks:
        labels = np.asarray(mask).reshape(-1)
        valid = np.isfinite(labels) & (labels >= 0) & (labels < num_classes)
        valid &= labels == np.floor(labels)
        if ignore_index is not None:
            valid &= labels != ignore_index
        class_counts += np.bincount(
            labels[valid].astype(np.int64), minlength=num_classes
        )

    present = class_counts > 0
    if not present.any():
        raise ValueError("masks contain no valid class labels")
    class_weights = np.zeros(num_classes, dtype=np.float64)
    inverse_frequency = 1.0 / class_counts[present]
    class_weights[present] = inverse_frequency / inverse_frequency.mean()

    return torch.tensor(class_weights, dtype=torch.float32)


def create_train_val_split(
    data_list: list, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42
) -> tuple[list, list, list]:
    """
    Split data into train, validation, and test sets

    Args:
        data_list: List of data samples
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        seed: Random seed

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    if train_ratio < 0 or val_ratio < 0 or train_ratio + val_ratio > 1:
        raise ValueError("split ratios must be non-negative and sum to at most 1")
    indices = np.random.default_rng(seed).permutation(len(data_list))

    train_size = int(len(data_list) * train_ratio)
    val_size = int(len(data_list) * val_ratio)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    train_data = [data_list[i] for i in train_indices]
    val_data = [data_list[i] for i in val_indices]
    test_data = [data_list[i] for i in test_indices]

    return train_data, val_data, test_data


def plot_data_distribution(data_dir: Path, save_dir: Path | None = None):
    """
    Plot data distribution statistics

    Args:
        data_dir: Directory containing processed data
        save_dir: Directory to save plots
    """
    if save_dir is None:
        save_dir = data_dir / "plots"
    save_dir.mkdir(exist_ok=True)

    # Load statistics
    stats_file = data_dir / "dataset_statistics.yaml"
    if not stats_file.exists():
        print(f"Statistics file not found: {stats_file}")
        return

    import yaml

    with open(stats_file, "r") as f:
        stats = yaml.safe_load(f)

    # Create plots
    _fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # S1 statistics
    if "s1" in stats:
        s1_stats = stats["s1"]
        axes[0, 0].bar(s1_stats.keys(), s1_stats.values())
        axes[0, 0].set_title("Sentinel-1 Statistics")
        axes[0, 0].tick_params(axis="x", rotation=45)

    # S2 statistics
    if "s2" in stats:
        s2_stats = stats["s2"]
        axes[0, 1].bar(s2_stats.keys(), s2_stats.values())
        axes[0, 1].set_title("Sentinel-2 Statistics")
        axes[0, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(save_dir / "data_statistics.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Data distribution plots saved to: {save_dir}")


def verify_data_integrity(data_dir: Path) -> bool:
    """
    Verify the integrity of processed data

    Args:
        data_dir: Directory containing processed data

    Returns:
        True if data is valid, False otherwise
    """
    required_files = ["dataset_statistics.yaml", "data_analysis.yaml"]
    required_dirs = ["train", "validation", "test"]

    # Check required files
    for file_name in required_files:
        file_path = data_dir / file_name
        if not file_path.exists():
            print(f"Missing required file: {file_path}")
            return False

    # Check data directories
    for dir_name in required_dirs:
        dir_path = data_dir / dir_name
        if not dir_path.exists():
            print(f"Missing data directory: {dir_path}")
            continue

        # Check if directory has processed data
        processed_file = dir_path / f"{dir_name}_processed.pt"
        if not processed_file.exists():
            print(f"Missing processed data file: {processed_file}")
            return False

    print("Data integrity check passed!")
    return True
