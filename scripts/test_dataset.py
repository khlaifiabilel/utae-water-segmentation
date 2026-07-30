#!/usr/bin/env python3
"""
Script to test the dataset implementation
Usage: python scripts/test_dataset.py
"""

from pathlib import Path

import yaml

from data.dataset import FloodDetectionDataset, create_data_loaders


def main() -> int:
    """Test dataset functionality"""
    print("Testing dataset implementation...")

    # Load config
    config_path = Path("config/training_config.yaml")
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        return 1

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_dir = Path(config["paths"]["data_dir"])

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Please run preprocessing first: python scripts/preprocess_data.py")
        return 1

    try:
        # Test dataset creation
        print("Creating datasets...")
        train_dataset = FloodDetectionDataset(data_dir, split="train")
        print(f"Train dataset: {len(train_dataset)} samples")

        # Test data loading
        print("Creating data loaders...")
        train_loader, _val_loader, _test_loader = create_data_loaders(
            config, data_dir, batch_size=2
        )

        # Test one batch
        print("Testing batch loading...")
        for batch in train_loader:
            print("Successfully loaded batch")
            print("Batch contents:")
            for key, value in batch.items():
                if hasattr(value, "shape"):
                    print(f"  {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"  {key}: {type(value)}")
            break

        # Test class distribution
        print("Computing class distribution...")
        class_dist = train_dataset.get_class_distribution()
        print(f"Class distribution: {class_dist}")

        print("\nAll dataset tests passed")

    except Exception as error:  # noqa: BLE001 - command boundary reports failures
        print(f"Dataset test failed: {error}")
        import traceback

        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
