from pathlib import Path

import pytest
import torch

from data.dataset import FloodDetectionDataset


def _write_split(path: Path, sample: dict) -> None:
    split_dir = path / "train"
    split_dir.mkdir(parents=True)
    torch.save([sample], split_dir / "train_processed.pt")


@pytest.mark.parametrize("time_steps", [None, 2])
def test_dataset_contract_for_spatial_and_temporal_inputs(
    tmp_path: Path, time_steps: int | None
) -> None:
    prefix = () if time_steps is None else (time_steps,)
    sample = {
        "s1_data": torch.ones(*prefix, 2, 8, 9),
        "s2_data": torch.ones(*prefix, 3, 8, 9),
        "mask": torch.zeros(8, 9),
    }
    _write_split(tmp_path, sample)
    dataset = FloodDetectionDataset(
        tmp_path,
        config={"model": {"s1_channels": 2, "s2_channels": 3}},
    )
    output = dataset[0]
    expected_time = 1 if time_steps is None else time_steps
    assert output["s1_data"].shape == (expected_time, 2, 8, 9)
    assert output["s2_data"].shape == (expected_time, 3, 8, 9)
    assert output["image"].shape == (expected_time, 5, 8, 9)
    assert output["mask"].shape == (8, 9)


def test_load_to_memory_does_not_mutate_cache(tmp_path: Path) -> None:
    _write_split(
        tmp_path,
        {
            "s1_data": torch.ones(2, 4, 4),
            "s2_data": torch.ones(3, 4, 4),
            "mask": torch.zeros(4, 4),
        },
    )
    dataset = FloodDetectionDataset(
        tmp_path,
        load_to_memory=True,
        config={"model": {"s1_channels": 2, "s2_channels": 3}},
    )
    first = dataset[0]
    first["s1_data"].zero_()
    assert torch.all(dataset[0]["s1_data"] == 1)


def test_dataset_rejects_mismatched_time(tmp_path: Path) -> None:
    _write_split(
        tmp_path,
        {
            "s1_data": torch.ones(2, 2, 4, 4),
            "s2_data": torch.ones(1, 3, 4, 4),
            "mask": torch.zeros(4, 4),
        },
    )
    dataset = FloodDetectionDataset(
        tmp_path, config={"model": {"s1_channels": 2, "s2_channels": 3}}
    )
    with pytest.raises(ValueError, match="temporal dimensions"):
        dataset[0]
