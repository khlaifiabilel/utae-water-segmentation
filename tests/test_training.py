import torch
from torch.utils.data import DataLoader, Dataset

from models.utae_water_segmentation import UTAE_WaterSegmentation
from train import train_epoch, validate_epoch


class SyntheticDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(index)
        return {
            "image": torch.randn(1, 3, 8, 8, generator=generator),
            "mask": torch.randint(0, 2, (8, 8), generator=generator),
        }


def test_train_and_validate_epoch_smoke() -> None:
    loader = DataLoader(SyntheticDataset(), batch_size=2)
    model = UTAE_WaterSegmentation(
        3,
        encoder_widths=(4, 8),
        out_conv=(),
        temporal_attention=False,
        spatial_attention=False,
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    assert train_epoch(model, loader, criterion, optimizer, "cpu") > 0
    assert validate_epoch(model, loader, criterion, "cpu") > 0
