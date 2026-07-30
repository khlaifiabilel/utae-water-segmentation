"""Sentinel-2-only inference for checkpoints trained with S2 inputs only."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torch.nn import functional as F
from tqdm import tqdm

from models.utae_water_segmentation import create_water_segmentation_model

logger = logging.getLogger(__name__)

S2_MEANS = np.array(
    [
        1226.21,
        1137.38,
        1139.82,
        1350.49,
        1932.94,
        2211.89,
        2154.36,
        2163.57,
        2246.07,
        2036.50,
        1465.38,
        986.64,
        231.95,
    ],
    dtype=np.float32,
)
S2_STDS = np.array(
    [
        572.41,
        582.87,
        675.54,
        675.60,
        736.38,
        878.58,
        905.21,
        943.75,
        955.51,
        978.05,
        825.01,
        729.21,
        365.72,
    ],
    dtype=np.float32,
)


def normalize_s2(
    s2_data: np.ndarray,
    means: Sequence[float] | None = None,
    stds: Sequence[float] | None = None,
) -> np.ndarray:
    """Normalize a [C,H,W] S2 array without modifying the caller's array."""
    if s2_data.ndim != 3:
        raise ValueError(f"Expected S2 shape [C,H,W], got {s2_data.shape}")
    means_array = np.asarray(S2_MEANS if means is None else means, dtype=np.float32)
    stds_array = np.asarray(S2_STDS if stds is None else stds, dtype=np.float32)
    channels = s2_data.shape[0]
    if len(means_array) < channels or len(stds_array) < channels:
        raise ValueError(f"Normalization statistics do not cover {channels} bands")
    if np.any(stds_array[:channels] == 0):
        raise ValueError("Normalization standard deviations must be nonzero")
    return (
        s2_data.astype(np.float32, copy=True) - means_array[:channels, None, None]
    ) / stds_array[:channels, None, None]


def prepare_s2_tensor(
    s2_data: np.ndarray, expected_channels: int, img_size: int | None = None
) -> torch.Tensor:
    """Validate, normalize, and create a [1,1,C,H,W] model input."""
    if s2_data.ndim != 3 or s2_data.shape[0] != expected_channels:
        actual = s2_data.shape[0] if s2_data.ndim == 3 else "invalid"
        raise ValueError(f"Expected {expected_channels} S2 bands, got {actual}")
    tensor = torch.from_numpy(normalize_s2(s2_data)).unsqueeze(0)
    if img_size is not None:
        if img_size <= 0:
            raise ValueError("img_size must be positive")
        tensor = F.interpolate(
            tensor, size=(img_size, img_size), mode="bilinear", align_corners=False
        )
    return tensor.unsqueeze(1)


def load_checkpoint(
    model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device | str
) -> torch.nn.Module:
    """Load a plain state dict or a common wrapped checkpoint."""
    checkpoint: Any = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state dictionary")
    state_dict = checkpoint.get(
        "model_state_dict", checkpoint.get("state_dict", checkpoint)
    )
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint state dictionary is invalid")
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def predict_water(
    model: torch.nn.Module,
    s2_path: str | Path,
    output_dir: str | Path,
    img_size: int | None = None,
    device: torch.device | str = "cpu",
    export_geojson: bool = True,
    export_png: bool = True,
    s2_channels: int | None = None,
) -> dict[str, Path | None]:
    """Generate raster, optional GeoJSON, and optional PNG outputs for one image."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    base_name = Path(s2_path).stem
    prediction_path = output_dir / f"{base_name}_water_prediction.tif"
    geojson_path = output_dir / f"{base_name}_water.geojson"
    png_path = output_dir / f"{base_name}_visualization.png"

    with rasterio.open(s2_path) as source:
        s2_data = source.read().astype(np.float32)
        profile = source.profile.copy()
        original_size = (source.height, source.width)
    expected_channels = s2_channels or getattr(model, "input_dim", None)
    if expected_channels is None:
        raise ValueError("s2_channels is required when the model has no input_dim")
    model_input = prepare_s2_tensor(s2_data, expected_channels, img_size).to(device)

    model.eval()
    with torch.no_grad():
        prediction = model(model_input).argmax(dim=1, keepdim=True).float()
        if prediction.shape[-2:] != original_size:
            prediction = F.interpolate(prediction, size=original_size, mode="nearest")
    prediction_array = prediction[0, 0].byte().cpu().numpy()

    profile.update(dtype=rasterio.uint8, count=1, compress="lzw", nodata=None)
    with rasterio.open(prediction_path, "w", **profile) as destination:
        destination.write(prediction_array, 1)

    if export_geojson:
        from utils.vectorize import raster_to_geojson

        raster_to_geojson(prediction_array, s2_path, geojson_path)

    if export_png:
        with rasterio.open(s2_path) as source:
            if source.count < 4:
                raise ValueError(
                    "At least four bands are required for RGB visualization"
                )
            rgb = np.stack((source.read(4), source.read(3), source.read(2)), axis=-1)
        rgb = np.clip(rgb / 3000, 0, 1)
        color_map = plt.colormaps["winter"].resampled(2)
        figure, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(rgb)
        axes[0].set_title("Sentinel-2 image")
        axes[1].imshow(prediction_array, cmap=color_map, vmin=0, vmax=1)
        axes[1].set_title("Water segmentation")
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(figure)

    return {
        "prediction": prediction_path,
        "geojson": geojson_path if export_geojson else None,
        "visualization": png_path if export_png else None,
    }


def build_s2_model(s2_channels: int, encoder_widths: Sequence[int]) -> torch.nn.Module:
    """Build an S2-only model; multimodal checkpoints are not compatible."""
    return create_water_segmentation_model(
        input_dim=s2_channels,
        temporal_length=1,
        encoder_widths=encoder_widths,
        temporal_attention=False,
    )


def batch_process(
    model_path: str | Path,
    s2_dir: str | Path,
    output_dir: str | Path,
    pattern: str,
    img_size: int | None,
    s2_channels: int,
    encoder_widths: Sequence[int],
    export_geojson: bool,
    export_png: bool,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(
        build_s2_model(s2_channels, encoder_widths), model_path, device
    )
    paths = sorted(Path(s2_dir).glob(pattern))
    logger.info("Found %d images", len(paths))
    for path in tqdm(paths, desc="Processing images"):
        predict_water(
            model,
            path,
            output_dir,
            img_size=img_size,
            device=device,
            export_geojson=export_geojson,
            export_png=export_png,
            s2_channels=s2_channels,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run S2-only water segmentation")
    parser.add_argument("--model", required=True, help="S2-only checkpoint path")
    parser.add_argument("--input", required=True, help="S2 GeoTIFF or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--batch", action="store_true", help="Process an input directory"
    )
    parser.add_argument("--pattern", default="*.tif", help="Batch input glob")
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--s2-channels", type=int, default=13)
    parser.add_argument("--encoder-widths", default="64,128,256,512")
    parser.add_argument("--no-geojson", action="store_true")
    parser.add_argument("--no-png", action="store_true")
    args = parser.parse_args()
    widths = tuple(int(width) for width in args.encoder_widths.split(","))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.batch:
        batch_process(
            args.model,
            args.input,
            args.output,
            args.pattern,
            args.img_size,
            args.s2_channels,
            widths,
            not args.no_geojson,
            not args.no_png,
        )
        return
    model = load_checkpoint(
        build_s2_model(args.s2_channels, widths), args.model, device
    )
    outputs = predict_water(
        model,
        args.input,
        args.output,
        img_size=args.img_size,
        device=device,
        export_geojson=not args.no_geojson,
        export_png=not args.no_png,
        s2_channels=args.s2_channels,
    )
    logger.info("Prediction saved to %s", outputs["prediction"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
