# U-TAE Water Segmentation

PyTorch code for binary water segmentation from temporal Sentinel-1 (S1) and
Sentinel-2 (S2) tensors. The model is a compact U-shaped encoder-decoder inspired
by U-TAE, with optional temporal attention at the deepest encoder feature. It is
not an implementation of PAPS or a knowledge-distillation pipeline.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

For development, install the test and lint dependencies instead:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

## Processed Data Contract

The configured directory has one serialized list per split:

```text
data/processed/
  train/train_processed.pt
  validation/validation_processed.pt
  test/test_processed.pt
```

Each list item contains `s1_data`, `s2_data`, and `mask`. A modality may be
`[C,H,W]` or `[T,C,H,W]`; both modalities must have equal temporal and spatial
dimensions and must match `model.s1_channels` and `model.s2_channels` in
`config/training_config.yaml`. The dataset returns canonical `s1_data` and
`s2_data` tensors plus `image` shaped `[T,C1+C2,H,W]` and `mask` shaped `[H,W]`.

The configured Hugging Face dataset may be gated or require authentication.
Downloading and preprocessing it is an explicit user action; tests do not access
the network or real data.

## Data Utilities

The default configuration names the gated
[`ibm-granite/granite-geospatial-uki-flooddetection`](https://huggingface.co/datasets/ibm-granite/granite-geospatial-uki-flooddetection)
dataset. The repository includes three data utilities:

```bash
python scripts/download-data.py
python scripts/preprocess_data.py --visualize
python scripts/test_dataset.py
```

`download-data.py` inspects the remote splits and saves metadata plus a small
set of samples under `data/raw/`. `preprocess_data.py` loads the configured
dataset, normalizes and resizes identified S1/S2/mask fields, and writes the
processed split files expected by training. `test_dataset.py` inspects those
processed files. These utilities require dataset access and were not exercised
by the offline synthetic test suite.

## Training

Review channel counts, widths, paths, and optimization settings, then run:

```bash
python train.py --config config/training_config.yaml
```

No trained checkpoints or published benchmark metrics are provided. The default
configuration is an example, not a claim of validated model quality.

See [`MODEL_CARD.md`](MODEL_CARD.md) for intended use, preprocessing
requirements, evaluation guidance, limitations, and checkpoint release
requirements.

## S2-Only Inference

`inference_s2.py` accepts a GeoTIFF and writes a class GeoTIFF, with optional PNG
and GeoJSON outputs. It requires a checkpoint trained with exactly the specified
S2-only channel count and architecture. A multimodal S1+S2 checkpoint is not
compatible.

```bash
python inference_s2.py --model s2_only.pth --input image.tif --output outputs \
  --s2-channels 13 --encoder-widths 64,128,256,512
```

## Verification

The synthetic CPU suite covers dataset shape validation and memory caching,
single- and multi-time-step model execution, backward propagation, training and
validation steps, ignored labels, metrics, checkpoint loading, and S2 tensor
preparation. CI runs the suite and code-quality checks on Python 3.10 and 3.12.
Run the same checks locally with:

```bash
python -m pytest -q
ruff check data/dataset.py models/utae_water_segmentation.py utils/losses.py \
  utils/metrics.py train.py inference_s2.py setup.py tests
ruff format --check data/dataset.py models/utae_water_segmentation.py \
  utils/losses.py utils/metrics.py train.py inference_s2.py setup.py tests
python -m compileall -q .
python -m pip check
```

## Attribution, License, and Security

The architecture is inspired by the U-TAE work and implementation by Vivien
Sainte Fare Garnot and collaborators; this repository is an independent,
simplified adaptation for water segmentation. See the
[official U-TAE/PaPs implementation](https://github.com/VSainteuf/utae-paps) and
the ICCV 2021 paper
[*Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks*](https://arxiv.org/abs/2107.07933)
for the original architecture and citation guidance.

This project is licensed under GPL-3.0; see `LICENSE`. Please report security
issues using `.github/SECURITY.md`, rather than opening a public vulnerability
issue.
