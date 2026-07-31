# U-TAE Water Segmentation Model Card

## Summary

This repository implements a compact U-shaped PyTorch model for binary water
segmentation from temporal Sentinel-1 and Sentinel-2 tensors. It is inspired by
U-TAE but is not the official U-TAE/PaPs implementation and does not implement
panoptic segmentation.

No trained weights, benchmark result, or production deployment is published
with this repository. This card documents the implementation and the conditions
required to evaluate a checkpoint; it is not evidence of model quality.

## Model Details

- Task: semantic segmentation of water and non-water pixels
- Framework: PyTorch
- Inputs: temporal tensors shaped `[B,T,C,H,W]`
- Modalities: Sentinel-1 SAR and Sentinel-2 multispectral imagery
- Output: per-pixel class logits shaped `[B,2,H,W]` by default
- Architecture: configurable encoder-decoder with optional temporal attention
- License: GPL-3.0 for this repository's original code

The channel counts, temporal length, encoder widths, attention settings, and
class count are configured in `config/training_config.yaml`. Checkpoint users
must reproduce the exact architecture and preprocessing configuration used for
training.

## Data

The default configuration references the gated
[`ibm-granite/granite-geospatial-uki-flooddetection`](https://huggingface.co/datasets/ibm-granite/granite-geospatial-uki-flooddetection)
dataset. Dataset access, licensing, geographic coverage, class definitions, and
usage terms are controlled by the dataset provider and are separate from this
repository's license.

The processed-data contract requires `s1_data`, `s2_data`, and `mask` tensors.
See the README for expected split files, dimensions, and channel validation.
The repository does not distribute source imagery or processed training data.

## Preprocessing

Training data must use a documented band order, scale, normalization method,
spatial resolution, temporal sampling policy, and nodata/cloud treatment.
These choices are part of the model definition and must accompany any released
checkpoint.

The current `inference_s2.py` path is S2-only and uses fixed per-band
normalization. It is compatible only with a checkpoint trained for the same
S2-only channel order, architecture, and normalization. It must not be used
with the default multimodal checkpoint configuration without an explicit,
validated conversion.

## Intended Use

The implementation is intended for:

- research and education in temporal Earth-observation segmentation;
- testing model, loss, metric, and geospatial inference code;
- controlled experiments with independently validated datasets and labels.

It is not validated for emergency response, flood warnings, navigation,
insurance, regulatory decisions, property-level risk assessment, or other
safety-critical or high-impact decisions.

## Evaluation

A checkpoint evaluation should report at least:

- the exact train/validation/test split and geographic separation policy;
- per-class IoU and F1 plus macro averages;
- pixel accuracy and class prevalence;
- performance by geography, season, sensor availability, and cloud condition;
- comparison against non-temporal and single-modality baselines;
- uncertainty or failure analysis around shorelines, shadows, snow, clouds,
  wetlands, and radar layover or speckle.

The metric implementation accumulates dataset-level confusion counts, supports
ignored labels, and excludes absent classes from macro averages. Results should
not be compared unless preprocessing, label definitions, and split methodology
are equivalent.

## Limitations And Risks

- No checkpoint or independently reproduced benchmark is currently available.
- The default dataset may be gated and may not represent other regions,
  climates, seasons, sensors, or flood types.
- Water labels can be ambiguous around mixed pixels, vegetation, shadows,
  coastlines, temporary inundation, and clouds.
- Resampling can move boundaries and alter small objects.
- Sentinel-1 and Sentinel-2 have different observation characteristics and
  acquisition schedules; missing or misregistered observations can degrade
  predictions.
- A visually plausible mask is not evidence of hydrological correctness.
- GeoTIFF nodata, CRS, transform, and band metadata must be validated before
  operational use.

## Reproducibility Requirements

Any future checkpoint release should include:

- a versioned configuration and preprocessing manifest;
- training code commit and environment information;
- dataset version and split identifiers;
- checkpoint checksum and architecture metadata;
- evaluation script, raw metric output, and qualitative examples;
- known failure cases and intended-use restrictions.

## Provenance

The architecture is an independent simplified adaptation inspired by the
U-TAE work by Vivien Sainte Fare Garnot and collaborators. Consult the
[`VSainteuf/utae-paps`](https://github.com/VSainteuf/utae-paps) repository and
the ICCV 2021 paper
[*Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks*](https://arxiv.org/abs/2107.07933)
for the original implementation and citation guidance.
