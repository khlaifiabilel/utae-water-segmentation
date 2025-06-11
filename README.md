# UTAE Water Segmentation

**Multi-modal water/land segmentation using UTAE-PAPS architecture with Sentinel-1 and Sentinel-2 data**

## Overview

This project implements a two-phase approach for water detection:

1. **Phase 1**: Train UTAE-PAPS on multi-modal data (Sentinel-1 + Sentinel-2) using the IBM Granite Geospatial UKI Flood Detection Dataset
2. **Phase 2**: Knowledge distillation to create a Sentinel-2 only model for practical deployment

## Features

- 🛰️ Multi-modal satellite data processing (S1 + S2)
- 🌊 Binary water/land segmentation
- 🧠 Knowledge distillation for S2-only inference
- 📊 Comprehensive evaluation metrics
- 🚀 Easy deployment and inference

## Installation

```bash
git clone https://github.com/khlaifiabilel/utae-water-segmentation.git
cd utae-water-segmentation
pip install -r requirements.txt
pip install -e .
```
## Quick Start

Training Multi-modal Model
```bash
python scripts/train_multimodal.py --config config/training_config.yaml
```

Training S2-only Model (Knowledge Distillation)
```bash
python scripts/train_s2_only.py --teacher-model experiments/checkpoints/best_multimodal.pth
```
Inference
```bash
python scripts/predict.py --model experiments/checkpoints/s2_only_model.pth --input /path/to/sentinel2/image
```

## Dataset
This project uses the IBM Granite Geospatial UKI Flood Detection Dataset from Hugging Face.
 
## Model Architecture
Based on UTAE-PAPS (U-Temporal Attention Encoder with Parcels-as-Points) adapted for water segmentation tasks.

## Citation
BibTeX
@misc{utae-water-segmentation,
  title={UTAE Water Segmentation: Multi-modal Water Detection using Temporal Attention},
  author={Bilel Khlaifi},
  year={2025},
  url={https://github.com/khlaifiabilel/utae-water-segmentation}
}

## Acknowledgments
Original UTAE-PAPS implementation by VSainteuf
IBM Granite Geospatial team for the flood detection dataset
