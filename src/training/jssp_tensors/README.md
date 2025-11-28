# JSSP Tensor-Based Solver Selection Training

Train CNN models on 3D tensor representations of Job Shop Scheduling Problem (JSSP) instances for solver selection.

## Overview

This module works with tensors of shape `(JOBS, MACHINES, 2)` where:
- **Channel 0**: Machine ID
- **Channel 1**: Processing duration

Tensors are padded to fixed size `(10, 10, 2)` for consistent CNN input.

## Key Differences from jssp_images

- **Input**: 3D tensors (10x10x2) instead of 2D images (128x128x1)
- **Architecture**: Adapted CNN with filters [32, 64, 128] for smaller spatial dimensions
- **Data Loading**: Custom tensor loader with padding logic

## Quick Start

```bash
# Classification
python -m src.training.jssp_tensors.cli \
  --csv data/jssp/datasets/jsp_cnn_data_gen_2/ground_truth.csv \
  --task classification \
  --epochs 30

# Regression
python -m src.training.jssp_tensors.cli \
  --csv data/jssp/datasets/jsp_cnn_data_gen_2/ground_truth.csv \
  --task regression \
  --folds 5
```

## Configuration

See [`config.yaml`](config.yaml:1) for all parameters:

```yaml
data:
  max_jobs: 10
  max_machines: 10
  n_channels: 2
  time_limit_s: 60.0

model:
  conv_filters: [32, 64, 128]
  dense_units: 256
  dropout_dense: 0.4
```

## Input Data Format

CSV must contain:
- `Image_Npy_Path`: Path to `.npy` tensor files
- `*_Runtime_s`: Runtime columns per solver
- `*_Score_S_rel`: (Optional) Score columns

Tensor files should contain arrays of shape `(J, M, 2)` where `J ≤ 10` and `M ≤ 10`.

## Architecture

```
Input: (10, 10, 2)
├─ Conv2D(32) + MaxPool
├─ Conv2D(64) + MaxPool
├─ Conv2D(128) + MaxPool
├─ Flatten
├─ Dense(256) + Dropout(0.4)
└─ Output (task-specific)
```

## Output Structure

```
training/jssp/results/jssp_tensors_cnn_classification_20241124_151730/
├── config.yaml
├── run_info.json
├── metrics_per_fold.csv
├── metrics_summary.json
├── accuracy_per_fold.png
└── fold_1/
    ├── fold1_metrics.json
    ├── fold1_y_true.npy
    ├── fold1_y_pred.npy
    └── fold1_confusion.png
```

## Related Modules

- **jssp_images**: Training on 2D grayscale images (128x128x1)
- **sat_images**: Training on SAT problem images
- **data_generation/jssp_tensors**: Generate tensor data from JSSP instances

## License

Part of the JSSP solver selection project.