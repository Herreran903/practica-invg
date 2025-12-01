# JSSP Tensor-Based Solver Selection Training

Train CNN models on 3D tensor representations of Job Shop Scheduling Problem (JSSP) instances for solver selection.

## Overview

This module works with JSSP tensors derived from `.dzn` instances:

- Data generation (`src/data_generation/jssp_tensors/`) saves **2D matrices** of
  shape `(JOBS, MACHINES)` containing **processing times**.
- The training pipeline loads these matrices and pads them into **3D tensors**
  of shape `(max_jobs, max_machines, n_channels)` where:
  - **Channel 0**: Processing duration (from the 2D matrix)
  - **Additional channels**: Currently unused / zero-padded (reserved for future features)

By default, tensors are padded to `(10, 10, 2)` for consistent CNN input, as
configured in [`config.yaml`](config.yaml:10).

## Key Differences from jssp_images

- **Input**: Padded 3D tensors `(max_jobs × max_machines × n_channels)` instead of fixed 2D images `(128×128×1)`
- **Architecture**: CNN tuned for small spatial grids (default 10×10 with filters [32, 64, 128])
- **Data Loading**: Custom tensor loader with padding logic from variable `(JOBS, MACHINES)` to fixed shape

## Quick Start

```bash
# Classification task
python -m src.training.jssp_tensors.cli \
  --csv data/jssp/datasets/jssp_cnn_data_tensors/ground_truth_jsp_generated_dataset.csv \
  --task classification \
  --epochs 30

# Multilabel task
python -m src.training.jssp_tensors.cli \
  --csv data/jssp/datasets/jssp_cnn_data_tensors/ground_truth_jsp_generated_dataset.csv \
  --task multilabel \
  --epochs 30
```

The CLI supports **classification** and **multilabel** tasks. Regression-style
experiments on tensors can be implemented via the Python API if needed, but are
not exposed as a CLI task.

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
- `Image_Npy_Path`: Path to `.npy` tensor files (produced by `src/data_generation/jssp_tensors`)
- `*_Runtime_s`: Runtime columns per solver
- `*_Score_S_rel`: (Optional) Score columns

Tensor files should typically contain **2D arrays** of shape `(J, M)` (processing
times), where `J ≤ max_jobs` and `M ≤ max_machines`. The training loader
[`load_tensor_npy`](src/training/jssp_tensors/data_utils.py:28) accepts:

- 2D: `(J, M)` → interpreted as a single duration channel, placed in channel 0
- 3D: `(J, M, C_in)` → first `min(C_in, n_channels)` channels copied

In all cases, tensors are padded to `(max_jobs, max_machines, n_channels)` as
configured in [`config.yaml`](src/training/jssp_tensors/config.yaml:10).

## Architecture

```
Input: (max_jobs, max_machines, n_channels)  # default: (10, 10, 2)
├─ Conv2D(32) + MaxPool
├─ Conv2D(64) + MaxPool
├─ Conv2D(128) + MaxPool
├─ Flatten
├─ Dense(256) + Dropout(0.4)
└─ Output (task-specific: softmax for classification, sigmoid for multilabel)
```

## Output Structure

```
results/jssp/tensors/jssp_tensors_cnn_classification_20241124_151730/
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

The parent directory (`results/jssp/tensors`) and run naming follow
[`config.yaml`](src/training/jssp_tensors/config.yaml:58) and the logic in
[`prepare_output_directory()`](src/training/jssp_tensors/cli.py:76).

## Related Modules

- **jssp_images**: Training on 2D grayscale images (128x128x1)
- **sat_images**: Training on SAT problem images
- **data_generation/jssp_tensors**: Generate tensor data from JSSP instances

## License

Part of the JSSP solver selection project.