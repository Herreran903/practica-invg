# JSSP Image-Based Solver Selection Training

This module provides a complete pipeline for training CNN models to select the best solver for Job Shop Scheduling Problem (JSSP) instances based on grayscale image representations (128x128x1).

## Overview

The module supports two main training tasks via the CLI:
- **Classification**: Select the single best solver for each instance
- **Multilabel**: Identify all viable solvers (runtime < time limit)

Internally, there are evaluation utilities that are also reused by other modules
(e.g., tensors), but the **JSSP image pipeline** itself only supports
**classification** and **multilabel**, not regression.

## Module Structure

```
src/training/jssp_images/
├── config.yaml           # Configuration file with all parameters
├── config_loader.py      # Configuration loading and validation
├── data_utils.py         # Data loading and preprocessing utilities
├── model_builder.py      # CNN architecture construction
├── training_loop.py      # K-Fold cross-validation logic
├── evaluation.py         # Metrics computation
├── visualization.py      # Result plotting functions
├── cli.py               # Command-line interface
├── __init__.py          # Module exports
└── README.md            # This file
```

## Features

- **Modular Design**: Each component has a single, well-defined responsibility
- **Configurable Architecture**: CNN layers, filters, dropout rates via YAML
- **K-Fold Cross-Validation**: Stratified for classification, standard for others
- **Baseline Comparison**: BSS (Baseline Single Solver) computed per fold
- **Comprehensive Metrics**: Task-specific evaluation (accuracy, F1, MAE)
- **Rich Visualizations**: Confusion matrices, PR curves, scatter plots
- **Reproducibility**: Fixed seeds for deterministic results
- **Professional CLI**: Clear help text, examples, validation

## Quick Start

### Basic Usage

```bash
# Classification task
python -m src.training.jssp_images.cli \
  --csv data/jssp/datasets/jssp_cnn_data_images/ground_truth_jsp_generated_dataset.csv \
  --task classification \
  --epochs 30 \
  --folds 5

# Multilabel task
python -m src.training.jssp_images.cli \
  --csv data/jssp/datasets/jssp_cnn_data_images/ground_truth_jsp_generated_dataset.csv \
  --task multilabel \
  --epochs 25

```

### With Custom Configuration

```bash
python -m src.training.jssp_images.cli \
  --config src/training/jssp_images/config.yaml \
  --csv data/jssp/datasets/jssp_cnn_data_images/ground_truth_jsp_generated_dataset.csv \
  --task classification \
  --epochs 50 \
  --batch_size 32 \
  --learning_rate 0.0001
```

### Filtering specific solvers with --solvers

The CSV [`ground_truth_jsp_generated_dataset.csv`](data/jssp/datasets/jssp_cnn_data_images/ground_truth_jsp_generated_dataset.csv) contains
runtime columns for the following solvers:

- `CBC_DEF_Runtime_s`
- `SCIP_DEF_Runtime_s`
- `HIGHS_DEF_Runtime_s`
- `CPLEX_DEF_Runtime_s`
- `GUROBI_DEF_Runtime_s`

Their corresponding `--solvers` names are:

- `CBC_DEF`
- `SCIP_DEF`
- `HIGHS_DEF`
- `CPLEX_DEF`
- `GUROBI_DEF`

#### Example: classification with a subset of solvers

Train a classifier using only `CBC_DEF`, `SCIP_DEF` and `CPLEX_DEF`:

```bash
python -m src.training.jssp_images.cli \
  --csv data/jssp/datasets/jssp_cnn_data_images/ground_truth_jsp_generated_dataset.csv \
  --task classification \
  --solvers CBC_DEF,SCIP_DEF,CPLEX_DEF \
  --epochs 30 \
  --folds 5
```

#### Example: multilabel with all available solvers

Train a multilabel model using all five solvers from the CSV:

```bash
python -m src.training.jssp_images.cli \
  --csv data/jssp/datasets/jssp_cnn_data_images/ground_truth_jsp_generated_dataset.csv \
  --task multilabel \
  --solvers CBC_DEF,SCIP_DEF,HIGHS_DEF,CPLEX_DEF,GUROBI_DEF \
  --epochs 25
```

If `--solvers` is omitted, the CLI will use all solver columns detected in the CSV.

## Configuration

The `config.yaml` file contains all training parameters organized into sections:

### Experiment Configuration
```yaml
experiment:
  name: "jssp_images_cnn"
  description: "CNN for JSSP solver selection from grayscale images"
```

### Data Configuration
```yaml
data:
  target_height: 128
  target_width: 128
  time_limit_s: 60.0
```

### Model Architecture
```yaml
model:
  # CNN that replicates the architecture from the paper (shared with SAT images)
  conv_filters: [32, 64, 128]
  conv_kernel_size: [3, 2, 2]   # per-block kernels: 3x3, 2x2, 2x2
  pool_size: 2
  dropout_conv: [0.1, 0.2, 0.3] # per-block dropout rates
  dense_units: [1000, 200]      # two dense layers: 1000 and 200 units
  dropout_dense: 0.5
```

### Training Parameters (paper configuration)
```yaml
training:
  # Number of epochs
  epochs: 25

  # Batch size (paper setting)
  batch_size: 128

  # SGD with Nesterov momentum (paper defaults)
  learning_rate: 0.03
  momentum: 0.9

  # Linear schedule applied at the beginning of each epoch
  #   lr(t+1)        = max(0, lr(t) - lr_step)
  #   momentum(t+1)  = min(0.99, momentum(t) + momentum_step)
  lr_step: 0.003       # Per-epoch decrement of learning rate
  momentum_step: 0.001 # Per-epoch increment of momentum

  # Cross-validation setup
  k_folds: 5
  early_stopping_patience: 6
  seed: 42
```

### Output Configuration
```yaml
output:
  parent_dir: "results/jssp/images"
  run_name: "jssp_images_cnn"
  append_task_to_dirname: true
  timestamp_format: "%Y%m%d_%H%M%S"
```

## CLI Arguments

### Required Arguments
- `--csv`: Path to CSV file with `Image_Npy_Path` and solver performance columns
- `--task`: Task type (`classification` or `multilabel`). Regression is supported via the Python API but is not exposed through the CLI.

### Optional Arguments
- `--config`: Path to configuration YAML (default: `src/training/jssp_images/config.yaml`)
- `--use_score`: Use `*_Score_S_rel` columns instead of `*_Runtime_s` for "best" solver
- `--epochs`: Number of training epochs (overrides config)
- `--batch_size`: Batch size (overrides config)
- `--folds`: Number of K-Fold splits (overrides config)
- `--learning_rate`: Learning rate (overrides config)
- `--out_parent`: Parent directory for output
- `--run_name`: Name prefix for this run
- `--seed`: Random seed for reproducibility

## Input Data Format

The CSV file must contain:
- `Image_Npy_Path`: Path to `.npy` file with image data (128x128 or will be resized)
- `*_Runtime_s`: Runtime columns for each solver (e.g., `CP_Runtime_s`, `LNS_Runtime_s`)
- `*_Score_S_rel`: (Optional) Score columns for each solver

Example CSV structure:
```csv
Instance_Name,Image_Npy_Path,CP_Runtime_s,LNS_Runtime_s,Gecode_Runtime_s
inst_001,data/images/inst_001.npy,12.5,8.3,15.2
inst_002,data/images/inst_002.npy,45.1,32.7,28.9
```

## Output Structure

Results are saved in timestamped directories:

```
results/jssp/images/jssp_images_cnn_classification_20241124_151730/
├── config.yaml                  # Configuration used
├── run_info.json               # Run metadata
├── metrics_per_fold.csv        # Detailed per-fold metrics
├── metrics_summary.json        # Aggregated statistics (per task)
├── accuracy_per_fold.png       # Cross-fold comparison
├── README.txt                  # Human-readable summary
└── fold_1/                     # Individual fold results
    ├── fold1_metrics.json
    ├── fold1_y_true.npy
    ├── fold1_y_pred.npy
    ├── fold1_confusion.png
    ├── fold1_class_bars.png
    └── fold1_cls_report.csv
```

For multilabel (and classification when `resolved_rate` is computed), `metrics_summary.json`
additionally reports aggregated resolved-rate statistics across folds:

- `resolved_rate_mean`, `resolved_rate_std`
- `resolved_rate_min`, `resolved_rate_max`

## Task-Specific Outputs

### Classification
- Confusion matrix heatmap
- Per-class accuracy bars
- Classification report (precision, recall, F1)
- Overall accuracy and macro F1

### Multilabel
- Precision-Recall curves per label
- F1-score bars per label
- Average Precision per label
- Micro and macro F1 scores


## Using as a Python Module

```python
from src.training.jssp_images import (
    load_config,
    detect_solver_cols,
    run_kfold,
)
import pandas as pd

# Load configuration
config = load_config("src/training/jssp_images/config.yaml")

# Load data
df = pd.read_csv("data/jssp/datasets/jssp_cnn_data_images/ground_truth_jsp_generated_dataset.csv")

# Detect solver columns
solver_cols = detect_solver_cols(df)

# Run K-Fold training
fold_results, summary = run_kfold(
    df=df,
    task="classification",
    solver_cols=solver_cols,
    use_score=False,
    config=config,
    root_outdir="output/my_experiment",
)

print(f"Mean accuracy: {summary['mean']:.4f} ± {summary['std']:.4f}")
```

## Architecture and Training Details (paper configuration)

For the JSSP image-based solver selection experiments described in the paper, the
training module uses a **fixed CNN + SGD configuration**:

1. **Input**
   - 128×128×1 grayscale images
   - Each image is **standardized per instance** to zero mean and unit standard
     deviation inside [`make_dataset`](src/training/jssp_images/data_utils.py:179).

2. **Convolutional blocks**
   - 3 consecutive blocks, each of the form:
     - `Conv2D(filters, kernel_size=3, padding="same", activation="relu")`
     - `MaxPooling2D(pool_size=2)`
     - `Dropout(rate=0.25)`
   - Default filters per block:
     - Block 1: 16 filters
     - Block 2: 32 filters
     - Block 3: 64 filters

3. **Dense part**
   - `Flatten`
   - `Dense(256, activation="relu")`
   - `Dropout(rate=0.5)`
   - `Dense(256, activation="relu")`

4. **Output layer**
   - **Multilabel (paper experiments):** `Dense(C, activation="sigmoid")`
   - Classification and regression variants remain available in
     [`build_cnn`](src/training/jssp_images/model_builder.py:18) for debugging/ablation:
     - Classification: `softmax` output with cross-entropy loss
     - Regression: `linear` output with MAE loss

5. **Optimizer and schedule**
   - Optimizer: `SGD` with **Nesterov momentum**
     - Initial learning rate: **0.03**
     - Initial momentum: **0.9**
   - Per-epoch linear schedule (implemented via
     [`SGDLearningRateMomentumScheduler`](src/training/jssp_images/training_loop.py:25)):
     - `learning_rate ← max(0, learning_rate − 0.003)`
     - `momentum     ← min(0.99, momentum + 0.001)`

6. **Batch size and epochs**
   - Batch size: **128**
   - Epochs: 25 (with early stopping on `val_loss`)

This configuration is the **default** when using the provided
[`config.yaml`](src/training/jssp_images/config.yaml:1) and corresponds exactly
to the CNN and training strategy described in the paper.

## Baseline Comparison

Each fold computes a Baseline Single Solver (BSS) metric:
- **BSS Strategy**: Select the solver with best average performance on training set
- **BSS Metric**: Apply this single solver to all validation instances
- **Comparison**: CNN performance vs BSS baseline

This provides context for model effectiveness beyond random selection.

## Reproducibility

The module ensures reproducibility through:
- Fixed random seeds (Python, NumPy, TensorFlow)
- Deterministic TensorFlow operations
- Stratified K-Fold for classification (maintains class balance)
- Saved configuration and run metadata

## Dependencies

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- PyYAML

## Troubleshooting

### "No columns *_Runtime_s found"
Ensure your CSV has solver performance columns ending with `_Runtime_s`.

### "No valid images found"
Check that `Image_Npy_Path` contains valid paths to `.npy` files. Paths can be relative to project root or absolute.

### "Minority class has fewer samples than folds"
The module automatically adjusts the number of folds if a class has too few samples. Consider using fewer folds or balancing your dataset.

### Memory issues
Reduce `batch_size` in config or via CLI argument.

## Related Modules

- **jssp_tensors**: Training on padded 3D tensor representations (max_jobs × max_machines × n_channels)
- **sat_images**: Training on SAT problem images
- **data_generation/jssp_images**: Generate training data from JSSP instances

## References

This module implements the training pipeline for the JSSP solver selection approach described in the project documentation. For data generation, see `src/data_generation/jssp_images/`.

## License

Part of the JSSP solver selection project.