# JSSP Image-Based Solver Selection Training

This module provides a complete pipeline for training CNN models to select the best solver for Job Shop Scheduling Problem (JSSP) instances based on grayscale image representations (128x128x1).

## Overview

The module supports three training tasks:
- **Classification**: Select the single best solver for each instance
- **Multilabel**: Identify all viable solvers (runtime < time limit)
- **Regression**: Predict runtime for each solver

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

# Regression task
python -m src.training.jssp_images.cli \
  --csv data/jssp/datasets/jsp_cnn_data_gen/ground_truth_jsp_generated_dataset.csv \
  --task regression \
  --use_score
```

### With Custom Configuration

```bash
python -m src.training.jssp_images.cli \
  --config src/training/jssp_images/config.yaml \
  --csv data/jssp/datasets/jsp_cnn_data_gen/ground_truth_jsp_generated_dataset.csv \
  --task classification \
  --epochs 50 \
  --batch_size 32 \
  --learning_rate 0.0001
```

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
  conv_filters: [16, 32, 64]
  conv_kernel_size: 3
  pool_size: 2
  dropout_conv: 0.25
  dense_units: 256
  dropout_dense: 0.5
```

### Training Parameters
```yaml
training:
  epochs: 25
  batch_size: 64
  learning_rate: 0.001
  k_folds: 5
  early_stopping_patience: 6
  seed: 42
```

### Output Configuration
```yaml
output:
  parent_dir: "training/jssp/results"
  run_name: "jssp_images_cnn"
  append_task_to_dirname: true
  timestamp_format: "%Y%m%d_%H%M%S"
```

## CLI Arguments

### Required Arguments
- `--csv`: Path to CSV file with `Image_Npy_Path` and solver performance columns
- `--task`: Task type (`classification`, `multilabel`, or `regression`)

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
training/jssp/results/jssp_images_cnn_classification_20241124_151730/
├── config.yaml                  # Configuration used
├── run_info.json               # Run metadata
├── metrics_per_fold.csv        # Detailed per-fold metrics
├── metrics_summary.json        # Aggregated statistics
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

### Regression
- Scatter plots (predicted vs actual) per solver
- MAE per solver
- Overall MAE across all solvers

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
df = pd.read_csv("data/jssp/datasets/jsp_cnn_data_gen/ground_truth_jsp_generated_dataset.csv")

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

## Architecture Details

The CNN architecture consists of:
1. **Input**: 128x128x1 grayscale images
2. **Convolutional Blocks**: 3 blocks of Conv2D + MaxPooling2D
   - Block 1: 16 filters, 3x3 kernel
   - Block 2: 32 filters, 3x3 kernel
   - Block 3: 64 filters, 3x3 kernel
3. **Dropout**: 0.25 after convolutions
4. **Flatten**: Convert to 1D
5. **Dense**: 256 units with ReLU
6. **Dropout**: 0.5 after dense
7. **Output**: Task-specific activation
   - Classification: Softmax
   - Multilabel: Sigmoid
   - Regression: Linear

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

- **jssp_tensors**: Training on 3D tensor representations (10x10x2)
- **sat_images**: Training on SAT problem images
- **data_generation/jssp_images**: Generate training data from JSSP instances

## References

This module implements the training pipeline for the JSSP solver selection approach described in the project documentation. For data generation, see `src/data_generation/jssp_images/`.

## License

Part of the JSSP solver selection project.