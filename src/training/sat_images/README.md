# SAT Image-Based Solver Selection Training

Train CNN models on grayscale image representations of SAT problem instances for solver selection, with SAT-specific metrics.

## Overview

This module extends jssp_images with SAT-specific features:
- **resolved_rate**: Percentage of instances successfully resolved
- **AST (Average Solving Time)**: Mean time including feature extraction
- **K-Fold repetitions**: Support for 5x5 cross-validation
- **Status columns**: Handles *_Status columns (OK/SAT/UNSAT/TIMEOUT)
- **Solver filtering**: Optional selection of specific solvers

## Quick Start

```bash
# Basic classification over ALL solvers present in the CSV (no --solvers filter)
python -m src.training.sat_images.cli \
  --csv data/sat/datasets/sat_cnn_data_images/ground_truth_aslib.csv \
  --task classification \
  --epochs 30

# 5x5 cross-validation (all solvers from CSV)
python -m src.training.sat_images.cli \
  --csv data/sat/datasets/sat_cnn_data_images/ground_truth_aslib.csv \
  --task classification \
  --folds 5 \
  --repeats 5 \
  --solvers ebglucose,glucose2,lingeling,clasp1,clasp2 \
  --time_limit 1200

# Filter specific solvers (names MUST match the 'algorithm' values in algorithm_runs.arff)
# Example for the SAT12-INDU scenario: ebglucose, glucose2, lingeling, clasp1, clasp2, ...
python -m src.training.sat_images.cli \
  --csv data/sat/datasets/sat_cnn_data_images/ground_truth_aslib.csv \
  --task multilabel
```

## Configuration

See [`config.yaml`](config.yaml:1) for all parameters:

```yaml
data:
  target_height: 128
  target_width: 128
  time_limit_s: 1200.0
  feat_time_column: null  # Feature extraction time

training:
  k_folds: 5
  k_fold_repeats: 1  # Set to 5 for 5x5

sat:
  valid_statuses: ["ok", "sat", "unsat"]
  invalid_statuses: ["timeout", "memout", "crash"]
```

## Architecture and training details (paper configuration)

The SAT image CNN uses the **paper-specific architecture and optimizer
configuration** described in Loreggia et al. (2016). The training loop is
shared with JSSP, but SAT overrides the architecture hyperparameters via
[`config.yaml`](src/training/sat_images/config.yaml:25) to match the paper.

1. **Input**
   - 128×128×1 grayscale images loaded from `.npy` files.
   - Each image is **standardized per instance to zero mean and unit variance**
     inside [`make_dataset`](src/training/jssp_images/data_utils.py:179).

2. **Convolutional blocks**
   Implemented in [`build_cnn`](src/training/jssp_images/model_builder.py:18):
 
   - 3 consecutive blocks, each of the form:
     - `Conv2D(filters, kernel_size=k_i, padding="same", activation="relu")`
     - `MaxPooling2D(pool_size=2)`
     - `Dropout(rate)` with a **block-specific** rate.
   - Paper-aligned filters, kernel sizes and dropout rates per block for SAT:
     - Block 1: 32 filters, kernel 3×3, `Dropout(0.1)`
     - Block 2: 64 filters, kernel 2×2, `Dropout(0.2)`
     - Block 3: 128 filters, kernel 2×2, `Dropout(0.3)`
   - These values are configured in:
     - `model.conv_filters: [32, 64, 128]`
     - `model.conv_kernel_size: [3, 2, 2]`
     - `model.dropout_conv: [0.1, 0.2, 0.3]`
     inside [`sat_images/config.yaml`](src/training/sat_images/config.yaml:25).

3. **Dense part**
  - `Flatten`
  - `Dense(1000, activation="relu")`
  - `Dropout(rate=0.5)`
  - `Dense(200, activation="relu")`

  These sizes correspond to:
  - `model.dense_units: [1000, 200]` in
    [`sat_images/config.yaml`](src/training/sat_images/config.yaml:25).

4. **Output layer**
   - For the **multilabel experiments in the paper** (recommended configuration):
     - `Dense(C, activation="sigmoid")` for C solvers, i.e., an independent
       binary output (capable / not capable) per solver.
   - Classification is also available in
     [`build_cnn`](src/training/jssp_images/model_builder.py:18) for additional
     experiments (softmax output).
     Regression **is not supported** in the SAT image pipeline.

5. **Optimizer and schedule**
  - Optimizer: `SGD` with **Nesterov momentum**, configured in
    [`config.yaml`](src/training/sat_images/config.yaml:25):
    - Initial learning rate: **0.03**
    - Initial momentum: **0.9**
  - Linear per-epoch schedule (implemented once in the shared training loop
    via [`SGDLearningRateMomentumScheduler`](src/training/jssp_images/training_loop.py:25),
    which SAT reuses):
    - `learning_rate ← max(0, learning_rate − 0.003)`
    - `momentum     ← min(0.99, momentum + 0.001)`

6. **Batch size and epochs**
  - Batch size: **128** (paper setting; see
    [`training.batch_size`](src/training/sat_images/config.yaml:45)).
  - Epochs: 25 (with EarlyStopping on `val_loss` and patience 6).

This configuration is the **default** when using the provided
[`config.yaml`](src/training/sat_images/config.yaml:1) and it replicates
the CNN, optimizer, activations, dropout, and training strategy described
in the paper for the SAT domain.

## SAT-Specific Metrics

### Resolved Rate
Percentage of instances where the predicted solver successfully resolved the instance (status OK/SAT/UNSAT or runtime < time_limit).

### AST (Average Solving Time)
Mean time = feature_extraction_time + solver_runtime

If solver fails or exceeds time limit, uses time_limit as penalty.

## Input Data Format

CSV must contain:
- `Image_Npy_Path`: Path to `.npy` image files (128x128)
- `*_Runtime_s`: Runtime columns per solver
- `*_Status`: (Optional) Status columns (OK/SAT/UNSAT/TIMEOUT/etc.)
- Feature time column (optional, configurable)

## Output Structure

```
training/sat/results/sat_images_cnn_classification_20241124_151730/
├── config.yaml
├── run_info.json
├── metrics_summary_GLOBAL.json      # Global stats across all folds × repeats
├── metrics_summary_per_rep.csv      # Per-repetition aggregated metrics
├── rep_1/
│   ├── metrics_per_fold.csv         # Per-fold metrics (incl. resolved_rate per fold)
│   ├── metrics_summary.json         # Aggregated stats for this repetition
│   ├── accuracy_per_fold.png
│   └── fold_1/
│       ├── fold1_metrics.json
│       ├── fold1_resolved_detail.csv  # SAT-specific per-instance resolution info
│       ├── fold1_y_true.npy
│       ├── fold1_y_pred.npy
│       └── fold1_confusion.png
└── rep_2/
    └── ...
```

For **classification** and **multilabel** SAT runs, both per-repetition and global
summaries aggregate not only the primary metric (`accuracy` or `f1_micro`) but also
the resolved-rate across folds:

- In each `rep_k/metrics_summary.json`:
  - `resolved_rate_mean`, `resolved_rate_std`
  - `resolved_rate_min`, `resolved_rate_max`
- In `metrics_summary_GLOBAL.json` (across all folds × repeats):
  - `resolved_rate_mean`, `resolved_rate_std`
  - `resolved_rate_min`, `resolved_rate_max`

## CLI Arguments

### SAT-Specific
- `--repeats`: K-Fold repetitions (e.g., 5 for 5x5)
- `--time_limit`: Time limit in seconds (overrides `data.time_limit_s` from [`config.yaml`](src/training/sat_images/config.yaml:9))
- `--feat_time_col`: Feature extraction time column name
- `--solvers`: Comma-separated solver names to filter

### Standard
- `--csv`: Path to CSV file
- `--task`: classification | multilabel
- `--epochs`, `--batch_size`, `--folds`, etc.

## Metrics by Task
 
### Classification
- Accuracy, F1-macro
- **resolved_rate**: % instances resolved
- **AST**: Average solving time
 
### Multilabel
- F1-micro, F1-macro, AP per label
- **resolved_rate**: % instances with ≥1 viable solver
- **AST**: Average time using best predicted solver

## Example: 5x5 Cross-Validation

```bash
python -m src.training.sat_images.cli \
  --csv data/sat/datasets/sat_cnn_data_images/ground_truth_aslib.csv \
  --task classification \
  --folds 5 \
  --repeats 5 \
  --time_limit 1800 \
  --feat_time_col Feature_Extract_s
```

This runs 25 folds total (5 folds × 5 repetitions) with different random seeds.

## Related Modules

- **jssp_images**: Training on JSSP images (base module)
- **jssp_tensors**: Training on JSSP tensors
- **data_generation/sat_images**: Generate SAT image data

## License

Part of the SAT solver selection project.
## Multilabel Training Examples

The following commands illustrate how to run the SAT images training CLI in **multilabel** mode.  
All examples use the same CLI entry point [`sat_images.cli`](src/training/sat_images/cli.py:125) and the default config [`config.yaml`](src/training/sat_images/config.yaml:1).

```bash
# Basic multilabel training over ALL solvers present in the CSV
python -m src.training.sat_images.cli \
  --csv data/sat/datasets/sat_cnn_data_images/ground_truth_aslib.csv \
  --task multilabel \
  --epochs 30 \
  --time_limit 1200
```

```bash
# 5x5 cross-validation in multilabel mode (all solvers from CSV)
python -m src.training.sat_images.cli \
  --csv data/sat/datasets/sat_cnn_data_images/ground_truth_aslib.csv \
  --task multilabel \
  --folds 5 \
  --repeats 5 \
  --time_limit 1200
```

```bash
# Multilabel training with an explicit subset of solvers
# IMPORTANT: Solver names MUST match the 'algorithm' values in algorithm_runs.arff
# Example (consistent with SAT12-INDU algorithms):
#   ebglucose, ebminisat, glucose2, glueminisat, lingeling, lrglshr,
#   minisatpsm, mphaseSAT64, precosat, qutersat, rcl, restartsat,
#   cryptominisat2011, spear-sw, spear-hw, eagleup, sparrow, marchrw,
#   mphaseSATm, satime11, tnm, mxc09, gnoveltyp2, sattime, sattimep,
#   clasp2, clasp1, picosat, mphaseSAT, sapperlot, sol

python -m src.training.sat_images.cli \
  --csv data/sat/datasets/sat_cnn_data_images/ground_truth_aslib.csv \
  --task multilabel \
  --time_limit 1800 \
  --solvers ebglucose,glucose2,lingeling,clasp1,clasp2
```