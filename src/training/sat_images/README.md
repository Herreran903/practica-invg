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
  --time_limit 1200

# Filter specific solvers (names MUST match the 'algorithm' values in algorithm_runs.arff)
# Example for the SAT12-INDU scenario: ebglucose, glucose2, lingeling, clasp1, clasp2, ...
python -m src.training.sat_images.cli \
  --csv data/sat/datasets/sat_cnn_data_images/ground_truth_aslib.csv \
  --task multilabel \
  --solvers ebglucose,glucose2,lingeling,clasp1,clasp2
```

## Configuration

See [`config.yaml`](config.yaml:1) for all parameters:

```yaml
data:
  target_height: 128
  target_width: 128
  time_limit_s: 1800.0
  feat_time_column: null  # Feature extraction time

training:
  k_folds: 5
  k_fold_repeats: 1  # Set to 5 for 5x5

sat:
  valid_statuses: ["ok", "sat", "unsat"]
  invalid_statuses: ["timeout", "memout", "crash"]
```

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
├── metrics_summary_GLOBAL.json
├── metrics_summary_per_rep.csv
├── rep_1/
│   ├── metrics_per_fold.csv
│   ├── metrics_summary.json
│   ├── accuracy_per_fold.png
│   └── fold_1/
│       ├── fold1_metrics.json
│       ├── fold1_resolved_detail.csv  # SAT-specific
│       ├── fold1_y_true.npy
│       ├── fold1_y_pred.npy
│       └── fold1_confusion.png
└── rep_2/
    └── ...
```

## CLI Arguments

### SAT-Specific
- `--repeats`: K-Fold repetitions (e.g., 5 for 5x5)
- `--time_limit`: Time limit in seconds (default: 1800)
- `--feat_time_col`: Feature extraction time column name
- `--solvers`: Comma-separated solver names to filter

### Standard
- `--csv`: Path to CSV file
- `--task`: classification | multilabel | regression
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

### Regression
- MAE (overall and per solver)
- No resolved_rate/AST

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
  --time_limit 1800
```

```bash
# 5x5 cross-validation in multilabel mode (all solvers from CSV)
python -m src.training.sat_images.cli \
  --csv data/sat/datasets/sat_cnn_data_images/ground_truth_aslib.csv \
  --task multilabel \
  --folds 5 \
  --repeats 5 \
  --time_limit 1800
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