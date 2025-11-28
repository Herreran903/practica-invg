# Solver Selection for SAT and JSSP using CNNs

This project implements CNN-based solver selection for SAT (Boolean Satisfiability) and JSSP (Job Shop Scheduling Problem) instances using image and tensor representations.

## 🎯 Overview

The project consists of two main pipelines:
1. **Data Generation**: Convert problem instances to image/tensor representations
2. **Training**: Train CNN models for solver selection

Both pipelines follow a professional, modular architecture with:
- ✅ Configuration via YAML files
- ✅ Command-line interfaces
- ✅ Comprehensive documentation
- ✅ Reusable Python modules

## 📋 Requirements

- Python 3.9+
- MiniZinc CLI (for JSSP data generation only)
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Key Dependencies
- TensorFlow 2.19+ / Keras 3.6+
- scikit-learn 1.6+
- PyYAML 6.0+
- pandas, numpy, matplotlib
- OR-Tools (for JSSP)

## 🏗️ Project Structure

```
├── src/
│   ├── data_generation/          # Modular data generation
│   │   ├── jssp_images/          # JSSP → grayscale images (128x128x1)
│   │   ├── jssp_tensors/         # JSSP → 3D tensors (10x10x2)
│   │   └── sat_images/           # SAT → grayscale images (128x128x1)
│   └── training/                 # Modular training pipelines
│       ├── jssp_images/          # Train on JSSP images
│       ├── jssp_tensors/         # Train on JSSP tensors
│       └── sat_images/           # Train on SAT images (with SAT-specific metrics)
├── data/                         # Generated datasets
│   ├── jssp/datasets/
│   └── sat/datasets/
├── training/                     # Training results
│   ├── jssp/results/
│   └── sat/results/
├── models/                       # MiniZinc models for JSSP
└── utils/                        # Utility scripts

```

## 🚀 Quick Start

### 1. Data Generation

#### JSSP Images (Grayscale 128x128x1)
```bash
# Academic dataset (JSPLIB instances)
python -m src.data_generation.jssp_images.cli \
  --config src/data_generation/jssp_images/config.yaml \
  --mode academic

# Generated dataset (random instances)
python -m src.data_generation.jssp_images.cli \
  --config src/data_generation/jssp_images/config.yaml \
  --mode generated
```

#### JSSP Tensors (3D 10x10x2)
```bash
python -m src.data_generation.jssp_tensors.cli \
  --config src/data_generation/jssp_tensors/config.yaml \
  --mode generated
```

#### SAT Images (Grayscale 128x128x1)
```bash
# Extract instances first
tar -xf data/sat/instances/sc2012-application.tar -C data/sat/instances/

# Generate dataset
python -m src.data_generation.sat_images.cli \
  --config src/data_generation/sat_images/config.yaml \
  --scenario_dir data/sat/aslib/sc2012-application \
  --instances_dir data/sat/instances/sc2012-application
```

### 2. Training

#### JSSP Images
```bash
# Classification (select best solver)
python -m src.training.jssp_images.cli \
  --csv data/jssp/datasets/jsp_cnn_data_gen/ground_truth.csv \
  --task classification \
  --epochs 30 \
  --folds 5

# Multilabel (identify viable solvers)
python -m src.training.jssp_images.cli \
  --csv data/jssp/datasets/jsp_cnn_data_gen/ground_truth.csv \
  --task multilabel \
  --epochs 25

# Regression (predict runtimes)
python -m src.training.jssp_images.cli \
  --csv data/jssp/datasets/jsp_cnn_data_gen/ground_truth.csv \
  --task regression \
  --epochs 30
```

#### JSSP Tensors
```bash
python -m src.training.jssp_tensors.cli \
  --csv data/jssp/datasets/jsp_cnn_data_gen_2/ground_truth.csv \
  --task classification \
  --epochs 30 \
  --folds 5
```

#### SAT Images (with SAT-specific metrics)
```bash
# Basic classification
python -m src.training.sat_images.cli \
  --csv data/sat/datasets/sat_cnn_data_gen_all/ground_truth.csv \
  --task classification \
  --epochs 30 \
  --folds 5

# 5x5 cross-validation with time limit
python -m src.training.sat_images.cli \
  --csv data/sat/datasets/sat_cnn_data_gen_all/ground_truth.csv \
  --task classification \
  --folds 5 \
  --repeats 5 \
  --time_limit 1800

# Filter specific solvers
python -m src.training.sat_images.cli \
  --csv data/sat/datasets/sat_cnn_data_gen_all/ground_truth.csv \
  --task multilabel \
  --solvers clasp,glucose,lingeling
```

## 📊 Supported Tasks

### Classification
Select the single best solver for each instance.
- **Metrics**: Accuracy, F1-macro
- **SAT only**: resolved_rate, AST

### Multilabel
Identify all viable solvers (runtime < time_limit).
- **Metrics**: F1-micro, F1-macro, Average Precision
- **SAT only**: resolved_rate, AST

### Regression
Predict runtime for each solver.
- **Metrics**: MAE (Mean Absolute Error)

## 🔧 Configuration

Each module has a `config.yaml` file with all parameters:

```yaml
# Example: src/training/jssp_images/config.yaml
data:
  target_height: 128
  target_width: 128
  time_limit_s: 60.0

model:
  conv_filters: [16, 32, 64]
  dense_units: 256
  dropout_dense: 0.5

training:
  epochs: 25
  batch_size: 64
  k_folds: 5
  learning_rate: 0.001
```

All parameters can be overridden via CLI arguments.

## 📈 SAT-Specific Features

The SAT training module includes additional metrics:

- **resolved_rate**: Percentage of instances successfully resolved
- **AST (Average Solving Time)**: Mean time including feature extraction
- **K-Fold repetitions**: Support for 5x5 cross-validation
- **Status handling**: Uses `*_Status` columns (OK/SAT/UNSAT/TIMEOUT)
- **Solver filtering**: Select specific solvers via `--solvers`

## 📚 Documentation

Each module has comprehensive documentation:
- [`src/data_generation/jssp_images/README.md`](src/data_generation/jssp_images/README.md)
- [`src/data_generation/jssp_tensors/README.md`](src/data_generation/jssp_tensors/README.md)
- [`src/data_generation/sat_images/README.md`](src/data_generation/sat_images/README.md)
- [`src/training/jssp_images/README.md`](src/training/jssp_images/README.md)
- [`src/training/jssp_tensors/README.md`](src/training/jssp_tensors/README.md)
- [`src/training/sat_images/README.md`](src/training/sat_images/README.md)

## 🛠️ Utilities

### Visualize .npy files
```bash
python utils/visualizador.py path/to/file.npy
```

### Analyze datasets
```bash
python utils/analiza_csv_dataset.py data/jssp/datasets/jsp_cnn_data_gen/ground_truth.csv
```

## 🔄 Complete Workflows

### JSSP End-to-End
```bash
# Generate data
python -m src.data_generation.jssp_images.cli --mode generated

# Train model
python -m src.training.jssp_images.cli \
  --csv data/jssp/datasets/jsp_cnn_data_gen/ground_truth.csv \
  --task classification \
  --epochs 30 \
  --folds 5
```

### SAT End-to-End
```bash
# Extract and generate data
tar -xf data/sat/instances/sc2012-application.tar -C data/sat/instances/
python -m src.data_generation.sat_images.cli \
  --scenario_dir data/sat/aslib/sc2012-application \
  --instances_dir data/sat/instances/sc2012-application

# Train with 5x5 cross-validation
python -m src.training.sat_images.cli \
  --csv data/sat/datasets/sat_cnn_data_gen_all/ground_truth.csv \
  --task classification \
  --folds 5 \
  --repeats 5 \
  --time_limit 1800
```

## ⚠️ Common Issues

### "CSV doesn't have 'Image_Npy_Path'"
Run data generation first to create the CSV with image paths.

### "No valid images found"
- Execute commands from project root
- Regenerate CSV if you moved files

### MiniZinc errors (JSSP only)
```bash
# Verify installation
minizinc --version
minizinc --solvers

# Ensure gecode or chuffed is available
```

### Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

## 📁 Output Structure

### Data Generation
```
data/[jssp|sat]/datasets/[dataset_name]/
├── ground_truth.csv          # Main dataset file
├── images/                   # .npy image files
│   ├── instance1.npy
│   └── instance2.npy
└── [instances]/              # Original instance files
```

### Training
```
training/[jssp|sat]/results/[run_name_timestamp]/
├── config.yaml               # Configuration used
├── run_info.json            # Run metadata
├── metrics_summary.json     # Aggregated results
├── metrics_per_fold.csv     # Per-fold details
├── [metric]_per_fold.png    # Visualization
└── fold_1/                  # Individual fold results
    ├── fold1_metrics.json
    ├── fold1_y_true.npy
    ├── fold1_y_pred.npy
    ├── fold1_confusion.png
    └── ...
```

## 🎓 Architecture

### JSSP Images CNN
- Input: 128x128x1 grayscale images
- Architecture: 3 Conv2D blocks (16→32→64 filters) + Dense(256)
- Tasks: Classification, Multilabel, Regression

### JSSP Tensors CNN
- Input: 10x10x2 tensors (padded from variable size)
- Architecture: 3 Conv2D blocks (32→64→128 filters) + Dense(256)
- Tasks: Classification, Multilabel, Regression

### SAT Images CNN
- Input: 128x128x1 grayscale images
- Architecture: Same as JSSP Images
- Tasks: Classification, Multilabel, Regression
- **Extra**: resolved_rate, AST metrics

## 📝 License

Part of the solver selection research project.

## 🤝 Contributing

This is a research project. For questions or issues, please refer to the module-specific READMEs.

## 📞 Support

- Check module READMEs for detailed documentation
- Review configuration files for available parameters
- Use `--help` flag on any CLI for usage information

```bash
python -m src.training.jssp_images.cli --help