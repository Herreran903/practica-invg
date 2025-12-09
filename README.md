# Solver Selection for SAT and JSSP using CNNs

This repository implements convolutional neural network (CNN) models to select good solvers for two domains:

- **SAT** (Boolean Satisfiability)
- **JSSP** (Job Shop Scheduling Problem)

Problem instances are converted to **images** or **tensors**, then CNNs are trained to predict which solver (or set of solvers) is expected to solve an instance within a time limit.

---

## 1. What You Get

- **Data generation pipelines**
  - **JSSP images**: Text-to-Image encoding of `model.mzn + .dzn` (grayscale 128×128).
  - **JSSP tensors**: Processing-time matrices (Jobs × Machines) padded into fixed-size tensors.
  - **SAT images**: Raw-byte encoding of CNF/XCSP/DZN instances (grayscale 128×128).

- **Training pipelines**
  - **Classification**: pick a single best solver.
  - **Multilabel**: pick all solvers that solve within a time limit.
  - **Regression (SAT only)**: predict runtime per solver.

- **Utilities**
  - Dataset imbalance analysis.
  - Tensor/image inspection for sanity checks.

All components are configurable via YAML and exposed as both **CLIs** and **Python modules**.

---

## 2. Minimal Setup

1. **Python**
   - Python **3.9+** (recommended: use a virtual environment such as `.venv`).

2. **Create and activate a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # Linux/macOS
   # .venv\Scripts\activate     # Windows (PowerShell or cmd)
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **MiniZinc (JSSP data generation only)**
   - Install the MiniZinc IDE or CLI from the official website.
   - Ensure a CP solver (e.g. **gecode** or **chuffed**) is available:
     ```bash
     minizinc --version
     minizinc --solvers
     ```

---

## 3. Repository Layout (High Level)

```text
├── src/
│   ├── data_generation/          # Data generation pipelines
│   │   ├── jssp_images/          # JSSP → grayscale images (Text-to-Image)
│   │   ├── jssp_tensors/         # JSSP → processing-time matrices → tensors
│   │   └── sat_images/           # SAT → grayscale images (raw bytes)
│   └── training/                 # Training pipelines
│       ├── jssp_images/          # Train on JSSP images
│       ├── jssp_tensors/         # Train on JSSP tensors
│       └── sat_images/           # Train on SAT images (SAT-specific metrics)
├── data/                         # Input and generated datasets
│   ├── jssp/
│   └── sat/
├── results/                      # Training outputs (configs, metrics, plots)
├── models/                       # MiniZinc models for JSSP
└── utils/                        # Utility scripts
```

Example encodings are shown at the repo root:

- `sat_image.png`: SAT instance as a grayscale image.
- `jssp_image.png`: JSSP instance via Text-to-Image (MiniZinc model + `.dzn` data).
- `jssp_tensor.png`: JSSP processing-time matrix visualised as a heatmap (before padding).

---

## 4. Most Important Configuration Files

Use these YAML files to control **paths, image/tensor shapes, solver sets, and time limits**. All of them can be overridden via CLI flags.

### 4.1 Data Generation

- **JSSP images**  
  `src/data_generation/jssp_images/config.yaml`
  - Where to read/write:
    - Root dataset folder under `data/jssp/datasets/`.
    - CSV filename (e.g. `ground_truth_jsp_generated_dataset.csv`).
    - Images subfolder (e.g. `images/`).
  - Encoding settings:
    - Target image size (typically **128×128**, 1 channel).
  - Solver benchmarking:
    - Candidate solvers and time limits for generating ground-truth metrics.

- **JSSP tensors**  
  `src/data_generation/jssp_tensors/config.yaml`
  - Dataset folders under `data/jssp/datasets/`.
  - Tensor conversion options:
    - Maximum jobs/machines for padding.
    - Standardisation (e.g. z-score) on processing times.
  - Solver benchmarking settings (same idea as JSSP images).

- **SAT images**  
  `src/data_generation/sat_images/config.yaml`
  - ASlib scenario paths and output dataset folders under `data/sat/datasets/`.
  - Image encoding:
    - Target height/width (typically 128×128) and channels.
  - ASlib integration:
    - Prefix maps to resolve instance IDs → file paths.
    - Time limits / status filtering for solver runs.

### 4.2 Training

- **JSSP images**  
  `src/training/jssp_images/config.yaml`
  - `data.image.*`:
    - Target height/width and channels (must match generated images).
  - `data.time_limit_s`:
    - Time limit used to define *viable* solvers (multilabel tasks).
  - `data.use_score`:
    - Whether to use score-based labels instead of pure runtimes.
  - `model.*`:
    - CNN architecture (number of conv layers, filters, dense layers, dropout).
    - Learning rate.
  - `training.*`:
    - Epochs, batch size, K-folds, and random seed.

- **JSSP tensors**  
  `src/training/jssp_tensors/config.yaml`
  - Tensor shape and padding parameters.
  - Same style `model.*` and `training.*` blocks as JSSP images.

- **SAT images**  
  `src/training/sat_images/config.yaml`
  - Same core fields as JSSP images:
    - `data.image.*`, `data.time_limit_s`, `model.*`, `training.*`.
  - SAT-specific:
    - Optional feature time column (for AST).
    - K-fold repetitions for repeated cross-validation.

---

## 5. Core Workflows (Copy–Paste Commands)

Commands assume you run them from the **project root**.

### 5.1 JSSP: Images End‑to‑End

```bash
# Generate JSSP image dataset (random/generated instances)
python -m src.data_generation.jssp_images.cli \
  --config src/data_generation/jssp_images/config.yaml \
  --mode generated

# Train classification model on generated JSSP images
python -m src.training.jssp_images.cli \
  --csv data/jssp/datasets/jssp_cnn_data_images/ground_truth_jsp_generated_dataset.csv \
  --task classification \
  --epochs 30 \
  --folds 5
```

Multilabel variant:

```bash
python -m src.training.jssp_images.cli \
  --csv data/jssp/datasets/jssp_cnn_data_images/ground_truth_jsp_generated_dataset.csv \
  --task multilabel \
  --epochs 25
```

### 5.2 JSSP: Tensors End‑to‑End

```bash
# Generate JSSP tensor dataset
python -m src.data_generation.jssp_tensors.cli \
  --config src/data_generation/jssp_tensors/config.yaml \
  --mode generated

# Train tensor-based model
python -m src.training.jssp_tensors.cli \
  --csv data/jssp/datasets/jssp_cnn_data_tensors/ground_truth_jsp_generated_dataset.csv \
  --task classification \
  --epochs 30 \
  --folds 5
```

Multilabel:

```bash
python -m src.training.jssp_tensors.cli \
  --csv data/jssp/datasets/jssp_cnn_data_tensors/ground_truth_jsp_generated_dataset.csv \
  --task multilabel \
  --epochs 30
```

### 5.3 SAT: ASlib → Images → Training

```bash
# 1) Extract SAT instances (if compressed)
tar -xf data/sat/instances/sc2012-application.tar -C data/sat/instances/

# 2) Generate SAT image dataset from ASlib scenario
python -m src.data_generation.sat_images.cli \
  --config src/data_generation/sat_images/config.yaml \
  --scenario_dir data/sat/aslib/sc2012-application \
  --instances_dir data/sat/instances/sc2012-application

# 3) Train SAT image model with 5×5 cross-validation
python -m src.training.sat_images.cli \
  --csv data/sat/datasets/sat_cnn_data_images/ground_truth_aslib.csv \
  --task classification \
  --folds 5 \
  --repeats 5 \
  --time_limit 1200
```

Example of multilabel training with a solver subset:

```bash
python -m src.training.sat_images.cli \
  --csv data/sat/datasets/sat_cnn_data_images/ground_truth_aslib.csv \
  --task multilabel \
  --solvers clasp,glucose,lingeling
```

---

## 6. Supported Tasks and Key Metrics

- **Classification**
  - Goal: select the single best solver.
  - Generic metrics: accuracy, macro‑F1.
  - SAT only: additional `resolved_rate` and `AST` (Average Solving Time).

- **Multilabel**
  - Goal: select *all* viable solvers (runtime < `time_limit_s`).
  - Generic metrics: micro‑F1, macro‑F1, average precision.
  - SAT only: `resolved_rate` and `AST` available.

- **Regression (SAT only, Python API)**
  - Goal: predict runtime for each solver.
  - Metric: MAE (mean absolute error).

For detailed metric definitions and plots, see the module-specific training READMEs.

---

## 7. Data and Results Layout

### 7.1 Data Generation Outputs

Typical dataset folder:

```text
data/[jssp|sat]/datasets/[dataset_name]/
├── ground_truth*.csv         # Main dataset file
├── images/                   # .npy image/tensor files
│   ├── instance1.npy
│   └── instance2.npy
└── [instances]/              # Original instances (.dzn, .cnf, .xml, ...)
```

CSV files include (among others):

- Path columns (e.g. `Image_Npy_Path`, `Tensor_Npy_Path`).
- Per‑solver runtime/status columns.
- Derived labels for classification/multilabel tasks.

### 7.2 Training Outputs

Each training run creates a timestamped folder:

```text
results/[jssp|sat]/[images|tensors]/[run_name_timestamp]/
├── config.yaml               # Configuration actually used
├── run_info.json             # Run metadata
├── metrics_summary.json      # Aggregated results
├── metrics_per_fold.csv      # Per-fold metrics
├── [metric]_per_fold.png     # Cross-fold plots
└── fold_1/                   # Individual fold details
    ├── fold1_metrics.json
    ├── fold1_y_true.npy
    ├── fold1_y_pred.npy
    ├── fold1_confusion.png
    └── ...
```

---

## 8. Module Documentation

For detailed options (all CLI flags, full config structure, implementation notes), refer to:

- `src/data_generation/jssp_images/README.md`
- `src/data_generation/jssp_tensors/README.md`
- `src/data_generation/sat_images/README.md`
- `src/training/jssp_images/README.md`
- `src/training/jssp_tensors/README.md`
- `src/training/sat_images/README.md`
- `utils/README.md`

---

## 9. Utilities (Quick Examples)

```bash
# Visualise a JSSP image
python -m utils.visualize_tensor \
  data/jssp/datasets/jssp_cnn_data_images/images/GEN_10x10_1_image.npy

# Visualise a JSSP tensor
python -m utils.visualize_tensor \
  data/jssp/datasets/jssp_cnn_data_tensors/images/GEN_10x10_1_tensor.npy

# Analyse solver/label imbalance in a JSSP CSV
python -m utils.dataset_imbalance \
  --csv data/jssp/datasets/jssp_cnn_data_images/ground_truth_jsp_generated_dataset.csv \
  --time-limit 60
```

See `utils/README.md` for all options.

---

## 10. Common Issues (Quick Checks)

- **CSV missing `Image_Npy_Path` / `Tensor_Npy_Path`**
  - Run the corresponding data generation script first.

- **"No valid images/tensors found"**
  - Ensure commands are run from the project root.
  - Regenerate CSVs if files were moved or renamed.

- **MiniZinc errors (JSSP only)**
  - Check that `minizinc` is in `PATH` and a CP solver is installed:
    ```bash
    minizinc --version
    minizinc --solvers
    ```

- **Import or dependency errors**
  - Reinstall dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## 11. License and Support

This code is part of a research project on solver selection.

When using commercial MIP solvers such as **Gurobi** or **CPLEX** for JSSP-related experiments, you must obtain and configure valid licenses yourself. These solvers are **not** bundled with this repository, and the default examples rely on freely available CP solvers (e.g. `gecode`, `chuffed`) via MiniZinc.

- For detailed usage, start from the module-specific READMEs under `src/` and `utils/`.
- Use `--help` on any CLI entry point to inspect available options, for example:
  ```bash
  python -m src.training.jssp_images.cli --help