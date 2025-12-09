# Utility Scripts

This folder contains small, standalone utilities to inspect datasets and `.npy`/tensor files produced by the SAT and JSSP pipelines.

Current public scripts:

- [`dataset_imbalance.py`](utils/dataset_imbalance.py) – analyze solver/label imbalance in CSV datasets
- [`visualize_tensor.py`](utils/visualize_tensor.py) – visualize `.npy`/tensor files and standard images

---

## `dataset_imbalance.py` – dataset imbalance analysis

Command-line tool to inspect how balanced a solver-selection dataset is.

It works with any CSV produced by the data-generation modules, for example:

- SAT images: `data/sat/datasets/sat_cnn_data_images/ground_truth_aslib.csv`
- JSSP images: `data/jssp/datasets/jssp_cnn_data_images/ground_truth_jsp_generated_dataset.csv`
- JSSP tensors: `data/jssp/datasets/jssp_cnn_data_tensors/ground_truth_jsp_generated_dataset.csv`

It automatically detects:

- `*_Runtime_s` – per-solver runtime columns
- optional `*_Status` – per-solver status columns
- optional `Winner_Key` – known-best solver label (if present)

### Basic usage

From the project root:

```bash
python -m utils.dataset_imbalance \
  --csv data/jssp/datasets/jssp_cnn_data_images/ground_truth_jsp_generated_dataset.csv \
  --time-limit 60
```

SAT CSV:

```bash
python -m utils.dataset_imbalance \
  --csv data/sat/datasets/sat_cnn_data_images/ground_truth_aslib.csv \
  --time-limit 1200
```

### Key options

```bash
python -m utils.dataset_imbalance --help
```

Common flags:

- `--csv PATH`  
  CSV file to analyze (required).

- `--time-limit SECONDS`  
  Time limit used to decide if a solver is **viable** for an instance (default 60.0).

- `--valid-statuses "ok,sat,unsat"`  
  Comma-separated list of status values considered successful.

- `--invalid-statuses "timeout,time_out,timedout,memout,crash,error,fail"`  
  Comma-separated list of status values considered failures.

### What it reports

For the given CSV, the script prints:

- Basic dataset info (rows, columns, missing values).
- `Winner_Key` distribution (if present).
- **Classification-style** label distribution:
  - Argmin over `*_Runtime_s`, treating invalid statuses as infinite (never winners).
- **Multilabel-style** viability:
  - For each instance, how many solvers are viable (`runtime < time_limit` and status not invalid).
  - Per-solver fraction of instances where the solver is viable.
  - Histogram of “number of viable solvers per instance”.
- Per-solver status summary:
  - Counts and percentages of valid / invalid / other / missing statuses.

Use this before training to detect extreme imbalance (e.g. one solver always winning, or almost no instances with multiple viable solvers).

---

## `visualize_tensor.py` – tensor/image viewer

Unified viewer for `.npy`/tensor artifacts and standard image formats. It replaces older visualization scripts and is the recommended way to quickly inspect inputs.

Typical uses:

- JSSP images: `Image_Npy_Path` from the JSSP image datasets
- JSSP tensors: `Image_Npy_Path` / `Tensor_Npy_Path` from the JSSP tensor datasets
- SAT images: `Image_Npy_Path` from the SAT image datasets
- Arbitrary `.npy`, `.npz`, `.pt`, `.pth`, `.pkl` tensors

### Supported file types

- **Tensor-like files**
  - `.npy`, `.npz` (NumPy)
  - `.pt`, `.pth` (PyTorch tensors or state dicts)
  - `.pkl` (pickled NumPy arrays / tensors / dict-like objects)

- **Standard images**
  - `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`

### Basic usage

From the project root:

```bash
# Visualize a JSSP image
python -m utils.visualize_tensor \
  data/jssp/datasets/jssp_cnn_data_images/images/GEN_10x10_1_image.npy
```

```bash
# Visualize a JSSP tensor (e.g. 10x10x2)
python -m utils.visualize_tensor \
  data/jssp/datasets/jssp_cnn_data_tensors/images/GEN_10x10_1_tensor.npy
```

```bash
# Visualize a SAT image
python -m utils.visualize_tensor \
  data/sat/datasets/sat_cnn_data_images/images/instance123__abcdef123456.npy
```

### Common options

Show available flags:

```bash
python -m utils.visualize_tensor --help
```

Useful combinations:

```bash
# Save the rendered image instead of opening a window
python -m utils.visualize_tensor \
  data/jssp/datasets/jssp_cnn_data_images/images/GEN_10x10_1_image.npy \
  --save outputs/gen_10x10_1_image.png \
  --no-show
```

```bash
# Visualize a specific key from an .npz or .pt/.pth/.pkl
python -m utils.visualize_tensor \
  checkpoints/weights.pt \
  --key features \
  --normalize
```

```bash
# Explicitly treat a 3D tensor as CHW and select a colormap
python -m utils.visualize_tensor \
  path/to/tensor.npy \
  --channel-order chw \
  --cmap gray
```

The script:

- Loads the array or tensor.
- Handles 2D or 3D arrays, inferring channel order by default.
- Optionally normalizes values to `[0, 1]` for display.
- Renders with Matplotlib and optionally saves the figure.

---

## Recommended workflow

- Run [`dataset_imbalance.py`](utils/dataset_imbalance.py) **before training** to understand solver/label imbalance and adjust time limits or solver subsets if needed.
- Run [`visualize_tensor.py`](utils/visualize_tensor.py) **after data generation** and when debugging training issues, to quickly check that `.npy` files have the expected shape and content.