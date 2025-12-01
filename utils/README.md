# Utility Scripts

This directory contains small standalone utilities that help analyze datasets and inspect `.npy` files produced by the data-generation pipelines. This README documents:

- [`utils/dataset_imbalance.py`](utils/dataset_imbalance.py)
- [`utils/visualize_tensor.py`](utils/visualize_tensor.py)

Other files in `utils/` are either internal helpers or legacy and are not covered here.

---

## `utils/dataset_imbalance.py`

[`utils/dataset_imbalance.py`](utils/dataset_imbalance.py) is a CLI tool to analyze **label / solver imbalance** in the CSVs produced by the data-generation modules (SAT and JSSP). It helps you understand:

- How often each solver is the (runtime) winner.
- How many **viable** solvers exist per instance under a given time limit.
- Basic quality and missing-value statistics.

### Expected Input

A CSV like:

- SAT images: `data/sat/datasets/sat_cnn_data_images/ground_truth_aslib.csv`
- JSSP images: `data/jssp/datasets/jssp_cnn_data_images/ground_truth_jsp_generated_dataset.csv`
- JSSP tensors: `data/jssp/datasets/jssp_cnn_data_tensors/ground_truth_jsp_generated_dataset.csv`

Required columns (the tool auto-detects):

- `*_Runtime_s` — per-solver runtime columns.
- Optionally `*_Status` — per-solver status columns (OK/TIMEOUT/…).
- Optionally `Winner_Key` — single-best solver label (if present).

### Basic Usage

From the project root:

```bash
python -m utils.dataset_imbalance \
  --csv data/jssp/datasets/jssp_cnn_data_images/ground_truth_jsp_generated_dataset.csv \
  --time-limit 60
```

For a SAT CSV:

```bash
python -m utils.dataset_imbalance \
  --csv data/sat/datasets/sat_cnn_data_images/ground_truth_aslib.csv \
  --time-limit 1200
```

### Key Options

- `--csv PATH`  
  Path to the CSV to analyze.

- `--time-limit SECONDS`  
  Time limit used to decide whether a solver is **viable** for an instance.

- `--valid-statuses ...`  
  Optional list of status values considered successful (defaults are sensible for SAT).

- `--invalid-statuses ...`  
  Optional list of status values considered failures (timeouts, crashes, etc.).

### What It Reports

Typical output includes:

- Basic dataset info (rows, columns, missing values).
- Label distribution if `Winner_Key` is present.
- **Classification-style** distribution:
  - Argmin over `*_Runtime_s` (fastest solver), with invalid statuses treated as infinite.
- **Multilabel-style viability**:
  - For each instance, how many solvers are viable (`runtime < time_limit` and status not invalid).
  - Per-solver frequency of being viable.
  - Distribution of number of viable solvers per instance.
- Per-solver status breakdown (valid / invalid / other / missing).

This is useful before training to check if your dataset is extremely imbalanced (e.g. one solver always wins, or almost no instances have multiple viable solvers).

---

## `utils/visualize_tensor.py`

[`utils/visualize_tensor.py`](utils/visualize_tensor.py) is a **unified viewer** for inspecting `.npy` / tensor files and standard image formats. It replaces older, more fragmented visualization scripts and is recommended for:

- Quickly inspecting JSSP images (`Image_Npy_Path` from `jssp_images`).
- Inspecting JSSP tensors (`Image_Npy_Path` / `Tensor_Npy_Path` from `jssp_tensors`).
- Inspecting SAT images (`Image_Npy_Path` from `sat_images`).
- Debugging arbitrary `.npy`, `.npz`, `.pt`, `.pth`, `.pkl` tensors.

### Supported Input Types

By filename extension:

- **NumPy / tensors**:
  - `.npy`, `.npz`
  - `.pt`, `.pth` (PyTorch)
  - `.pkl` (pickled arrays / tensors / dicts)

- **Standard images**:
  - `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif` (via Pillow)

Internally, it:

- Loads the array or tensor.
- Handles 2D or 3D arrays, inferring channels (`H×W` or `H×W×C`).
- Optionally normalizes to `[0, 1]` for display.
- Plots with Matplotlib.

### Basic Usage

From the project root:

```bash
python -m utils.visualize_tensor path/to/file.npy
```

Examples:

```bash
# Visualize a JSSP image (Text-to-Image)
python -m utils.visualize_tensor \
  data/jssp/datasets/jssp_cnn_data_images/images/GEN_10x10_1_image.npy

# Visualize a JSSP tensor (padded 10x10x2 or 10x10x1)
python -m utils.visualize_tensor \
  data/jssp/datasets/jssp_cnn_data_tensors/images/GEN_10x10_1_tensor.npy

# Visualize a SAT image
python -m utils.visualize_tensor \
  data/sat/datasets/sat_cnn_data_images/images/instance123__abcdef123456.npy
```

### Common Options (if implemented in the script)

Depending on the version of [`utils/visualize_tensor.py`](utils/visualize_tensor.py), typical CLI flags include:

- `--show` / `--no-show` — whether to display the figure interactively.
- `--save PATH` — save the rendered image to a file.
- `--normalize` — apply min–max normalization to `[0, 1]` before plotting.
- `--transpose` — treat input as `C×H×W` instead of `H×W×C` (common for some tensor formats).
- `--channel N` — select a specific channel to display for multi-channel tensors.

(If in doubt, run `python -m utils.visualize_tensor --help` to see the exact set of supported options.)

### When to Use It

- **After data generation**: sanity-check a few random `.npy` files to confirm shapes and intensities:
  - JSSP images: expect `(128, 128)` or `(128, 128, 1)`.
  - JSSP tensors: expect `(max_jobs, max_machines, n_channels)`, typically `(10, 10, 2)`.
  - SAT images: expect `(128, 128)` or `(128, 128, 1)`.

- **During debugging**: if a training run crashes with shape issues, open the offending `.npy` to confirm its dimensions and channel ordering.

---

## Recommended Workflow

- Use [`utils/dataset_imbalance.py`](utils/dataset_imbalance.py) **before training** to understand solver/label imbalance and adjust your experimental design (e.g., solver subsets, time limits).
- Use [`utils/visualize_tensor.py`](utils/visualize_tensor.py) **after data generation** and when debugging training issues, to visually check that inputs look as expected and have the correct shapes.