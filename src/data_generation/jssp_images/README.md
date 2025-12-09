# JSSP Image Data Generation

This folder generates supervised datasets for Job Shop Scheduling Problems (JSSP) where each instance is encoded as a grayscale image and paired with solver performance metrics in a CSV file.

It supports two modes:

- **academic** – JSPLIB benchmark instances  
- **generated** – randomly generated, balanced instances

Image encoding is text‑based: each image is derived from the concatenation of the MiniZinc CP model (`.mzn`) and a JSSP instance (`.dzn`).

---

## Folder contents

- [`cli.py`](src/data_generation/jssp_images/cli.py) – command‑line entry point for the full pipeline
- [`config.yaml`](src/data_generation/jssp_images/config.yaml) – configuration for models, solvers, outputs, and instance generation
- [`prepare_academic_dataset.py`](src/data_generation/jssp_images/prepare_academic_dataset.py) – builds the academic (JSPLIB) dataset and CSV
- [`prepare_generated_dataset.py`](src/data_generation/jssp_images/prepare_generated_dataset.py) – builds the synthetic dataset and CSV
- [`image_converter.py`](src/data_generation/jssp_images/image_converter.py) – text‑to‑image conversion (`.mzn + .dzn → .npy`)
- [`minizinc_solver.py`](src/data_generation/jssp_images/minizinc_solver.py) – MiniZinc integration and solver orchestration
- [`jssp_instance_utils.py`](src/data_generation/jssp_images/jssp_instance_utils.py) – helpers to load and generate JSSP instances

---

## Inputs

Configured in [`config.yaml`](src/data_generation/jssp_images/config.yaml):

- **MiniZinc models**
  - CP model: [`models/jssp/model.mzn`](models/jssp/model.mzn)
  - optional MIP model: [`models/jssp/model_linear.mzn`](models/jssp/model_linear.mzn)
- **Academic mode**
  - JSPLIB instances provided via the `job-shop-lib` Python package
  - time limit, penalty factor, and instance list in the `academic.*` section
- **Generated mode**
  - grid of `(n_jobs, n_machines, n_instances)` in `generated.generation_cases`
  - per‑run time limits and random seeds in `generated.*`
  - list of CP/MIP solvers in `solver_candidates`

MiniZinc must be installed and at least one solver must be available via `minizinc --solvers`.

---

## Outputs

All outputs are written under `data/jssp/datasets/` (relative to the project root).

Typical layout:

```text
data/jssp/datasets/
├── jsp_cnn_data_acad/          # academic mode
│   ├── *.dzn                   # MiniZinc data files
│   ├── ground_truth_jsp_academic.csv
│   └── images/
│       └── *_image.npy
└── jssp_cnn_data_images/       # generated mode
    ├── *.dzn
    ├── ground_truth_jsp_generated_dataset.csv
    └── images/
        └── *_image.npy
```

Image format:

- type: NumPy array saved as `.npy`
- shape: `(H, W)` with default `128 × 128` pixels (configurable)
- values: `[0, 255]` grayscale, derived from CP model + instance text
- no normalization by default (matching the paper setup)

---

## CLI usage (from project root)

Run the full pipeline via [`cli.py`](src/data_generation/jssp_images/cli.py):

```bash
python -m src.data_generation.jssp_images.cli --mode <academic|generated> [options]
```

### 1. Basic runs

Academic JSPLIB dataset with default configuration:

```bash
python -m src.data_generation.jssp_images.cli --mode academic
```

Generated dataset with default configuration:

```bash
python -m src.data_generation.jssp_images.cli --mode generated
```

Both commands:

1. build the CSV dataset (running MiniZinc solvers unless `--skip-solvers` is used)
2. convert `(.mzn + .dzn)` to grayscale image `.npy` files

---

### 2. Use a custom configuration

Point to a different YAML file, for example to change instance sizes, time limits, or solvers:

```bash
python -m src.data_generation.jssp_images.cli \
  --mode academic \
  --config src/data_generation/jssp_images/config.yaml
```

```bash
python -m src.data_generation.jssp_images.cli \
  --mode generated \
  --config path/to/custom_jssp_images_config.yaml
```

---

### 3. Override image size

Override `image.target_size` from [`config.yaml`](src/data_generation/jssp_images/config.yaml) at runtime:

```bash
python -m src.data_generation.jssp_images.cli \
  --mode academic \
  --image-size 256
```

```bash
python -m src.data_generation.jssp_images.cli \
  --mode generated \
  --image-size 64
```

---

### 4. Re‑encode an existing CSV (skip solvers)

If you already have a CSV with solver metrics and `.dzn` files, you can regenerate images without running MiniZinc again by using `--skip-solvers` and `--csv`.

Generated dataset:

```bash
python -m src.data_generation.jssp_images.cli \
  --mode generated \
  --skip-solvers \
  --csv data/jssp/datasets/jssp_cnn_data_images/ground_truth_jsp_generated_dataset.csv
```

Academic dataset (adjust the CSV name to your setup):

```bash
python -m src.data_generation.jssp_images.cli \
  --mode academic \
  --skip-solvers \
  --csv data/jssp/datasets/jsp_cnn_data_acad/ground_truth_jsp_academic.csv
```

In both cases, the script:

1. reads the CSV
2. locates the corresponding `.dzn` instances
3. creates/overwrites `images/*.npy` for each row

---

## Minimal dependencies

Python packages (see also [`requirements.txt`](requirements.txt)):

```bash
pip install numpy pandas Pillow PyYAML job-shop-lib minizinc
```

System requirements:

- MiniZinc command‑line tools installed (e.g. bundle or package manager)
- at least one solver available (CBC, SCIP, HiGHS, or others configured in [`config.yaml`](src/data_generation/jssp_images/config.yaml))

Unavailable solvers are handled gracefully: their metrics in the CSV are marked as `NA`/`inf`, and the rest of the pipeline still runs.

---

## Where this data is used

The generated JSSP image datasets are consumed by the training pipelines under [`src/training/jssp_images`](src/training/jssp_images/README.md) to train and evaluate CNN‑based models on JSSP solver selection or performance prediction tasks.