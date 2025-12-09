# JSSP Tensor Data Generation

This folder generates supervised datasets for Job Shop Scheduling Problems (JSSP) where each instance is encoded as a 2D tensor (matrix) of processing times and paired with solver performance metrics in a CSV file.

It supports two modes:

- **academic** – JSPLIB benchmark instances  
- **generated** – randomly generated, balanced instances

Each instance is represented as a matrix with shape `(num_jobs, num_machines)`, optionally standardized with z‑score.

---

## Folder contents

- [`cli.py`](src/data_generation/jssp_tensors/cli.py) – command‑line entry point for the tensor pipeline
- [`config.yaml`](src/data_generation/jssp_tensors/config.yaml) – configuration for models, solvers, outputs, and instance generation
- [`prepare_academic_dataset.py`](src/data_generation/jssp_tensors/prepare_academic_dataset.py) – builds the academic (JSPLIB) dataset and CSV
- [`prepare_generated_dataset.py`](src/data_generation/jssp_tensors/prepare_generated_dataset.py) – builds the synthetic dataset and CSV
- [`tensor_converter.py`](src/data_generation/jssp_tensors/tensor_converter.py) – `.dzn → 2D tensor .npy` conversion and CSV update
- [`__init__.py`](src/data_generation/jssp_tensors/__init__.py) – module exports

Instance generation and MiniZinc execution are reused from the image pipeline in [`src/data_generation/jssp_images`](src/data_generation/jssp_images/README.md).

---

## Inputs

Configured in [`config.yaml`](src/data_generation/jssp_tensors/config.yaml):

- **MiniZinc models**
  - CP model: [`models/jssp/model.mzn`](models/jssp/model.mzn)
  - optional MIP model: [`models/jssp/model_linear.mzn`](models/jssp/model_linear.mzn)
- **Academic mode**
  - JSPLIB instances via the `job-shop-lib` Python package
  - instance list, time limit, and penalty factor under `academic.*`
- **Generated mode**
  - grid of `(n_jobs, n_machines, n_instances)` in `generated.generation_cases`
  - time limits and random seeds in `generated.*`
  - solver set in `solver_candidates`
- **Tensor options**
  - `tensor.standardize: true|false` to enable or disable per‑instance z‑score standardization

MiniZinc must be installed and at least one solver must be available via `minizinc --solvers`.

---

## Outputs

All outputs are written under `data/jssp/datasets/` (relative to the project root).

Typical layout:

```text
data/jssp/datasets/
├── jsp_cnn_data_acad_tensors/      # academic mode
│   ├── *.dzn
│   ├── ground_truth_jsp_academic.csv
│   └── images/
│       └── *_tensor.npy            # 2D tensors
└── jssp_cnn_data_tensors/          # generated mode
    ├── *.dzn
    ├── ground_truth_jsp_generated_dataset.csv
    └── images/
        └── *_tensor.npy
```

Tensor format:

- type: NumPy array saved as `.npy`
- shape: `(num_jobs, num_machines)` (varies per instance)
- dtype: `float32`
- content: processing times for each operation
- optional: z‑score standardization (mean ≈ 0, std ≈ 1) applied per instance

During conversion, [`tensor_converter.py`](src/data_generation/jssp_tensors/tensor_converter.py) also updates the CSV with:

- `Tensor_Npy_Path` – tensor file path
- `Image_Npy_Path` – alias pointing to the same tensor file so training code can reuse the CSV

---

## CLI usage (from project root)

Run the full pipeline via [`cli.py`](src/data_generation/jssp_tensors/cli.py):

```bash
python -m src.data_generation.jssp_tensors.cli --mode <academic|generated> [options]
```

### 1. Basic runs

Academic JSPLIB dataset with the default configuration:

```bash
python -m src.data_generation.jssp_tensors.cli --mode academic
```

Generated dataset with the default configuration:

```bash
python -m src.data_generation.jssp_tensors.cli --mode generated
```

Both commands:

1. build the CSV dataset (running MiniZinc solvers)
2. convert `.dzn` instances to 2D tensor `.npy` files

---

### 2. Use a custom configuration file

Select a specific YAML file, for example to change instance sizes, time limits, or solvers:

```bash
python -m src.data_generation.jssp_tensors.cli \
  --mode academic \
  --config src/data_generation/jssp_tensors/config.yaml
```

```bash
python -m src.data_generation.jssp_tensors.cli \
  --mode generated \
  --config path/to/custom_jssp_tensors_config.yaml
```

---

### 3. Control standardization (z‑score)

Standardization is enabled by default (`tensor.standardize: true` in [`config.yaml`](src/data_generation/jssp_tensors/config.yaml)). Disable it from the CLI:

Academic, without standardization:

```bash
python -m src.data_generation.jssp_tensors.cli \
  --mode academic \
  --no-standardize
```

Generated, without standardization:

```bash
python -m src.data_generation.jssp_tensors.cli \
  --mode generated \
  --no-standardize
```

---

### 4. Re‑encode an existing CSV (skip solvers)

If you already have a CSV with solver metrics and `.dzn` files, you can regenerate tensors without running MiniZinc again by using `--skip-solvers` and `--csv`.

Generated dataset:

```bash
python -m src.data_generation.jssp_tensors.cli \
  --mode generated \
  --skip-solvers \
  --csv data/jssp/datasets/jssp_cnn_data_tensors/ground_truth_jsp_generated_dataset.csv
```

Academic dataset:

```bash
python -m src.data_generation.jssp_tensors.cli \
  --mode academic \
  --skip-solvers \
  --csv data/jssp/datasets/jsp_cnn_data_acad_tensors/ground_truth_jsp_academic.csv
```

In both cases the script:

1. reads the CSV
2. locates the `.dzn` instance files
3. creates/overwrites `images/*_tensor.npy` for each row and updates the CSV paths

---

### 5. Typical end‑to‑end run

1. Adjust [`config.yaml`](src/data_generation/jssp_tensors/config.yaml) if needed (models, solvers, instance sizes, time limits).
2. Generate the dataset, for example in generated mode:

   ```bash
   python -m src.data_generation.jssp_tensors.cli --mode generated
   ```

3. Use the CSV and tensors for training, for example:

   ```python
   import numpy as np
   import pandas as pd

   df = pd.read_csv(
       "data/jssp/datasets/jssp_cnn_data_tensors/ground_truth_jsp_generated_dataset.csv"
   )

   tensor = np.load(df.iloc[0]["Tensor_Npy_Path"])
   print(tensor.shape)
   ```

---

## Minimal dependencies

Python packages (see also [`requirements.txt`](requirements.txt)):

```bash
pip install numpy pandas PyYAML job-shop-lib minizinc
```

System requirements:

- MiniZinc command‑line tools installed (bundle or package manager)
- at least one solver available (e.g. Gecode, Chuffed, CBC, SCIP, HiGHS, or others configured in [`config.yaml`](src/data_generation/jssp_tensors/config.yaml))

Unavailable solvers are handled gracefully at CSV level (their metrics are marked as `NA`/`inf` and the pipeline continues).

---

## Where this data is used

The generated JSSP tensor datasets are consumed by the training pipelines under [`src/training/jssp_tensors`](src/training/jssp_tensors/README.md) to train and evaluate models that operate directly on tensor (matrix) representations of JSSP instances.