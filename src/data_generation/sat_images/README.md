# SAT Image Data Generation

This folder generates supervised datasets for SAT (Boolean Satisfiability) problems where each instance is encoded as a grayscale image and paired with solver performance metrics in a CSV file.

It works on **existing ASlib scenarios** (no instance generation) and:

- parses solver runtimes from ASlib
- resolves paths to raw SAT instances
- converts each instance file into a fixed‑size grayscale image

---

## Folder contents

- [`cli.py`](src/data_generation/sat_images/cli.py) – command‑line entry point for the full pipeline
- [`config.yaml`](src/data_generation/sat_images/config.yaml) – configuration for outputs, image size, and ASlib path mapping
- [`prepare_aslib_dataset.py`](src/data_generation/sat_images/prepare_aslib_dataset.py) – builds the ground‑truth CSV from an ASlib scenario
- [`image_converter.py`](src/data_generation/sat_images/image_converter.py) – SAT file → image `.npy` conversion
- [`aslib_parser.py`](src/data_generation/sat_images/aslib_parser.py) – reads `algorithm_runs.arff` and related metadata
- [`instance_resolver.py`](src/data_generation/sat_images/instance_resolver.py) – resolves instance IDs to concrete file paths
- [`__init__.py`](src/data_generation/sat_images/__init__.py) – module exports

---

## Inputs

Configured in [`config.yaml`](src/data_generation/sat_images/config.yaml):

- **ASlib scenario directory** (when not using `--skip-aslib`)
  - Required file: `algorithm_runs.arff`
  - Optional: `description.txt` with timeout
- **Instances directory**
  - Plaintext SAT instances: CNF (`*.cnf`), XCSP, DZN, etc.
  - Ideally **already decompressed** (no `.gz`, `.bz2`, `.xz`, `.zip`, `.lzma`)
- **Image parameters**
  - `image.target_size` – square size for grayscale images (default `128`)
- **ASlib settings**
  - `aslib.default_timeout_s` – default timeout if not present in scenario
  - `aslib.prefix_map` – mapping from instance ID prefixes to subdirectories for path resolution

---

## Outputs

All outputs are written under `data/sat/datasets/` (relative to the project root).

Default layout:

```text
data/sat/datasets/
└── sat_cnn_data_images/
    ├── ground_truth_aslib.csv
    └── images/
        └── <instance_name>__<hash>.npy
```

CSV columns (main ones):

- `Instance_Id` – original ASlib instance identifier
- `Instance_Name` – base filename without extension
- `Raw_Text_Path` – absolute path to the raw instance file
- `Time_Limit_s` – timeout used
- `Winner_Key` – best solver for this instance
- `{SOLVER}_Runtime_s` – runtime per solver (seconds)
- `{SOLVER}_Status` – run status per solver
- `Image_Npy_Path` – path to the generated image `.npy`

Image format:

- type: NumPy array saved as `.npy`
- shape: `(H, W)` with default `128 × 128`
- dtype: `float32`
- content: byte/ASCII intensities from the instance text, reshaped and resized
- normalization:
  - **disabled by default** (raw `[0..255]` for paper‑like images)
  - optional z‑score normalization via `--normalize`

---

## CLI usage (from project root)

Run the full pipeline via [`cli.py`](src/data_generation/sat_images/cli.py):

```bash
python -m src.data_generation.sat_images.cli [options]
```

### 1. Basic run: process one ASlib scenario

```bash
python -m src.data_generation.sat_images.cli \
  --scenario-dir data/sat/aslib/sc2012-application \
  --instances-dir data/sat/instances/sc2012-application
```

This will:

1. read `algorithm_runs.arff` from `--scenario-dir`
2. build `ground_truth_aslib.csv` under `data/sat/datasets/sat_cnn_data_images/`
3. create grayscale images under `data/sat/datasets/sat_cnn_data_images/images/`

---

### 2. Custom output directory

```bash
python -m src.data_generation.sat_images.cli \
  --scenario-dir data/sat/aslib/sc2012-application \
  --instances-dir data/sat/instances/sc2012-application \
  --output-dir data/sat/datasets/my_sat_images
```

Outputs:

- CSV at `data/sat/datasets/my_sat_images/ground_truth_aslib.csv`
- images under `data/sat/datasets/my_sat_images/images/`

---

### 3. Custom image size and normalization

Increase image size to `256 × 256`:

```bash
python -m src.data_generation.sat_images.cli \
  --scenario-dir data/sat/aslib/sc2012-application \
  --instances-dir data/sat/instances/sc2012-application \
  --image-size 256
```

Enable per‑image z‑score normalization:

```bash
python -m src.data_generation.sat_images.cli \
  --scenario-dir data/sat/aslib/sc2012-application \
  --instances-dir data/sat/instances/sc2012-application \
  --normalize
```

Combine both:

```bash
python -m src.data_generation.sat_images.cli \
  --scenario-dir data/sat/aslib/sc2012-application \
  --instances-dir data/sat/instances/sc2012-application \
  --image-size 256 \
  --normalize
```

---

### 4. Use an existing CSV (skip ASlib processing)

If you already have a CSV with solver metrics, skip the ASlib step and only generate images:

```bash
python -m src.data_generation.sat_images.cli \
  --skip-aslib \
  --csv data/sat/datasets/sat_cnn_data_images/ground_truth_aslib.csv \
  --instances-dir data/sat/instances/sc2012-application
```

---

### 5. Override timeout or provide explicit instance mapping

Custom timeout (seconds):

```bash
python -m src.data_generation.sat_images.cli \
  --scenario-dir data/sat/aslib/sc2012-application \
  --instances-dir data/sat/instances/sc2012-application \
  --timeout 3600
```

Explicit instance mapping via CSV (`instance_id,file_path`):

```bash
python -m src.data_generation.sat_images.cli \
  --scenario-dir data/sat/aslib/sc2012-application \
  --instances-dir data/sat/instances/sc2012-application \
  --instance-map-csv data/sat/instance_mapping.csv
```

---

### 6. Typical end‑to‑end run

1. Extract ASlib scenario to `data/sat/aslib/<scenario_name>/`
2. Extract plaintext instances to `data/sat/instances/<scenario_name>/`
3. Optionally adjust [`config.yaml`](src/data_generation/sat_images/config.yaml)
4. Run:

   ```bash
   python -m src.data_generation.sat_images.cli \
     --scenario-dir data/sat/aslib/sc2012-application \
     --instances-dir data/sat/instances/sc2012-application
   ```

5. Use the CSV and images for training, e.g.:

   ```python
   import numpy as np
   import pandas as pd

   df = pd.read_csv("data/sat/datasets/sat_cnn_data_images/ground_truth_aslib.csv")

   image = np.load(df.iloc[0]["Image_Npy_Path"])
   print(image.shape)
   print(df.iloc[0]["Winner_Key"])
   ```

---

## Optional: decompress instances

For best results, instance files under `data/sat/instances/...` should be plain text (no compression). You can use the helper script [`scripts/decompress_instances.py`](scripts/decompress_instances.py) to expand archives:

Dry‑run (show what would be decompressed):

```bash
python3 scripts/decompress_instances.py data/sat/instances/sc2012-application --dry-run
```

Decompress and keep the original archives:

```bash
python3 scripts/decompress_instances.py data/sat/instances/sc2012-application
```

---

## Minimal dependencies

Python packages (see also [`requirements.txt`](requirements.txt)):

```bash
pip install numpy pandas Pillow PyYAML
```

There are no direct solver dependencies here (solver runs come from ASlib), but you need:

- an ASlib scenario with `algorithm_runs.arff`
- access to the corresponding instance files on disk

---

## Where this data is used

The generated SAT image datasets are consumed by the training pipelines under [`src/training/sat_images`](src/training/sat_images/README.md) to train and evaluate CNN‑based models on SAT algorithm selection or performance prediction tasks.