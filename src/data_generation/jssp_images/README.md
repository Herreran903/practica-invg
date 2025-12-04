# JSSP Images Data Generation

This module generates datasets for Job Shop Scheduling Problems (JSSP) with grayscale image representations. It supports two modes: **academic** (using JSPLIB benchmark instances) and **generated** (creating random balanced instances).

## Overview

The JSSP Images generator converts JSSP problem instances into grayscale images that can be used as input features for machine learning models. The pipeline:

1. **Loads or generates JSSP instances** (academic benchmarks or random instances)
2. **Converts instances to MiniZinc format** (.dzn files)
3. **Runs multiple solvers** to collect performance metrics
4. **Generates ground truth CSV** with solver runtimes and scores
5. **Converts instances to grayscale images** (128x128 .npy files) using a **Text-to-Image** encoding that concatenates the MiniZinc CP model (`.mzn`) with each instance (`.dzn`) and maps the resulting text bytes to pixel intensities

## Directory Structure

```
src/data_generation/jssp_images/
├── config.yaml                      # Central configuration file
├── config_loader.py                 # Configuration loading utilities
├── jssp_instance_utils.py          # Instance loading and generation
├── minizinc_solver.py              # MiniZinc solver execution
├── prepare_academic_dataset.py     # Academic mode pipeline
├── prepare_generated_dataset.py    # Generated mode pipeline
├── image_converter.py              # Text-to-image conversion
├── cli.py                          # Command-line interface
├── __init__.py                     # Module exports
└── README.md                       # This file
```

## Input Requirements

### Academic Mode
- **JSPLIB benchmark instances**: Automatically loaded via `job-shop-lib`
- **MiniZinc model**: `models/jssp/model.mzn` (CP model)
- **Solvers**: At least one MiniZinc solver installed (e.g., Gecode, Chuffed)

### Generated Mode
- **MiniZinc models**: 
  - CP model: `models/jssp/model.mzn`
  - MIP model (optional): `models/jssp/model_linear.mzn`
- **Solvers**: Multiple solvers for benchmarking (CP and/or MIP)

## Output Structure

Both modes produce the following outputs in `data/jssp/datasets/`:

```
jsp_cnn_data_acad/         # Academic mode (JSPLIB benchmarks)
jssp_cnn_data_images/      # Generated mode (random instances, Text-to-Image)
├── *.dzn                           # Instance files in MiniZinc format
├── ground_truth_*.csv              # Ground truth with solver metrics
└── images/
    └── *_image.npy                 # Grayscale images (128x128 float32)
```

### CSV Columns

**Academic Mode:**
- `Instance_Name`: Name of the benchmark instance
- `Raw_Text_Path`: Path to the .dzn file
- `N_Jobs`, `N_Machines`: Instance dimensions
- `Best_Makespan_Found`: Best makespan across all solvers
- `Optimum`: Known optimal makespan
- `{SOLVER}_Runtime_s`: Runtime for each solver
- `{SOLVER}_Score_S_rel`: Relative performance score for each solver

**Generated Mode:**
- `Instance_Name`: Generated instance name (e.g., GEN_6x6_1)
- `Raw_Text_Path`: Path to the .dzn file
- `N_Jobs`, `N_Machines`: Instance dimensions
- `Time_Limit_s`: Time limit used for this run
- `Seed`: Random seed used
- `Winner_Key`: Best performing solver for this configuration
- `{SOLVER}_Runtime_s`: Runtime for each solver
- `{SOLVER}_Makespan`: Makespan found by each solver
- `{SOLVER}_Wall_s`: Wall-clock time for each solver

### Image Format

Images are stored as NumPy arrays (.npy files):
- **Shape**: (128, 128) by default (configurable)
- **Data type**: float32
- **Content**: Grayscale encoding of the **concatenated MiniZinc CP model (`.mzn`) and instance data (`.dzn`)**. The combined text is encoded as bytes (`uint8`), reshaped into the largest possible square matrix, and resized to the target size (values in [0, 255]).
- **Normalization**: None by default (paper-like). Optional per-image z-score via `normalize=True` when using the Text-to-Image converter in [`image_converter.py`](src/data_generation/jssp_images/image_converter.py)
- **Visualization**: To reproduce paper-like figures, display with `cmap="gray"` and fixed range `vmin=0, vmax=255` (e.g., matplotlib)

## Configuration

All parameters are centralized in [`config.yaml`](config.yaml). Key sections:

### Models
```yaml
models:
  cp_model: "models/jssp/model.mzn"
  mip_model: "models/jssp/model_linear.mzn"
```

### Output Directories
```yaml
output:
  base_dir: "data/jssp/datasets"
  academic_dir: "jsp_cnn_data_acad"
  generated_dir: "jssp_cnn_data_images"
```

### Image Parameters
```yaml
image:
  target_size: 128  # Size of square grayscale images
```

### Academic Mode
```yaml
academic:
  instances: ["ft06", "ft10", "la01", "abz5"]
  time_limit_ms: 180000
  penalty_factor_k: 10.0
```

### Generated Mode
```yaml
generated:
  time_limits_ms: [5000, 30000, 60000]
  random_seeds: [1, 2, 3]
  generation_cases:
    - [4, 4, 5]   # 5 instances of 4x4
    - [6, 6, 5]   # 5 instances of 6x6
    - [8, 8, 5]   # 5 instances of 8x8
    - [10, 10, 5] # 5 instances of 10x10
```

## Usage

### From Project Root

**Academic Mode (JSPLIB benchmarks):**
```bash
python -m src.data_generation.jssp_images.cli --mode academic
```

**Generated Mode (random instances):**
```bash
python -m src.data_generation.jssp_images.cli --mode generated
```

**With custom configuration:**
```bash
python -m src.data_generation.jssp_images.cli --mode academic --config path/to/config.yaml
```

**Custom image size:**
```bash
python -m src.data_generation.jssp_images.cli --mode generated --image-size 256
```

**Skip solver execution (only convert existing dataset):**
```bash
python -m src.data_generation.jssp_images.cli --mode academic --skip-solvers --csv data/jssp/datasets/jsp_cnn_data_acad/ground_truth_jsp_academic.csv
```

### Typical Workflow

1. **Configure parameters** in `config.yaml`
2. **Run data generation**:
   ```bash
   python -m src.data_generation.jssp_images.cli --mode generated
   ```
3. **Check outputs**:
   - CSV: `data/jssp/datasets/jssp_cnn_data_images/ground_truth_jsp_generated_dataset.csv`
   - Images: `data/jssp/datasets/jssp_cnn_data_images/images/*.npy`
4. **Use in training**:
   ```python
   import numpy as np
   import pandas as pd
   
   # Load dataset
   df = pd.read_csv('data/jssp/datasets/jsp_cnn_data_gen/ground_truth_jsp_generated_dataset.csv')
   
   # Load an image
   image = np.load(df.iloc[0]['Image_Npy_Path'])
   print(image.shape)  # (128, 128)
   ```

## Dependencies
 
Required Python packages:
- `numpy`: Array operations and image storage
- `pandas`: CSV handling
- `Pillow (PIL)`: Image resizing
- `PyYAML`: Configuration file parsing
- `job-shop-lib`: JSSP instance loading and generation
 
Install Python dependencies:
```bash
pip install numpy pandas Pillow PyYAML job-shop-lib
```
 
### MiniZinc and solver backends
 
This module relies on MiniZinc to benchmark multiple solvers. The execution
is orchestrated by [`minizinc_solver.py`](src/data_generation/jssp_images/minizinc_solver.py:1),
which implements a **hybrid** strategy:
 
1. First, it tries to use the official Python `minizinc` package (if installed).
2. If that fails (e.g., solver not registered, missing DLL, Python package
   not available), it automatically **falls back** to calling the `minizinc`
   binary via `subprocess`.
 
#### Recommended MiniZinc installation
 
There are two common ways to install MiniZinc:
 
- **Official MiniZinc bundle (recommended for CP)**
  - MiniZincIDE for macOS / Linux / Windows: includes CP drivers such as
    Gecode and Chuffed already configured.
  - On macOS it is typically located at:
    - `/Applications/MiniZincIDE.app/Contents/Resources/minizinc`
  - This is usually the easiest way to get CP solvers working.
 
- **Package manager installation (e.g., Homebrew)**
  - On macOS:
    ```bash
    brew install minizinc
    ```
  - This usually provides mainly MIP solvers (CBC, SCIP, HiGHS, etc.) and
    **does not include CP drivers** (Gecode/Chuffed).
 
At runtime, the code tries to locate the `minizinc` binary in the following order:
 
1. The bundle binary (if present and accessible).
2. A binary pointed to by the `MINIZINC_BIN` environment variable (if set).
3. A binary found on the system `PATH` (plain `minizinc`).
 
#### Supported solvers and licenses
 
The configuration files
[`config.yaml`](src/data_generation/jssp_images/config.yaml:1)
and
[`src/data_generation/jssp_tensors/config.yaml`](src/data_generation/jssp_tensors/config.yaml:1)
define both CP and MIP solvers:
 
- **CP (theoretically supported)**:
  - `gecode` (`GECODE_FF`)
  - `chuffed` (`CHUFFED_IO`)
  - `cp-sat` / OR-Tools CP-SAT (`CPSAT_FF`, requires a dedicated MiniZinc driver)
 
- **MIP (typical backends)**:
  - CBC / COIN-BC (`CBC_DEF`) – open source
  - SCIP (`SCIP_DEF`) – free academic license
  - HiGHS (`HIGHS_DEF`) – open source
  - CPLEX (`CPLEX_DEF`) – **requires an IBM CPLEX license**
  - Gurobi (`GUROBI_DEF`) – **requires a Gurobi license**
 
In practice, which solvers can actually run depends on:
 
- What `minizinc --solvers` reports (i.e., which drivers are really installed).
- The licensing and DLL environment:
  - CPLEX and Gurobi need their shared libraries (DLLs/.dylib) configured
    and a valid license.
  - HiGHS requires that `libhighs` is locatable by MiniZinc.
 
#### Behavior when some solvers are missing
 
The module is designed to **degrade gracefully**:
 
- If the Python MiniZinc API cannot use a solver (e.g., `minizinc.Solver.lookup("gecode")`
  fails or `solve()` raises a DLL error), it prints a short message
  `[MiniZinc-Python] ... Falling back to CLI.` and tries to execute the same
  solver via the `minizinc` binary.
- If the binary also does not recognize that solver (for example, there is
  no driver with tag `gecode`), that solver is marked as failed with:
  - `returncode != 0`
  - `solved_binary = 0`
  - `makespan = inf`
 
During dataset generation:
 
- The columns corresponding to that solver (`{KEY}_Runtime_s`,
  `{KEY}_Makespan`, `{KEY}_Wall_s`) are filled with:
  - `"NA"` for runtimes and wall-clock time.
  - `"NA"` or `"inf"` for makespan, depending on context.
- Other solvers that do run successfully fill their metrics with numeric values,
  and `Winner_Key` is computed exclusively among solvers that actually solved
  the instance (`solved_binary == 1`).
- The experiment **does not stop** just because one or more solvers are missing.
 
This allows running the pipeline even in “partial” environments (for example,
only CBC/SCIP/HiGHS installed) without treating it as a failure.
 
#### Reproducing full vs. basic results
 
- **Basic, fully open-source setup**:
  - Install MiniZinc (bundle or brew) + Python `minizinc`.
  - Ensure at least the following solvers work:
    - CBC (`coin-bc`)
    - SCIP (`scip`)
    - HiGHS (`highs`)
  - The generated datasets will include columns for these solvers, and the
    training pipelines can use them directly or filter them via `--solvers`.
 
- **Full reproduction with proprietary solvers (CPLEX/Gurobi)**:
  - In addition to the above, install:
    - IBM CPLEX Studio and configure its DLL (or use a MiniZinc bundle that
      ships with CPLEX support).
    - Gurobi and configure a valid license.
  - Verify that `minizinc --solvers` lists both `cplex` and `gurobi`.
  - In this environment, the `CPLEX_DEF_*` and `GUROBI_DEF_*` columns in the
    CSV will contain valid metrics and can be included in training or analysis.
 
In all cases, if a specific solver is not available in the current environment,
the code will mark it as “not available” at the metrics level (NA/inf) without
aborting the pipeline or flooding the console with repeated error messages.

## Module API

You can also use the module programmatically:

```python
from src.data_generation.jssp_images import (
    load_config,
    prepare_academic_dataset,
    prepare_generated_dataset,
    convert_dataset_to_images,
)

# Load configuration
config = load_config("src/data_generation/jssp_images/config.yaml")

# Generate academic dataset
csv_path = prepare_academic_dataset(config)

# Convert to images (Text-to-Image: model.mzn + .dzn)
convert_dataset_to_images(
    csv_path=csv_path,
    cp_model_path=config.cp_model_path,
    target_size=128,
)
```

## Troubleshooting

### MiniZinc not found
```
ERROR: MiniZinc not found in PATH
```
**Solution**: Install MiniZinc and add it to your system PATH.

### No solvers available
```
ERROR: No usable solvers found
```
**Solution**: Install at least one MiniZinc solver (e.g., `minizinc --install gecode`).

### job-shop-lib import error
```
ImportError: Cannot import job_shop_lib
```
**Solution**: Install the library: `pip install job-shop-lib`

### Instance not found
```
ValueError: Failed to load benchmark instance 'ft06'
```
**Solution**: Ensure the instance name is correct and available in JSPLIB.

### Configuration file not found
```
FileNotFoundError: Configuration file not found
```
**Solution**: Ensure you're running from the project root and the config path is correct.

## Performance Notes

- **Academic mode**: Typically takes 1-5 minutes per instance (depends on solvers and time limits)
- **Generated mode**: Can take 30+ minutes for large configurations (multiple instances × time limits × seeds × solvers)
- **Image conversion**: Very fast (~1 second per instance)

## Customization

### Adding New Solvers

Edit `config.yaml` and add to `solver_candidates`:

```yaml
solver_candidates:
  - solver_id: "your-solver"
    key: "YOUR_SOLVER_KEY"
    type: "cp"  # or "mip"
    options: {}  # use the solver's default configuration (no custom heuristics)
```

### Changing Instance Sizes

Edit `config.yaml` under `generated.generation_cases`:

```yaml
generation_cases:
  - [15, 15, 10]  # 10 instances of 15x15
  - [20, 20, 5]   # 5 instances of 20x20
```

### Modifying Image Size

Either in `config.yaml`:
```yaml
image:
  target_size: 256
```

Or via CLI:
```bash
python -m src.data_generation.jssp_images.cli --mode academic --image-size 256
```

## Related Modules

- **JSSP Tensors**: `src/data_generation/jssp_tensors/` - Generates tensor representations instead of images
- **SAT Images**: `src/data_generation/sat_images/` - Similar pipeline for SAT problems
- **Training**: `training/jssp/train_images.py` - Uses the generated datasets for model training

## License

Part of the practica-invg project.