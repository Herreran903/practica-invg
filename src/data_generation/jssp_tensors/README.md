# JSSP Tensors Data Generation

This module generates datasets for Job Shop Scheduling Problems (JSSP) with 2D tensor/matrix representations. It supports two modes: **academic** (using JSPLIB benchmark instances) and **generated** (creating random balanced instances).

## Overview

The JSSP Tensors generator converts JSSP problem instances into 2D matrices that can be used as input features for machine learning models. The pipeline:

1. **Loads or generates JSSP instances** (academic benchmarks or random instances)
2. **Converts instances to MiniZinc format** (.dzn files)
3. **Runs multiple solvers** to collect performance metrics
4. **Generates ground truth CSV** with solver runtimes and scores
5. **Converts instances to 2D tensors** (shape: num_jobs × num_machines, stored as .npy files)

## Key Difference from jssp_images

While `jssp_images` converts instances to fixed-size grayscale images (128×128), `jssp_tensors` creates **variable-size 2D matrices** where:
- Shape: `(num_jobs, num_machines)` - preserves original problem dimensions
- Each cell contains the **processing time** for that operation
- Optional z-score standardization: `(x - mean) / std`

This representation follows the **CONVJSSP style** and is more suitable for models that can handle variable input sizes.

## Directory Structure

```
src/data_generation/jssp_tensors/
├── config.yaml                      # Central configuration file
├── config_loader.py                 # Configuration management utilities
├── prepare_academic_dataset.py     # Academic mode pipeline (JSPLIB)
├── prepare_generated_dataset.py    # Generated mode pipeline (random instances)
├── tensor_converter.py             # .dzn to tensor conversion
├── cli.py                          # Command-line interface
├── __init__.py                     # Module exports
└── README.md                       # This file
```

**Note:** This module reuses `jssp_instance_utils.py` and `minizinc_solver.py` from `jssp_images` since the instance handling and solver logic is identical.

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
jsp_cnn_data_acad_tensors/  (or jssp_cnn_data_tensors/)
├── *.dzn                           # Instance files in MiniZinc format
├── ground_truth_*.csv              # Ground truth with solver metrics
└── images/
    └── *_tensor.npy                # 2D tensors (variable size, float32)
```

### Tensor Format

Tensors are stored as NumPy arrays (.npy files):
- **Shape**: `(num_jobs, num_machines)` - varies per instance
- **Data type**: float32
- **Content**: Processing times for each operation
- **Normalization**: Optional z-score (mean=0, std=1)

Example:
```python
import numpy as np

# Load a tensor
tensor = np.load('GEN_6x6_1_tensor.npy')
print(tensor.shape)  # (6, 6) for a 6×6 instance
print(tensor.dtype)  # float32
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
- `Tensor_Npy_Path`: Path to the tensor .npy file
- `Image_Npy_Path`: Alias of `Tensor_Npy_Path`, added so that training pipelines that expect an `Image_Npy_Path` column can directly reuse this CSV

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
- `Tensor_Npy_Path`: Path to the tensor .npy file
- `Image_Npy_Path`: Alias of `Tensor_Npy_Path`, added so that training pipelines that expect an `Image_Npy_Path` column can directly reuse this CSV

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
  academic_dir: "jsp_cnn_data_acad_tensors"
  generated_dir: "jssp_cnn_data_tensors"
```

### Tensor Parameters
```yaml
tensor:
  standardize: true  # Apply z-score normalization
```

### Academic Mode
```yaml
academic:
  instances: ["ft06", "ft10", "la01", "abz5"]
  time_limit_ms: 60000
  penalty_factor_k: 10.0
```

### Generated Mode
```yaml
generated:
  time_limits_ms: [5000, 30000, 60000]
  random_seeds: [1, 2, 3]
  generation_cases:
    - [4, 4, 5]   # 5 instances of 4×4
    - [6, 6, 5]   # 5 instances of 6×6
    - [8, 8, 5]   # 5 instances of 8×8
    - [10, 10, 5] # 5 instances of 10×10
```

## Usage

### From Project Root

**Academic Mode (JSPLIB benchmarks):**
```bash
python -m src.data_generation.jssp_tensors.cli --mode academic
```

**Generated Mode (random instances):**
```bash
python -m src.data_generation.jssp_tensors.cli --mode generated
```

**With custom configuration:**
```bash
python -m src.data_generation.jssp_tensors.cli --mode academic --config path/to/config.yaml
```

**Disable standardization:**
```bash
python -m src.data_generation.jssp_tensors.cli --mode generated --no-standardize
```

**Skip solver execution (only convert existing dataset):**
```bash
python -m src.data_generation.jssp_tensors.cli --mode academic --skip-solvers --csv data/jssp/datasets/jsp_cnn_data_acad_tensors/ground_truth_jsp_academic.csv
```

### Typical Workflow

1. **Configure parameters** in `config.yaml`
2. **Run data generation**:
   ```bash
   python -m src.data_generation.jssp_tensors.cli --mode generated
   ```
3. **Check outputs**:
   - CSV: `data/jssp/datasets/jssp_cnn_data_tensors/ground_truth_jsp_generated_dataset.csv`
   - Tensors: `data/jssp/datasets/jssp_cnn_data_tensors/images/*.npy`
4. **Use in training**:
   ```python
   import numpy as np
   import pandas as pd
   
   # Load dataset
   df = pd.read_csv('data/jssp/datasets/jssp_cnn_data_tensors/ground_truth_jsp_generated_dataset.csv')
   
   # Load a tensor
   tensor = np.load(df.iloc[0]['Tensor_Npy_Path'])
   print(f"Shape: {tensor.shape}")  # e.g., (6, 6)
   print(f"Mean: {tensor.mean():.3f}")  # ~0 if standardized
   print(f"Std: {tensor.std():.3f}")    # ~1 if standardized
   ```

## Dependencies

Required Python packages:
- `numpy`: Array operations and tensor storage
- `pandas`: CSV handling
- `PyYAML`: Configuration file parsing
- `job-shop-lib`: JSSP instance loading and generation

External dependencies:
- **MiniZinc**: Constraint programming solver framework
- **Solvers**: At least one solver (Gecode, Chuffed, CP-SAT, CBC, SCIP, HiGHS, CPLEX, or Gurobi)

Install Python dependencies:
```bash
pip install numpy pandas PyYAML job-shop-lib
```

Install MiniZinc:
- Download from: https://www.minizinc.org/
- Ensure `minizinc` is in your PATH

## Module API

You can also use the module programmatically:

```python
from src.data_generation.jssp_tensors import (
    load_config,
    prepare_academic_dataset,
    prepare_generated_dataset,
    convert_dataset_to_tensors
)

# Load configuration
config = load_config("src/data_generation/jssp_tensors/config.yaml")

# Generate academic dataset
csv_path = prepare_academic_dataset(config)

# Convert to tensors
convert_dataset_to_tensors(csv_path, standardize=True)
```

## Comparison: Tensors vs Images

| Aspect | jssp_tensors | jssp_images |
|--------|--------------|-------------|
| **Output shape** | Variable `(n_jobs, n_machines)` | Fixed `(128, 128)` |
| **Content** | Processing times (standardized) | Text encoding of concatenated MiniZinc model (`.mzn`) and instance (`.dzn`) |
| **Representation** | Direct problem encoding | Indirect text encoding |
| **Model compatibility** | Requires variable-size support | Works with fixed-size CNNs |
| **Information preservation** | Exact problem structure | Lossy compression |
| **Use case** | Models handling variable inputs | Standard CNN architectures |

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

### Standardization error
```
ValueError: Standard deviation is zero
```
**Solution**: This occurs when all processing times are identical. Either:
- Use `--no-standardize` flag
- Check your instance generation parameters

### Configuration file not found
```
FileNotFoundError: Configuration file not found
```
**Solution**: Ensure you're running from the project root and the config path is correct.

## Performance Notes

- **Academic mode**: Typically takes 1-5 minutes per instance (depends on solvers and time limits)
- **Generated mode**: Can take 30+ minutes for large configurations (multiple instances × time limits × seeds × solvers)
- **Tensor conversion**: Very fast (~0.1 seconds per instance)

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
  - [15, 15, 10]  # 10 instances of 15×15
  - [20, 20, 5]   # 5 instances of 20×20
```

### Disabling Standardization

Either in `config.yaml`:
```yaml
tensor:
  standardize: false
```

Or via CLI:
```bash
python -m src.data_generation.jssp_tensors.cli --mode academic --no-standardize
```

## Related Modules

- **JSSP Images**: `src/data_generation/jssp_images/` - Generates fixed-size grayscale images
- **SAT Images**: `src/data_generation/sat_images/` - Similar pipeline for SAT problems
- **Training**: `training/jssp/train_tensor.py` - Uses the generated tensor datasets for model training

## License

Part of the practica-invg project.