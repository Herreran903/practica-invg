# SAT Images Data Generation

This module generates datasets for SAT (Boolean Satisfiability) problems with grayscale image representations from ASlib scenarios. It processes algorithm performance data and converts SAT instance files to images for machine learning applications.

## Overview

The SAT Images generator processes ASlib (Algorithm Selection Library) scenarios to create datasets suitable for algorithm selection and performance prediction tasks. The pipeline:

1. **Loads ASlib scenario** (algorithm_runs.arff and description.txt)
2. **Parses solver performance data** (runtimes and statuses)
3. **Pivots data by instance×solver** and identifies best solver per instance
4. **Resolves paths to raw instance files** (CNF, XCSP, DZN, etc.)
5. **Generates ground truth CSV** with solver performance metrics
6. **Converts instances to grayscale images** (128×128 .npy files)

## Key Difference from JSSP Modules

While JSSP modules **generate** problem instances, SAT Images **processes existing benchmarks** from ASlib scenarios. It focuses on:
- Parsing ARFF files (ASlib format)
- Handling multiple SAT solvers and their performance data
- Resolving complex instance path structures
- Converting various SAT file formats (CNF, XCSP, etc.) to images

## Directory Structure

```
src/data_generation/sat_images/
├── config.yaml                      # Central configuration file
├── config_loader.py                 # Configuration management utilities
├── aslib_parser.py                  # ASlib ARFF and description.txt parsing
├── instance_resolver.py             # Instance file path resolution
├── prepare_aslib_dataset.py         # ASlib scenario processing pipeline
├── image_converter.py               # SAT file to grayscale image conversion
├── cli.py                           # Command-line interface
├── __init__.py                      # Module exports
└── README.md                        # This file
```

## Input Requirements

### ASlib Scenario
An ASlib scenario directory must contain:
- **algorithm_runs.arff**: Solver performance data (required)
- **description.txt**: Scenario metadata including timeout (optional)

### Instance Files
Directory containing raw SAT instance files:
- **CNF files**: DIMACS CNF format
- **XCSP files**: XML-based constraint format
- **DZN files**: MiniZinc data format
- Other text-based formats

## Output Structure

```
data/sat/datasets/sat_cnn_data_gen/
├── ground_truth_aslib.csv          # Ground truth with solver metrics
└── images/
    └── *__<hash>.npy               # Grayscale images (128×128, float32)
```

### CSV Columns

- `Instance_Id`: Original instance identifier from ASlib
- `Instance_Name`: Base filename without extension
- `Raw_Text_Path`: Absolute path to raw instance file
- `Time_Limit_s`: Timeout used for this scenario
- `Winner_Key`: Best performing solver for this instance
- `{SOLVER}_Runtime_s`: Runtime for each solver (seconds)
- `{SOLVER}_Status`: Status for each solver (OK, TIMEOUT, etc.)
- `Image_Npy_Path`: Path to the generated image file

### Image Format

Images are stored as NumPy arrays (.npy files):
- **Shape**: (128, 128) by default (configurable)
- **Data type**: float32
- **Content**: Raw byte/ASCII intensities from the instance file, reshaped and resized
- **Normalization**: None by default (paper-like). Optional z-score via CLI flag `--normalize` or by passing `normalize=True` to [`sat_images.image_converter.convert_dataset_to_images()`](src/data_generation/sat_images/image_converter.py:84)
- **Visualization**: To reproduce paper-like figures, display with `cmap="gray"` and fixed range `vmin=0, vmax=255` (e.g., matplotlib)
- **Naming**: `<instance_name>__<hash>.npy` (hash prevents collisions)

## Configuration

All parameters are centralized in [`config.yaml`](config.yaml). Key sections:

### Output Directories
```yaml
output:
  base_dir: "data/sat/datasets"
  default_output_dir: "sat_cnn_data_gen"
```

### Image Parameters
```yaml
image:
  target_size: 128  # Size of square grayscale images
```

### ASlib Parameters
```yaml
aslib:
  default_timeout_s: 5000.0  # Default if not in description.txt
  
  # Prefix mapping for resolving instance paths
  prefix_map:
    "SAT-Race-2010-CNF": "Application SAT+UNSAT/SAT Race 2010"
    "SATCompetition2007": "SATCompetition2007/industrial"
    # ... more mappings
```

## Usage

### From Project Root

**Basic usage:**
```bash
python -m src.data_generation.sat_images.cli \
  --scenario-dir data/sat/aslib/sc2012-application \
  --instances-dir data/sat/instances/sc2012-application
```

**With custom output directory:**
```bash
python -m src.data_generation.sat_images.cli \
  --scenario-dir data/sat/aslib/sc2012-application \
  --instances-dir data/sat/instances/sc2012-application \
  --output-dir data/sat/datasets/my_custom_output
```

**With custom image size:**
```bash
python -m src.data_generation.sat_images.cli \
  --scenario-dir data/sat/aslib/sc2012-application \
  --instances-dir data/sat/instances/sc2012-application \
  --image-size 256
```

**Skip ASlib processing (only convert existing dataset):**
```bash
python -m src.data_generation.sat_images.cli \
  --skip-aslib \
  --csv data/sat/datasets/sat_cnn_data_gen/ground_truth_aslib.csv \
  --instances-dir data/sat/instances/sc2012-application
```

**With instance mapping CSV:**
```bash
python -m src.data_generation.sat_images.cli \
  --scenario-dir data/sat/aslib/sc2012-application \
  --instances-dir data/sat/instances/sc2012-application \
  --instance-map-csv data/sat/instance_mapping.csv
```

### Instance decompression (recommended)

When your ASlib instances are still compressed (`*.cnf.gz`, `*.bz2`, `*.xz`, `*.zip`, `*.lzma`), the standard flow now prefers plaintext files only. You can decompress everything recursively in two ways:

1) Python CLI (safe default, handles .zip multi-member archives)

- Script: [`scripts/decompress_instances.py`](scripts/decompress_instances.py:1)
- Dry run (no changes):
  ```bash
  python3 scripts/decompress_instances.py data/sat/instances/sc2012-application --dry-run
  ```
- Decompress (keep archives):
  ```bash
  python3 scripts/decompress_instances.py data/sat/instances/sc2012-application
  ```
- Decompress and delete archives after success:
  ```bash
  python3 scripts/decompress_instances.py data/sat/instances/sc2012-application --delete
  ```
- Overwrite existing plaintext targets (use with care):
  ```bash
  python3 scripts/decompress_instances.py data/sat/instances/sc2012-application --overwrite
  ```

Safety notes:
- By default, existing targets like `foo.cnf` will NOT be overwritten.
- Use `--dry-run` first to preview actions.
- For `.zip` files with multiple members, the script can extract the full internal tree with `--extract-zip-multi`.

2) One-liner find + native tools (fast, shell-based)

- Count compressed files:
  ```bash
  find "data/sat/instances/sc2012-application" -type f \
    -regex '.*\.\(gz\|bz2\|xz\|zip\|lzma\)$' | wc -l
  ```
- Decompress .gz (keep archives; add `-d` instead of `-dk` to delete):
  ```bash
  find "data/sat/instances/sc2012-application" -type f -name '*.gz' -print0 \
    | xargs -0 -I{} gzip -dk "{}"
  ```
- Decompress .bz2 (keep archives):
  ```bash
  find "data/sat/instances/sc2012-application" -type f -name '*.bz2' -print0 \
    | xargs -0 -I{} bzip2 -dk "{}"
  ```
- Decompress .xz (keep archives):
  ```bash
  find "data/sat/instances/sc2012-application" -type f -name '*.xz' -print0 \
    | xargs -0 -I{} unxz -k "{}"
  ```
- Decompress .lzma (keep archives):
  ```bash
  find "data/sat/instances/sc2012-application" -type f -name '*.lzma' -print0 \
    | xargs -0 -I{} unlzma -k "{}"
  ```
- Extract .zip (never overwrite existing targets due to -n):
  ```bash
  find "data/sat/instances/sc2012-application" -type f -name '*.zip' -print0 \
    | xargs -0 -I{} unzip -n "{}" -d "$(dirname "{}")"
  ```

Precautions:
- The flags `-k` (keep) and `-n` (no overwrite) protect existing plaintext (e.g., `foo.cnf`). Remove `-k` (or use `-d` for gzip/bzip2) if you want to delete archives afterward.
- For `.zip` with many files, prefer the Python script for better control.

After decompression, the CLI will show diagnostics and should report few or zero compressed files remaining. If many remain, rerun with the Python script’s `--delete` to clean up.

### Typical Workflow

1. **Prepare ASlib scenario and instances**:
   - Extract ASlib scenario to `data/sat/aslib/<scenario_name>/`
   - Extract instance files to `data/sat/instances/<scenario_name>/`

2. **Configure parameters** in `config.yaml` (optional)

3. **Run data generation**:
   ```bash
   python -m src.data_generation.sat_images.cli \
     --scenario-dir data/sat/aslib/sc2012-application \
     --instances-dir data/sat/instances/sc2012-application
   ```

4. **Check outputs**:
   - CSV: `data/sat/datasets/sat_cnn_data_gen/ground_truth_aslib.csv`
   - Images: `data/sat/datasets/sat_cnn_data_gen/images/*.npy`

5. **Use in training**:
   ```python
   import numpy as np
   import pandas as pd
   
   # Load dataset
   df = pd.read_csv('data/sat/datasets/sat_cnn_data_gen/ground_truth_aslib.csv')
   
   # Load an image
   image = np.load(df.iloc[0]['Image_Npy_Path'])
   print(f"Shape: {image.shape}")  # (128, 128)
   print(f"Winner: {df.iloc[0]['Winner_Key']}")
   ```

## Dependencies

Required Python packages:
- `numpy`: Array operations and image storage
- `pandas`: CSV and data handling
- `Pillow (PIL)`: Image resizing
- `PyYAML`: Configuration file parsing

Install dependencies:
```bash
pip install numpy pandas Pillow PyYAML
```

## Module API

You can also use the module programmatically:

```python
from src.data_generation.sat_images import (
    load_config,
    prepare_aslib_dataset,
    convert_dataset_to_images
)

# Load configuration
config = load_config("src/data_generation/sat_images/config.yaml")

# Process ASlib scenario
csv_path = prepare_aslib_dataset(
    scenario_dir="data/sat/aslib/sc2012-application",
    out_csv="data/sat/datasets/sat_cnn_data_gen/ground_truth_aslib.csv",
    instances_dir="data/sat/instances/sc2012-application",
    timeout_s=5000.0
)

# Convert to images
convert_dataset_to_images(
    csv_path=csv_path,
    instances_root="data/sat/instances/sc2012-application",
    target_size=128,
    prefix_map=config.prefix_map
)
```

## ASlib Scenarios

Common ASlib scenarios for SAT:

| Scenario | Description | Typical Location |
|----------|-------------|------------------|
| **sc2012-application** | SAT Competition 2012 - Application track | `data/sat/aslib/sc2012-application` |
| **sc2012-hard-combinatorial** | SAT Competition 2012 - Hard Combinatorial | `data/sat/aslib/sc2012-hard-combinatorial` |
| **sc2012-random** | SAT Competition 2012 - Random track | `data/sat/aslib/sc2012-random` |

## Troubleshooting

### ARFF file not found
```
FileNotFoundError: algorithm_runs.arff not found
```
**Solution**: Ensure the scenario directory contains `algorithm_runs.arff`.

### Missing columns in ARFF
```
ValueError: Missing required columns
```
**Solution**: Verify the ARFF file has columns: instance_id, algorithm, runtime, runstatus.

### Instance files not found
```
Warning: Many instances have empty Raw_Text_Path
```
**Solution**: 
- Check `--instances-dir` points to correct location
- Verify instance files are extracted (not in .tar archives)
- Use `--instance-map-csv` for explicit path mapping
- Update `prefix_map` in config.yaml if needed

### Sparse ARFF format error
```
NotImplementedError: Sparse ARFF format not supported
```
**Solution**: This parser only handles dense ARFF format. Convert sparse ARFF to dense format first.

### Configuration file not found
```
FileNotFoundError: Configuration file not found
```
**Solution**: Ensure you're running from project root and config path is correct.

## Performance Notes

- **ASlib processing**: Fast (seconds to minutes depending on scenario size)
- **Image conversion**: ~0.1-0.5 seconds per instance
- **Total time**: Typically 5-30 minutes for a full scenario (depends on number of instances)

## Customization

### Adding New Prefix Mappings

Edit `config.yaml` to add mappings for your instance directory structure:

```yaml
aslib:
  prefix_map:
    "MyPrefix": "path/to/instances/subdirectory"
```

### Changing Image Size

Either in `config.yaml`:
```yaml
image:
  target_size: 256
```

Or via CLI:
```bash
python -m src.data_generation.sat_images.cli ... --image-size 256
```

### Custom Timeout

Override timeout from description.txt:
```bash
python -m src.data_generation.sat_images.cli ... --timeout 3600
```

## Comparison with JSSP Modules

| Aspect | sat_images | jssp_images/jssp_tensors |
|--------|------------|--------------------------|
| **Input** | ASlib scenarios (existing benchmarks) | Generated or JSPLIB instances |
| **Data source** | ARFF files with solver runs | MiniZinc solver execution |
| **Instance types** | CNF, XCSP, DZN (various) | DZN (MiniZinc format) |
| **Solver data** | Pre-computed (from ARFF) | Computed on-the-fly |
| **Path resolution** | Complex (prefix mapping) | Simple (direct paths) |
| **Use case** | Algorithm selection | Performance prediction |

## Related Modules

- **JSSP Images**: `src/data_generation/jssp_images/` - Generates JSSP datasets with grayscale images
- **JSSP Tensors**: `src/data_generation/jssp_tensors/` - Generates JSSP datasets with 2D tensors
- **Training**: `training/sat/train_images.py` - Uses generated SAT datasets for model training

## License

Part of the practica-invg project.