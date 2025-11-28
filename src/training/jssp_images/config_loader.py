"""
Configuration loader for JSSP Images CNN Training.

This module handles loading and validating training configuration from YAML files.
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate training configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing all configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required configuration keys are missing
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Validate required top-level keys
    required_keys = ['experiment', 'data', 'model', 'training', 'output']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    # Validate data configuration
    if 'csv_path' not in config['data']:
        raise ValueError("Missing 'csv_path' in data configuration")
    
    # Validate training task
    valid_tasks = ['classification', 'multilabel', 'regression']
    task = config['training'].get('task')
    if task not in valid_tasks:
        raise ValueError(f"Invalid task '{task}'. Must be one of: {valid_tasks}")
    
    return config


def get_default_config_path() -> str:
    """
    Get the default configuration file path.
    
    Returns:
        Path to the default config.yaml in this module's directory
    """
    module_dir = Path(__file__).parent
    return str(module_dir / 'config.yaml')

def merge_cli_args(config: Dict[str, Any], cli_args: Any) -> Dict[str, Any]:
    """
    Merge CLI arguments into configuration, with CLI taking precedence.
    """
    # 👇 convertir Namespace -> dict si hace falta
    if not isinstance(cli_args, dict):
        cli_args = vars(cli_args)

    cli_to_config_map = {
        'csv': ('data', 'csv_path'),
        'task': ('training', 'task'),
        'use_score': ('data', 'use_score'),
        'epochs': ('training', 'epochs'),
        'batch_size': ('training', 'batch_size'),
        'folds': ('training', 'k_folds'),
        'learning_rate': ('model', 'learning_rate'),
        'results_dir': ('output', 'results_dir'),
        'run_name': ('experiment', 'name'),
    }
    
    for cli_key, config_path in cli_to_config_map.items():
        if cli_key in cli_args and cli_args[cli_key] is not None:
            current = config
            for key in config_path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[config_path[-1]] = cli_args[cli_key]
    
    return config


def resolve_paths(config: Dict[str, Any], project_root: str = None) -> Dict[str, Any]:
    """
    Resolve relative paths in configuration to absolute paths.
    
    Args:
        config: Configuration dictionary
        project_root: Project root directory (defaults to current working directory)
        
    Returns:
        Configuration with resolved absolute paths
    """
    if project_root is None:
        project_root = os.getcwd()
    
    # Resolve CSV path
    csv_path = config['data']['csv_path']
    if not os.path.isabs(csv_path):
        config['data']['csv_path'] = os.path.normpath(
            os.path.join(project_root, csv_path)
        )
    
    # Resolve results directory
    results_dir = config['output']['results_dir']
    if not os.path.isabs(results_dir):
        config['output']['results_dir'] = os.path.normpath(
            os.path.join(project_root, results_dir)
        )
    
    return config