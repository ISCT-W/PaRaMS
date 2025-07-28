#!/usr/bin/env python3
"""
Path Configuration for PaRaMS Project

This module centralizes all path configurations for models and datasets.
Modify the constants below to match your actual storage directories.
"""

import os
from typing import Optional

# =============================================================================
# MAIN CONFIGURATION - MODIFY THESE PATHS TO MATCH YOUR SETUP
# =============================================================================

# Root directory where all models and data are stored
# Modify this to point to your actual storage location
MODEL_DATA_ROOT = "/gs/bs/tga-mdl/Wei_mdl/params"  

# Use environment variable if available
# MODEL_DATA_ROOT = os.environ.get('PARAMS_MODEL_ROOT', '/path/to/your/model_data_storage')

# =============================================================================
# DERIVED PATHS - These should not need modification
# =============================================================================

# Model checkpoint directories
CHECKPOINTS_ROOT = os.path.join(MODEL_DATA_ROOT, "checkpoints")
MODIFIED_MODELS_ROOT = os.path.join(MODEL_DATA_ROOT, "modified_models")

# Data directory
DATA_ROOT = os.path.join(MODEL_DATA_ROOT, "data")

# Results directory - moved to repository root
work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
RESULTS_ROOT = os.path.join(work_dir, "results")
EVALUATION_RESULTS_ROOT = os.path.join(RESULTS_ROOT, "evaluation")
MODIFICATION_RESULTS_ROOT = os.path.join(RESULTS_ROOT, "modification")

# Legacy logs root (deprecated, use RESULTS_ROOT instead)
LOGS_ROOT = RESULTS_ROOT

# =============================================================================
# PATH HELPER FUNCTIONS
# =============================================================================

def get_checkpoint_path(model: str, task: str, checkpoint_type: str = "finetuned") -> str:
    """
    Get the path to a model checkpoint.
    
    Args:
        model (str): Model architecture (e.g., 'ViT-B-32')
        task (str): Task name (e.g., 'Cars') or 'zeroshot' for pretrained
        checkpoint_type (str): Type of checkpoint ('finetuned' or 'zeroshot')
    
    Returns:
        str: Full path to the checkpoint file
    """
    if task == "zeroshot" or checkpoint_type == "zeroshot":
        return os.path.join(CHECKPOINTS_ROOT, model, "zeroshot.pt")
    else:
        return os.path.join(CHECKPOINTS_ROOT, model, task, f"{checkpoint_type}.pt")

def get_modified_model_path(model: str, task: str, protection_type: str, 
                          scaling_type: Optional[str] = None) -> str:
    """
    Get the path to a PaRaMS-protected model.
    
    Args:
        model (str): Model architecture (e.g., 'ViT-B-32')
        task (str): Task name (e.g., 'Cars')
        protection_type (str): Protection configuration (e.g., 'perm_nonsymmetric')
        scaling_type (str): Scaling type for filename (e.g., 'nonsymmetric')
    
    Returns:
        str: Full path to the protected model file
    """
    if scaling_type is None:
        scaling_type = protection_type.split('_')[-1] if '_' in protection_type else protection_type
    
    filename = f"{task}_PaRaMSed_{scaling_type}.pt"
    return os.path.join(MODIFIED_MODELS_ROOT, model, task, protection_type, filename)

def get_evaluation_log_path(
    merge_method: str,
    model: str, 
    victim_task: str,
    protection_type: str
) -> str:
    """
    Get the path for evaluation log files using the new naming convention.
    
    Args:
        merge_method (str): Merging method ('TA', 'TIES', 'AdaMerging', etc.)
        model (str): Model architecture ('ViT-B-32', 'ViT-B-16', 'ViT-L-14')
        victim_task (str): Victim task name ('Cars', 'MNIST', 'DTD', etc.)
        protection_type (str): PaRaMS protection type ('perm_nonsymmetric', etc.)
    
    Returns:
        str: Full path to the evaluation log file
        
    Example:
        >>> get_evaluation_log_path('TA', 'ViT-B-32', 'Cars', 'perm_nonsymmetric')
        '/path/to/repo/results/evaluation/TA_ViT-B-32_Cars_perm-nonsymmetric.log'
    """
    # Format protection type: replace underscores with dashes
    formatted_protection = protection_type.replace('_', '-')
    
    # Generate filename according to new naming rule
    filename = f"{merge_method}_{model}_{victim_task}_{formatted_protection}.log"
    
    return os.path.join(EVALUATION_RESULTS_ROOT, filename)

def get_modification_log_path(model: str, task: str, protection_type: str) -> str:
    """
    Get the path for modification log files using the new naming convention.
    
    Args:
        model (str): Model architecture
        task (str): Task name  
        protection_type (str): Protection type
        
    Returns:
        str: Full path to the modification log file
        
    Example:
        >>> get_modification_log_path('ViT-B-32', 'Cars', 'perm_nonsymmetric')
        '/path/to/repo/results/modification/ViT-B-32/Cars_perm-nonsymmetric.log'
    """
    formatted_protection = protection_type.replace('_', '-')
    
    # Create model subdirectory path
    model_dir = os.path.join(MODIFICATION_RESULTS_ROOT, model)
    
    # Generate filename: {victim}_{params_type}.log
    filename = f"{task}_{formatted_protection}.log"
    
    return os.path.join(model_dir, filename)

# Legacy function for backward compatibility
def get_log_path(model: str, task: str, experiment_name: Optional[str] = None) -> str:
    """
    Legacy log path function - deprecated.
    Use get_evaluation_log_path() or get_modification_log_path() instead.
    """
    if experiment_name:
        return os.path.join(LOGS_ROOT, model, task, experiment_name)
    else:
        return os.path.join(LOGS_ROOT, model, task)

def ensure_dir_exists(path: str) -> str:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path (str): Directory path to check/create
        
    Returns:
        str: The same path (for chaining)
    """
    os.makedirs(path, exist_ok=True)
    return path

def validate_paths() -> bool:
    """
    Validate that the configured paths are accessible.
    
    Returns:
        bool: True if all paths are valid, False otherwise
    """
    try:
        # Check if root directory exists or can be created
        ensure_dir_exists(MODEL_DATA_ROOT)
        ensure_dir_exists(CHECKPOINTS_ROOT)
        ensure_dir_exists(MODIFIED_MODELS_ROOT)
        ensure_dir_exists(DATA_ROOT)
        
        # Check results directories
        ensure_dir_exists(RESULTS_ROOT)
        ensure_dir_exists(EVALUATION_RESULTS_ROOT)
        ensure_dir_exists(MODIFICATION_RESULTS_ROOT)
        
        return True
    except Exception as e:
        print(f"❌ Path validation failed: {e}")
        print(f"Please check your MODEL_DATA_ROOT configuration: {MODEL_DATA_ROOT}")
        return False

# =============================================================================
# PATH VERIFICATION
# =============================================================================

def print_path_config():
    """Print current path configuration for debugging."""
    print("🗂️  PaRaMS Path Configuration:")
    print("=" * 50)
    print(f"Model & Data Root: {MODEL_DATA_ROOT}")
    print(f"Checkpoints:       {CHECKPOINTS_ROOT}")
    print(f"Modified Models:   {MODIFIED_MODELS_ROOT}")
    print(f"Data:              {DATA_ROOT}")
    print(f"Results Root:      {RESULTS_ROOT}")
    print(f"Evaluation Logs:   {EVALUATION_RESULTS_ROOT}")
    print(f"Modification Logs: {MODIFICATION_RESULTS_ROOT}")
    print("=" * 50)
    
    # Check accessibility
    paths_valid = validate_paths()
    if paths_valid:
        print("✅ All paths are accessible")
    else:
        print("❌ Some paths are not accessible")
        print("💡 Please update MODEL_DATA_ROOT in src/config/paths.py")
    
    return paths_valid

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Print current configuration
    print_path_config()
    
    # Example path generation
    print("\n📁 Example Paths:")
    print("-" * 30)
    
    # Checkpoint paths
    print("Checkpoints:")
    print(f"  Pretrained ViT-B-32: {get_checkpoint_path('ViT-B-32', 'zeroshot')}")
    print(f"  Cars finetuned:      {get_checkpoint_path('ViT-B-32', 'Cars')}")
    
    # Modified model paths
    print("\nModified Models:")
    print(f"  Cars nonsymmetric:   {get_modified_model_path('ViT-B-32', 'Cars', 'perm_nonsymmetric')}")
    print(f"  MNIST symmetric:     {get_modified_model_path('ViT-B-32', 'MNIST', 'perm_symmetric')}")
    
    # Log paths
    print("\nLogs:")
    print(f"  Cars evaluation:     {get_log_path('ViT-B-32', 'Cars', 'victim_eval_20240727')}")