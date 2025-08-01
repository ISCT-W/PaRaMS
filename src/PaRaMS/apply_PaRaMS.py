#!/usr/bin/env python3
"""
Apply PaRaMS Protection to Models

This script demonstrates how to apply PaRaMS protection to trained models
using the new unified interface with different scaling variants.
"""

import sys
import os
import argparse
import warnings

import numpy as np
import torch

# Suppress torch.load FutureWarning about weights_only
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


# Add project root to path for module imports
import sys
from pathlib import Path

# Find project root - look for setup.py or requirements.txt as markers
current_path = Path(__file__).absolute()
project_root = None
for parent in [current_path.parent.parent.parent, Path.cwd()]:
    if (parent / 'setup.py').exists() or (parent / 'requirements.txt').exists():
        project_root = parent
        break

if project_root is None:
    # Fallback to 3 levels up from this file
    project_root = current_path.parent.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.PaRaMS.PaRaMS_algorithm import params as PaRaMS_params
from src.task_vectors.args import parse_arguments
from src.task_vectors.utils import create_log_dir
from src.config.paths import (
    get_checkpoint_path, get_modified_model_path, get_modification_log_path,
    ensure_dir_exists, print_path_config, validate_paths
)

def apply_params_protection(victim_task, model, enable_perm=True, enable_scaling=True, scaling_type='diagonal'):
    """Apply PaRaMS protection to a specific model and task"""
    
    layers = 12 if model in ['ViT-B-16', 'ViT-B-32'] else 24
    
    # Model paths using centralized path configuration
    victim_task_checkpoint = get_checkpoint_path(model, victim_task, 'finetuned')
    pretrained_checkpoint = get_checkpoint_path(model, 'zeroshot')
    
    # Determine protection type for logging
    if enable_perm and enable_scaling:
        protection_type = f'perm_{scaling_type}'
    elif enable_perm:
        protection_type = 'perm_only'
    elif enable_scaling:
        protection_type = f'{scaling_type}_only'
    else:
        protection_type = 'unprotected'
    
    # Create log file for this protection process
    log_path = get_modification_log_path(model, victim_task, protection_type)
    ensure_dir_exists(os.path.dirname(log_path))
    log = create_log_dir(os.path.dirname(log_path), os.path.basename(log_path))
    
    # Log and print protection info
    protection_info = f"Applying PaRaMS protection: {model}/{victim_task} with {scaling_type} (perm: {enable_perm})"
    log.info(protection_info)
    log.info(f"Source checkpoint: {victim_task_checkpoint}")
    log.info(f"Pretrained checkpoint: {pretrained_checkpoint}")
    log.info(f"Protection configuration: perm={enable_perm}, scaling={enable_scaling}, type={scaling_type}")
    
    # Load models with proper error handling
    victim_encoder = torch.load(victim_task_checkpoint)
    pretrained_encoder = torch.load(pretrained_checkpoint)
    
    # Configure scaling based on type
    if scaling_type == 'diagonal':
        scaling_config = {
            'scaling_type': 'diagonal',
            'scale_min': 0.05,
            'scale_max': 20.0,
            'mode': 'uniform'
        }
    elif scaling_type == 'symmetric':
        scaling_config = {
            'scaling_type': 'symmetric',
            'n_heads': 12 if layers == 12 else 16,  # Adjust heads for different models
            'scale_min': 0.05,
            'scale_max': 20.0
        }
    elif scaling_type == 'nonsymmetric':
        scaling_config = {
            'scaling_type': 'nonsymmetric',
            'n_heads': 12 if layers == 12 else 16,
            'scale_min': 0.1,
            'scale_max': 15.0,
            'eye_eps': 1e-3
        }
    elif scaling_type == 'none':
        # When scaling is disabled, use default config (won't be used anyway)
        scaling_config = {
            'scaling_type': 'diagonal',
            'scale_min': 0.05,
            'scale_max': 20.0,
            'mode': 'uniform'
        }
    else:
        raise ValueError(f"Unknown scaling_type: {scaling_type}")
    
    # Apply PaRaMS protection using the unified interface
    protected_params = PaRaMS_params(
        model_state_dict=victim_encoder.state_dict(),
        pretrained_state_dict=pretrained_encoder.state_dict() if enable_perm else None,
        num_layers=layers,
        enable_permutation=enable_perm,
        enable_scaling=enable_scaling,
        scaling_config=scaling_config,
        rng_seed=0
    )
    
    # Load the protected parameters back to the model
    victim_encoder.load_state_dict(protected_params)
    
    # Get save path using centralized path configuration
    save_model = get_modified_model_path(model, victim_task, protection_type, scaling_type)
    ensure_dir_exists(os.path.dirname(save_model))
    
    # Save model - prefer saving state_dict for better compatibility
    if hasattr(victim_encoder, '_state_dict'):
        # This is our wrapper class, save the state dict
        torch.save(victim_encoder._state_dict, save_model)
        log.info("Saved model as state_dict for better compatibility")
    else:
        # This is a full model object
        torch.save(victim_encoder, save_model)
    
    # Log and print completion info
    completion_info = f"Protected model saved: {save_model}"
    print(f"✅ {os.path.basename(save_model)}")
    log.info(completion_info)
    log.info(f"Protection completed successfully for {victim_task}")
    
    return save_model, log


# Removed duplicate parser - now using the unified one from src.task_vectors.args


if __name__ == "__main__":
    # Use the unified parser from task_vectors.args
    args = parse_arguments()
    
    # Configuration from command line arguments
    victim_tasks = args.tasks if args.tasks else ['Cars']
    models = args.models if args.models else ['ViT-B-32'] 
    
    # Handle scaling types - use scaling_types if provided, otherwise use scaling_type
    if args.scaling_types:
        scaling_types = args.scaling_types
    else:
        scaling_types = [args.scaling_type]
    
    perm_settings = [not args.no_permutation]
    scaling_settings = [not args.no_scaling]

    # Validate paths before starting
    if not validate_paths():
        print(" Path validation failed. Please check your configuration in src/config/paths.py")
        exit(1)
    
    # Print current path configuration
    print_path_config()
    
    # Set up data location from config if not provided
    if not hasattr(args, 'data_location') or not args.data_location:
        from src.config.paths import DATA_ROOT
        args.data_location = DATA_ROOT
    
    print("PaRaMS Protection Suite")
    print("Applying all three scaling variants for comprehensive protection")
    print("="*80)
    
    total_models = 0
    for victim_task in victim_tasks:
        for model in models:
            args.model = model
            args.save = f'checkpoints/{model}'
            
            for perm in perm_settings:
                for scaling in scaling_settings:
                    # Apply all three scaling types when scaling is enabled
                    if scaling:
                        for scaling_type in scaling_types:
                            try:
                                saved_model, log = apply_params_protection(
                                    victim_task=victim_task,
                                    model=model,
                                    enable_perm=perm,
                                    enable_scaling=scaling,
                                    scaling_type=scaling_type
                                )
                                total_models += 1
                            except Exception as e:
                                print(f" Error protecting {victim_task} with {scaling_type}: {e}")
                    else:
                        # When scaling is disabled, just apply permutation
                        try:
                            saved_model, log = apply_params_protection(
                                victim_task=victim_task,
                                model=model,
                                enable_perm=perm,
                                enable_scaling=False,
                                scaling_type='none'
                            )
                            total_models += 1
                        except Exception as e:
                            print(f" Error protecting {victim_task} (perm only): {e}")
    
    print("\n" + "="*80)
    print(f"Protection completed! Generated {total_models} protected models")
    print("\nProtection variants created:")
    print("• Diagonal scaling: Fast, basic protection")
    print("• Symmetric scaling: Balanced protection and performance") 
    print("• Nonsymmetric scaling: Maximum protection against sophisticated attacks")
    from src.config.paths import MODIFIED_MODELS_ROOT
    print(f"\nAll models saved in {MODIFIED_MODELS_ROOT}/ directory with descriptive names")
    print("You can now evaluate the protection effectiveness against different merging attacks!")
    print("\nUsage examples:")
    print("  python src/PaRaMS/apply_PaRaMS.py --tasks Cars DTD RESISC45 SVHN MNIST GTSRB EuroSAT --models ViT-B-32 --scaling-types diagonal")
    print("  python src/PaRaMS/apply_PaRaMS.py --no-permutation --scaling-types symmetric nonsymmetric")
    print("  python src/PaRaMS/apply_PaRaMS.py --output-dir my_protected_models")
