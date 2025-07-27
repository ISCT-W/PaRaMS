#!/usr/bin/env python3
"""
Apply PaRaMS Protection to Models

This script demonstrates how to apply PaRaMS protection to trained models
using the new unified interface with different scaling variants.
"""

import sys
import os
import argparse

work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
os.chdir(work_dir)
sys.path.append(work_dir)

import numpy as np
import torch

import src.PaRaMS.PaRaMS as PaRaMS
from src.task_vectors.args import parse_arguments

def apply_params_protection(victim_task, model, enable_perm=True, enable_scaling=True, scaling_type='diagonal', output_dir='modified_models'):
    """Apply PaRaMS protection to a specific model and task"""
    
    layers = 12 if model in ['ViT-B-16', 'ViT-B-32'] else 24
    
    # Model paths
    victim_task_checkpoint = f'checkpoints/{model}/{victim_task}/finetuned.pt'
    pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'
    
    print(f"\n Applying PaRaMS:")
    print(f"   Task: {victim_task}")
    print(f"   Model: {model}")
    print(f"   Permutation: {enable_perm}")
    print(f"   Scaling: {enable_scaling} ({scaling_type})")
    
    # Load models
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
    protected_params = PaRaMS.params(
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
    
    # Create save path based on protection methods used
    save_path = f'{output_dir}/{model}/{victim_task}/'
    if enable_perm and enable_scaling:
        save_path += f'perm_{scaling_type}/'
    elif enable_perm:
        save_path += 'perm_only/'
    elif enable_scaling:
        save_path += f'{scaling_type}_only/'
    else:
        save_path += 'unprotected/'
    
    os.makedirs(save_path, exist_ok=True)
    model_name = f'{victim_task}_PaRaMSed_{scaling_type}.pt'
    save_model = os.path.join(save_path, model_name)
    torch.save(victim_encoder, save_model)
    print(f"Protected model saved to: {save_model}")
    
    return save_model


def parse_protection_args():
    """Parse command line arguments for PaRaMS protection"""
    parser = argparse.ArgumentParser(description='Apply PaRaMS protection to models')
    parser.add_argument('--tasks', nargs='+', default=['Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD'], 
                       help='List of tasks to protect (default: Cars)')
    parser.add_argument('--models', nargs='+', default=['ViT-B-32'],
                       help='List of models to protect (default: ViT-B-32)')
    parser.add_argument('--scaling-types', nargs='+', 
                       choices=['diagonal', 'symmetric', 'nonsymmetric'],
                       default=['nonsymmetric'],
                       help='Scaling types to apply (default: nonsymmetric)')
    parser.add_argument('--no-permutation', action='store_true',
                       help='Disable permutation protection', default=False)
    parser.add_argument('--no-scaling', action='store_true', 
                       help='Disable scaling protection', default=False)
    parser.add_argument('--output-dir', default='modified_models',
                       help='Output directory for protected models')
    return parser.parse_args()


if __name__ == "__main__":
    # Parse both task vector args and protection args
    protection_args = parse_protection_args()
    args = parse_arguments()
    
    # Configuration from command line arguments
    victim_tasks = protection_args.tasks
    models = protection_args.models
    scaling_types = protection_args.scaling_types
    perm_settings = [not protection_args.no_permutation]
    scaling_settings = [not protection_args.no_scaling]

    # Set up arguments for task vectors
    args.data_location = 'data'
    
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
                                saved_model = apply_params_protection(
                                    victim_task=victim_task,
                                    model=model,
                                    enable_perm=perm,
                                    enable_scaling=scaling,
                                    scaling_type=scaling_type,
                                    output_dir=protection_args.output_dir
                                )
                                total_models += 1
                            except Exception as e:
                                print(f"Error protecting {victim_task} with {scaling_type}: {e}")
                    else:
                        # When scaling is disabled, just apply permutation
                        try:
                            saved_model = apply_params_protection(
                                victim_task=victim_task,
                                model=model,
                                enable_perm=perm,
                                enable_scaling=False,
                                scaling_type='none',
                                output_dir=protection_args.output_dir
                            )
                            total_models += 1
                        except Exception as e:
                            print(f"Error protecting {victim_task} (perm only): {e}")
    
    print("\n" + "="*80)
    print(f"Protection completed! Generated {total_models} protected models")
    print("\nProtection variants created:")
    print("• Diagonal scaling: Fast, basic protection")
    print("• Symmetric scaling: Balanced protection and performance") 
    print("• Nonsymmetric scaling: Maximum protection against sophisticated attacks")
    print(f"\nAll models saved in {protection_args.output_dir}/ directory with descriptive names")
    print("You can now evaluate the protection effectiveness against different merging attacks!")
    print("\nUsage examples:")
    print("  python apply_PaRaMS.py --tasks Cars DTD --models ViT-B-32 --scaling-types diagonal")
    print("  python apply_PaRaMS.py --no-permutation --scaling-types symmetric nonsymmetric")
    print("  python apply_PaRaMS.py --output-dir my_protected_models")
