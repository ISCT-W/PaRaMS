#!/usr/bin/env python3
"""
Automated Victim Task Merging Evaluation

This script automatically evaluates model merging between a specified victim task 
and all other available tasks. It supports different PaRaMS protection variants 
and logs comprehensive evaluation results.
"""

import os
import argparse
import json
import warnings
from datetime import datetime
from typing import List, Dict, Optional

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

from src.task_vectors.args import parse_arguments
from src.task_vectors.eval import eval_single_dataset
from src.task_vectors.task_vectors import TaskVector
from src.task_vectors.utils import create_log_dir
from src.config.paths import (
    get_checkpoint_path, get_modified_model_path, get_evaluation_log_path,
    ensure_dir_exists, print_path_config, validate_paths, DATA_ROOT
)


class AutoVictimEvaluator:
    """
    Automated evaluator for victim task protection against merging attacks.
    
    This class provides comprehensive evaluation of PaRaMS protection effectiveness
    by testing a specified victim task against all other available tasks under
    different scaling factors and protection configurations.
    """
    
    def __init__(self, victim_task: str, model: str, 
                 scaling_factors: Optional[List[float]] = None,
                 protection_types: Optional[List[str]] = None):
        """
        Initialize the automated victim evaluator.
        
        Args:
            victim_task (str): The victim task to protect and evaluate
            model (str): Model architecture (e.g., 'ViT-B-32', 'ViT-B-16', 'ViT-L-14')
            scaling_factors (List[float]): Task arithmetic scaling factors to test
            protection_types (List[str]): PaRaMS protection variants to evaluate
        """
        self.victim_task = victim_task
        self.model = model
        self.scaling_factors = scaling_factors or [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        self.protection_types = protection_types or ['nonsymmetric']
        
        # All available tasks (excluding victim)
        self.all_tasks = ['Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
        self.free_rider_tasks = [task for task in self.all_tasks if task != victim_task]
        
        # Initialize arguments
        self.args = parse_arguments()
        self.args.data_location = DATA_ROOT
        self.args.model = model
        self.args.save = get_checkpoint_path(model, 'temp').replace('/temp/finetuned.pt', '')  # Get base checkpoint dir
        
        # Create timestamp for this evaluation session
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Results storage
        self.results = {
            'victim_task': victim_task,
            'model': model,
            'timestamp': self.timestamp,
            'config': {
                'scaling_factors': self.scaling_factors,
                'protection_types': self.protection_types
            },
            'evaluations': {}
        }
        
    def _get_checkpoint_paths(self, free_rider_task: str, protection_type: str = None) -> Dict[str, str]:
        """Get file paths for different model checkpoints."""
        paths = {
            'victim_benign': get_checkpoint_path(self.model, self.victim_task, 'finetuned'),
            'free_rider': get_checkpoint_path(self.model, free_rider_task, 'finetuned'),
            'pretrained': get_checkpoint_path(self.model, 'zeroshot')
        }
        
        if protection_type:
            protection_config = f'perm_{protection_type}'
            paths['victim_protected'] = get_modified_model_path(
                self.model, self.victim_task, protection_config, protection_type
            )
        else:
            # Fallback to old naming for compatibility
            paths['victim_protected'] = get_modified_model_path(
                self.model, self.victim_task, 'perm_scaling', 'scaling'
            )
        
        return paths
    
    def _check_file_existence(self, paths: Dict[str, str]) -> bool:
        """Check if all required checkpoint files exist."""
        missing_files = []
        for name, path in paths.items():
            if not os.path.exists(path):
                missing_files.append(f"{name}: {path}")
        
        if missing_files:
            print(f"Missing checkpoint files:")
            for missing in missing_files:
                print(f"   {missing}")
            return False
        return True
    
    def evaluate_single_protection_type(self, protection_type: str) -> Dict:
        """
        Evaluate victim task protection for a specific protection type.
        
        Args:
            protection_type (str): Type of PaRaMS protection ('diagonal', 'symmetric', 'nonsymmetric')
            
        Returns:
            Dict: Evaluation results for this protection type
        """
        print(f"\n Evaluating {protection_type} protection for {self.victim_task}")
        print("="*60)
        
        # Create log directory using new evaluation log naming
        log_file_path = get_evaluation_log_path('AutoTA', self.model, self.victim_task, protection_type)
        log_dir = os.path.dirname(log_file_path)
        ensure_dir_exists(log_dir)
        
        # Create logger with new naming convention
        log = create_log_dir(log_dir, os.path.basename(log_file_path))
        log.info(f"Starting automated victim evaluation: {self.victim_task} vs All Tasks")
        log.info(f"Model: {self.model}, Protection: {protection_type}")
        log.info(f"Timestamp: {self.timestamp}")
        
        protection_results = {
            'protection_type': protection_type,
            'free_rider_evaluations': {}
        }
        
        for free_rider_task in self.free_rider_tasks:
            print(f"\n Testing against free-rider: {free_rider_task}")
            log.info(f"\n{'='*50}")
            log.info(f"Free-rider task: {free_rider_task}")
            log.info(f"{'='*50}")
            
            # Get checkpoint paths
            paths = self._get_checkpoint_paths(free_rider_task, protection_type)
            
            # Check file existence
            if not self._check_file_existence(paths):
                log.error(f"Missing checkpoint files for {free_rider_task}, skipping...")
                continue
            
            try:
                # Create task vectors
                victim_benign_tv = TaskVector(paths['pretrained'], paths['victim_benign'])
                victim_protected_tv = TaskVector(paths['pretrained'], paths['victim_protected'])
                free_rider_tv = TaskVector(paths['pretrained'], paths['free_rider'])
                
                # Load models for single task evaluation
                free_rider_encoder = torch.load(paths['free_rider'], weights_only=False).to(self.args.device)
                protected_victim_encoder = torch.load(paths['victim_protected'], weights_only=False).to(self.args.device)
                
                # Evaluate single task performance
                log.info(f" Single Task Performance:")
                protected_victim_single = eval_single_dataset(protected_victim_encoder, self.victim_task, self.args)
                free_rider_single = eval_single_dataset(free_rider_encoder, free_rider_task, self.args)
                
                log.info(f"Protected victim ({self.victim_task}): {protected_victim_single['top1'] * 100:.2f}%")
                log.info(f"Free-rider ({free_rider_task}): {free_rider_single['top1'] * 100:.2f}%")
                
                # Initialize free-rider results
                free_rider_results = {
                    'free_rider_task': free_rider_task,
                    'single_task_performance': {
                        'protected_victim': protected_victim_single['top1'] * 100,
                        'free_rider': free_rider_single['top1'] * 100
                    },
                    'scaling_factor_results': {}
                }
                
                # Test different scaling factors
                log.info(f"\nTask Arithmetic Evaluation:")
                for scaling_factor in self.scaling_factors:
                    log.info(f"\n{'*'*20} Scaling Factor: {scaling_factor} {'*'*20}")
                    
                    # Create merged task vectors
                    benign_merged_tv = sum([victim_benign_tv, free_rider_tv])
                    protected_merged_tv = sum([victim_protected_tv, free_rider_tv])
                    
                    # Apply to pretrained model
                    benign_merged_encoder = benign_merged_tv.apply_to(paths['pretrained'], scaling_coef=scaling_factor).to(self.args.device)
                    protected_merged_encoder = protected_merged_tv.apply_to(paths['pretrained'], scaling_coef=scaling_factor).to(self.args.device)
                    
                    # Evaluate merged models
                    # Benign merge results
                    benign_merged_victim = eval_single_dataset(benign_merged_encoder, self.victim_task, self.args)
                    benign_merged_fr = eval_single_dataset(benign_merged_encoder, free_rider_task, self.args)
                    
                    # Protected merge results  
                    protected_merged_victim = eval_single_dataset(protected_merged_encoder, self.victim_task, self.args)
                    protected_merged_fr = eval_single_dataset(protected_merged_encoder, free_rider_task, self.args)
                    
                    # Log results
                    log.info(f"Benign TA - Victim: {benign_merged_victim['top1'] * 100:.2f}%, Free-rider: {benign_merged_fr['top1'] * 100:.2f}%")
                    log.info(f"Protected TA - Victim: {protected_merged_victim['top1'] * 100:.2f}%, Free-rider: {protected_merged_fr['top1'] * 100:.2f}%")
                    
                    # Calculate protection effectiveness
                    victim_preservation = (protected_merged_victim['top1'] / benign_merged_victim['top1']) * 100 if benign_merged_victim['top1'] > 0 else 0
                    fr_disruption = 100 - (protected_merged_fr['top1'] / benign_merged_fr['top1']) * 100 if benign_merged_fr['top1'] > 0 else 100
                    
                    log.info(f"Victim preservation: {victim_preservation:.1f}%")
                    log.info(f"Free-rider disruption: {fr_disruption:.1f}%")
                    
                    # Store scaling factor results
                    free_rider_results['scaling_factor_results'][scaling_factor] = {
                        'benign_merge': {
                            'victim_acc': benign_merged_victim['top1'] * 100,
                            'free_rider_acc': benign_merged_fr['top1'] * 100
                        },
                        'protected_merge': {
                            'victim_acc': protected_merged_victim['top1'] * 100,
                            'free_rider_acc': protected_merged_fr['top1'] * 100
                        },
                        'protection_metrics': {
                            'victim_preservation_pct': victim_preservation,
                            'free_rider_disruption_pct': fr_disruption
                        }
                    }
                
                # Store free-rider results
                protection_results['free_rider_evaluations'][free_rider_task] = free_rider_results
                log.info(f" Completed evaluation against {free_rider_task}")
                
            except Exception as e:
                log.error(f"Error evaluating against {free_rider_task}: {str(e)}")
                print(f"Error with {free_rider_task}: {str(e)}")
                continue
        
        log.info(f"\nCompleted {protection_type} protection evaluation for {self.victim_task}")
        return protection_results
    
    def run_full_evaluation(self) -> Dict:
        """
        Run complete evaluation for all protection types.
        
        Returns:
            Dict: Complete evaluation results
        """
        # Validate paths before starting
        if not validate_paths():
            print(" Path validation failed. Please check your configuration in src/config/paths.py")
            return {}
        
        print(f"\n Starting automated victim evaluation")
        print_path_config()
        print(f"\nVictim Task: {self.victim_task}")
        print(f"Model: {self.model}")
        print(f"Protection Types: {', '.join(self.protection_types)}")
        print(f"Free-rider Tasks: {', '.join(self.free_rider_tasks)}")
        print(f"Scaling Factors: {self.scaling_factors}")
        
        for protection_type in self.protection_types:
            protection_results = self.evaluate_single_protection_type(protection_type)
            self.results['evaluations'][protection_type] = protection_results
        
        # Save comprehensive results in evaluation directory
        results_filename = f'AutoTA_{self.model}_{self.victim_task}_summary_{self.timestamp}.json'
        from src.config.paths import EVALUATION_RESULTS_ROOT
        results_file = os.path.join(EVALUATION_RESULTS_ROOT, results_filename)
        ensure_dir_exists(os.path.dirname(results_file))
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nEvaluation completed!")
        print(f"Results saved to: {results_file}")
        
        return self.results
    
    def print_summary(self):
        """Print a summary of evaluation results."""
        print(f"\nEVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Victim Task: {self.victim_task}")
        print(f"Model: {self.model}")
        print(f"Timestamp: {self.timestamp}")
        
        for protection_type in self.protection_types:
            if protection_type not in self.results['evaluations']:
                continue
                
            print(f"\n {protection_type.upper()} PROTECTION:")
            print(f"{'-'*40}")
            
            protection_results = self.results['evaluations'][protection_type]
            
            for fr_task, fr_results in protection_results['free_rider_evaluations'].items():
                print(f"\n vs {fr_task}:")
                
                # Get best scaling factor result (highest victim preservation)
                best_sf = None
                best_preservation = -1
                
                for sf, sf_results in fr_results['scaling_factor_results'].items():
                    preservation = sf_results['protection_metrics']['victim_preservation_pct']
                    if preservation > best_preservation:
                        best_preservation = preservation
                        best_sf = sf
                
                if best_sf is not None:
                    best_results = fr_results['scaling_factor_results'][best_sf]
                    print(f"   Best scaling factor: {best_sf}")
                    print(f"   Victim preservation: {best_results['protection_metrics']['victim_preservation_pct']:.1f}%")
                    print(f"   Free-rider disruption: {best_results['protection_metrics']['free_rider_disruption_pct']:.1f}%")


# Removed duplicate parser - now using the unified one from src.task_vectors.args


if __name__ == "__main__":
    args = parse_arguments()
    
    # Set defaults for auto evaluation if not provided
    if not args.victim_task:
        print("--victim-task is required for auto evaluation")
        exit(1)
    
    if not args.model:
        print("--model is required for auto evaluation")
        exit(1)
    
    if not args.protection_types:
        args.protection_types = ['nonsymmetric']
    
    if not args.scaling_factors:
        args.scaling_factors = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    # Create evaluator
    evaluator = AutoVictimEvaluator(
        victim_task=args.victim_task,
        model=args.model,
        scaling_factors=args.scaling_factors,
        protection_types=args.protection_types,
    )
    
    if not args.summary_only:
        # Run full evaluation
        results = evaluator.run_full_evaluation()
        
        # Print summary
        evaluator.print_summary()
    else:
        print("Summary-only mode not implemented yet. Please run full evaluation.")

"""
Example usage (now using unified argument parser): 
  # Run full evaluation for a victim task
  python src/eval_merging/auto_victim_eval_TA.py --victim-task Cars --model ViT-B-32

  # Define protection types & scaling factors
  python src/eval_merging/auto_victim_eval_TA.py \
      --victim-task Cars \
      --model ViT-B-32 \
      --protection-types symmetric nonsymmetric \
      --scaling-factors 0.3 0.8

  # All arguments are now unified in src.task_vectors.args
  # Output directory configured in src/config/paths.py
"""