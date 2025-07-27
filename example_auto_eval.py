#!/usr/bin/env python3
"""
Example usage of the automated victim evaluation system.

This script demonstrates how to use the AutoVictimEvaluator to test
PaRaMS protection effectiveness across different scenarios.
"""

import sys
import os

# Add project root to path
work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(work_dir)

from src.eval_merging.auto_victim_eval import AutoVictimEvaluator

def example_single_victim_evaluation():
    """Example: Evaluate protection for a single victim task."""
    print("🎯 Example 1: Single victim task evaluation")
    print("="*50)
    
    # Create evaluator for Cars task on ViT-B-32
    evaluator = AutoVictimEvaluator(
        victim_task='Cars',
        model='ViT-B-32',
        scaling_factors=[0.4, 0.5, 0.6, 0.7],  # Test fewer scaling factors for speed
        protection_types=['nonsymmetric'],  # Test only strongest protection
        output_dir='example_logs'
    )
    
    # Run evaluation
    results = evaluator.run_full_evaluation()
    
    # Print summary
    evaluator.print_summary()
    
    return results

def example_multi_protection_evaluation():
    """Example: Compare different protection types."""
    print("\n🛡️  Example 2: Multi-protection comparison")
    print("="*50)
    
    # Create evaluator comparing all three protection types
    evaluator = AutoVictimEvaluator(
        victim_task='MNIST',
        model='ViT-B-32',
        scaling_factors=[0.5, 0.6, 0.7],  # Focus on mid-range scaling
        protection_types=['diagonal', 'symmetric', 'nonsymmetric'],  # Compare all types
        output_dir='example_logs'
    )
    
    # Run evaluation
    results = evaluator.run_full_evaluation()
    
    # Print summary
    evaluator.print_summary()
    
    return results

def example_custom_configuration():
    """Example: Custom evaluation configuration."""
    print("\n⚙️  Example 3: Custom configuration")
    print("="*50)
    
    # Create evaluator with custom settings
    evaluator = AutoVictimEvaluator(
        victim_task='DTD',
        model='ViT-B-16',  # Different model
        scaling_factors=[0.3, 0.5, 0.8],  # Custom scaling factors
        protection_types=['symmetric'],  # Only symmetric protection
        output_dir='custom_eval_logs'
    )
    
    # Run evaluation
    results = evaluator.run_full_evaluation()
    
    # Print summary
    evaluator.print_summary()
    
    return results

if __name__ == "__main__":
    print("🚀 PaRaMS Automated Victim Evaluation Examples")
    print("=" * 60)
    
    try:
        # Run example 1: Single victim evaluation
        results1 = example_single_victim_evaluation()
        
        # Run example 2: Multi-protection comparison
        results2 = example_multi_protection_evaluation()
        
        # Run example 3: Custom configuration
        results3 = example_custom_configuration()
        
        print("\n🎉 All examples completed successfully!")
        print("\nTo run individual evaluations from command line:")
        print("python src/eval_merging/auto_victim_eval.py --victim-task Cars --model ViT-B-32")
        print("python src/eval_merging/auto_victim_eval.py --victim-task MNIST --model ViT-B-32 --protection-types diagonal symmetric")
        
    except KeyboardInterrupt:
        print("\n⏹️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        print("Make sure you have:")
        print("1. Generated protected models using apply_PaRaMS.py")
        print("2. All required checkpoint files are available")
        print("3. Correct file paths in the configuration")