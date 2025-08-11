import os
import argparse

import torch

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--perm",
        type=bool,
        default=True,
        help="Bool=False for ablation",
    )
    parser.add_argument(
        "--scale",
        type=bool,
        default=True,
        help="Bool=False for ablation",
    )
    parser.add_argument(
        "--victim_task",
        type=str,
        default=None,
        help='Define the victim task',
    )
    parser.add_argument(
        "--free_rider_task",
        type=str,
        default=None,
        help='Define the free-rider task',
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help='Baseline model',
    )
    parser.add_argument(
        "--scaling_coef",
        type=float,
        default=0.8,
        help='Scaling factor in task arithmetic',
    )
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT. "
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        type=lambda x: x.split(","),
        help="Which dataset(s) to patch on.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only."
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--ls",
        type=float,
        default=0.0,
        help="Label smoothing."
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default='/gscratch/efml/gamaga/.cache/open_clip',
        help='Directory for caching models from OpenCLIP'
    )
    
    # PaRaMS specific arguments
    parser.add_argument(
        "--scaling-type",
        type=str,
        choices=['diagonal', 'symmetric', 'nonsymmetric'],
        default='nonsymmetric',
        help='Type of attention scaling to apply (default: nonsymmetric)'
    )
    parser.add_argument(
        "--scaling-types",
        nargs='+',
        choices=['diagonal', 'symmetric', 'nonsymmetric'],
        default=None,
        help='Multiple scaling types to apply (overrides --scaling-type)'
    )
    parser.add_argument(
        "--tasks",
        nargs='+',
        default=None,
        help='List of tasks to process (for batch operations)'
    )
    parser.add_argument(
        "--models",
        nargs='+',
        default=None,
        help='List of models to process (for batch operations)'
    )
    parser.add_argument(
        "--no-permutation",
        action='store_true',
        help='Disable permutation protection',
        default=False
    )
    parser.add_argument(
        "--no-scaling",
        action='store_true',
        help='Disable scaling protection',
        default=False
    )
    
    # Auto evaluation specific arguments
    parser.add_argument(
        "--victim-task",
        type=str,
        choices=['Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD'],
        help='Victim task to evaluate (for auto evaluation)'
    )
    parser.add_argument(
        "--free-rider-task",
        type=str,
        choices=['Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD'],
        help='Free rider task for binary evaluation'
    )
    parser.add_argument(
        "--protection-types",
        nargs='+',
        choices=['diagonal', 'symmetric', 'nonsymmetric'],
        default=None,
        help='PaRaMS protection types to evaluate (for auto evaluation)'
    )
    parser.add_argument(
        "--scaling-factors",
        nargs='+',
        type=float,
        default=None,
        help='Task arithmetic scaling factors to test (for auto evaluation)'
    )
    parser.add_argument(
        "--summary-only",
        action='store_true',
        help='Only print summary without running evaluation',
        default=False
    )
    parser.add_argument(
        "--sparsity-ratio",
        type=float,
        default=0.2,
        help='Sparsity ratio for WANDA pruning (if not specified, uses original protected model without WANDA)'
    )
    parser.add_argument(
        "--use-recovered",
        action='store_true',
        help='Use recovered model (with _recovered suffix) instead of original protected model',
        default=False
    )
    parser.add_argument(
        "--recovery-method",
        type=str,
        choices=['rowwise', 'blockwise', 'svd'],
        default='svd',
        help='Attention recovery method: rowwise (for diagonal), blockwise (original), or svd (numerically stable, default)'
    )
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args
