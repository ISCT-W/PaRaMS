import os
import sys

from src.task_vectors.args import parse_arguments
from src.task_vectors.eval import eval_single_dataset
from src.task_vectors.task_vectors import TaskVector
from src.task_vectors.utils import create_log_dir
from src.config.paths import (
    get_checkpoint_path, get_modified_model_path, get_evaluation_log_path,
    ensure_dir_exists, print_path_config, validate_paths, DATA_ROOT
)

# Validate paths before starting
if not validate_paths():
    print(" Path validation failed. Please check your configuration in src/config/paths.py")
    exit(1)

print_path_config()

args = parse_arguments()
args.data_location = DATA_ROOT
models = ['ViT-B-16', 'ViT-B-32', 'ViT-L-14']
for model in models:
    args.model = model
    all_tasks = ['MNIST', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'DTD']
    victim_task = 'MNIST'
    scaling_factor = 0.3

    other_tasks = all_tasks.copy()
    other_tasks.remove(victim_task)
    args.save = get_checkpoint_path(model, 'temp').replace('/temp/finetuned.pt', '')  # Get base checkpoint dir
    
    # Use new evaluation log path with MultiTA naming
    protection_type = 'perm-scaling'  # Default protection type
    log_file_path = get_evaluation_log_path('MultiTA', model, victim_task, protection_type)
    args.logs_path = os.path.dirname(log_file_path)
    ensure_dir_exists(args.logs_path)

    pretrained_checkpoint = get_checkpoint_path(model, 'zeroshot')
    victim_task_checkpoint = get_checkpoint_path(model, victim_task, 'finetuned')
    victim_PaRaMSed_checkpoint = get_modified_model_path(model, victim_task, 'perm_scaling', 'scaling')

    benign_task_vectors = [TaskVector(pretrained_checkpoint,
                                      get_checkpoint_path(model, each_task, 'finetuned')) for each_task in all_tasks]
    benign_task_vectors.append(TaskVector(pretrained_checkpoint, victim_task_checkpoint))
    PaRaMSed_task_vectors = [TaskVector(pretrained_checkpoint,
                                        get_checkpoint_path(model, each_task, 'finetuned')) for each_task in all_tasks]
    PaRaMSed_task_vectors.append(TaskVector(pretrained_checkpoint, victim_PaRaMSed_checkpoint))

    log = create_log_dir(args.logs_path, os.path.basename(log_file_path))

    benign_ta_sum = sum(benign_task_vectors)
    PaRaMSed_ta_sum = sum(PaRaMSed_task_vectors)

    benign_image_encoder = benign_ta_sum.apply_to(pretrained_checkpoint, scaling_coef=scaling_factor)
    PaRaMSed_image_encoder = PaRaMSed_ta_sum.apply_to(pretrained_checkpoint, scaling_coef=scaling_factor)
    log.info('*' * 20 + 'scaling_coef:' + str(scaling_factor) + '*' * 20)

    for dataset in all_tasks:
        benign_metrics = eval_single_dataset(benign_image_encoder, dataset, args)
        PaRaMSed_metrics = eval_single_dataset(PaRaMSed_image_encoder, dataset, args)
        benign_value = benign_metrics.get('top1') * 100
        PaRaMSed_value = PaRaMSed_metrics.get('top1') * 100
        log.info('Benign TA on: ' + str(dataset) + ': ' + f'{benign_value:.2f}"%')
        log.info('PaRaMSed TA on: ' + str(dataset) + ': ' + f'{PaRaMSed_value:.2f}"%')
