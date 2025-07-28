import os
import sys

import torch

from src.eval_merging.bin_layerwise_adamerging_fromscratch import create_log_dir
from src.task_vectors.args import parse_arguments
from src.task_vectors.eval import eval_single_dataset
from src.task_vectors.task_vectors import TaskVector
from src.config.paths import (
    get_checkpoint_path, get_modified_model_path, get_evaluation_log_path,
    ensure_dir_exists, print_path_config, validate_paths, DATA_ROOT
)

# Validate paths before starting
if not validate_paths():
    print(" Path validation failed. Please check your configuration in src/config/paths.py")
    exit(1)

# Print current path configuration
print_path_config()

args = parse_arguments()
args.data_location = DATA_ROOT
exam_datasets_all = ['Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
models = ['ViT-B-16', 'ViT-B-32', 'ViT-L-14']
scaling_factors = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for victim_task in exam_datasets_all:
    for free_rider_task in exam_datasets_all:
        if victim_task == free_rider_task:
            continue
        for model in models:
            args.model = model
            args.save = get_checkpoint_path(model, 'temp').replace('/temp/finetuned.pt', '')  # Get base checkpoint dir
            
            # Determine protection type from modified model path
            # Assume we're using perm_scaling for now (can be made configurable)
            protection_type = 'perm-scaling'
            
            # Use new evaluation log path with TA naming
            log_file_path = get_evaluation_log_path('TA', model, victim_task, protection_type)
            args.logs_path = os.path.dirname(log_file_path)
            ensure_dir_exists(args.logs_path)

            victim_task_checkpoint = get_checkpoint_path(model, victim_task, 'finetuned')
            victim_PaRaMSed_checkpoint = get_modified_model_path(model, victim_task, 'perm_scaling', 'scaling')
            free_rider_checkpoint = get_checkpoint_path(model, free_rider_task, 'finetuned')
            pretrained_checkpoint = get_checkpoint_path(model, 'zeroshot')

            log = create_log_dir(args.logs_path, os.path.basename(log_file_path))

            victim_benign_task_vector = TaskVector(pretrained_checkpoint, victim_task_checkpoint)
            victim_PaRaMSed_task_vector = TaskVector(pretrained_checkpoint, victim_PaRaMSed_checkpoint)
            fr_task_vector = TaskVector(pretrained_checkpoint, free_rider_checkpoint)

            ta_benign = sum([victim_benign_task_vector, fr_task_vector])
            ta_params = sum([victim_PaRaMSed_task_vector, fr_task_vector])

            fr_encoder = torch.load(free_rider_checkpoint)
            PaRaMSed_vic_encoder = torch.load(victim_PaRaMSed_checkpoint)

            # benign single
            log.info(f"free-rider: {free_rider_task}.")
            benign_vic = eval_single_dataset(PaRaMSed_vic_encoder, victim_task, args)
            log.info(f"Victim benign: {benign_vic['top1'] * 100:.2f}%")
            benign_fr = eval_single_dataset(fr_encoder, free_rider_task, args)
            log.info(f"free-rider benign: {benign_fr['top1'] * 100:.2f}%")

            for scaling_factor in scaling_factors:
                log.info('*' * 20 + 'scaling_coef:' + str(scaling_factor) + '*' * 20)
                benign_ta_encoder = ta_benign.apply_to(pretrained_checkpoint, scaling_coef=scaling_factor)
                params_ta_encoder = ta_params.apply_to(pretrained_checkpoint, scaling_coef=scaling_factor)

                # benign ta vic, benign ta fr
                benign_ta_vic = eval_single_dataset(benign_ta_encoder, victim_task, args)
                log.info(f"SC: {scaling_factor}. TA victim benign: {benign_ta_vic['top1'] * 100:.2f}%")
                benign_ta_fr = eval_single_dataset(benign_ta_encoder, free_rider_task, args)
                log.info(f"SC: {scaling_factor}. TA free-rider benign: {benign_ta_fr['top1'] * 100:.2f}%")

                # modified ta vic, modified ta fr (PaRaMS_personal, scaling, perm)
                params_ta_vic = eval_single_dataset(params_ta_encoder, victim_task, args)
                log.info(f"SC: {scaling_factor}. TA PaRaMSed victim: {params_ta_vic['top1'] * 100:.2f}%")
                params_ta_fr = eval_single_dataset(params_ta_encoder, free_rider_task, args)
                log.info(f"SC: {scaling_factor}. TA PaRaMSed free-rider: {params_ta_fr['top1'] * 100:.2f}%")

                log.info(f"Evaluation on Model {model} -- Victim: {victim_task}, "
                         f"free-rider: {free_rider_task} on scaling: {scaling_factor} finished.")


