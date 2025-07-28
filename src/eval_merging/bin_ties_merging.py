import os
import sys

from src.eval_merging.bin_layerwise_adamerging_fromscratch import create_log_dir
from src.task_vectors.args import parse_arguments
from src.task_vectors.eval import eval_single_dataset
from ties_merging_utils import *
from src.config.paths import (
    get_checkpoint_path, get_modified_model_path, get_evaluation_log_path,
    ensure_dir_exists, print_path_config, validate_paths, DATA_ROOT
)

if __name__ == "__main__":
    # Validate paths before starting
    if not validate_paths():
        print(" Path validation failed. Please check your configuration in src/config/paths.py")
        exit(1)
    
    print_path_config()
    
    args = parse_arguments()
    args.data_location = DATA_ROOT
    exam_datasets_all = ['Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
    models = ['ViT-B-16', 'ViT-B-32', 'ViT-L-14']
    for victim_task in exam_datasets_all:
        for free_rider_task in exam_datasets_all:
            if victim_task == free_rider_task:
                continue
            cur_tasks = [victim_task, free_rider_task]
            for model in models:
                args.model = model
                args.save = get_checkpoint_path(model, 'temp').replace('/temp/finetuned.pt', '')  # Get base checkpoint dir
                
                # Use new evaluation log path with TIES naming
                protection_type = 'perm-scaling'
                log_file_path = get_evaluation_log_path('TIES', model, victim_task, protection_type)
                args.logs_path = os.path.dirname(log_file_path)
                ensure_dir_exists(args.logs_path)
                log = create_log_dir(args.logs_path, os.path.basename(log_file_path))
                args.log = log
                # Use centralized path configuration
                pretrained_checkpoint = get_checkpoint_path(model, 'zeroshot')
                free_rider_checkpoint = get_checkpoint_path(model, free_rider_task, 'finetuned')
                victim_task_checkpoint = get_checkpoint_path(model, victim_task, 'finetuned')
                victim_PaRaMSed_checkpoint = get_modified_model_path(model, victim_task, 'perm_scaling', 'scaling')
                victim_adaptive_checkpoint = f'modified_models/{model}/{victim_task}/perm_adaptive/PaRaMS_adaptive_{free_rider_task}.pt'

                benign_ft_checkpoints = [torch.load(victim_task_checkpoint).state_dict(),
                                         torch.load(free_rider_checkpoint).state_dict()]
                paramsed_ft_checkpoints = [torch.load(victim_PaRaMSed_checkpoint).state_dict(),
                                           torch.load(free_rider_checkpoint).state_dict()]
                adaptive_ft_checkpoints = [torch.laod(victim_adaptive_checkpoint).state_dict(),
                                           torch.load(free_rider_checkpoint).state_dict()]
                pretrain_state_dict = torch.load(pretrained_checkpoint).state_dict()

                checkpoints = [benign_ft_checkpoints, paramsed_ft_checkpoints, adaptive_ft_checkpoints]
                for ckpt in checkpoints:
                    log.info('*' * 20 + 'Starting benign, paramsed, adaptive evaluation.' + '*' * 20)
                    check_parameterNamesMatch(ckpt + [pretrain_state_dict])

                    remove_keys = []
                    flatten_ft_ckpts = torch.vstack([state_dict_to_vector(pt, remove_keys) for pt in ckpt])
                    flatten_pretrain = state_dict_to_vector(pretrain_state_dict, remove_keys)
                    task_vector_flatten = flatten_ft_ckpts - flatten_pretrain

                    K = 20
                    merge_func = "dis-sum"
                    scaling_factor = 0.8

                    merged_task_vector = ties_merging(task_vector_flatten, reset_thresh=K, merge_func=merge_func)
                    merged_ckpt = flatten_pretrain + scaling_factor * merged_task_vector
                    merged_stat_dict = vector_to_state_dict(merged_ckpt, pretrain_state_dict, remove_keys)

                    image_encoder = torch.load(pretrained_checkpoint)
                    image_encoder.load_state_dict(merged_stat_dict, strict=False)

                    for dataset in cur_tasks:
                        metrics = eval_single_dataset(image_encoder, dataset, args)
                        score = metrics['top1'] * 100
                        log.info(str(dataset) + f': {score:.2f}%')
