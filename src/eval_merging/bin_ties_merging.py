import os
import sys

from src.eval_merging.bin_layerwise_adamerging_fromscratch import create_log_dir

work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
os.chdir(work_dir)
sys.path.append(work_dir)

from src.task_vectors.args import parse_arguments
from src.task_vectors.eval import eval_single_dataset
from ties_merging_utils import *

if __name__ == "__main__":
    args = parse_arguments()
    args.data_location = 'data'
    exam_datasets_all = ['Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
    models = ['ViT-B-16', 'ViT-B-32', 'ViT-L-14']
    for victim_task in exam_datasets_all:
        for free_rider_task in exam_datasets_all:
            if victim_task == free_rider_task:
                continue
            cur_tasks = [victim_task, free_rider_task]
            for model in models:
                args.model = model
                args.save = f'checkpoints/{model}'
                args.logs_path = f'logs/{model}/{victim_task}/free_rider_{free_rider_task}/'
                os.makedirs(os.path.dirname(args.logs_path), exist_ok=True)
                log = create_log_dir(args.logs_path, f'TiesMerging_log.txt')
                args.log = log
                # checkpoint
                pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'
                free_rider_checkpoint = f'checkpoints/{model}/{free_rider_task}/finetuned.pt'
                victim_task_checkpoint = f'checkpoints/{model}/{victim_task}/finetuned.pt'
                victim_PaRaMSed_checkpoint = f'modified_models/{model}/{victim_task}/perm_scaling/{victim_task}_PaRaMSed.pt'
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
