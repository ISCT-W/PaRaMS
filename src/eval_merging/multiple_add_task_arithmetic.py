import os
import sys

work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
os.chdir(work_dir)
sys.path.append(work_dir)

from src.task_vectors.args import parse_arguments
from src.task_vectors.eval import eval_single_dataset
from src.task_vectors.task_vectors import TaskVector
from src.task_vectors.utils import create_log_dir

args = parse_arguments()
args.data_location = 'data'
models = ['ViT-B-16', 'ViT-B-32', 'ViT-L-14']
for model in models:
    args.model = model
    all_tasks = ['MNIST', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'DTD']
    victim_task = 'MNIST'
    scaling_factor = 0.3

    other_tasks = all_tasks.copy()
    other_tasks.remove(victim_task)
    args.save = f'checkpoints/{model}'
    args.logs_path = f'logs/{model}/{victim_task}/1+{len(other_tasks)}/'
    os.makedirs(os.path.dirname(args.logs_path), exist_ok=True)

    pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'
    victim_task_checkpoint = f'checkpoints/{model}/{victim_task}/finetuned.pt'
    victim_PaRaMSed_checkpoint = f'modified_models/{model}/{victim_task}/perm_scaling/{victim_task}_PaRaMSed.pt'

    benign_task_vectors = [TaskVector(pretrained_checkpoint,
                                      f'checkpoints/{model}/{each_task}/finetuned.pt') for each_task in all_tasks]
    benign_task_vectors.append(TaskVector(pretrained_checkpoint, victim_task_checkpoint))
    PaRaMSed_task_vectors = [TaskVector(pretrained_checkpoint,
                                        f'checkpoints/{model}/{each_task}/finetuned.pt') for each_task in all_tasks]
    PaRaMSed_task_vectors.append(TaskVector(pretrained_checkpoint, victim_PaRaMSed_checkpoint))

    log = create_log_dir(args.logs_path, 'task_arithmetic.txt')

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
