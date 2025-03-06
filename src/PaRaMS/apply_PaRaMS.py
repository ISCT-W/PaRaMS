import sys
import os

work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
os.chdir(work_dir)
sys.path.append(work_dir)

import numpy as np
import torch
from jax import random

import src.PaRaMS.PaRaMS as PaRaMS
from src.task_vectors.args import parse_arguments

if __name__ == "__main__":
    args = parse_arguments()
    victim_tasks = ['Cars']
    models = ['ViT-B-16', 'ViT-B-32', 'ViT-L-14']
    perm_setting = [True]
    scaling_setting = [True]

    for victim_task in victim_tasks:
        for model in models:
            layers = 12 if model in ['ViT-B-16', 'ViT-B-32'] else 24
            for perm in perm_setting:
                for scaling in scaling_setting:
                    if not perm or not scaling:
                        continue

                    args.model = model
                    args.data_location = 'data'
                    args.save = f'checkpoints/{model}'

                    victim_task_checkpoint = f'checkpoints/{model}/{victim_task}/finetuned.pt'
                    pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'

                    victim_encoder = torch.load(victim_task_checkpoint)
                    pretrained_encoder = torch.load(pretrained_checkpoint)

                    print(f"PaRaMS performing on {victim_task}, model: {model}...")

                    victim_params = {name: param.clone() for name, param in victim_encoder.state_dict().items() if
                                     'mlp.c' in name}
                    pt_params = {name: param.clone() for name, param in pretrained_encoder.state_dict().items() if
                                 'mlp.c' in name}
                    full_victim_params = {name: param.clone() for name, param in victim_encoder.state_dict().items()}

                    if perm:
                        perm_spec = PaRaMS.vit_permutation_spec_MLP(num_layers=layers)
                        rng = random.PRNGKey(0)
                        permutation, _, _, _ = PaRaMS.parameter_rearrangement(rng, perm_spec, victim_params, pt_params)
                        permuted_victim_MLP_params = {k: torch.tensor(np.array(v)) for k, v in
                                                      PaRaMS.apply_permutation(perm_spec, permutation,
                                                                               victim_params).items()}

                        full_victim_params.update(permuted_victim_MLP_params)
                        victim_encoder.load_state_dict(full_victim_params)

                    if scaling:
                        for layer in range(layers):
                            full_victim_params = PaRaMS.apply_attention_qkvw_scaling(full_victim_params,
                                                                                     layer_idx=layer,
                                                                                     scale_min=0.05,
                                                                                     scale_max=20)
                        victim_encoder.load_state_dict(full_victim_params)

                    save_path = f'modified_models/{model}/{victim_task}/'
                    if perm and scaling:
                        save_path += 'perm_scaling/'
                    elif perm:
                        save_path += 'perm/'
                    else:
                        save_path += 'scaling/'
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    model_name = f'{victim_task}_PaRaMSed.pt'
                    save_model = os.path.join(save_path, model_name)
                    torch.save(victim_encoder, save_model)
                    print(f"Model saved to {save_model}")
