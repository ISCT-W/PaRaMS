import os
import sys

import torch
import tqdm

from src.datasets.common import maybe_dictionarize, get_dataloader_shuffle
from src.datasets.registry import get_dataset
from src.task_vectors.utils import create_log_dir

work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
os.chdir(work_dir)
sys.path.append(work_dir)

from src.task_vectors.args import parse_arguments
from src.task_vectors.eval import eval_single_dataset_preprocess_head
from src.task_vectors.task_vectors import TaskVector
from src.task_vectors.heads import get_classification_head


def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)


def make_functional(mod):
    orig_params = tuple(mod.parameters())
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names


def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, initial_weights=None):
        super(ModelWrapper, self).__init__()
        self.initial_weights = initial_weights
        self.model = model

        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        features = self.model(images)
        return features


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


class AdaMerging(torch.nn.Module):
    def __init__(self, paramslist, model, names, exam_datasets, args):
        super(AdaMerging, self).__init__()
        self.paramslist = paramslist
        self.args = args
        self.model = model
        self.names = names
        self.pretrain_lambdas = torch.ones(len(paramslist[0]), 1)
        prior = 0.3
        self.lambdas_raw = torch.nn.Parameter(torch.ones(len(paramslist[0]), len(paramslist) - 1) * prior)

        self.classifier = []
        for dataset_name in exam_datasets:
            classification_head = get_classification_head(self.args, dataset_name)
            layer_name = 'classifier_{}'.format(dataset_name)
            self.add_module(layer_name, classification_head.to(self.args.device))
            self.classifier.append(layer_name)

    def lambdas(self):
        task_lambdas = torch.clamp(self.lambdas_raw, min=0.0, max=1.0)
        lambdass = torch.cat((self.pretrain_lambdas, task_lambdas), 1)
        return lambdass

    def collect_trainable_params(self):
        return [self.lambdas_raw]

    def get_classification_head(self, dataset_name):
        layer_name = 'classifier_{}'.format(dataset_name)
        return getattr(self, layer_name)

    def get_image_encoder(self):
        if hasattr(self, 'alpha'):
            if self.alpha.size()[0] == 1:  # task-wise merging
                params = tuple(
                    sum(tuple(pi * alphai for pi, alphai in zip(p, self.alpha[0].cpu())))
                    for j, p in enumerate(zip(*self.paramslist))
                )
            else:  # layer-wise merging
                params = tuple(
                    sum(tuple(pi * alphai for pi, alphai in zip(p, self.alpha[j].cpu())))
                    for j, p in enumerate(zip(*self.paramslist))
                )
        else:
            alph = self.lambdas()
            params = tuple(
                sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[j].cpu())))
                for j, p in enumerate(zip(*self.paramslist))
            )
        params = tuple(p.cuda(self.args.device) for p in params)
        load_weights(self.model, self.names, params)
        return self.model

    def forward(self, inp, dataset_name):
        if hasattr(self, 'alpha'):
            if self.alpha.size()[0] == 1:  # task-wise merging
                params = tuple(
                    sum(tuple(pi * alphai for pi, alphai in zip(p, self.alpha[0].cpu())))
                    for j, p in enumerate(zip(*self.paramslist))
                )
            else:  # layer-wise merging
                params = tuple(
                    sum(tuple(pi * alphai for pi, alphai in zip(p, self.alpha[j].cpu())))
                    for j, p in enumerate(zip(*self.paramslist))
                )
        else:
            alph = self.lambdas()
            params = tuple(
                sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[j].cpu())))
                for j, p in enumerate(zip(*self.paramslist))
            )
        params = tuple(p.cuda(self.args.device) for p in params)
        load_weights(self.model, self.names, params)
        feature = self.model(inp)
        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        out = classification_head(feature)
        return out


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

                pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'
                free_rider_checkpoint = f'checkpoints/{model}/{free_rider_task}/finetuned.pt'
                victim_task_checkpoint = f'checkpoints/{model}/{victim_task}/finetuned.pt'
                victim_PaRaMSed_checkpoint = f'modified_models/{model}/{victim_task}/perm_scaling/{victim_task}_PaRaMSed.pt'
                benign_task_vectors = [TaskVector(pretrained_checkpoint, victim_task_checkpoint),
                                       TaskVector(pretrained_checkpoint, free_rider_checkpoint)]
                PaRaMSed_task_vectors = [TaskVector(pretrained_checkpoint, victim_PaRaMSed_checkpoint),
                                         TaskVector(pretrained_checkpoint, free_rider_checkpoint)]

                pretrained_encoder = torch.load(pretrained_checkpoint)
                pretrained_model_dic = pretrained_encoder.state_dict()

                log = create_log_dir(args.logs_path, f'LayerWiseAdaMerging_log.txt')
                args.log = log
                model = ModelWrapper(pretrained_encoder, cur_tasks)
                model = model.to(args.device)
                _, names = make_functional(model)
                benign_param_list = []
                benign_param_list += [tuple(v.detach().requires_grad_().cpu() for _, v in pretrained_model_dic.items())]
                benign_param_list += [tuple(v.detach().requires_grad_().cpu() for _, v in tv.vector.items()) for i, tv
                                      in
                                      enumerate(benign_task_vectors)]

                PaRaMSed_param_list = []
                PaRaMSed_param_list += [
                    tuple(v.detach().requires_grad_().cpu() for _, v in pretrained_model_dic.items())]
                PaRaMSed_param_list += [tuple(v.detach().requires_grad_().cpu() for _, v in tv.vector.items()) for i, tv
                                        in
                                        enumerate(PaRaMSed_task_vectors)]

                torch.cuda.empty_cache()
                benign_adamerging_mtl_model = AdaMerging(benign_param_list, model, names, cur_tasks)
                PaRaMSed_adamerging_mtl_model = AdaMerging(PaRaMSed_param_list, model, names, cur_tasks)

                for training_mtl_model in [benign_adamerging_mtl_model, PaRaMSed_adamerging_mtl_model]:
                    log.info('Training AdaMerging on first: benign and second: PaRaMSed.')
                    print('init benign lambda:')
                    print(training_mtl_model.lambdas())
                    print('collect_trainable_params:')
                    print(list(training_mtl_model.collect_trainable_params()))

                    epochs = 20
                    optimizer = torch.optim.Adam(training_mtl_model.collect_trainable_params(), lr=1e-3,
                                                 betas=(0.9, 0.999),
                                                 weight_decay=0.)

                    Total_ACC = 0.
                    for dataset_name in cur_tasks:
                        image_encoder = training_mtl_model.get_image_encoder()
                        classification_head = training_mtl_model.get_classification_head(dataset_name)
                        metrics = eval_single_dataset_preprocess_head(image_encoder, classification_head, dataset_name,
                                                                      args)
                        Total_ACC += metrics['top1']
                        log.info('Eval: init: ' + ' dataset: ' + str(dataset_name) + ' ACC: ' + str(metrics['top1']))
                    log.info('Eval: init: ' + ' Avg ACC:' + str(Total_ACC / len(cur_tasks)) + '\n')

                    for epoch in range(epochs):
                        losses = 0.
                        for dataset_name in cur_tasks:
                            dataset = get_dataset(dataset_name, pretrained_encoder.val_preprocess,
                                                  location=args.data_location,
                                                  batch_size=16)
                            dataloader = get_dataloader_shuffle(dataset)

                            for i, data in enumerate(tqdm.tqdm(dataloader)):
                                data = maybe_dictionarize(data)
                                x = data['images'].to(args.device)
                                y = data['labels'].to(args.device)

                                outputs = training_mtl_model(x, dataset_name)
                                loss = softmax_entropy(outputs).mean(0)
                                losses += loss

                                if i > 0:
                                    break

                        optimizer.zero_grad()
                        losses.backward()
                        optimizer.step()

                        print(list(training_mtl_model.lambdas().data))

                        if ((epoch + 1) % 500) == 0:
                            log.info(str(list(training_mtl_model.lambdas().data)))

                            Total_ACC = 0.
                            for dataset_name in cur_tasks:
                                image_encoder = training_mtl_model.get_image_encoder()
                                classification_head = training_mtl_model.get_classification_head(dataset_name)
                                metrics = eval_single_dataset_preprocess_head(image_encoder, classification_head,
                                                                              dataset_name,
                                                                              args)
                                Total_ACC += metrics['top1']
                                log.info(
                                    'Eval: Epoch: ' + str(epoch) + ' dataset: ' + str(dataset_name) + ' ACC: ' + str(
                                        metrics['top1']))
                            log.info(
                                'Eval: Epoch: ' + str(epoch) + ' Avg ACC:' + str(Total_ACC / len(cur_tasks)) + '\n')
