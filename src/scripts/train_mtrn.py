import argparse
import logging

from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from systems import EpicActionRecogintionShapleyClassifier

from models.esvs import _MTRN
from datasets.pickle_dataset import MultiPickleDataset
from frame_sampling import RandomSampler

from ipdb import launch_ipdb_on_exception
from tqdm import tqdm

import plotly.graph_objects as go
import numpy as np
import pickle

parser = argparse.ArgumentParser(
    description="Extract per-frame features from given dataset and backbone",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("pickle_dir", type=Path, help="Path to pickle file to save features")
parser.add_argument("results_pkl", type=Path, help="Path to save training results")
parser.add_argument("model_params_dir", type=Path, help="Path to save model parameters (not file name)")
parser.add_argument("--max_frames", type=int, default=8, help="max frames to train models for")
parser.add_argument("--batch_size", type=int, default=512, help="mini-batch size of frame features to run through ")
parser.add_argument("--epoch", type=int, default=100, help="How many epochs to do over the dataset")
# parser.add_argument("--test", type=bool, default=False, help="Set test mode to true or false on the RandomSampler")
# parser.add_argument("--log_interval", type=int, default=10, help="How many iterations between outputting running loss")
# parser.add_argument("--n_frames", type=int, help="Number of frames for 2D CNN backbone")
# parser.add_argument("--save_fig", type=Path, help="Save a graph showing lr / loss")

def no_collate(args):
    return args

def train_test_loader(dataset: MultiPickleDataset, batch_size: int, val_split: float) -> Tuple[DataLoader, DataLoader]:
    idxs = list(range(len(dataset)))
    split = int(np.floor(val_split * len(dataset)))
    np.random.shuffle(idxs)

    train_idx, test_idx = idxs[split:], idxs[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    return DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=no_collate), DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, collate_fn=no_collate)

def main(args):

    models = [_MTRN(frame_count=i) for i in range(1,args.max_frames+1)]
    optimisers = [Adam(m.parameters(), lr=1e-4) for m in models]
    frame_samplers = [RandomSampler(frame_count=m.frame_count, snippet_length=1, test=False) for m in models]

    dataset = MultiPickleDataset(args.pickle_dir)

    results = train(
        args.epoch,
        args.max_frames,
        args.batch_size,
        args.model_params_dir,
        dataset,
        models,
        optimisers,
        frame_samplers
    )

    with open(args.results_pkl, 'wb') as f:
        pickle.dump(results, f)
    

def test():
    return 0

def train(
    epochs: int,
    max_frames: int,
    batch_size: int,
    model_path: Path,
    dataset: Dataset,
    models: List[nn.Module],
    optimisers: List[Adam],
    frame_samplers: List[RandomSampler]
):
    assert len(models) == len(optimisers)
    assert len(models) == len(frame_samplers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float

    trainloader, testloader = train_test_loader(dataset, batch_size, 0.3)
    writer = SummaryWriter(f'datasets/epic/runs/epic_mtrn_max-frames:{max_frames}'f'_epochs:{epochs}'f'_batch_size{batch_size}')

    training_result = []
    testing_result = []

    for m, o, f in zip(models, optimisers, frame_samplers):
        classifier = EpicActionRecogintionShapleyClassifier(
            m,
            device,
            o,
            f,
            trainloader,
            testloader
        )

        model_train_results = {
            'running_loss': [],
            'running_acc1': [],
            'running_acc5': [],
            'epoch_loss': [],
            'epoch_acc1': [],
            'epoch_acc5': []
        }
        model_test_results = {
            'running_loss': [],
            'running_acc1': [],
            'running_acc5': [],
            'epoch_loss': [],
            'epoch_acc1': [],
            'epoch_acc5': []
        }

        for epoch in tqdm(
            range(epochs),
            unit=" epoch",
            dynamic_ncols=True
        ):

            train_result = classifier.train_step()

            epoch_loss = sum(train_result[f'{m.frame_count}_loss']) / len(trainloader)
            epoch_acc1 = sum(train_result[f'{m.frame_count}_acc1']) / len(trainloader)
            epoch_acc5 = sum(train_result[f'{m.frame_count}_acc5']) / len(trainloader)

            model_train_results['running_loss'].append(train_result[f'{m.frame_count}_loss'])
            model_train_results['running_acc1'].append(train_result[f'{m.frame_count}_acc1'])
            model_train_results['running_acc5'].append(train_result[f'{m.frame_count}_acc5'])
            model_train_results['epoch_loss'].append(epoch_loss)
            model_train_results['epoch_acc1'].append(epoch_acc1)
            model_train_results['epoch_acc5'].append(epoch_acc5)

            writer.add_scalar(f'training loss frames={m.frame_count}', epoch_loss, epoch)
            writer.add_scalars('combined training loss', {f'loss frames={m.frame_count}': epoch_loss}, epoch)
            writer.add_scalars(f'training accuracy frames={m.frame_count}', {'acc1': epoch_acc1, 'acc5': epoch_acc5}, epoch)
            writer.add_scalars('combined training accuracy', {f'acc1 frames={m.frame_count}': epoch_acc1, f'acc5 frames={m.frame_count}': epoch_acc5}, epoch)

            test_result = classifier.train_step()

            epoch_loss_ = sum(test_result[f'{m.frame_count}_loss']) / len(testloader)
            epoch_acc1_ = sum(test_result[f'{m.frame_count}_acc1']) / len(testloader)
            epoch_acc5_ = sum(test_result[f'{m.frame_count}_acc5']) / len(testloader)

            model_test_results['running_loss'].append(test_result[f'{m.frame_count}_loss'])
            model_test_results['running_acc1'].append(test_result[f'{m.frame_count}_acc1'])
            model_test_results['running_acc5'].append(test_result[f'{m.frame_count}_acc5'])
            model_test_results['epoch_loss'].append(epoch_loss_)
            model_test_results['epoch_acc1'].append(epoch_acc1_)
            model_test_results['epoch_acc5'].append(epoch_acc5_)

            writer.add_scalar(f'testing loss frames={m.frame_count}', epoch_loss_, epoch)
            writer.add_scalars('combined testing loss', {f'loss frames={m.frame_count}': epoch_loss_}, epoch)
            writer.add_scalars(f'testing accuracy frames={m.frame_count}', {'acc1': epoch_acc1_, 'acc5': epoch_acc5_}, epoch)
            writer.add_scalars('combined testing accuracy', {f'acc1 frames={m.frame_count}': epoch_acc1_, f'acc5 frames={m.frame_count}': epoch_acc5_}, epoch)

        training_result.append(model_train_results)
        testing_result.append(model_test_results)

        classifier.save_parameters(model_path / f'mtrn-frames={m.frame_count}.pt')
    
    return {'training': training_result, 'testing': testing_result}

    # if args.save_params:
    #     classifier.save_parameters(args.save_params)

    # loss = np.concatenate(loss)

    # if args.save_fig:
    #     x = np.linspace(1, len(loss), len(loss), dtype=int)

    #     fig = go.Figure()

    #     fig.add_trace(go.Scatter(
    #         x=x,
    #         y=loss
    #     ))

    #     fig.update_layout(
    #         xaxis_title='batched steps',
    #         yaxis_title='loss',
    #         title='training performance'
    #     )
    #     fig.update_yaxes(type='log')
    #     fig.write_image(args.save_fig)


if __name__ == "__main__":
    main(parser.parse_args())

