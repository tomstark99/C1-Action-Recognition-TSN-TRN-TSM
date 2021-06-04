import argparse
import logging

from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.optim import SGD
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from systems import EpicActionRecogintionShapleyClassifier

from models.esvs import Net
from datasets.pickle_dataset import PickleDataset
from frame_sampling import RandomSampler

from ipdb import launch_ipdb_on_exception

import plotly.graph_objects as go
import numpy as np

parser = argparse.ArgumentParser(
    description="Extract per-frame features from given dataset and backbone",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("pickle_dir", type=Path, help="Path to pickle file to save features")
parser.add_argument("--n_frames", type=int, default=8, help="Number of frames for 2D CNN backbone")
parser.add_argument("--save_params", type=Path, help="Save model parameters")
parser.add_argument("--test", type=bool, default=False, help="Set test mode to true or false on the RandomSampler")
parser.add_argument("--batch_size", type=int, default=128, help="mini-batch size of frame features to run through ")
parser.add_argument("--epoch", type=int, default=100, help="How many epochs to do over the dataset")
parser.add_argument("--log_interval", type=int, default=10, help="How many iterations between outputting running loss")
parser.add_argument("--save_fig", type=Path, help="Save a graph showing lr / loss")

def train_test_loader(dataset: PickleDatset, batch_size: int, val_split: float) -> Tuple[DataLoader, DataLoader]:

    idxs = list(range(len(dataset)))
    split = int(np.floor(val_split * len(dataset)))
    np.random.shuffle(idxs)

    train_idx, test_idx = idxs[split:], idxs[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    return DataLoader(dataset, batch_size=batch_size, sampler=train_sampler), DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float

    frame_sampler = RandomSampler(frame_count=args.n_frames, snippet_length=1, test=False)

    dataset = PickleDataset(args.pickle_dir, frame_sampler)
    trainloader, testloader = train_test_loader(dataset, args.batch_size, 0.3)

    model = Net(frame_count=args.n_frames).to(device)
    optimiser = Adam(model.parameters(), lr=3e-4)
    classifier = EpicActionRecogintionShapleyClassifier(
        model, 
        device,
        optimiser,
        trainloader,
        testloader,
        log_interval=args.log_interval
    )

    if args.test:
        test()
    else:
        train(classifier, args)
    

def test():
    return 0

def train(
    classifier: EpicActionRecogintionShapleyClassifier,
    args
):
    lr = 3e-4
    loss = []

    for epoch in range(args.epoch):
        loss.append(classifier.train())

    if args.save_params:
        classifier.save_parameters(args.save_params)

    loss = np.concatenate(loss)

    if args.save_fig:
        x = np.linspace(1, len(loss), len(loss), dtype=int)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x,
            y=loss
        ))

        fig.update_layout(
            xaxis_title='batched steps',
            yaxis_title='loss',
            title='training performance'
        )
        fig.update_yaxes(type='log')
        fig.write_image(args.save_fig)


if __name__ == "__main__":
    main(parser.parse_args())

