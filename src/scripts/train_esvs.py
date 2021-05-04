import argparse
import logging

from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset

from systems import EpicActionRecogintionShapleyClassifier

from models.esvs import Net
from datasets.pickle_dataset import PickleDataset
from frame_sampling import RandomSampler

from ipdb import launch_ipdb_on_exception

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description="Extract per-frame features from given dataset and backbone",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("pickle_dir", type=Path, help="Path to pickle file to save features")
parser.add_argument("--n_frames", type=int, default=8, help="Number of frames for 2D CNN backbone")
parser.add_argument("--test", type=bool, default=False, help="Set test mode to true or false on the RandomSampler")
parser.add_argument("--batch_size", type=int, default=128, help="mini-batch size of frame features to run through ")
parser.add_argument("--log_interval", type=int, default=10, help="How many iterations between outputting running loss")
parser.add_argument("--save_fig", type=Path, help="Save a graph showing lr / loss")

def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float

    frame_sampler = RandomSampler(frame_count=args.n_frames, snippet_length=1, test=False)

    dataset = PickleDataset(args.pickle_dir, frame_sampler)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = Net(frame_count=args.n_frames).to(device)

    lr = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    loss = []

    for l in lr:
        optimiser = SGD(model.parameters(), lr=l, momentum=0.9)
        classifier = EpicActionRecogintionShapleyClassifier(model, dataloader, optimiser, device, log_interval=args.log_interval)
        running_loss = 0.0
        for epoch in range(2):
            running_loss = classifier.train(epoch)

        loss.append(running_loss)

    if args.save_fig:
        assert len(lr) == len(loss)

        fig, ax = plt.subplots(figsize=(12,7))
        print(lr, loss)
        ax.set_xscale('log')
        ax.plot(lr, loss)
        fig.savefig(args.save_fig)

if __name__ == "__main__":
    main(parser.parse_args())

