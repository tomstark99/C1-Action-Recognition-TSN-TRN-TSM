import argparse
import logging

from gulpio2 import GulpDirectory
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset

from omegaconf import OmegaConf

from systems import EpicActionRecognitionSystem
from systems import EpicActionRecogintionDataModule

from features.feature_extractor import FeatureExtractor
from features.pkl import PickleFeatureWriter

parser = argparse.ArgumentParser(
    description="Extract per-frame features from given dataset and backbone",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("gulp_dir", type=Path, help="Path to gulp directory")
parser.add_argument("checkpoint", type=Path, help="Path to model checkpoint")
parser.add_argument("features_pickle", type=Path, help="Path to pickle file to save features")
parser.add_argument("--batch_size", type=int, default=128, help="Max frames to run through backbone 2D CNN at a time")
parser.add_argument("--feature_dim", type=int, default=256, help="Number of features expected from frame")

def main(args):

    # SETUP TORCH VARIABLES
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float

    # LOAD IN SAVED CHECKPOINT
    ckpt = torch.load(args.checkpoint, map_location='cpu')

    # CREATE CONFIG FROM CHECKPOINT
    cfg = OmegaConf.create(ckpt['hyper_parameters'])
    OmegaConf.set_struct(cfg, False)

    # SET GULP DIRECTORY
    cfg.data._root_gulp_dir = str(args.gulp_dir)

    # CREATE MODEL
    model = EpicActionRecognitionSystem(cfg)
    model.load_state_dict(ckpt['state_dict'])

    rgb_train = GulpDirectory(args.gulp_dir)

    extractor = FeatureExtractor(model.model.to(device), device, dtype, frame_batch_size=args.batch_size)
    total_instances = extract_features_to_pkl(
        rgb_train, extractor, args.features_pickle, args.feature_dim
    )

    print(f"extracted {total_instances} features.")

def extract_features_to_pkl(
    gulp_dir: GulpDirectory,
    feature_extractor: FeatureExtractor,
    features_path: Path,
    feature_dim: int
):
    total_instances = 0

    feature_writer = PickleFeatureWriter(features_path, features_dim=feature_dim)

    total_instances += feature_extractor.extract(gulp_dir, feature_writer)

    feature_writer.save()
    return total_instances

if __name__ == "__main__":
    main(parser.parse_args())

