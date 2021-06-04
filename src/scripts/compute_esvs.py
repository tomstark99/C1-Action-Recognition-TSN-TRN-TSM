import argparse
import logging

from typing import Dict, Any, List, Tuple
from torch.utils.data import Dataset
from pathlib import Path

from torchvideo.samplers import frame_idx_to_list
from frame_sampling import RandomSampler

import torch
from gulpio2 import GulpDirectory

from omegaconf import OmegaConf

from systems import EpicActionRecognitionSystem
from systems import EpicActionRecogintionDataModule

import pickle
import numpy as np

from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Compute ESVs given a trained model",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("gulp_dir", type=Path, help="Path to gulp directory")
parser.add_argument("checkpoint", type=Path, help="Path to model checkpoint")
parser.add_argument("esvs_pickle", type=Path, help="Path to pickle file to save features")
parser.add_argument("--sample_n_frames", type=int, default=8, help="How many frames to sample to compute ESVs for")

def main(args):

    # TODO: Implement
    """
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    dtype = torch.float

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg = OmegaConf.create(ckpt['hyper_parameters'])
    OmegaConf.set_struct(cfg, False)

    cfg.data._root_gulp_dir = str(args.gulp_dir)

    model = EpicActionRecognitionSystem(cfg)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    """

    rgb_train = GulpDirectory(args.gulp_dir)

    n_frames = args.sample_n_frames
    frame_sampler = RandomSampler(frame_count=n_frames, snippet_length=1, test=True)

    def subsample_frames(video: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        video_length = len(video)
        if video_length < n_frames:
            raise ValueError(f"Video too short to sample {n_frames} from")
        sample_idxs = np.array(frame_idx_to_list(frame_sampler.sample(video_length)))
        return sample_idxs, video[sample_idxs]

    data = {
        "labels": [],
        "uids": [],
        "sequence_idxs": [],
        "sequence_lengths": [],
        "scores": [],
        "shapley_values": [],
    }

    for i, c in tqdm(
        enumerate(rgb_train),
        unit=" chunk",
        total=rgb_train.num_chunks,
        dynamic_ncols=True
    ):
        for video, rgb_meta in c:
            labels = {
                'verb': rgb_meta['verb_class'],
                'noun': rgb_meta['noun_class']
            }

            try:
                sample_idx, sample_video = subsample_frames(np.array(video))
            except ValueError:
                print(
                    f"{uid} is too short ({len(video)} frames) to sample {n_frames} "
                    f"frames from."
                )
                continue

            # TODO: implement scores from model idx out of bound error
            """
            with torch.no_grad():
                out = model(sample_video.to(device))
                out = out.cpu().numpy()
            """

            result_scores = torch.softmax(torch.rand((1,397)), dim=-1)

            scores = {
                'verb': result_scores[:,:97].numpy(),#.cpu().numpy(),
                'noun': result_scores[:,97:].numpy()#.cpu().numpy()
            }

            # TODO: implement compute esvs

            result_esvs = torch.softmax(torch.rand((1,n_frames,397)), dim=-1)

            esvs = {
                'verb': result_esvs[:,:,:97].numpy(),
                'noun': result_esvs[:,:,97:].numpy()
            }

            data["labels"].append(labels)
            data["uids"].append(rgb_meta['narration_id'])
            data["sequence_idxs"].append(sample_idx)
            data["sequence_lengths"].append(rgb_meta['num_frames'])
            data["scores"].append(scores)
            data["shapley_values"].append(esvs)
    
    def collate(vs: List[Any]):
        try:
            return np.stack(vs)
        except ValueError:
            return vs

    data_to_persist = {k: collate(vs) for k, vs in data.items()}

    with open(args.esvs_pickle, 'wb') as f:
        pickle.dump(data_to_persist, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main(parser.parse_args())
