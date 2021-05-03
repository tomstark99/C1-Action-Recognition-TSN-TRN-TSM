from typing import Dict, Any
from tqdm import tqdm
from gulpio2 import GulpDirectory

import numpy as np
import torch as t
import torch.nn as nn

from .pkl import PickleFeatureWriter

class FeatureExtractor:
    """
    Extracts image features from a 2D CNN backbone for every frame in all videos.
    """
    
    def __init__(
        self, 
        backbone_2d: nn.Module, 
        device: t.device, 
        dtype: t.float, 
        frame_batch_size: int = 128
    ):
        self.model = backbone_2d
        self.device = device
        self.dtype = dtype
        self.frame_batch_size = frame_batch_size
    
    def extract(self, dataset: GulpDirectory, feature_writer: PickleFeatureWriter) -> int:
        total_instances = 0
        self.model.eval()
        for i, c in tqdm(
            enumerate(dataset),
            unit=" chunk",
            total=dataset.num_chunks,
            dynamic_ncols=True
        ):
            if i > feature_writer.chunk_no:
                print(f"chunks_finished: {feature_writer.chunk_no}, "f"iteration: {i}")
                for j, (batch_input, batch_labels) in tqdm(
                    enumerate(c),
                    unit=" video",
                    total=len(c.meta_dict),
                    dynamic_ncols=True
                ):
                
                    batch_input = np.array(batch_input).transpose(0,3,1,2)
                    batch_input = t.tensor(batch_input, device=self.device, dtype=self.dtype)
                    batch_input = batch_input.unsqueeze(0)

                    batch_size, n_frames = batch_input.shape[:2]
                    flattened_batch_input = batch_input.view((-1, *batch_input.shape[2:]))

                    n_chunks = int(np.ceil(len(flattened_batch_input)/128))
                    chunks = t.chunk(flattened_batch_input, n_chunks, dim=0)
                    flatten_batch_features = []
                    for chunk in chunks:
                        chunk = chunk.unsqueeze(0)
                        with t.no_grad():
                            chunk_features = self.model.features(chunk.to(self.device))
                            chunk_features = self.model.new_fc(chunk_features)
                            flatten_batch_features.append(chunk_features.squeeze(0))
                    flatten_batch_features = t.cat(flatten_batch_features, dim=0)
                    batch_features = flatten_batch_features.view((batch_size, 
                                                                n_frames, 
                                                                *flatten_batch_features.shape[1:]))

                    total_instances += batch_size
                    self._append(batch_features, batch_labels, batch_size, feature_writer)
                feature_writer.save(chunk_no=i)
            else:
                pass
        return total_instances

    def _append(self, batch_features, batch_labels, batch_size, feature_writer):
        batch_narration_id = batch_labels['narration_id']
        assert len([batch_narration_id]) == batch_size
        assert len([batch_labels]) == batch_size
        assert len(batch_features) == batch_size
        batch_features = batch_features.squeeze(0).cpu().numpy()

        feature_writer.append(batch_narration_id, batch_features, batch_labels)

