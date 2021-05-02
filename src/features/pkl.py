from typing import Dict, Any
from .feature_store import FeatureWriter
from pathlib import Path

import pickle
import numpy as np

class PickleFeatureWriter(FeatureWriter):
    
    def __init__(self, pkl_path: Path, features_dim: int):
        self.pkl_path = pkl_path
        self.features_dim = features_dim
        self.narration_ids = []
        self.features = []
        self.labels = []
        
    def append(self, narration_id: str, features: np.ndarray, labels: Dict[str, Any]) -> None:
        assert features.shape[1] == self.features_dim
        self.narration_ids.append(narration_id)
        self.features.append(features)
        self.labels.append(labels)
        
    def save(self):
        with open(self.pkl_path, 'wb') as f:
            pickle.dump({
                'narration_id': self.narration_ids,
                'features': np.concatenate(self.features),
                'labels': self.labels
            }, f)
