import torch as t
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self, frame_count: int):
        super().__init__()
        self.frame_count = frame_count
        self.fc1 = nn.Linear(256 * frame_count, 512)
        self.fc2 = nn.Linear(512, 397)
    
    def forward(self, x):
        x = x.view(-1, 256 * self.frame_count)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
