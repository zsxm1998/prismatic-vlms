import torch
import torch.nn as nn


class BBoxEncoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )

    def forward(self, x: torch.Tensor):
        return self.encoder(x)
    

class BBoxDecoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 4)
        )

    def forward(self, x: torch.Tensor):
        return self.decoder(x).clamp(min=0, max=1)