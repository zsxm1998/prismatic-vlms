import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class BBoxEncoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, hidden_size // 4),
            RMSNorm(hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, hidden_size // 2),
            RMSNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size),
            RMSNorm(hidden_size),
        )

    def forward(self, x: torch.Tensor):
        return self.encoder(x)
    

class BBoxDecoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            RMSNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            RMSNorm(hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 4),
        )

    def forward(self, x: torch.Tensor):
        return self.decoder(x).clamp(min=0, max=1)