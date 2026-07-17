import torch
import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self, 
        weights: list[torch.Tensor],
    ) -> torch.Tensor:
        concat = torch.cat(tensors=weights, dim=1)
        return torch.sum(input=concat, dim=1, keepdim=False)