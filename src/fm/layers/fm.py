import torch
import torch.nn as nn


class FactorizationMachineLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self, 
        embeddings: list[torch.Tensor],
    ) -> torch.Tensor:
        stacked = torch.stack(tensors=embeddings, dim=1)
        summed_square = torch.sum(input=stacked, dim=1) ** 2
        squared_sum = torch.sum(input=stacked ** 2, dim=1)
        predictive = 0.5 * (summed_square - squared_sum)
        return torch.sum(input=predictive, dim=1)