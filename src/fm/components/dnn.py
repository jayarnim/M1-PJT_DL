import torch
import torch.nn as nn
from ctr.model.functions import fc_block


class DeepNeuralNetworksLayer(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        hidden_dim: list[int],
        dropout: float,
        **kwargs, 
    ):
        super().__init__()

        kwargs = dict(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        components = list(fc_block(**kwargs))
        self.mlp = nn.Sequential(*components)

        kwargs = dict(
            in_features=hidden_dim[-1],
            out_features=1,
        )
        self.linear = nn.Linear(**kwargs)

    def forward(
        self, 
        embeddings: list[torch.Tensor],
    ) -> torch.Tensor:
        concat = torch.cat(tensors=embeddings, dim=1)
        predictive = self.mlp(concat)
        return self.linear(predictive).squeeze(-1)