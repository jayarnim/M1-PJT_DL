import torch
import torch.nn as nn
from .layers.linear import LinearLayer
from .layers.cin import CompressedInteractionNetworksLayer
from components.base import BaseModel
from components.feature_embedding import build as build_feature_embedding
from components.feature_map import FeatureMap


class CompressedInteractionNetworks(BaseModel):
    def __init__(
        self,
        embedding_dim: int, 
        channels: list[int],
        dropout: float,
        feature_map: FeatureMap,
    ):
        super().__init__(locals())

        self.weight = build_feature_embedding(
            embedding_dim=1,
            feature_map=feature_map,
        )
        self.linear = LinearLayer()

        self.embedding = build_feature_embedding(
            embedding_dim=embedding_dim,
            feature_map=feature_map,
        )
        self.cin = CompressedInteractionNetworksLayer(
            dim=embedding_dim,
            in_channels=len(feature_map),
            out_channels=channels,
            dropout=dropout,
        )

        self.bias = nn.Parameter(
            data=torch.zeros(1),
            requires_grad=True,
        )

    def forward(
        self, 
        X: torch.Tensor,
    ) -> torch.Tensor:
        weights = self.weight(X)
        main_effect = self.linear(weights)

        embeddings = self.embedding(X)
        interaction_effect = self.cin(embeddings)

        return self.bias + main_effect + interaction_effect