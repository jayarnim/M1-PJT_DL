import torch
import torch.nn as nn
from .components.linear import LinearLayer
from .components.fm import FactorizationMachineLayer
from .components.dnn import DeepNeuralNetworksLayer
from ctr.model.base import BaseModel
from ctr.model.feature_embedding import build_feature_embedding
from ctr.featuremap import FeatureMap


class DeepFactorizationMachine(BaseModel):
    def __init__(
        self,
        embedding_dim: int, 
        hidden_dim: list[int],
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
        self.fm = FactorizationMachineLayer()
        self.dnn = DeepNeuralNetworksLayer(
            input_dim=embedding_dim*len(feature_map),
            hidden_dim=hidden_dim,
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
        interaction_effect = (
            self.fm(embeddings) 
            + self.dnn(embeddings)
        )

        return self.bias + main_effect + interaction_effect