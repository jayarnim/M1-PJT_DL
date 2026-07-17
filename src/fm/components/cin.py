import torch
import torch.nn as nn


class CompressedInteractionNetworksLayer(nn.Module):
    def __init__(
        self, 
        dim: int,
        in_channels: int,
        out_channels: list[int],
        dropout: float,
        **kwargs, 
    ):
        super().__init__()

        kwargs = dict(
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
        )
        components = list(conv_block(**kwargs))
        self.cnn = nn.ModuleList(components)

        kwargs = dict(
            in_features=sum(out_channels),
            out_features=1,
        )
        self.linear = nn.Linear(**kwargs)

    def forward(
        self, 
        embeddings: list[torch.Tensor],
    ) -> torch.Tensor:
        X0 = torch.stack(tensors=embeddings, dim=1)
        H = X0

        Xs = []
        for block in self.cnn:
            Z = torch.einsum('bid,bjd->bijd', X0, H)    # (B,M,D) * (B,N,D) -> (B,M,N,D)
            B, M, N, D = Z.shape
            Z = Z.view(B, M * N, D)                     # (B,M,N,D) -> (B,K,D)
            H = block(Z)                                # (B,K,D)
            Xs.append(H.sum(dim=-1))                    # (B,K)

        concat = torch.cat(Xs, dim=-1)                  # (B,K*len(block))
        return self.linear(concat).squeeze(-1)          # (B,)


def conv_block(
    dim,
    in_channels,
    out_channels,
    dropout,
):
    IN_CHANNELS = in_channels * in_channels

    for channels in out_channels:
        kwargs = dict(
            in_channels=IN_CHANNELS,
            out_channels=channels,
            kernel_size=1,
        )

        NORM_SHAPE = (channels, dim)

        yield nn.Sequential(
            nn.Conv1d(**kwargs),
            nn.LayerNorm(NORM_SHAPE),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        IN_CHANNELS = in_channels * channels